"""Where a job's time goes, for its one "job finished" line in the server log (on a host, that log is the only place
to see it). The AI calls log their own times; this adds the steps around them, the CPU time the process used, and how
long the host held the CPU back. That tells slow storage or a slow start apart from a CPU the host keeps busy."""
from __future__ import annotations

import time
from pathlib import Path

PROC_CGROUP = Path("/proc/self/cgroup")  # which control group (cgroup) this process is in
CGROUP_ROOT = Path("/sys/fs/cgroup")  # where Linux shows each cgroup's counters


class Steps:
    """The wall time of each named step of one job, in order. A step runs from the previous mark (or the start) to
    its own mark."""

    def __init__(self) -> None:
        self.last = time.perf_counter()
        self.done: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.done.append((name, now - self.last))
        self.last = now

    def summary(self) -> str:
        return ", ".join(f"{name} {seconds:.1f}s" for name, seconds in self.done)


def _cpu_stat_files() -> list[Path]:
    """cpu.stat of this process's own cgroup (v2, then v1), then the top of the cgroup mounts, where a container
    usually sees its own cgroup."""
    files = []
    try:
        lines = PROC_CGROUP.read_text().splitlines()
    except (OSError, ValueError):  # not Linux, or unreadable
        lines = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _number, controllers, path = parts
        inside = path.strip().lstrip("/")
        if not controllers:  # cgroup v2: one tree for every controller
            files.append(CGROUP_ROOT / inside / "cpu.stat")
        elif "cpu" in controllers.split(","):  # cgroup v1: the cpu controller's own tree
            files += [CGROUP_ROOT / folder / inside / "cpu.stat" for folder in (controllers, "cpu", "cpu,cpuacct")]
    files += [CGROUP_ROOT / "cpu.stat", CGROUP_ROOT / "cpu" / "cpu.stat", CGROUP_ROOT / "cpu,cpuacct" / "cpu.stat"]
    return list(dict.fromkeys(files))


def held_back_seconds() -> float | None:
    """How long the host has held this process's CPU back so far (its cgroup's CPU limit), or None when it does not
    say."""
    for file in _cpu_stat_files():
        try:
            text = file.read_text()
        except (OSError, ValueError):  # missing, not allowed, or not text
            continue
        values = {}
        for line in text.splitlines():  # "throttled_usec 2500000"
            key, _space, value = line.partition(" ")
            values[key] = value.strip()
        try:
            if "throttled_usec" in values:  # cgroup v2, in microseconds
                return int(values["throttled_usec"]) / 1e6
            if "throttled_time" in values:  # cgroup v1, in nanoseconds
                return int(values["throttled_time"]) / 1e9
        except ValueError:
            continue
    return None


def cpu_now() -> tuple[float, float | None]:
    """(CPU seconds this process has used, seconds its CPU was held back, or None when the host does not say)."""
    return time.process_time(), held_back_seconds()


def report(steps: Steps, cpu_before: tuple[float, float | None]) -> str:
    """'setup 0.0s, AI 8.2s, ...; CPU used 0.4s, held back 6.0s'. CPU covers the whole process (any other request
    at the same time too); 'held back' only appears when the host says."""
    used, held = cpu_now()
    text = f"{steps.summary() or 'no steps'}; CPU used {used - cpu_before[0]:.1f}s"
    if held is not None and cpu_before[1] is not None:
        text += f", held back {held - cpu_before[1]:.1f}s"
    return text
