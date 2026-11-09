"""Where a job's time goes, for its one "job finished" line in the server log (on a host, that log is the only place
to see it). The AI calls log their own times; this adds the steps around them and what the process went through
meanwhile:

- CPU used: the work itself.
- waited for a CPU: time the process's threads were ready to run but got no CPU (a busy shared machine, a lower
  priority or a CPU limit), from /proc/self/task/*/schedstat.
- quota throttled: time the CPU limit of the process's own cgroup held it back (cpu.stat).
- database and fonts loaded: how many database steps and font loads ran, and how long they took in all.

A step much longer than these explain was waiting on something else: storage outside the database and fonts, a lock,
or the host pausing the process. Each figure covers the whole process, so other requests at the same time count too.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

PROC_CGROUP = Path("/proc/self/cgroup")  # which control group (cgroup) this process is in
CGROUP_ROOT = Path("/sys/fs/cgroup")  # where Linux shows each cgroup's counters
PROC_TASKS = Path("/proc/self/task")  # one folder per thread of this process


class Steps:
    """The wall time of each named step of one job, in order. A step runs from the previous mark (or the start) to
    its own mark."""

    def __init__(self) -> None:
        self.started = self.last = time.perf_counter()
        self.done: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.done.append((name, now - self.last))
        self.last = now

    def total(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> str:
        return ", ".join(f"{name} {seconds:.1f}s" for name, seconds in self.done)


class Tally:
    """How many times something ran in this process and how long it took in all (database steps, font loads), so a
    job line can say how much of the job it was. Two threads adding at once may rarely lose one: it is a diagnostic."""

    def __init__(self, label: str, unit: str) -> None:
        self.label, self.unit = label, unit
        self.count, self.seconds = 0, 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.seconds += seconds


DATABASE = Tally("database", "steps")
FONTS = Tally("fonts loaded", "times")
TALLIES = (DATABASE, FONTS)


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, ValueError):  # missing, not allowed, or not text
        return None


def _cpu_stat_files() -> list[Path]:
    """The CPU counters of this process's own cgroup (on cgroup v2 cpu.stat.local first: from Linux 6.6 it also counts
    a parent's limit), then the top of the cgroup mounts, where a container usually sees its own cgroup."""
    files = []
    for line in (_read(PROC_CGROUP) or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _number, controllers, path = parts
        inside = path.strip().lstrip("/")
        if not controllers:  # cgroup v2: one tree for every controller
            files += [CGROUP_ROOT / inside / "cpu.stat.local", CGROUP_ROOT / inside / "cpu.stat"]
        elif "cpu" in controllers.split(","):  # cgroup v1: the cpu controller's own tree
            files += [CGROUP_ROOT / folder / inside / "cpu.stat" for folder in (controllers, "cpu", "cpu,cpuacct")]
    files += [CGROUP_ROOT / "cpu.stat", CGROUP_ROOT / "cpu" / "cpu.stat", CGROUP_ROOT / "cpu,cpuacct" / "cpu.stat"]
    return list(dict.fromkeys(files))


def throttled_seconds() -> float | None:
    """How long the CPU limit of this process's cgroup has held it back so far, or None when the host does not
    say."""
    for file in _cpu_stat_files():
        values = {}
        for line in (_read(file) or "").splitlines():  # "throttled_usec 2500000"
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


def waited_for_cpu_seconds() -> float | None:
    """How long this process's threads have been ready to run but waiting for a CPU, summed over the threads alive
    now (the second figure of each thread's schedstat, in nanoseconds), or None when Linux does not say."""
    try:
        threads = list(PROC_TASKS.iterdir())
    except OSError:
        return None
    total, seen = 0, False
    for thread in threads:
        fields = (_read(thread / "schedstat") or "").split()
        if len(fields) >= 2 and fields[1].isdigit():
            total, seen = total + int(fields[1]), True
    return total / 1e9 if seen else None


@dataclass(frozen=True)
class Snapshot:
    cpu: float
    waited: float | None
    throttled: float | None
    tallies: tuple[tuple[int, float], ...]


def snapshot() -> Snapshot:
    return Snapshot(time.process_time(), waited_for_cpu_seconds(), throttled_seconds(),
                    tuple((tally.count, tally.seconds) for tally in TALLIES))


def report(steps: Steps, before: Snapshot) -> str:
    """'setup 0.0s, AI 8.2s, ...; CPU used 0.4s, waited for a CPU 0.0s, quota throttled 0.0s; database 9 steps 0.1s;
    fonts loaded 4 times 0.2s'. A figure the host does not provide is left out."""
    now = snapshot()
    figures = [f"CPU used {now.cpu - before.cpu:.1f}s"]
    if now.waited is not None and before.waited is not None:  # a thread that ended takes its wait along: never < 0
        figures.append(f"waited for a CPU {max(0.0, now.waited - before.waited):.1f}s")
    if now.throttled is not None and before.throttled is not None:
        figures.append(f"quota throttled {max(0.0, now.throttled - before.throttled):.1f}s")
    parts = [steps.summary() or "no steps", ", ".join(figures)]
    for tally, (count_before, seconds_before), (count, seconds) in zip(TALLIES, before.tallies, now.tallies):
        parts.append(f"{tally.label} {count - count_before} {tally.unit} {seconds - seconds_before:.1f}s")
    return "; ".join(parts)
