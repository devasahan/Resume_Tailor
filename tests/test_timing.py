"""Where a job's time goes, in the server log: one "job finished" line per tailoring or cover letter with each step's
time and the CPU used, plus a line for each slow database step or request, and the warm-up when the server starts.
On a host that log is the only place to see why a run was slow."""
import logging
import re

from app import main, prompts, timing
from app.llm.base import LLMClient, LLMError
from app.storage import db as db_module

from .conftest import SAMPLE_JD
from .test_api import save_profile

FINISHED = re.compile(r"(\w+) job finished in ([\d.]+)s: (.*); CPU used ([\d.]+)s(, held back ([\d.]+)s)?$")


def finished_lines(caplog, kind):
    return [FINISHED.match(r.getMessage()) for r in caplog.records if r.getMessage().startswith(f"{kind} job finished")]


def steps_of(line):
    """[(name, seconds), ...] from the line's step list."""
    return [(name, float(seconds)) for name, seconds in re.findall(r"([a-zA-Z][a-zA-Z ]*?) ([\d.]+)s", line.group(3))]


def test_a_tailoring_and_a_cover_letter_each_log_their_steps(client, caplog):
    save_profile(client)
    with caplog.at_level(logging.INFO, logger="app"):
        run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()
        client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})
    [tailored] = finished_lines(caplog, "tailor")
    assert [name for name, _ in steps_of(tailored)] == [
        "setup", "models", "thread wait", "bullet plan", "prompts", "AI", "layout", "checks", "saving"]
    assert sum(seconds for _, seconds in steps_of(tailored)) <= float(tailored.group(2)) + 0.1  # the steps add up
    [covered] = finished_lines(caplog, "cover_letter")
    assert [name for name, _ in steps_of(covered)] == [
        "reading the run", "cover prompts", "cover AI", "cost record", "checks", "saving"]


class Down(LLMClient):
    async def generate_structured(self, **kwargs):
        raise LLMError("the provider is down")


def test_a_failed_tailoring_logs_the_steps_it_reached(client, caplog):
    save_profile(client)
    client.app.state.router.overrides["demo-concise"] = Down()
    with caplog.at_level(logging.INFO, logger="app"):
        client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"})
    [line] = finished_lines(caplog, "tailor")
    names = [name for name, _ in steps_of(line)]
    assert names[:5] == ["setup", "models", "thread wait", "bullet plan", "prompts"]
    assert names[5] == "until it failed" and "AI" not in names  # the failed call is not reported as AI time


def cgroup(tmp_path, monkeypatch, membership, stat_path, stat):
    """A made-up /proc/self/cgroup and cpu.stat, as Linux shows them."""
    proc, root = tmp_path / "proc_cgroup", tmp_path / "cgroup"
    proc.write_text(membership)
    if stat_path is not None:
        (root / stat_path).mkdir(parents=True)
        (root / stat_path / "cpu.stat").write_text(stat)
    monkeypatch.setattr(timing, "PROC_CGROUP", proc)
    monkeypatch.setattr(timing, "CGROUP_ROOT", root)


def test_held_back_cpu_comes_from_the_process_own_cgroup_v2(tmp_path, monkeypatch):
    cgroup(tmp_path, monkeypatch, "0::/user.slice/site.scope\n", "user.slice/site.scope",
           "usage_usec 900000\nnr_throttled 12\nthrottled_usec 2500000\n")
    assert timing.held_back_seconds() == 2.5


def test_held_back_cpu_comes_from_the_cpu_controller_in_cgroup_v1(tmp_path, monkeypatch):
    cgroup(tmp_path, monkeypatch, "4:memory:/site\n3:cpu,cpuacct:/site\n", "cpu,cpuacct/site",
           "nr_periods 40\nnr_throttled 3\nthrottled_time 1500000000\n")
    assert timing.held_back_seconds() == 1.5


def test_a_container_sees_its_own_cgroup_at_the_top(tmp_path, monkeypatch):
    cgroup(tmp_path, monkeypatch, "3:cpu,cpuacct:/docker/0123abcd\n", "cpu", "throttled_time 250000000\n")
    assert timing.held_back_seconds() == 0.25  # the path from /proc is not mounted inside: the top is this cgroup


def test_without_cgroup_figures_the_line_shows_only_the_cpu_used(tmp_path, monkeypatch):
    cgroup(tmp_path, monkeypatch, "0::/\n", None, "")
    assert timing.held_back_seconds() is None
    steps = timing.Steps()
    steps.mark("setup")
    line = timing.report(steps, timing.cpu_now())
    assert re.fullmatch(r"setup [\d.]+s; CPU used [\d.]+s", line)


def test_a_cpu_held_back_during_the_job_is_reported(tmp_path, monkeypatch):
    cgroup(tmp_path, monkeypatch, "0::/\n", "", "throttled_usec 1000000\n")
    before = timing.cpu_now()
    (tmp_path / "cgroup" / "cpu.stat").write_text("throttled_usec 7500000\n")
    assert timing.report(timing.Steps(), before).endswith(", held back 6.5s")


def test_slow_database_steps_and_requests_are_named(client, caplog, monkeypatch):
    monkeypatch.setattr(db_module, "SLOW_DB_SECONDS", 0)
    monkeypatch.setattr(main, "SLOW_REQUEST_SECONDS", 0)
    with caplog.at_level(logging.INFO, logger="app"):
        client.get("/api/settings")
        client.get("/api/jobs/not-a-job")
    messages = [r.getMessage() for r in caplog.records]
    assert any(re.fullmatch(r"slow database step get_settings: [\d.]+s", m) for m in messages)
    assert any(re.fullmatch(r"slow request GET /api/settings: [\d.]+s", m) for m in messages)
    assert not any(m.startswith("slow request GET /api/jobs/") for m in messages)  # job checks wait on purpose


def test_quick_steps_and_requests_are_not_logged(client, caplog):
    with caplog.at_level(logging.INFO, logger="app"):
        client.get("/api/settings")
    assert not any(r.getMessage().startswith("slow") for r in caplog.records)


def test_the_warm_up_reads_the_prompts_and_loads_the_fonts_once(caplog):
    prompts._read.cache_clear()
    with caplog.at_level(logging.INFO, logger="app"):
        main.warm_up()
    files = len(list(prompts._DIR.glob("*.md")))
    assert prompts._read.cache_info().currsize == files
    assert any(re.fullmatch(rf"warm-up done in [\d.]+s \({files} prompt files, the fonts and the page layout code\)",
                            r.getMessage()) for r in caplog.records)


def test_a_failed_warm_up_never_stops_the_server(caplog, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("font file missing")

    monkeypatch.setattr(main, "render_resume_pdf", broken)
    with caplog.at_level(logging.INFO):
        main.warm_up()  # does not raise
    assert any("Warm-up failed" in r.getMessage() for r in caplog.records)
