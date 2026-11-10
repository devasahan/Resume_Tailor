"""Where a job's time goes, in the server log: one "job finished" line per tailoring or cover letter with each step's
time, the CPU used and waited for, and the database steps and font loads, plus a line for each slow database step or
request, and the warm-up when the server starts. On a host that log is the only place to see why a run was slow."""
import asyncio
import logging
import re
import sqlite3
import time

import pytest

from app import main, prompts, timing
from app.llm.base import LLMClient, LLMError

from .conftest import SAMPLE_JD
from .test_api import save_profile

FINISHED = re.compile(
    r"(?P<kind>\w+) job finished in (?P<total>[\d.]+)s: (?P<steps>.*); CPU used (?P<cpu>[\d.]+)s"
    r"(, waited for a CPU (?P<waited>[\d.]+)s)?(, quota throttled (?P<throttled>[\d.]+)s)?"
    r"; database (?P<db>\d+) steps (?P<db_s>[\d.]+)s; fonts loaded (?P<fonts>\d+) times (?P<fonts_s>[\d.]+)s$")


def finished_line(caplog, kind):
    [line] = [FINISHED.match(r.getMessage()) for r in caplog.records if r.getMessage().startswith(f"{kind} job finished")]
    assert line, "the job line does not have the expected shape"
    return line


def steps_of(line):
    """[(name, seconds), ...] from the line's step list."""
    return [(name, float(seconds)) for name, seconds in re.findall(r"([a-zA-Z][a-zA-Z ]*?) ([\d.]+)s", line["steps"])]


def add_up(line):
    """The printed steps add up to the printed total, each rounded to 0.1 s (so up to 0.05 s off each)."""
    steps = steps_of(line)
    assert abs(sum(seconds for _, seconds in steps) - float(line["total"])) <= 0.05 * (len(steps) + 1) + 1e-9


def test_a_tailoring_and_a_cover_letter_each_log_their_steps(client, caplog):
    save_profile(client)
    with caplog.at_level(logging.INFO, logger="app"):
        run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()
        client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})
    tailored = finished_line(caplog, "tailor")
    assert [name for name, _ in steps_of(tailored)] == [
        "setup", "models", "thread wait", "bullet plan", "prompts", "AI", "layout", "checks", "saving"]
    add_up(tailored)
    assert int(tailored["db"]) >= 3 and int(tailored["fonts"]) >= 2  # settings, models, saving; plan and layout
    covered = finished_line(caplog, "cover_letter")
    assert [name for name, _ in steps_of(covered)] == [
        "reading the run", "cover models", "cover prompts", "cover AI", "cost record", "reading the run again",
        "checks", "saving"]
    add_up(covered)


class Down(LLMClient):
    async def generate_structured(self, **kwargs):
        raise LLMError("the provider is down")


def test_a_failed_tailoring_logs_the_steps_it_reached(client, caplog):
    save_profile(client)
    client.app.state.router.overrides["demo-concise"] = Down()
    with caplog.at_level(logging.INFO, logger="app"):
        client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"})
    line = finished_line(caplog, "tailor")
    names = [name for name, _ in steps_of(line)]
    assert names == ["setup", "models", "thread wait", "bullet plan", "prompts", "until it failed", "saving"]
    add_up(line)  # the failed call is in 'until it failed', not reported as AI time


def test_work_that_stops_with_an_error_still_adds_up(client, caplog, monkeypatch):
    """An error the work does not catch (a locked database while saving, a bug) closes the open step, and the line
    is written before the job's own bookkeeping, so a slow database there never breaks the sum."""
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()

    async def stops_halfway(router, db, run_id, *, steps, **kwargs):
        steps.mark("reading the run")
        await asyncio.sleep(0.3)
        raise RuntimeError("database is locked")

    db, finish_job = client.app.state.db, client.app.state.db.finish_job

    def slow_finish(*args, **kwargs):
        time.sleep(0.3)
        return finish_job(*args, **kwargs)

    monkeypatch.setattr(main, "write_cover_letter", stops_halfway)
    monkeypatch.setattr(db, "finish_job", slow_finish)
    with caplog.at_level(logging.INFO, logger="app"), pytest.raises(RuntimeError):
        client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})
    line = finished_line(caplog, "cover_letter")
    steps = steps_of(line)
    assert [name for name, _ in steps] == ["reading the run", "until the error"] and steps[1][1] >= 0.2
    add_up(line)


def test_a_cover_letter_the_model_could_not_write_ends_at_saving(client, caplog):
    """The usual failure: the work saves what happened and reports the error itself, with every step marked."""
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()
    for spec in client.app.state.registry.all():
        client.app.state.router.overrides[spec.id] = Down()
    with caplog.at_level(logging.INFO, logger="app"):
        assert client.post(f"/api/runs/{run['run_id']}/cover-letter", json={}).status_code == 502
    line = finished_line(caplog, "cover_letter")
    assert [name for name, _ in steps_of(line)] == [
        "reading the run", "cover models", "cover prompts", "cover AI failed", "cost record", "reading the run again",
        "checks", "saving"]
    add_up(line)


def test_a_failed_save_after_a_failed_model_call_names_both(client, caplog, monkeypatch):
    save_profile(client)
    client.app.state.router.overrides["demo-concise"] = Down()

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(client.app.state.db, "save_run", locked)
    with caplog.at_level(logging.INFO, logger="app"), pytest.raises(sqlite3.OperationalError):
        client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"})
    line = finished_line(caplog, "tailor")
    assert [name for name, _ in steps_of(line)][-2:] == ["until it failed", "until the error"]
    add_up(line)


def fake_proc(tmp_path, monkeypatch, membership="0::/\n", stat_path=None, stat="", files=None):
    """A made-up /proc/self/cgroup and cgroup files, as Linux shows them."""
    proc, root = tmp_path / "proc_cgroup", tmp_path / "cgroup"
    proc.write_text(membership)
    for path, text in ({stat_path: stat} if stat_path is not None else {}).items() | (files or {}).items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    monkeypatch.setattr(timing, "PROC_CGROUP", proc)
    monkeypatch.setattr(timing, "CGROUP_ROOT", root)
    monkeypatch.setattr(timing, "PROC_TASKS", tmp_path / "no_tasks")


def test_quota_throttling_comes_from_the_process_own_cgroup_v2(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch, "0::/user.slice/site.scope\n", "user.slice/site.scope/cpu.stat",
              "usage_usec 900000\nnr_throttled 12\nthrottled_usec 2500000\n")
    assert timing.throttled_seconds() == 2.5


def test_cpu_stat_local_wins_because_it_also_counts_a_parent_limit(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch, "0::/site.scope\n", files={
        "site.scope/cpu.stat": "throttled_usec 0\n", "site.scope/cpu.stat.local": "throttled_usec 4000000\n"})
    assert timing.throttled_seconds() == 4.0


def test_quota_throttling_comes_from_the_cpu_controller_in_cgroup_v1(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch, "4:memory:/site\n3:cpu,cpuacct:/site\n", "cpu,cpuacct/site/cpu.stat",
              "nr_periods 40\nnr_throttled 3\nthrottled_time 1500000000\n")
    assert timing.throttled_seconds() == 1.5


def test_a_container_sees_its_own_cgroup_at_the_top(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch, "3:cpu,cpuacct:/docker/0123abcd\n", "cpu/cpu.stat", "throttled_time 250000000\n")
    assert timing.throttled_seconds() == 0.25  # the path from /proc is not mounted inside: the top is this cgroup


def threads(tmp_path, monkeypatch, *schedstats):
    tasks = tmp_path / "tasks"
    for number, text in enumerate(schedstats):
        (tasks / str(100 + number)).mkdir(parents=True)
        (tasks / str(100 + number) / "schedstat").write_text(text)
    monkeypatch.setattr(timing, "PROC_TASKS", tasks)
    return tasks


def test_the_wait_for_a_cpu_is_summed_over_the_threads(tmp_path, monkeypatch):
    threads(tmp_path, monkeypatch, "8000000 2500000000 40\n", "1000000 500000000 3\n", "not numbers\n")
    assert timing.waited_for_cpu_seconds() == 3.0


def test_a_cpu_wait_or_throttling_during_the_job_is_reported(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch, "0::/\n", "cpu.stat", "throttled_usec 1000000\n")
    tasks = threads(tmp_path, monkeypatch, "1 1000000000 1\n")
    before = timing.snapshot()
    (tmp_path / "cgroup" / "cpu.stat").write_text("throttled_usec 7500000\n")
    (tasks / "100" / "schedstat").write_text("1 7000000000 9\n")
    assert ", waited for a CPU 6.0s, quota throttled 6.5s;" in timing.report(timing.Steps(), before)


def test_a_thread_that_ends_takes_only_its_own_wait_along(tmp_path, monkeypatch):
    """Idle worker threads end on their own: one that waited 3 s long before the job must not hide the 2 s another
    thread waits during it."""
    fake_proc(tmp_path, monkeypatch)
    tasks = threads(tmp_path, monkeypatch, "1 3000000000 1\n", "1 1000000000 1\n")
    before = timing.snapshot()
    (tasks / "100" / "schedstat").unlink()
    (tasks / "100").rmdir()
    (tasks / "101" / "schedstat").write_text("1 3000000000 2\n")
    (tasks / "102").mkdir()
    (tasks / "102" / "schedstat").write_text("1 500000000 1\n")  # a thread started during the job counts from 0
    assert ", waited for a CPU 2.5s" in timing.report(timing.Steps(), before)


def test_a_thread_that_ended_never_makes_the_wait_negative(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch)
    tasks = threads(tmp_path, monkeypatch, "1 1000000000 1\n", "1 3000000000 1\n")
    before = timing.snapshot()
    (tasks / "101" / "schedstat").unlink()
    (tasks / "101").rmdir()
    assert ", waited for a CPU 0.0s" in timing.report(timing.Steps(), before)


def test_without_these_files_the_line_shows_what_is_known(tmp_path, monkeypatch):
    fake_proc(tmp_path, monkeypatch)
    assert timing.throttled_seconds() is None and timing.waited_for_cpu_seconds() is None
    steps = timing.Steps()
    steps.mark("setup")
    line = timing.report(steps, timing.snapshot())
    assert re.fullmatch(r"setup [\d.]+s; CPU used [\d.]+s; database 0 steps 0\.0s; fonts loaded 0 times 0\.0s", line)


def test_slow_database_steps_and_requests_are_named_by_route(client, caplog, monkeypatch):
    from app.storage import db as db_module

    monkeypatch.setattr(db_module, "SLOW_DB_SECONDS", 0)
    monkeypatch.setattr(main, "SLOW_REQUEST_SECONDS", 0)
    with caplog.at_level(logging.INFO, logger="app"):
        client.get("/api/settings")
        client.get("/static/app.js")
        client.get("/api/runs/987654/resume.pdf")
        client.get("/jane.doe%40example.com%0A2026-11-09 21:40:00,000 app.main: tailor job finished in 0.1s: forged")
        client.get("/api/jobs/not-a-job")
    messages = [r.getMessage() for r in caplog.records]
    assert any(re.fullmatch(r"slow database step get_settings: [\d.]+s", m) for m in messages)
    assert any(re.fullmatch(r"slow request GET /api/settings: [\d.]+s", m) for m in messages)
    assert any(re.fullmatch(r"slow request GET /static: [\d.]+s", m) for m in messages)  # the mounted folder
    assert any(re.fullmatch(r"slow request GET /api/runs/\{run_id\}/resume\.pdf: [\d.]+s", m) for m in messages)
    assert any(re.fullmatch(r"slow request GET \(no route\): [\d.]+s", m) for m in messages)  # never the address
    assert not any("jane" in m or "\n" in m or "987654" in m for m in messages)
    assert not any(m.startswith("slow request GET /api/jobs/") for m in messages)  # job checks wait on purpose


def test_quick_steps_and_requests_are_not_logged(client, caplog):
    with caplog.at_level(logging.INFO, logger="app"):
        client.get("/api/settings")
    assert not any(r.getMessage().startswith("slow") for r in caplog.records)


def test_the_warm_up_reads_the_prompts_and_loads_the_layout_code(caplog):
    prompts._read.cache_clear()
    loads = timing.FONTS.count
    with caplog.at_level(logging.INFO, logger="app"):
        main.warm_up()
    files = len(list(prompts._DIR.glob("*.md")))
    assert prompts._read.cache_info().currsize == files and timing.FONTS.count == loads + 1
    assert any(re.fullmatch(rf"warm-up done in [\d.]+s \({files} prompt files, the font and page layout code\)",
                            r.getMessage()) for r in caplog.records)


def test_a_failed_warm_up_never_stops_the_server(caplog, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("font file missing")

    monkeypatch.setattr(main, "render_resume_pdf", broken)
    with caplog.at_level(logging.INFO):
        main.warm_up()  # does not raise
    assert any("Warm-up failed" in r.getMessage() for r in caplog.records)
