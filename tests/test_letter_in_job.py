"""The cover letter is written inside the resume's job: it starts as soon as the resume text is final (no shorter draft
can come any more) and is written while the pages are laid out, so "Create resume & cover letter" is one wait of
about the resume's own time, and one run row holds both. A letter that cannot be written never fails the resume. Each
run also keeps where its time went (the admin's line on the Tailor page), and the draft no longer asks for the job's
responsibilities."""
import asyncio
import io
import json
import logging
import sqlite3
import threading
import time
from wsgiref.simple_server import make_server

import httpx
import pytest
from a2wsgi import ASGIMiddleware
from pypdf import PdfReader

from app import main
from app.llm.base import CostCapError, LLMClient, LLMResult, ProviderError, Usage
from app.llm.fake_client import FakeClient
from app.llm.schema import strict_json_schema
from app.main import create_app
from app.pipeline import fit, generate, tailor
from app.pipeline.tailor import analysis_terms, run_tailor, write_cover_letter
from app.prompts import tailor_system
from app.schemas import Company, Profile, TailorOutput, TailorResult
from app.storage.db import COVER_MARK, Database

from .conftest import ADMIN_EMAIL, ADMIN_PASSWORD, SAMPLE_JD, log_in
from .test_api import docx_bytes, save_profile
from .test_deploy import Quiet
from .test_jobs import answer
from .test_pipeline import Priced
from .test_time_limit import Hanging, SlowRewrite
from .test_timing import add_up, finished_line, steps_of


class Paced(LLMClient):
    """The demo model, each kind of answer taking its own time (`delays`); every call's kind, prompt, start and end
    (or that it was cancelled) are kept."""

    def __init__(self, inner, delays=None):
        self.inner, self.delays, self.calls = inner, delays or {}, []

    async def generate_structured(self, **kwargs):
        call = {"kind": kwargs["schema_name"], "user": kwargs["user"], "start": time.perf_counter(), "cancelled": False}
        self.calls.append(call)
        try:
            await asyncio.sleep(self.delays.get(kwargs["schema_name"], 0.0))
        except asyncio.CancelledError:
            call["cancelled"] = True
            raise
        result = await self.inner.generate_structured(**kwargs)
        call["end"] = time.perf_counter()
        return result

    def of(self, kind):
        return [call for call in self.calls if call["kind"] == kind]


def slow_layout(monkeypatch, seconds):
    """try_styles taking `seconds` longer (a slow host); each layout's start and end are kept."""
    real, times = tailor.try_styles, []

    def layout(*args, **kwargs):
        began = time.perf_counter()
        time.sleep(seconds)
        found = real(*args, **kwargs)
        times.append((began, time.perf_counter()))
        return found

    monkeypatch.setattr(tailor, "try_styles", layout)
    return times


def tailored(router, db, profile, model="demo-concise", **kwargs):
    return asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=model,
                                  save=False, **kwargs))


def career(jobs: int) -> Profile:
    return Profile(name="Avery Lindqvist", companies=[
        Company(name=f"Harbor Works {i}", start=f"Jan {2023 - 3 * i}",
                end="Present" if i == 0 else f"Dec {2025 - 3 * i}")
        for i in range(jobs)])


# ------------------------------------------------------------------ the letter runs alongside the layout
def test_the_letter_is_written_while_the_layout_runs(router, db, registry, profile, monkeypatch):
    """At the time limit no shorter draft fits after the draft, so the letter starts right after the AI step: the
    job then waits for the longer of the layout and the letter, not for both."""
    db.set_settings({"resume_time_limit_s": 5})
    client = Paced(FakeClient(registry.get("demo-concise")), {"tailor_output": 0.3, "cover_letter": 0.6})
    router.overrides["demo-concise"] = client
    layouts = slow_layout(monkeypatch, 0.5)
    result = tailored(router, db, profile, cover_letter=True)
    ended = time.perf_counter()
    assert result.status == "ok" and result.resume is not None and result.cover_letter is not None
    [draft], [letter] = client.of("tailor_output"), client.of("cover_letter")
    (layout_start, layout_end), *_ = layouts
    assert draft["end"] <= letter["start"] < layout_end  # written during the layout, after the draft
    assert min(letter["end"], layout_end) - max(letter["start"], layout_start) >= 0.4  # at the same time
    one_after_another = ((draft["end"] - draft["start"]) + (layout_end - layout_start)
                         + (letter["end"] - letter["start"]))
    assert ended - draft["start"] < one_after_another - 0.3  # clearly less than one after the other
    timing = result.timing
    wait = sum(seconds for name, seconds in timing.steps if name == "cover letter wait")
    assert timing.letter_seconds >= 0.55 and wait < timing.letter_seconds  # the admin's line says "alongside"
    assert [call.kind for call in timing.calls] == ["tailor_output", "cover_letter"]
    assert all(call.result == "ok" for call in timing.calls) and 0.55 <= timing.calls[1].seconds < 1.5
    assert tailor._LETTER_SECONDS["demo-concise"] == timing.calls[1].seconds  # what the next run keeps free


def test_a_letter_waits_for_the_shorter_draft_and_quotes_it(router, db, registry):
    """Without a time limit a draft too long for two pages is written again, shorter: a whole new draft, with new
    figures. The letter starts only after it and quotes the resume as it stays."""
    marker = ("Rebuilt the Quillmark ledger service with staged rollouts and clear ownership, so finance teams closed "
              "each month without manual fixes or late corrections.")

    class MarksTheShorterDraft(Paced):
        async def generate_structured(self, **kwargs):
            result = await super().generate_structured(**kwargs)
            if kwargs["schema_name"] == "tailor_output" and len(self.of("tailor_output")) == 2:
                result.data["sections"]["experience"][0]["bullets"][0] = marker
            return result

    client = MarksTheShorterDraft(FakeClient(registry.get("demo-detailed")))
    router.overrides["demo-detailed"] = client
    result = tailored(router, db, career(9), model="demo-detailed", cover_letter=True, time_limit=0)
    assert result.status == "ok" and "The model shortened the bullets to fit two pages." in result.notes
    assert [call["kind"] for call in client.calls] == ["tailor_output", "tailor_revision", "tailor_output",
                                                       "cover_letter"]  # the same order as before
    [letter], second = client.of("cover_letter"), client.of("tailor_output")[1]
    assert "Quillmark" in letter["user"] and letter["start"] >= second["end"]
    assert result.resume.experience[0].bullets[0] == marker and result.cover_letter is not None
    assert [name for name, _ in result.timing.steps][-4:] == ["layout", "shorter draft", "cover letter wait", "checks"]


def test_the_bake_off_keeps_its_order_and_keeps_no_time_free(router, db, registry, profile):
    """time_limit=0: nothing is kept free, and the calls come in the same order as before (draft, rewrite, letter)."""
    client = SlowRewrite(FakeClient(registry.get("demo-concise")), 0.2)  # a thin draft: the rewrite pass is asked
    router.overrides["demo-concise"] = client
    tailor._LETTER_SECONDS["demo-concise"] = 500.0  # would leave no time for the rewrite with a limit
    result = tailored(router, db, profile, cover_letter=True, time_limit=0)
    assert client.calls == ["tailor_output", "tailor_revision", "cover_letter"] and result.cover_letter is not None
    assert not any("time limit" in note for note in result.notes)


# ------------------------------------------------------------------ a letter that fails keeps the resume
class LetterFails(LLMClient):
    """The demo model, with every cover letter failing the way `how` says: a provider error, a bug, or no answer."""

    def __init__(self, inner, how):
        self.inner, self.how, self.letters = inner, how, 0

    async def generate_structured(self, **kwargs):
        if kwargs["schema_name"] == "cover_letter":
            self.letters += 1
            if self.how == "provider":
                raise ProviderError("HTTP 503: the model is overloaded", 503)
            if self.how == "bug":
                raise RuntimeError("an adapter bug")
            await asyncio.sleep(60)
        return await self.inner.generate_structured(**kwargs)


@pytest.mark.parametrize("how,note", [
    ("provider", "Cover letter failed: demo-concise: HTTP 503: the model is overloaded"),
    ("bug", "Cover letter failed: unexpected error: RuntimeError: an adapter bug"),
    ("hanging", "Cover letter failed: demo-concise: did not answer within 1 seconds"),
])
def test_a_letter_that_fails_keeps_the_resume(router, db, registry, profile, monkeypatch, how, note):
    monkeypatch.setattr(tailor, "SWITCH_AFTER_SECONDS", 1.0)
    monkeypatch.setattr(tailor, "LETTER_WAIT_SECONDS", 1.0)  # the resume waits for its letter only this long
    db.set_settings({"resume_time_limit_s": 1})
    client = LetterFails(FakeClient(registry.get("demo-concise")), how)
    router.overrides["demo-concise"] = client
    started = time.perf_counter()
    result = tailored(router, db, profile, cover_letter=True)
    assert time.perf_counter() - started < 4  # never the 60 s the hanging letter would take
    assert result.status == "ok" and result.resume is not None and result.cover_letter is None
    assert client.letters == 1 and note in result.notes
    assert result.attempts == 2  # the draft and the letter


def test_a_resume_that_fails_after_its_letter_started_stops_the_letter(router, db, registry, profile, monkeypatch):
    db.set_settings({"resume_time_limit_s": 5})  # the letter starts right after the AI step
    client = Paced(FakeClient(registry.get("demo-concise")), {"cover_letter": 5.0})
    router.overrides["demo-concise"] = client

    def broken(*args, **kwargs):
        time.sleep(0.2)  # the letter is on its way by now
        raise RuntimeError("the layout broke")

    monkeypatch.setattr(tailor, "try_styles", broken)

    async def run():
        result = await run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id="demo-concise",
                                  cover_letter=True, save=False)
        return result, [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    started = time.perf_counter()
    result, left = asyncio.run(run())
    assert time.perf_counter() - started < 3  # the 5 s letter was not waited for
    assert result.status == "error" and "the layout broke" in result.error and result.cover_letter is None
    assert left == []  # nothing is left running
    [letter] = client.of("cover_letter")
    assert letter["cancelled"]
    stopped = result.timing.calls[-1]
    assert (stopped.kind, stopped.result) == ("cover_letter", "stopped") and stopped.input > 100  # its prompt is billed
    assert result.input_tokens == sum(call.input for call in result.timing.calls)


def test_a_cancelled_job_stops_its_letter_too(router, db, registry, profile, monkeypatch):
    """Ctrl+C (or a host's restart) cancels the job: the letter it started is cancelled with it."""
    db.set_settings({"resume_time_limit_s": 5})
    client = Paced(FakeClient(registry.get("demo-concise")), {"cover_letter": 5.0})
    router.overrides["demo-concise"] = client
    slow_layout(monkeypatch, 1.0)

    async def run():
        job = asyncio.create_task(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                             model_id="demo-concise", cover_letter=True, save=False))
        while not client.of("cover_letter"):
            await asyncio.sleep(0.01)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    assert asyncio.run(run()) == [] and client.of("cover_letter")[0]["cancelled"]


# ------------------------------------------------------------------ the time a letter takes is kept free
def test_the_rewrite_pass_leaves_room_for_the_letter(router, db, registry, profile):
    """A letter remembered at 25 s leaves only 1 s of the 30 s limit: the rewrite pass (6 s at least) is skipped.
    Without a letter it runs. Each letter written updates the time remembered for its model."""
    tailor._LETTER_SECONDS["demo-concise"] = 25.0
    client = SlowRewrite(FakeClient(registry.get("demo-concise")), 0.3)
    router.overrides["demo-concise"] = client
    with_letter = tailored(router, db, profile, cover_letter=True)
    assert client.calls == ["tailor_output", "cover_letter"] and with_letter.cover_letter is not None
    assert any("rewrite pass was skipped" in note for note in with_letter.notes)
    assert tailor._LETTER_SECONDS["demo-concise"] < 1  # the demo's letter took no time
    client = SlowRewrite(FakeClient(registry.get("demo-concise")), 0.3)
    router.overrides["demo-concise"] = client
    tailor._LETTER_SECONDS["demo-concise"] = 25.0
    alone = tailored(router, db, profile)
    assert client.calls == ["tailor_output", "tailor_revision"] and alone.retries == 1
    assert tailor._LETTER_SECONDS["demo-concise"] == 25.0


def test_the_draft_cost_not_yet_saved_counts_for_the_letter(router, db, registry, profile, monkeypatch):
    """Spent today $0.10, the draft $0.022 (not saved until the run is), the cap $0.11: no letter is written."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    calls = []

    class Counted(Priced):
        async def generate_structured(self, **kwargs):
            calls.append(kwargs["schema_name"])
            return await super().generate_structured(**kwargs)

    router.overrides["claude-sonnet-5-5"] = Counted(FakeClient(registry.get("demo-concise")))
    db.save_run(TailorResult(requested_model="other", cost_usd=0.1), None, "")
    db.set_settings({"daily_cost_cap_usd": 0.11})
    result = tailored(router, db, profile, model="claude-sonnet-5-5", cover_letter=True)
    assert result.status == "ok" and result.resume is not None and result.cover_letter is None
    assert "cover_letter" not in calls and result.cost_usd >= 0.022
    assert any(note.startswith("Cover letter not written: Daily cost cap of $0.11 reached") for note in result.notes)


def test_the_cost_cap_check_counts_what_is_not_saved_yet(router, db):
    db.save_run(TailorResult(requested_model="other", cost_usd=1.0), None, "")
    db.set_settings({"daily_cost_cap_usd": 1.5})
    router.check_cost_cap()
    router.check_cost_cap(0.4)
    with pytest.raises(CostCapError):
        router.check_cost_cap(0.5)


# ------------------------------------------------------------------ Add / Rewrite cover letter
def test_add_cover_letter_waits_for_a_model_as_long_as_a_resume_does(router, db, registry, profile, monkeypatch):
    run = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id="demo-concise"))
    monkeypatch.setattr(tailor, "SWITCH_AFTER_SECONDS", 1.0)
    monkeypatch.setattr(tailor, "LAST_MODEL_SECONDS", 1.0)  # Add cover letter waits as long as a resume does
    db.set_settings({"resume_time_limit_s": 1})
    router.overrides["demo-concise"] = Hanging()
    started = time.perf_counter()
    fresh, error = asyncio.run(write_cover_letter(router, db, run.run_id, jd=SAMPLE_JD, model_id="demo-concise"))
    assert time.perf_counter() - started < 3 and fresh.cover_letter is None
    assert error == "Cover letter failed: demo-concise: did not answer within 1 seconds" and error in fresh.notes
    router.overrides["demo-concise"] = FakeClient(registry.get("demo-concise"))
    generate._RESTING_UNTIL.clear()
    fresh, error = asyncio.run(write_cover_letter(router, db, run.run_id, jd=SAMPLE_JD, model_id="demo-concise"))
    assert error is None and fresh.cover_letter is not None
    assert not any(note.startswith(tailor.LETTER_NOTES) for note in fresh.notes)  # the earlier failure is cleared


# ------------------------------------------------------------------ one job on the web
def test_one_job_brings_both_documents_with_its_timing_for_the_admin_only(client, user_client, caplog):
    save_profile(client)
    with caplog.at_level(logging.INFO, logger="app"):
        run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "cover_letter": True}).json()
    assert run["status"] == "ok" and run["cover_letter"]["paragraphs"] and run["resume"]
    line = finished_line(caplog, "tailor")
    names = [name for name, _ in steps_of(line)]
    assert names == ["setup", "models", "thread wait", "bullet plan", "prompts", "AI", "layout", "cover letter wait",
                     "checks", "saving"]
    add_up(line)
    timing = run["timing"]
    assert [name for name, _ in timing["steps"]] == names[:-1]  # saved before the saving step
    assert abs(sum(seconds for _, seconds in timing["steps"]) - timing["total_seconds"]) < 0.05
    assert [call["kind"] for call in timing["calls"]][-1] == "cover_letter"
    assert run["input_tokens"] == sum(call["input"] for call in timing["calls"])  # one row: the letter's tokens too
    rows = client.app.state.db.list_runs()
    assert len(rows) == 1 and rows[0]["cover_letter"] is True and rows[0]["task"] == "resume"
    with sqlite3.connect(client.app.state.db.path) as raw:
        stored = raw.execute("SELECT result FROM runs WHERE id = ?", (run["run_id"],)).fetchone()[0]
    assert stored.count(COVER_MARK) == 1  # History's has_cover finds the letter, never the timing record
    client.app.state.db.set_settings({"resume_model": "demo-concise", "cover_letter_model": "demo-concise"})
    save_profile(user_client)
    mine = user_client.post("/api/tailor", json={"jd": SAMPLE_JD, "cover_letter": True}).json()
    assert mine["status"] == "ok" and mine["cover_letter"]["paragraphs"] and "timing" not in mine
    assert "timing" not in user_client.get(f"/api/runs/{mine['run_id']}").json()
    saved = TailorResult.model_validate(client.get(f"/api/runs/{mine['run_id']}").json())
    assert set(main.public_run(saved)) == set(mine)
    assert client.get(f"/api/runs/{mine['run_id']}").json()["timing"]["calls"]  # the admin sees anyone's


def test_a_run_without_its_letter_is_not_marked_as_having_one(client):
    save_profile(client)
    client.app.state.router.overrides["demo-concise"] = LetterFails(
        FakeClient(client.app.state.registry.get("demo-concise")), "provider")
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "cover_letter": True}).json()
    assert run["status"] == "ok" and run["cover_letter"] is None
    assert run["timing"]["calls"][-1]["kind"] == "cover_letter" and run["timing"]["calls"][-1]["result"] == "failed"
    with sqlite3.connect(client.app.state.db.path) as raw:
        stored = raw.execute("SELECT result FROM runs WHERE id = ?", (run["run_id"],)).fetchone()[0]
    assert COVER_MARK not in stored and client.app.state.db.list_runs()[0]["cover_letter"] is False


def test_one_job_brings_both_documents_behind_the_web_tab_bridge_too(tmp_path, monkeypatch):
    """Path B on PythonAnywhere (WSGI through a2wsgi): the job, with its letter, goes on after the request answered."""
    monkeypatch.setattr(main, "JOB_WAIT_SECONDS", 0)
    db = Database(tmp_path / "w.db")
    db.set_settings({"parser_model": "rules"})
    server = make_server("127.0.0.1", 0, ASGIMiddleware(create_app(db=db)), handler_class=Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as http:
            log_in(http, ADMIN_EMAIL, ADMIN_PASSWORD)
            assert http.post("/api/resumes", files={"file": ("base.docx", docx_bytes())}).status_code == 200
            started = http.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "cover_letter": True})
            assert started.status_code == 202
            run = answer(http, started)
            assert run.status_code == 200 and run.json()["status"] == "ok"
            assert run.json()["cover_letter"]["paragraphs"] and run.json()["timing"]["letter_seconds"] >= 0
    finally:
        server.shutdown()


def test_the_pages_say_one_step_and_the_limit_covers_the_letter(client):
    page = client.get("/tailor").text
    assert "Writing your resume and cover letter" in page and "cover_letter: true" in page
    assert "The cover letter could not be written this time; press Add cover letter to try again." in page
    assert "style.display" in page and 'id="timing"' in page
    assert "Time limit for a resume and its cover letter" in client.get("/settings").text


# ------------------------------------------------------------------ where the time went
class Answers(LLMClient):
    """Gives each answer in turn: an exception is raised, a dict is returned with the usage given."""

    def __init__(self, *answers):
        self.answers, self.usage = list(answers), Usage(1200, 300, 100)

    async def generate_structured(self, **kwargs):
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if answer == "hang":
            await asyncio.sleep(60)
        return LLMResult(data=answer, usage=self.usage, served_model="m")


def parse_number(data):
    if "n" not in data:
        raise ValueError("no number")
    return data["n"]


def test_every_attempt_is_recorded_with_its_time_tokens_and_result(router):
    router.overrides["demo-detailed"] = Answers(ProviderError("HTTP 503", 503))
    router.overrides["demo-concise"] = Answers({"x": 1}, {"n": 7})
    out = asyncio.run(generate.generate_validated(
        router, ["demo-detailed", "demo-concise"], system="s", user="u", schema={}, schema_name="job_facts",
        parse=parse_number))
    assert out.value == 7
    assert [(c["model"], c["kind"], c["result"]) for c in out.calls] == [
        ("demo-detailed", "job_facts", "failed"), ("demo-concise", "job_facts", "unusable"),
        ("demo-concise", "job_facts", "ok")]
    assert out.calls[2] | {"seconds": 0} == {"model": "demo-concise", "kind": "job_facts", "seconds": 0, "input": 1200,
                                             "output": 300, "thinking": 100, "result": "ok"}
    assert out.calls[0]["input"] == 0 and len(out.calls) == out.attempts


def test_a_call_with_no_answer_in_time_and_a_stopped_follow_up_are_recorded(router, monkeypatch):
    monkeypatch.setattr(generate, "MIN_FOLLOW_UP_SECONDS", 0.1)
    router.overrides["demo-concise"] = Answers("hang")
    with pytest.raises(generate.GenerationFailed) as failed:
        asyncio.run(generate.generate_validated(
            router, ["demo-concise"], system="s" * 400, user="u" * 400, schema={}, schema_name="job_facts",
            parse=parse_number, last_after=0.2))
    [call] = failed.value.outcome.calls
    assert call["result"] == "no answer" and call["input"] == 200 and 0.15 <= call["seconds"] < 1  # its prompt
    router.overrides["demo-concise"] = Answers({"n": 1}, "hang")
    follow = generate.Ask(user="fix", schema={}, schema_name="job_facts_fix", parse=parse_number)
    out = asyncio.run(generate.generate_validated(
        router, ["demo-concise"], system="s", user="u", schema={}, schema_name="job_facts", parse=parse_number,
        refine=lambda value: follow, deadline=time.perf_counter() + 0.3))
    assert out.value == 1 and [(c["kind"], c["result"]) for c in out.calls] == [("job_facts", "ok"),
                                                                                 ("job_facts_fix", "stopped")]


def test_a_shorter_draft_stopped_at_the_limit_is_recorded_and_paid_for(router, db, registry, profile, monkeypatch):
    monkeypatch.setattr(tailor, "MIN_SHORTER_SECONDS", 0.1)
    monkeypatch.setattr(tailor, "CHECKS_RESERVE_SECONDS", 0.0)
    monkeypatch.setattr(generate, "MIN_FOLLOW_UP_SECONDS", 60)  # no rewrite pass
    db.set_settings({"resume_time_limit_s": 3})

    class SlowShorter(Paced):
        async def generate_structured(self, **kwargs):
            if kwargs["schema_name"] == "tailor_output" and self.of("tailor_output"):
                self.delays["tailor_output"] = 5.0  # the second draft
            return await super().generate_structured(**kwargs)

    router.overrides["demo-detailed"] = SlowShorter(FakeClient(registry.get("demo-detailed")))
    result = tailored(router, db, profile, model="demo-detailed")
    assert result.status == "ok" and result.fits
    assert any("shorter bullets ran past the time limit" in note for note in result.notes)
    first, stopped = result.timing.calls
    assert (stopped.kind, stopped.result) == ("tailor_output", "stopped") and 0.2 < stopped.seconds < 3.5
    assert stopped.input > 1000 and result.input_tokens == first.input + stopped.input


def test_an_older_run_without_a_timing_record_still_loads():
    old = json.loads(TailorResult(requested_model="demo-concise").model_dump_json())
    old.pop("timing")
    assert TailorResult.model_validate(old).timing is None


# ------------------------------------------------------------------ the draft asks for less
def test_the_draft_no_longer_asks_for_the_responsibilities():
    schema = strict_json_schema(TailorOutput)
    analysis = schema["properties"]["jd_analysis"]
    assert "key_responsibilities" not in analysis["properties"] and "key_responsibilities" not in analysis["required"]
    assert {"seniority", "domain", "must_have_skills", "keywords"} <= set(analysis["required"])
    assert "key_responsibilities" not in json.dumps(schema)
    assert "the main responsibilities" not in tailor_system()


def test_a_run_saved_with_responsibilities_still_loads_and_uses_them(router, db, profile):
    stored = json.dumps({
        "requested_model": "demo-concise", "status": "ok", "prompt_version": "v34",
        "analysis": {"target_role": "Payroll Analyst", "seniority": "Senior", "domain": "Payroll",
                     "must_have_skills": ["SQL"], "nice_to_have_skills": [], "keywords": ["Workday"],
                     "key_responsibilities": ["Run the monthly payroll close"]},
    })
    old = TailorResult.model_validate_json(stored)
    assert old.analysis.key_responsibilities == ["Run the monthly payroll close"] and old.timing is None
    assert "Run the monthly payroll close" in analysis_terms(old.analysis)
    run_id = db.save_run(old, None, "")
    assert db.get_run(run_id).analysis.key_responsibilities == ["Run the monthly payroll close"]
    new = tailored(router, db, profile)
    assert new.analysis.key_responsibilities == [] and new.prompt_version == "v35"


# ------------------------------------------------------------------ the fitted PDF reads like the preview
def test_the_fitted_pdf_is_titled_cv_in_eu_style(router, db, profile, monkeypatch):
    documents = []
    real = tailor.try_styles

    def layout(resume, paper, document="Resume"):
        documents.append(document)
        return real(resume, paper, document)

    monkeypatch.setattr(tailor, "try_styles", layout)
    for style in ("eu", "us"):
        assert tailored(router, db, profile, spelling=style).status == "ok"
    assert documents == ["CV", "Resume"]
    resume = tailored(router, db, profile).resume
    for pdf in (real(resume, "Letter", "CV")[2], fit.trim_to_fit(resume, "A4", "CV")[3],
                fit.render_smallest(resume, "Letter", "CV")[2]):
        reader = PdfReader(io.BytesIO(pdf))
        assert reader.metadata.title == "Jordan Rivera - CV" and reader.trailer["/Root"]["/Lang"] == "en-GB"
    assert PdfReader(io.BytesIO(real(resume, "Letter")[2])).metadata.title == "Jordan Rivera - Resume"


# ------------------------------------------------------------------ bounded waits (review cycle 7)
def test_the_resume_waits_for_its_letter_at_most_the_letter_wait_in_all(router, db, registry, profile, monkeypatch):
    """Two letter models (a backup in Settings), both silent: the finished resume is held LETTER_WAIT_SECONDS in all,
    not that long for each model or each attempt."""
    monkeypatch.setattr(tailor, "LETTER_WAIT_SECONDS", 2.0)  # each model alone would get 1 s and 2 s: 3 s in all
    db.set_settings({"fallback_models": ["demo-detailed"]})
    router.overrides["demo-concise"] = LetterFails(FakeClient(registry.get("demo-concise")), "hanging")
    router.overrides["demo-detailed"] = LetterFails(FakeClient(registry.get("demo-detailed")), "hanging")
    result = tailored(router, db, profile, cover_letter=True)
    assert result.status == "ok" and result.resume is not None and result.cover_letter is None
    assert 1.9 < result.timing.letter_seconds < 2.6  # one deadline for the whole letter
    assert any(note.startswith("Cover letter failed: ") and note.endswith("did not answer within 2 seconds")
               for note in result.notes)
    letters = [call for call in result.timing.calls if call.kind == "cover_letter"]
    assert letters and letters[-1].result in ("stopped", "no answer")  # the stopped call is counted (and billed)


def test_a_backup_letter_model_gets_its_turn_within_the_letter_wait(router, db, registry, profile, monkeypatch):
    monkeypatch.setattr(tailor, "LETTER_WAIT_SECONDS", 2.0)  # the backup is asked after half of it
    db.set_settings({"fallback_models": ["demo-detailed"]})
    router.overrides["demo-concise"] = LetterFails(FakeClient(registry.get("demo-concise")), "hanging")
    result = tailored(router, db, profile, cover_letter=True)
    assert result.status == "ok" and result.cover_letter is not None
    assert [call.model for call in result.timing.calls if call.kind == "cover_letter"] == ["demo-concise",
                                                                                           "demo-detailed"]


def test_with_no_time_limit_a_silent_rewrite_pass_still_ends(router, db, registry, profile, monkeypatch):
    """With Settings > Time limit 0 (and for a required fix) a follow-up has no deadline: its model's longest wait
    still ends it, and the first draft is kept."""
    monkeypatch.setattr(tailor, "LAST_MODEL_SECONDS", 1.0)
    client = SlowRewrite(FakeClient(registry.get("demo-concise")), 30.0)
    router.overrides["demo-concise"] = client
    started = time.perf_counter()
    result = tailored(router, db, profile, time_limit=0)
    assert time.perf_counter() - started < 4 and result.status == "ok"
    assert client.calls == ["tailor_output", "tailor_revision"]
    assert not any("time limit" in note for note in result.notes)


def test_with_no_time_limit_a_silent_shorter_draft_still_ends(router, db, registry, monkeypatch):
    monkeypatch.setattr(tailor, "LAST_MODEL_SECONDS", 1.0)

    class SilentShorterDraft(Paced):
        async def generate_structured(self, **kwargs):
            if kwargs["schema_name"] == "tailor_output" and len(self.of("tailor_output")) == 1:
                self.calls.append({"kind": "tailor_output", "start": time.perf_counter(), "end": None})
                await asyncio.sleep(60)  # the shorter draft never comes
            return await super().generate_structured(**kwargs)

    router.overrides["demo-detailed"] = SilentShorterDraft(FakeClient(registry.get("demo-detailed")))
    started = time.perf_counter()
    result = tailored(router, db, career(9), model="demo-detailed", time_limit=0)
    assert time.perf_counter() - started < 8 and result.status == "ok"
    assert "Asking the model for shorter bullets failed, so the text was trimmed instead." in result.notes
