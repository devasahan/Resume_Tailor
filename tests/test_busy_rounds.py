"""A busy model (Google's HTTP 503 "This model is currently experiencing high demand") hands over to the next model at
once. When every model was busy they take turns, the chosen model first, after short pauses, and after about a minute
and a half of busy answers the run stops with a plain message (app/pipeline/generate.py BUSY_ROUND_PAUSES). A busy
rewrite pass is asked again the same way. The owner's run waited 2:50 for two busy Google models (a minute of retries
each) and then failed."""
import asyncio
import json
import shutil
import subprocess
import time

import httpx
import pytest

from app.llm import router as router_module
from app.llm.base import InvalidOutputError, LLMClient, ProviderError
from app.llm.fake_client import FakeClient
from app.llm.openai_compat_client import OpenAICompatClient
from app.main import BUSY_MESSAGE, failure_message
from app.parsing.resume_parser import read_resume
from app.pipeline import generate, tailor
from app.pipeline.generate import Ask, GenerationFailed, generate_validated
from app.pipeline.tailor import run_tailor

from .conftest import SAMPLE_JD, SAMPLE_RESUME_TEXT

JD = "A remote job paying $100,000 a year."
HIGH_DEMAND = ("This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try "
               "again later.")
MAIN, BACKUP = "gemini-3.8-flash", "gemini-3.5-flash-lite"


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    """Placeholder key names so the real models count as set up; every call goes to a scripted client instead."""
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")


@pytest.fixture
def clock(monkeypatch):
    """The pauses between rounds pass at once: each moves a fake clock on and is recorded."""
    state = {"now": 1000.0, "pauses": [], "order": []}

    async def pause(seconds):
        state["pauses"].append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(generate, "_sleep", pause)
    monkeypatch.setattr(generate, "_clock", lambda: state["now"])
    return state


def busy(label="Gemini 3.8 Flash", status=503):
    return ProviderError(f"{label}: HTTP {status}: {HIGH_DEMAND}", status, busy=True)


class Script(LLMClient):
    """Answers each call with the next item (the last one repeats): an exception is raised, "ok" is the demo model's
    answer. Each call takes `seconds` on the fake clock (Google took about 6 s to say "busy")."""

    def __init__(self, registry, name, items, clock, seconds=0.0):
        self.inner = FakeClient(registry.get("demo-concise"))
        self.name, self.items, self.clock, self.seconds, self.calls = name, list(items), clock, seconds, 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        self.clock["order"].append(self.name)
        self.clock["now"] += self.seconds
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return await self.inner.generate_structured(**kwargs)


def scripted(router, registry, clock, main, backup, seconds=0.0):
    router.overrides[MAIN] = Script(registry, MAIN, main, clock, seconds)
    router.overrides[BACKUP] = Script(registry, BACKUP, backup, clock, seconds)
    return router.overrides[MAIN], router.overrides[BACKUP]


def ask(router, chain=(MAIN, BACKUP), **kwargs):
    return asyncio.run(generate_validated(
        router, list(chain), system="s", user="u", schema={}, schema_name="job_facts",
        parse=kwargs.pop("parse", lambda data: data), context={"jd": JD}, **kwargs))


def test_a_busy_model_hands_over_at_once(router, registry, clock):
    main, backup = scripted(router, registry, clock, [busy()], ["ok"])
    out = ask(router)
    assert out.final_model_id == BACKUP and out.chain == [MAIN, BACKUP]
    assert clock["pauses"] == [] and main.calls == 1 and backup.calls == 1  # no minute of retries first


def test_the_chosen_model_answers_as_soon_as_it_is_free(router, registry, clock):
    scripted(router, registry, clock, [busy(), "ok"], [busy("Gemini 3.5 Flash-Lite")])
    out = ask(router)
    assert out.final_model_id == MAIN and clock["order"] == [MAIN, BACKUP, MAIN] and clock["pauses"] == [2.0]
    assert out.chain == [MAIN, BACKUP] and out.attempts == 3 and len(out.errors) == 2  # each model listed once


def test_while_every_model_stays_busy_they_take_turns_then_stop_with_a_plain_message(router, registry, clock):
    scripted(router, registry, clock, [busy()], [busy("Gemini 3.5 Flash-Lite")], seconds=6.0)
    with pytest.raises(GenerationFailed) as info:
        ask(router)
    # 12 s a round; no round starts when its pause would end past 100 s: 92 s in all instead of 2:50
    assert clock["pauses"] == [2.0, 5.0, 10.0, 15.0] and clock["now"] - 1000 == 92
    assert clock["order"] == [MAIN, BACKUP] * 5  # the chosen model first in every round
    text = str(info.value)
    assert text.startswith(
        "Google's AI service is overloaded right now: Gemini 3.8 Flash and Gemini 3.5 Flash-Lite answered \"busy\" "
        "(HTTP 503) every time they were asked (5 times each in 1 min 32 s). This is on Google's side, not a problem in "
        "the app: try again in a few minutes. A backup model from another company (Settings > Fallback models) keeps "
        "the app working when Google is busy. Last answers: gemini-3.8-flash: Gemini 3.8 Flash: HTTP 503: This model "
        "is currently experiencing high demand.")
    assert text.count("HTTP 503: This model") == 2 and len(text) <= 900  # the run keeps 900 characters
    assert len(info.value.outcome.errors) == 10 and info.value.outcome.chain == [MAIN, BACKUP]
    assert failure_message(text) == BUSY_MESSAGE  # users see only "busy, try again in a few minutes"


def test_with_no_time_left_for_a_pause_no_round_is_added(router, registry, clock):
    scripted(router, registry, clock, [busy()], [busy("Gemini 3.5 Flash-Lite")], seconds=6.0)
    with pytest.raises(GenerationFailed) as info:
        ask(router, busy_give_up=10.0)
    assert clock["pauses"] == [] and clock["order"] == [MAIN, BACKUP]
    assert str(info.value).startswith("Google's AI service is overloaded right now: Gemini 3.8 Flash and Gemini 3.5 "
                                      "Flash-Lite answered \"busy\" (HTTP 503). This is on Google's side")


def test_a_model_that_failed_another_way_is_not_asked_again(router, registry, clock):
    refused = ProviderError("Gemini 3.8 Flash: HTTP 400: API key not valid. Please pass a valid API key.", 400)
    main, backup = scripted(router, registry, clock, [refused], [busy("Gemini 3.5 Flash-Lite"), busy(), "ok"])
    out = ask(router)
    assert out.final_model_id == BACKUP and main.calls == 1 and backup.calls == 3 and clock["pauses"] == [2.0, 5.0]


def test_a_used_up_quota_still_rests_and_is_not_asked_again(router, registry, clock):
    quota = ProviderError("Gemini 3.8 Flash: the API key's quota is used up", 429, quota=True)
    main, _backup = scripted(router, registry, clock, [quota], [busy("Gemini 3.5 Flash-Lite"), "ok"])
    out = ask(router)
    assert out.final_model_id == BACKUP and main.calls == 1 and generate.resting(MAIN)


FIX = Ask(user="fix", schema={}, schema_name="job_facts", parse=lambda data: "fixed")


def test_a_busy_follow_up_is_asked_again_after_a_pause(router, registry, clock):
    """The rewrite pass answered "busy" once: it is asked again 2 s later, as the minute of retries inside the call
    once did, so one 503 does not drop the pass (or the fix that takes a certification name out)."""
    main, backup = scripted(router, registry, clock, ["ok", busy(), "ok"], ["ok"])
    out = ask(router, parse=lambda data: "draft", refine=lambda value: FIX)
    assert out.value == "fixed" and out.final_model_id == MAIN and clock["pauses"] == [2.0]
    assert main.calls == 3 and backup.calls == 0 and out.attempts == 3


def test_a_required_follow_up_is_asked_again_even_past_the_time_limit(router, registry, clock):
    required = Ask(user="fix", schema={}, schema_name="job_facts", parse=lambda data: "fixed", required=True)
    main, _backup = scripted(router, registry, clock, ["ok", busy(), busy(), "ok"], ["ok"])
    out = ask(router, parse=lambda data: "draft", refine=lambda value: required, deadline=time.perf_counter() - 1)
    assert out.value == "fixed" and clock["pauses"] == [2.0, 5.0] and main.calls == 4


def test_a_follow_up_that_stays_busy_keeps_the_first_answer(router, registry, clock):
    main, _backup = scripted(router, registry, clock, ["ok", busy()], ["ok"], seconds=6.0)
    out = ask(router, parse=lambda data: "draft", refine=lambda value: FIX)
    # 6 s per busy answer: no pause starts that would end past 100 s of busy answers and pauses
    assert out.value == "draft" and out.final_model_id == MAIN and clock["pauses"] == [2.0, 5.0, 10.0, 15.0, 20.0]
    assert main.calls == 7 and out.follow_up == ""


def test_a_busy_follow_up_under_the_time_limit_is_asked_again_only_when_it_still_fits(router, registry, clock):
    main, _backup = scripted(router, registry, clock, ["ok", busy(), "ok"], ["ok"])
    out = ask(router, parse=lambda data: "draft", refine=lambda value: FIX, deadline=time.perf_counter() + 60)
    assert out.value == "fixed" and clock["pauses"] == [2.0] and main.calls == 3
    # 7 s left: a pause of 2 s and a follow-up of at least 6 s would not finish in time, so the first answer is kept
    clock["pauses"].clear()
    main, _backup = scripted(router, registry, clock, ["ok", busy(), "ok"], ["ok"])
    out = ask(router, parse=lambda data: "draft", refine=lambda value: FIX, deadline=time.perf_counter() + 7)
    assert out.value == "draft" and clock["pauses"] == [] and main.calls == 2


def test_a_busy_follow_up_whose_pause_ends_past_the_time_limit_is_not_sent(router, registry, monkeypatch):
    """The event loop stood still during the pause (on the Web-tab site it moves only while a request is served), so
    the pause ended past the time limit: the follow-up is not sent again only to be stopped at once, and nothing more is
    billed or shown as a stopped call."""
    monkeypatch.setattr(generate, "MIN_FOLLOW_UP_SECONDS", 0.3)
    monkeypatch.setattr(generate, "BUSY_ROUND_PAUSES", (0.05,) * 6)
    deadline = time.perf_counter() + 1.5

    async def late(seconds):
        await asyncio.sleep(deadline - time.perf_counter() + 0.05)

    monkeypatch.setattr(generate, "_sleep", late)
    main = router.overrides[MAIN] = Script(registry, MAIN, ["ok", busy(), "ok"], {"now": 0.0, "order": []})
    out = ask(router, parse=lambda data: "draft", refine=lambda value: FIX, deadline=deadline)
    assert out.value == "draft" and out.follow_up == "stopped" and main.calls == 2
    assert [call["result"] for call in out.calls] == ["ok", "failed"]
    assert out.input_tokens == out.calls[0]["input"]  # only the first answer's prompt


def test_time_spent_on_other_failures_leaves_the_busy_time_alone(router, registry, clock):
    """The chosen model said "busy" once; the backup then took 94 s to give two unusable answers. The chosen model is
    still asked again: only busy answers and pauses count toward the 100 s."""
    router.overrides[MAIN] = Script(registry, MAIN, [busy(), "ok"], clock, seconds=6.0)
    unusable = InvalidOutputError("Gemini 3.5 Flash-Lite returned an empty answer")
    router.overrides[BACKUP] = Script(registry, BACKUP, [unusable], clock, seconds=47.0)
    out = ask(router)
    assert out.final_model_id == MAIN and clock["order"] == [MAIN, BACKUP, BACKUP, MAIN]
    assert clock["pauses"] == [2.0] and clock["now"] - 1000 == 108


def test_a_turn_that_ends_busy_counts_in_full(router, registry, clock):
    """Each round the chosen model gave an unusable answer (30 s), then "busy" when asked again (30 s): it is asked
    again in the next round, so its whole turn counts toward the 100 s, and the step stops after 2 rounds (2 min 6 s),
    not after 3 (3 min 13 s)."""
    unusable = InvalidOutputError("Gemini 3.8 Flash returned an empty answer")
    router.overrides[MAIN] = Script(registry, MAIN, [unusable, busy()] * 6, clock, seconds=30.0)
    router.overrides[BACKUP] = Script(registry, BACKUP, [busy("Gemini 3.5 Flash-Lite")], clock, seconds=2.0)
    with pytest.raises(GenerationFailed) as info:
        ask(router)
    assert clock["pauses"] == [2.0] and clock["now"] - 1000 == 126
    assert clock["order"] == [MAIN, MAIN, BACKUP] * 2
    assert str(info.value).startswith("Google's AI service is overloaded right now")


def test_one_round_keeps_every_error_in_order(router, registry, clock):
    refused = ProviderError("Gemini 3.8 Flash: HTTP 400: API key not valid.", 400)
    scripted(router, registry, clock, [refused], [busy("Gemini 3.5 Flash-Lite")])
    with pytest.raises(GenerationFailed) as info:
        ask(router, busy_give_up=0.0)
    assert str(info.value) == (f"{MAIN}: Gemini 3.8 Flash: HTTP 400: API key not valid.; {BACKUP}: Gemini 3.5 "
                               f"Flash-Lite: HTTP 503: {HIGH_DEMAND}")


def test_after_more_rounds_each_model_is_named_once_with_how_often_it_was_asked(router, registry, clock):
    refused = ProviderError("Gemini 3.8 Flash: HTTP 400: API key not valid.", 400)
    scripted(router, registry, clock, [busy(), refused], [busy("Gemini 3.5 Flash-Lite")], seconds=6.0)
    with pytest.raises(GenerationFailed) as info:
        ask(router)
    assert str(info.value) == (f"{MAIN}: Gemini 3.8 Flash: HTTP 400: API key not valid. (asked 2 times); {BACKUP}: "
                               f"Gemini 3.5 Flash-Lite: HTTP 503: {HIGH_DEMAND} (asked 6 times)")


def test_models_of_two_companies_busy_together(router, registry, clock):
    router.overrides["claude-sonnet-5-5"] = Script(registry, "claude", [busy("Anthropic API error", 529)], clock)
    router.overrides[MAIN] = Script(registry, MAIN, [busy()], clock)
    with pytest.raises(GenerationFailed) as info:
        ask(router, chain=(MAIN, "claude-sonnet-5-5"))
    text = str(info.value)
    assert text.startswith("The AI services are overloaded right now: Gemini 3.8 Flash and Claude Sonnet 5.5 answered "
                           "\"busy\" (HTTP 503 and 529) every time they were asked (7 times each in 1 min 17 s). This is "
                           "on the AI companies' side, not a problem in the app: try again in a few minutes. Last "
                           "answers: ")
    assert "A backup model from another company" not in text  # there is one already


def limited(label="Gemini 3.8 Flash"):
    return ProviderError(f"{label}: HTTP 429: Too many requests, slow down.", 429, busy=True)


def test_too_many_requests_is_called_an_overload_or_the_keys_rate_limit(router, registry, clock):
    """Some providers answer 429 when the key's rate limit is reached, others when they are overloaded ("The service
    may be temporarily overloaded"): the app cannot tell which, so the message names both."""
    scripted(router, registry, clock, [limited()], [limited("Gemini 3.5 Flash-Lite")])
    with pytest.raises(GenerationFailed) as info:
        ask(router, busy_give_up=0.0)
    assert str(info.value).startswith(
        "Google's AI service is overloaded or limiting this app's requests right now: Gemini 3.8 Flash and Gemini 3.5 "
        "Flash-Lite answered \"too many requests\" (HTTP 429). This is on Google's side (an overload, or the rate "
        "limit of the app's API key), not a problem in the app: try again in a few minutes. A backup model from "
        "another company (Settings > Fallback models) keeps the app working when Google is busy. Last answers: ")

    router.overrides["claude-sonnet-5-5"] = Script(registry, "claude", [limited("Anthropic API error")], clock)
    router.overrides[MAIN] = Script(registry, MAIN, [busy()], clock)
    with pytest.raises(GenerationFailed) as info:
        ask(router, chain=(MAIN, "claude-sonnet-5-5"), busy_give_up=0.0)
    assert str(info.value).startswith(
        "The AI services are overloaded or limiting this app's requests right now: Gemini 3.8 Flash and Claude Sonnet "
        "5.5 answered \"busy\" (HTTP 429 and 503). This is on the AI companies' side (an overload, or the rate limit "
        "of the app's API key), not a problem in the app: try again in a few minutes. Last answers: ")


def test_a_backup_resting_for_its_quota_is_named_instead_of_advising_one(router, registry, clock):
    """GPT-6.1 Sol (another company) is ticked as a backup but its quota is used up: the message says so, and does not
    advise adding a backup from another company, which is there already."""
    generate._rest("gpt-6.1-sol", "its API key's quota is used up", 3600)
    scripted(router, registry, clock, [busy()], [busy("Gemini 3.5 Flash-Lite")])
    with pytest.raises(GenerationFailed) as info:
        ask(router, chain=(MAIN, "gpt-6.1-sol", BACKUP), busy_give_up=0.0)
    text = str(info.value)
    assert text.startswith(
        "Google's AI service is overloaded right now: Gemini 3.8 Flash and Gemini 3.5 Flash-Lite answered \"busy\" "
        "(HTTP 503). This is on Google's side, not a problem in the app: try again in a few minutes. GPT-6.1 Sol was "
        "not asked, because its API key's quota is used up for now. Last answers: ")
    assert "A backup model from another company" not in text


def test_through_the_real_gemini_client(router, registry, clock):
    """The real adapter: Google's 503 is not retried inside the call (no sleeps there), the backup is asked at once,
    and the chosen model's answer is used as soon as it is free."""
    replies = {MAIN: [503, 200], BACKUP: [503]}
    sleeps = []

    async def no_sleep(seconds):
        sleeps.append(seconds)

    def client(model_id):
        def handler(request):
            clock["order"].append(model_id)
            codes = replies[model_id]
            code = codes.pop(0) if len(codes) > 1 else codes[0]
            if code == 503:
                return httpx.Response(503, json={"error": {"code": 503, "message": HIGH_DEMAND, "status": "UNAVAILABLE"}})
            return httpx.Response(200, json={"model": model_id, "usage": {"prompt_tokens": 9, "completion_tokens": 3},
                                             "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}]})
        return OpenAICompatClient(registry.get(model_id), transport=httpx.MockTransport(handler), sleep=no_sleep)

    router.overrides[MAIN], router.overrides[BACKUP] = client(MAIN), client(BACKUP)
    out = ask(router)
    assert out.value == {"a": 1} and out.final_model_id == MAIN
    assert clock["order"] == [MAIN, BACKUP, MAIN] and clock["pauses"] == [2.0] and sleeps == []


def test_a_resume_while_google_is_busy(router, registry, db, profile, clock):
    """End to end, with the owner's setup (Gemini 3.8 Flash, Gemini 3.5 Flash-Lite as backup)."""
    db.set_settings({"fallback_models": [BACKUP]})
    scripted(router, registry, clock, [busy()], ["ok"])
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=MAIN,
                                    save=False))
    assert result.status == "ok" and result.fallback_chain == [MAIN, BACKUP] and clock["pauses"] == []
    assert any(note.startswith(f"{MAIN} could not be used (Gemini 3.8 Flash: HTTP 503: This model is currently "
                               f"experiencing high demand.") and note.endswith(f"so {BACKUP} made this resume.")
               for note in result.notes)

    scripted(router, registry, clock, [busy()], [busy("Gemini 3.5 Flash-Lite")], seconds=6.0)
    failed = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=MAIN,
                                    save=False))
    assert failed.status == "error" and failed.error.startswith("Google's AI service is overloaded right now")


def test_the_upload_reader_waits_less_then_uses_the_rules(router, registry, db, clock):
    router.overrides[BACKUP] = Script(registry, BACKUP, [busy("Gemini 3.5 Flash-Lite")], clock, seconds=6.0)
    profile, method, notes = asyncio.run(read_resume(router, db, SAMPLE_RESUME_TEXT, BACKUP))
    assert method == "rules" and profile.name == "Jordan Rivera"
    assert clock["pauses"] == [2.0, 5.0, 10.0] and clock["now"] - 1000 == 41  # 45 s at most, then the rules
    assert notes[0].startswith("The model could not parse the resume (Google's AI service is overloaded right now")


def test_settings_says_to_add_another_company(client):
    page = client.get("/settings").text
    assert "When that company's AI service is busy (for example Google's \"high demand\")" in page
    assert "Paste another company's API key under API keys above" in page


class ByKind(LLMClient):
    """Answers each kind of request (its schema_name) with the next item of that kind's list (the last one repeats):
    an exception is raised, "ok" is the demo model's answer, ("slow", s) is that answer after s seconds."""

    def __init__(self, registry, scripts):
        self.inner = FakeClient(registry.get("demo-concise"))
        self.scripts, self.sent = {kind: list(items) for kind, items in scripts.items()}, []

    async def generate_structured(self, **kwargs):
        items = self.scripts[kwargs["schema_name"]]
        item = items.pop(0) if len(items) > 1 else items[0]
        self.sent.append(kwargs["schema_name"])
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            await asyncio.sleep(item[1])
        return await self.inner.generate_structured(**kwargs)


def test_a_letter_stopped_in_a_later_round_names_and_bills_the_model_on_its_way(router, registry, db, profile,
                                                                                monkeypatch):
    """The letter's chosen model was busy and its backup refused, so the chosen model was asked again after a pause,
    and was still writing when the letter's time ran out: the note and the timing line name that model (not the
    backup, the last one listed), for the time its call ran (not the pause before it)."""
    db.set_settings({"fallback_models": [BACKUP]})
    monkeypatch.setattr(generate, "BUSY_ROUND_PAUSES", (0.4,) * 6)
    monkeypatch.setattr(tailor, "LETTER_WAIT_SECONDS", 1.0)
    refused = ProviderError("Gemini 3.5 Flash-Lite: HTTP 400: API key not valid.", 400)
    router.overrides[MAIN] = ByKind(registry, {"tailor_output": ["ok"], "tailor_revision": ["ok"],
                                               "cover_letter": [busy(), ("slow", 5.0)]})
    router.overrides[BACKUP] = ByKind(registry, {"cover_letter": [refused]})
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=MAIN,
                                    save=False, cover_letter=True, time_limit=0))
    assert result.status == "ok" and result.cover_letter is None
    assert f"Cover letter failed: {MAIN}: did not answer within 1 seconds" in result.notes
    letters = [(call.model, call.result) for call in result.timing.calls if call.kind == "cover_letter"]
    assert letters == [(MAIN, "failed"), (BACKUP, "failed"), (MAIN, "stopped")]
    stopped = result.timing.calls[-1]
    assert stopped.input > 0 and stopped.seconds < 0.9  # about 0.6 s: the 0.4 s pause was not on its way


def test_a_letter_whose_time_runs_out_between_busy_answers_says_busy(router, registry, db, profile, monkeypatch):
    db.set_settings({"fallback_models": [BACKUP]})
    monkeypatch.setattr(generate, "BUSY_ROUND_PAUSES", (5.0,) * 6)  # the letter's 1 s ends during the first pause
    monkeypatch.setattr(tailor, "LETTER_WAIT_SECONDS", 1.0)
    router.overrides[MAIN] = ByKind(registry, {"tailor_output": ["ok"], "tailor_revision": ["ok"],
                                               "cover_letter": [busy()]})
    router.overrides[BACKUP] = ByKind(registry, {"cover_letter": [busy("Gemini 3.5 Flash-Lite")]})
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=MAIN,
                                    save=False, cover_letter=True, time_limit=0))
    assert result.status == "ok" and result.cover_letter is None
    note = next(note for note in result.notes if note.startswith("Cover letter failed"))
    assert note.startswith(f"Cover letter failed: still busy after 1 seconds ({BACKUP}: Gemini 3.5 Flash-Lite: HTTP "
                           "503: This model is currently experiencing high demand.")
    letters = [(call.model, call.result) for call in result.timing.calls if call.kind == "cover_letter"]
    assert letters == [(MAIN, "failed"), (BACKUP, "failed")]  # nothing was on its way, nothing stopped or billed


def test_a_letter_note_quotes_the_busy_answer_it_waited_out(router, registry, db, profile, monkeypatch):
    """The chosen letter model was busy and the backup, asked after it, was refused: only the chosen model was due
    again when the letter's time ran out in the pause. The note quotes that busy answer, not the backup's refusal."""
    db.set_settings({"fallback_models": [BACKUP]})
    monkeypatch.setattr(generate, "BUSY_ROUND_PAUSES", (5.0,) * 6)  # the letter's 1 s ends during the first pause
    monkeypatch.setattr(tailor, "LETTER_WAIT_SECONDS", 1.0)
    refused = ProviderError("Gemini 3.5 Flash-Lite: HTTP 400: API key not valid.", 400)
    router.overrides[MAIN] = ByKind(registry, {"tailor_output": ["ok"], "tailor_revision": ["ok"],
                                               "cover_letter": [busy()]})
    router.overrides[BACKUP] = ByKind(registry, {"cover_letter": [refused]})
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=MAIN,
                                    save=False, cover_letter=True, time_limit=0))
    assert result.status == "ok" and result.cover_letter is None
    note = next(note for note in result.notes if note.startswith("Cover letter failed"))
    assert note.startswith(f"Cover letter failed: still busy after 1 seconds ({MAIN}: Gemini 3.8 Flash: HTTP 503: "
                           "This model is currently experiencing high demand.")
    assert "API key not valid" not in note
    letters = [(call.model, call.result) for call in result.timing.calls if call.kind == "cover_letter"]
    assert letters == [(MAIN, "failed"), (BACKUP, "failed")]


def test_a_time_limit_during_a_pause_bills_no_call(router, registry, db, profile, monkeypatch):
    """The shorter draft answered "busy" and the time limit ran out during the pause before it was asked again: no
    request was on its way, so nothing more is billed or shown as stopped."""
    monkeypatch.setattr(tailor, "MIN_SHORTER_SECONDS", 0.1)
    first_layout = []

    def try_styles(*args, **kwargs):  # the first draft does not fit two pages, so a shorter one is asked for
        first_layout.append(1)
        return None if len(first_layout) == 1 else real_try_styles(*args, **kwargs)

    real_try_styles = tailor.try_styles
    monkeypatch.setattr(tailor, "try_styles", try_styles)

    async def long_pause(seconds):
        await asyncio.sleep(30)

    monkeypatch.setattr(generate, "_sleep", long_pause)
    router.overrides[MAIN] = ByKind(registry, {"tailor_output": ["ok", busy()], "tailor_revision": ["ok"]})
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=MAIN,
                                    save=False, allow_fallback=False, time_limit=5.5))  # 1.5 s after the checks
    assert router.overrides[MAIN].sent == ["tailor_output", "tailor_output"]
    drafts = [call.result for call in result.timing.calls if call.kind == "tailor_output"]
    assert drafts == ["ok", "failed"]  # no "stopped" call that was never sent
    assert any(note.startswith("Asking the model for shorter bullets ran past the time limit") for note in result.notes)


def test_run_check_asks_a_busy_model_once_more(client, registry, monkeypatch):
    monkeypatch.setattr(router_module, "CHECK_BUSY_PAUSE", 0.0)
    replies, sent = [503, 200], []

    def handler(request):
        code = replies.pop(0) if len(replies) > 1 else replies[0]
        sent.append(code)
        if code == 503:
            return httpx.Response(503, json={"error": {"code": 503, "message": HIGH_DEMAND, "status": "UNAVAILABLE"}})
        answer = {"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}
        return httpx.Response(200, json={"model": MAIN, "usage": {"prompt_tokens": 9, "completion_tokens": 3},
                                         "choices": [answer]})

    client.app.state.router.overrides[MAIN] = OpenAICompatClient(registry.get(MAIN),
                                                                  transport=httpx.MockTransport(handler))
    row = client.post("/api/models/check", json={"models": [MAIN]}).json()["results"][0]
    assert row["ok"] is True and sent == [503, 200]  # a passing spike is not a broken key or model
    replies[:], sent[:] = [503], []
    row = client.post("/api/models/check", json={"models": [MAIN]}).json()["results"][0]
    assert row["ok"] is False and sent == [503, 503] and "experiencing high demand" in row["error"]


NOTE_HARNESS = r"""
const fs = require('fs');
const [pagePath, casesPath] = process.argv.slice(2);
const page = fs.readFileSync(pagePath, 'utf8');
const code = page.slice(page.indexOf('const OTHER_COMPANY'), page.indexOf('async function init()'));
const el = (tag, attrs) => attrs.text;  // the note's text stands for the element
const notes = JSON.parse(fs.readFileSync(casesPath, 'utf8')).map(c => {
  const sameKeyNote = new Function('models', 'el', code + '\nreturn sameKeyNote;')(c.models, el);
  return sameKeyNote({ fallback_models: c.fallbacks }, c.resume);
});
console.log(JSON.stringify(notes));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="runs the Settings page's own script with Node.js")
def test_the_settings_warning_is_shown_for_the_owners_setup(client, registry, tmp_path, monkeypatch):
    """The page's own sameKeyNote, run with /api/models as the page gets it."""
    for spec in registry.all():  # only the keys each case names
        if spec.key_env:
            monkeypatch.delenv(spec.key_env, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "placeholder-not-a-key")
    google_only = client.get("/api/models").json()["models"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    with_claude = client.get("/api/models").json()["models"]
    cases = [
        {"models": google_only, "resume": MAIN, "fallbacks": [BACKUP]},  # the owner's setup
        {"models": with_claude, "resume": MAIN, "fallbacks": [BACKUP]},
        {"models": with_claude, "resume": MAIN, "fallbacks": ["claude-sonnet-5-5"]},
        {"models": google_only, "resume": "demo-concise", "fallbacks": []},
    ]
    (tmp_path / "settings.html").write_text(client.get("/settings").text, encoding="utf-8")
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "note.js").write_text(NOTE_HARNESS, encoding="utf-8")
    run = subprocess.run(
        ["node", str(tmp_path / "note.js"), str(tmp_path / "settings.html"), str(tmp_path / "cases.json")],
        capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 0, run.stderr
    notes = json.loads(run.stdout)
    assert notes[0] == (
        "No backup model uses a different API key from the resume model (GEMINI_API_KEY). When that company's AI "
        "service is busy (for example Google's \"high demand\") or the key's quota is used up, tailoring stops. Paste "
        "another company's API key under API keys above (for example ANTHROPIC_API_KEY, for Claude Sonnet 5.5), then "
        "tick that model here.")
    assert notes[1].endswith("tailoring stops. Tick a backup with another key, for example Claude Sonnet 5.5.")
    assert notes[2] is None and notes[3] is None
