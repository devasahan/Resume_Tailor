"""A busy model (Google's HTTP 503 "This model is currently experiencing high demand") hands over to the next model at
once. When every model was busy they take turns, the chosen model first, after short pauses, and after about a minute
and a half the run stops with a plain message (app/pipeline/generate.py BUSY_ROUND_PAUSES). The owner's run waited
2:50 for two busy Google models (a minute of retries each) and then failed."""
import asyncio

import httpx
import pytest

from app.llm.base import LLMClient, ProviderError
from app.llm.fake_client import FakeClient
from app.llm.openai_compat_client import OpenAICompatClient
from app.main import BUSY_MESSAGE, failure_message
from app.parsing.resume_parser import read_resume
from app.pipeline import generate
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


def test_a_busy_follow_up_keeps_the_first_answer(router, registry, clock):
    main, _backup = scripted(router, registry, clock, ["ok", busy()], ["ok"])
    follow = Ask(user="fix", schema={}, schema_name="job_facts", parse=lambda data: "fixed")
    out = ask(router, parse=lambda data: "draft", refine=lambda value: follow)
    assert out.value == "draft" and out.final_model_id == MAIN and main.calls == 2 and clock["pauses"] == []


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
