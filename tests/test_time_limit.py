"""The time limit for a resume (Settings, default 30 s): the first draft always comes back; the extra rewrite pass and
the shorter-draft call run only when they can finish in time, and are stopped at the limit."""
import asyncio
import time

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import generate, tailor
from app.pipeline.tailor import run_tailor

from .conftest import SAMPLE_JD


@pytest.fixture(autouse=True)
def fresh_follow_up_times(monkeypatch):
    """How long the last rewrite took is remembered per process: each test starts without it."""
    monkeypatch.setattr(generate, "_FOLLOW_UP_SECONDS", {})


class SlowRewrite(LLMClient):
    """The demo model, with a rewrite pass (the follow-up) that takes `delay` seconds and a first draft that takes
    `first` seconds; every call is counted."""

    def __init__(self, inner, delay, first=0.0):
        self.inner, self.delay, self.first, self.calls = inner, delay, first, []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs["schema_name"])
        if kwargs["schema_name"] == "tailor_revision":
            await asyncio.sleep(self.delay)
        elif len(self.calls) == 1:
            await asyncio.sleep(self.first)
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and len(self.calls) == 1:
            result.data["sections"]["summary"] = "Data engineer."  # too thin: the rewrite pass is asked for
        return result


def tailor_with(router, db, registry, profile, *, limit, delay, first=0.0, reserve=0.0, minimum=0.1, monkeypatch,
                model="demo-concise"):
    monkeypatch.setattr(tailor, "CHECKS_RESERVE_SECONDS", reserve)
    monkeypatch.setattr(generate, "MIN_FOLLOW_UP_SECONDS", minimum)
    db.set_settings({"resume_time_limit_s": limit})
    client = SlowRewrite(FakeClient(registry.get(model)), delay, first)
    router.overrides[model] = client
    started = time.perf_counter()
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id=model, save=False))
    return result, client, time.perf_counter() - started


def test_a_slow_rewrite_pass_is_stopped_at_the_limit(router, db, registry, profile, monkeypatch):
    result, client, took = tailor_with(router, db, registry, profile, limit=1, delay=5, monkeypatch=monkeypatch)
    assert result.status == "ok" and client.calls == ["tailor_output", "tailor_revision"]
    assert took < 3  # not the 5 s the rewrite would have taken
    assert any("rewrite pass was stopped to finish within the time limit of 1 seconds" in n for n in result.notes)


def test_a_rewrite_that_cannot_finish_after_a_slow_draft_is_not_started(router, db, registry, profile,
                                                                         monkeypatch):
    """A rewrite takes about half as long as the draft: after a 1.6 s draft with 2 s allowed, the 0.4 s left are
    not enough, so none is started (in the owner's log one was started with 7 s left and stopped, wasted)."""
    result, client, took = tailor_with(router, db, registry, profile, limit=2, delay=5, first=1.6,
                                       monkeypatch=monkeypatch)
    assert result.status == "ok" and client.calls == ["tailor_output"] and took < 2.5
    assert any("rewrite pass was skipped" in n for n in result.notes)


def test_the_time_the_last_rewrite_took_is_remembered(router, db, registry, profile, monkeypatch):
    result, client, _took = tailor_with(router, db, registry, profile, limit=1, delay=5, monkeypatch=monkeypatch)
    assert client.calls == ["tailor_output", "tailor_revision"]  # started, then stopped at the limit
    assert generate._FOLLOW_UP_SECONDS[("demo-concise", "tailor_revision")] >= 0.5  # it needed at least that
    generate._FOLLOW_UP_SECONDS[("demo-concise", "tailor_revision")] = 50.0  # as if the last one took 50 s
    result, client, _took = tailor_with(router, db, registry, profile, limit=30, delay=0, monkeypatch=monkeypatch)
    assert client.calls == ["tailor_output"] and any("rewrite pass was skipped" in n for n in result.notes)
    generate._FOLLOW_UP_SECONDS.clear()
    result, client, _took = tailor_with(router, db, registry, profile, limit=30, delay=0.2, monkeypatch=monkeypatch)
    assert client.calls == ["tailor_output", "tailor_revision"]
    assert 0.2 <= generate._FOLLOW_UP_SECONDS[("demo-concise", "tailor_revision")] < 5  # a finished one: its time


def test_no_rewrite_pass_starts_without_enough_time(router, db, registry, profile, monkeypatch):
    result, client, _took = tailor_with(router, db, registry, profile, limit=1, delay=0, minimum=60,
                                        monkeypatch=monkeypatch)
    assert result.status == "ok" and client.calls == ["tailor_output"]
    assert any("rewrite pass was skipped" in n for n in result.notes)


def test_no_limit_always_polishes(router, db, registry, profile, monkeypatch):
    result, client, _took = tailor_with(router, db, registry, profile, limit=0, delay=0.2, minimum=60,
                                        monkeypatch=monkeypatch)
    assert client.calls == ["tailor_output", "tailor_revision"] and result.retries == 1
    assert not any("time limit" in n for n in result.notes)


def test_a_fast_rewrite_pass_still_runs_within_the_limit(router, db, registry, profile, monkeypatch):
    result, client, _took = tailor_with(router, db, registry, profile, limit=30, delay=0.1, monkeypatch=monkeypatch)
    assert client.calls == ["tailor_output", "tailor_revision"] and not any("time limit" in n for n in result.notes)


def test_a_follow_up_that_removes_a_certification_claim_ignores_the_limit(router, registry, monkeypatch):
    """A required follow-up runs even with no time left (a false credential claim matters more than seconds)."""
    calls = []

    class Recording(LLMClient):
        def __init__(self, inner):
            self.inner = inner

        async def generate_structured(self, **kwargs):
            calls.append(kwargs["schema_name"])
            return await self.inner.generate_structured(**kwargs)

    router.overrides["demo-concise"] = Recording(FakeClient(registry.get("demo-concise")))
    follow = generate.Ask(user="fix", schema={}, schema_name="job_facts", parse=lambda data: "fixed", required=True)
    out = asyncio.run(generate.generate_validated(
        router, ["demo-concise"], system="s", user="u", schema={}, schema_name="job_facts", parse=lambda data: "first",
        context={"jd": "A remote job paying $100,000 a year."}, refine=lambda value: follow,
        deadline=time.perf_counter() - 100))
    assert out.value == "fixed" and calls == ["job_facts", "job_facts"] and out.follow_up == ""


def test_the_limit_is_a_setting(client):
    assert client.get("/api/settings").json()["settings"]["resume_time_limit_s"] == 30
    assert client.put("/api/settings", json={"resume_time_limit_s": 45}).status_code == 200
    assert client.get("/api/settings").json()["settings"]["resume_time_limit_s"] == 45
    assert client.put("/api/settings", json={"resume_time_limit_s": 0.5}).json()["settings"]["resume_time_limit_s"] == 1
    for bad in ("-5", '"soon"', "Infinity", "NaN", "1e999"):  # never 'no limit' by mistake, never a server error
        sent = client.put("/api/settings", content=f'{{"resume_time_limit_s": {bad}}}',
                          headers={"content-type": "application/json"})
        assert sent.status_code == 422, bad
    assert client.put("/api/settings", content='{"daily_cost_cap_usd": Infinity}',
                      headers={"content-type": "application/json"}).status_code == 422
    assert client.get("/api/settings").json()["settings"]["resume_time_limit_s"] == 1
    assert 'id="resume_time_limit_s"' in client.get("/settings").text



def test_a_stopped_rewrite_is_still_paid_for(router, db, registry, profile, monkeypatch):
    """The provider got the whole prompt before the call was stopped: the run's tokens count it (the run's cost,
    Spent today and the daily cap all add up runs)."""
    skipped, _client, _took = tailor_with(router, db, registry, profile, limit=1, delay=5, minimum=60,
                                          monkeypatch=monkeypatch)
    stopped, client, _took = tailor_with(router, db, registry, profile, limit=1, delay=5, monkeypatch=monkeypatch)
    assert client.calls == ["tailor_output", "tailor_revision"]
    assert stopped.input_tokens > skipped.input_tokens + 1000  # about the rewrite prompt's size


def test_one_slow_rewrite_does_not_block_every_later_one(router, db, registry, profile, monkeypatch):
    generate._FOLLOW_UP_SECONDS[("demo-concise", "tailor_revision")] = 50.0  # a provider hiccup once
    tailor_with(router, db, registry, profile, limit=30, delay=0, monkeypatch=monkeypatch)
    assert generate._FOLLOW_UP_SECONDS[("demo-concise", "tailor_revision")] == 40.0  # shrinks each time it blocks


def test_no_shorter_draft_without_a_drafts_time_left(router, db, registry, profile, monkeypatch):
    """A shorter draft is a whole new draft: after a 1 s first draft with about 0.6 s left, none is started, and the
    text is trimmed to two pages at once."""
    monkeypatch.setattr(tailor, "MIN_SHORTER_SECONDS", 0.1)
    result, client, took = tailor_with(router, db, registry, profile, limit=1.6, delay=0, first=1.0, minimum=60,
                                       monkeypatch=monkeypatch, model="demo-detailed")
    assert client.calls == ["tailor_output"] and result.fits and took < 4
    assert any("no time left" in n for n in result.notes)


def test_the_bake_off_compares_models_without_a_time_limit(router, db, registry, profile, monkeypatch):
    from app.pipeline.compare import compare

    monkeypatch.setattr(generate, "MIN_FOLLOW_UP_SECONDS", 0.1)
    db.set_settings({"resume_time_limit_s": 1})
    client = SlowRewrite(FakeClient(registry.get("demo-concise")), 1.5)
    router.overrides["demo-concise"] = client
    [result] = asyncio.run(compare(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                   model_ids=["demo-concise"], save=False))
    assert client.calls == ["tailor_output", "tailor_revision"] and result.retries == 1
    assert not any("time limit" in n for n in result.notes)  # each model's own rewrite, however long it takes


# ------------------------------------------------------------------ a model that does not answer
class Hanging(LLMClient):
    """A model that never answers (an overloaded provider, a request stuck in the network)."""

    def __init__(self):
        self.calls = 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(60)


def run_with_backup(router, db, profile, monkeypatch, *, limit=1, fallback=True):
    monkeypatch.setattr(tailor, "SWITCH_EXTRA_SECONDS", 0.0)
    monkeypatch.setattr(tailor, "LAST_MODEL_SECONDS", 1.0)
    db.set_settings({"resume_time_limit_s": limit, "fallback_models": ["demo-concise"] if fallback else []})
    started = time.perf_counter()
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-detailed", save=False))
    return result, time.perf_counter() - started


def test_a_follow_up_skipped_with_no_time_left_never_crashes(router):
    """The owner's log: after a slow first model, the rewrite was skipped with -241.9 s left and a KeyError threw the
    finished draft away."""
    follow = generate.Ask(user="fix", schema={}, schema_name="job_facts", parse=lambda data: "fixed")
    out = asyncio.run(generate.generate_validated(
        router, ["demo-concise"], system="s", user="u", schema={}, schema_name="job_facts", parse=lambda data: "draft",
        context={"jd": "A remote job paying $100,000 a year."}, refine=lambda value: follow,
        deadline=time.perf_counter() - 240))
    assert out.value == "draft" and out.follow_up == "skipped" and generate._FOLLOW_UP_SECONDS == {}


def test_a_model_that_does_not_answer_is_left_for_the_backup(router, db, profile, monkeypatch):
    hanging = Hanging()
    router.overrides["demo-detailed"] = hanging
    result, took = run_with_backup(router, db, profile, monkeypatch)
    assert result.status == "ok" and result.fallback_chain == ["demo-detailed", "demo-concise"] and took < 4
    assert any("demo-detailed could not be used (did not answer within 1 seconds), so demo-concise made this resume"
               in note for note in result.notes)
    assert result.input_tokens > 0 and generate.resting("demo-detailed")  # its prompt was sent: it counts
    # the next resume does not wait for it again: it rests, and the backup answers at once
    again, took = run_with_backup(router, db, profile, monkeypatch)
    assert hanging.calls == 1 and again.fallback_chain == ["demo-concise"] and took < 2
    assert any("it is resting after a recent failure" in note for note in again.notes)
    generate._RESTING_UNTIL["demo-detailed"] = 0.0  # its rest is over: it is tried again
    run_with_backup(router, db, profile, monkeypatch)
    assert hanging.calls == 2


def test_the_last_model_is_given_a_bounded_wait_and_always_tried(router, db, profile, monkeypatch):
    hanging = Hanging()
    router.overrides["demo-detailed"] = hanging
    generate._RESTING_UNTIL["demo-detailed"] = time.monotonic() + 300  # resting, but nothing else to use
    result, took = run_with_backup(router, db, profile, monkeypatch, fallback=False)
    assert hanging.calls == 1 and result.status == "error" and took < 3
    assert "did not answer within 1 seconds" in result.error


def test_without_a_time_limit_a_slow_answer_is_awaited(router, registry):
    class Slow(LLMClient):
        async def generate_structured(self, **kwargs):
            await asyncio.sleep(0.3)
            return await FakeClient(registry.get("demo-concise")).generate_structured(**kwargs)

    router.overrides["demo-detailed"] = Slow()
    out = asyncio.run(generate.generate_validated(
        router, ["demo-detailed", "demo-concise"], system="s", user="u", schema={}, schema_name="job_facts",
        parse=lambda data: data, context={"jd": "A remote job paying $100,000 a year."}))
    assert out.final_model_id == "demo-detailed"


def test_a_failed_attempt_is_logged_with_its_reason(router, caplog):
    from app.llm.base import ProviderError

    class Down(LLMClient):
        async def generate_structured(self, **kwargs):
            raise ProviderError("Gemini 3.8 Flash: HTTP 503: The model is overloaded", 503)

    router.overrides["demo-detailed"] = Down()
    with caplog.at_level("INFO", logger="app.pipeline.generate"):
        out = asyncio.run(generate.generate_validated(
            router, ["demo-detailed", "demo-concise"], system="s", user="u", schema={}, schema_name="job_facts",
            parse=lambda data: data, context={"jd": "A remote job paying $100,000 a year."}))
    assert out.final_model_id == "demo-concise"
    assert "demo-detailed job_facts attempt 1 failed after" in caplog.text and "HTTP 503" in caplog.text
    assert "demo-detailed rests for 5 minutes (HTTP 503)" in caplog.text and generate.resting("demo-detailed")



def test_a_used_up_quota_rests_the_model_as_long_as_the_provider_says(router, caplog):
    from app.llm.base import ProviderError

    class OutOfQuota(LLMClient):
        def __init__(self, retry_after):
            self.retry_after = retry_after

        async def generate_structured(self, **kwargs):
            raise ProviderError("Gemini 3.8 Flash: the API key's quota is used up", 429, quota=True,
                                retry_after=self.retry_after)

    def ask():
        return asyncio.run(generate.generate_validated(
            router, ["demo-detailed", "demo-concise"], system="s", user="u", schema={}, schema_name="job_facts",
            parse=lambda data: data, context={"jd": "A remote job paying $100,000 a year."}))

    router.overrides["demo-detailed"] = OutOfQuota(33.0)
    with caplog.at_level("INFO", logger="app.pipeline.generate"):
        assert ask().final_model_id == "demo-concise"
    assert "rests for 33 seconds (its API key's quota is used up)" in caplog.text
    assert 30 < generate._RESTING_UNTIL["demo-detailed"] - time.monotonic() <= 33
    generate._RESTING_UNTIL.clear()
    router.overrides["demo-detailed"] = OutOfQuota(None)  # the provider did not say: an hour
    ask()
    assert 3500 < generate._RESTING_UNTIL["demo-detailed"] - time.monotonic() <= 3600
    generate._RESTING_UNTIL.clear()
    router.overrides["demo-detailed"] = OutOfQuota(86400.0)  # never longer than an hour: a reload is not needed
    ask()
    assert 3500 < generate._RESTING_UNTIL["demo-detailed"] - time.monotonic() <= 3600
    assert tailor.LAST_MODEL_SECONDS == 60


def test_settings_warns_when_every_backup_uses_the_same_key(client):
    page = client.get("/settings").text
    assert "No backup model uses a different API key from the resume model" in page
    models = client.get("/api/models").json()["models"]
    flash = next(m for m in models if m["id"] == "gemini-3.8-flash")
    assert flash["key_env"] == "GEMINI_API_KEY"  # the variable's name only, never a key
    assert all(not m["key_env"] or m["key_env"].endswith("_API_KEY") for m in models)  # names, no values


def test_the_note_keeps_the_quota_advice_whole(router, db, profile, monkeypatch):
    from app.llm.base import ProviderError

    message = ("Gemini 3.8 Flash: the API key's quota is used up (Google: 'You exceeded your current quota, please "
               "check your plan and billing details.'). Check the plan and billing at aistudio.google.com, or wait "
               "until it resets.")

    class OutOfQuota(LLMClient):
        async def generate_structured(self, **kwargs):
            raise ProviderError(message, 429, quota=True)

    router.overrides["demo-detailed"] = OutOfQuota()
    result, _ = run_with_backup(router, db, profile, monkeypatch)
    assert result.status == "ok"
    assert any(f"demo-detailed could not be used ({message}), so demo-concise made this resume" in note
               for note in result.notes)
