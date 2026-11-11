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


def tailor_with(router, db, registry, profile, *, limit, delay, first=0.0, reserve=0.0, minimum=0.1, monkeypatch):
    monkeypatch.setattr(tailor, "CHECKS_RESERVE_SECONDS", reserve)
    monkeypatch.setattr(generate, "MIN_FOLLOW_UP_SECONDS", minimum)
    db.set_settings({"resume_time_limit_s": limit})
    client = SlowRewrite(FakeClient(registry.get("demo-concise")), delay, first)
    router.overrides["demo-concise"] = client
    started = time.perf_counter()
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", save=False))
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
    assert client.put("/api/settings", json={"resume_time_limit_s": -5}).json()["settings"]["resume_time_limit_s"] == 0
    assert client.put("/api/settings", json={"resume_time_limit_s": "soon"}).status_code == 422
    assert 'id="resume_time_limit_s"' in client.get("/settings").text
