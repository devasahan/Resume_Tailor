import asyncio

import pytest

from app.llm.base import LLMClient, LLMResult, ProviderError, Usage
from app.llm.fake_client import FakeClient
from app.pipeline.compare import compare
from app.pipeline.tailor import run_tailor
from app.schemas import TailorResult


def run(coro):
    return asyncio.run(coro)


def tailor(router, db, profile, jd, model="demo-concise", **kw):
    return run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id=model, **kw))


def test_demo_run_is_ok_one_page_and_keeps_fixed_facts(router, db, profile, jd):
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and result.fits and result.pages == 1
    assert result.resume.profile == profile  # name, contact, companies, dates, education untouched
    assert [e.company_index for e in result.resume.experience] == [0, 1, 2]
    assert result.analysis.target_role.startswith("Senior Data Engineer")
    assert result.run_id and db.get_run(result.run_id).resume.headline == result.resume.headline
    assert result.coverage.score > 0.5 and result.prompt_version
    assert result.attempts == 1 and result.retries == 0


def test_titles_form_a_career_path(router, db, profile, jd):
    titles = [e.job_title for e in tailor(router, db, profile, jd).resume.experience]
    assert titles == ["Senior Data Engineer", "Data Engineer", "Junior Data Engineer"]


def test_long_text_is_brought_to_one_page(router, db, profile, jd):
    result = tailor(router, db, profile, jd, model="demo-detailed")
    assert result.status == "ok" and result.pages == 1 and result.fits
    assert result.notes  # something had to be done (smaller font, shorter bullets or trimming)
    assert any(f.level == "verify" and "30%" in f.reason for f in result.flags)  # invented number is flagged


def test_invalid_output_is_retried_with_feedback(router, db, profile, jd):
    result = tailor(router, db, profile, jd, model="demo-flaky")
    assert result.status == "ok" and result.attempts == 2 and result.retries == 1


class Boom(LLMClient):
    async def generate_structured(self, **kwargs):
        raise ProviderError("provider is down", 503)


def test_fallback_model_takes_over(router, db, profile, jd):
    router.overrides["demo-flaky"] = Boom()
    db.set_settings({"fallback_models": ["demo-concise"]})
    result = tailor(router, db, profile, jd, model="demo-flaky", allow_fallback=True)
    assert result.status == "ok"
    assert result.fallback_chain == ["demo-flaky", "demo-concise"] and result.served_model == "demo-concise"
    assert result.requested_model == "demo-flaky"


def test_no_fallback_reports_the_error(router, db, profile, jd):
    router.overrides["demo-flaky"] = Boom()
    db.set_settings({"fallback_models": ["demo-concise"]})
    result = tailor(router, db, profile, jd, model="demo-flaky", allow_fallback=False)
    assert result.status == "error" and "provider is down" in result.error and result.resume is None


def test_cost_cap_stops_new_runs(router, db, profile, jd):
    db.set_settings({"daily_cost_cap_usd": 0.5})
    db.save_run(TailorResult(requested_model="m", cost_usd=0.6), None, "")
    result = tailor(router, db, profile, jd)
    assert result.status == "error" and "cap" in result.error.lower()


class Priced(LLMClient):
    """Delegates to the demo model but reports fixed token usage, to check the cost maths."""

    def __init__(self, inner):
        self.inner = inner

    async def generate_structured(self, **kwargs):
        r = await self.inner.generate_structured(**kwargs)
        return LLMResult(data=r.data, usage=Usage(1000, 2000), served_model="claude-sonnet-5-5-20260101", raw_text=r.raw_text)


def test_cost_is_computed_from_registry_prices(router, db, registry, profile, jd, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    router.overrides["claude-sonnet-5-5"] = Priced(FakeClient(registry.get("demo-concise")))
    result = tailor(router, db, profile, jd, model="claude-sonnet-5-5")
    assert result.status == "ok"
    assert result.cost_usd == pytest.approx(1000 * 2 / 1e6 + 2000 * 10 / 1e6)  # $2 in / $10 out per 1M
    assert result.served_model == "claude-sonnet-5-5-20260101"  # what the API reported, kept for the log
    assert result.input_tokens == 1000 and result.output_tokens == 2000


def test_cover_letter(router, db, profile, jd):
    result = tailor(router, db, profile, jd, cover_letter=True, company_name="Acme", hiring_manager="Sam Lee")
    assert result.cover_letter and len(result.cover_letter.paragraphs) >= 2
    assert (result.cover_letter.company_name, result.cover_letter.hiring_manager) == ("Acme", "Sam Lee")


def test_compare_runs_models_side_by_side_without_fallback(router, db, profile, jd):
    results = run(compare(router, db, profile=profile, profile_id=None, jd=jd, model_ids=["demo-concise", "demo-detailed", "demo-flaky"]))
    assert [r.requested_model for r in results] == ["demo-concise", "demo-detailed", "demo-flaky"]
    assert all(r.status == "ok" for r in results)
    assert results[0].resume.experience[0].bullets != results[1].resume.experience[0].bullets


class Silent(LLMClient):
    """A provider that does not report token usage."""

    def __init__(self, inner):
        self.inner = inner

    async def generate_structured(self, **kwargs):
        r = await self.inner.generate_structured(**kwargs)
        return LLMResult(data=r.data, usage=Usage(0, 0), served_model="x", raw_text=r.raw_text)


def test_missing_usage_is_called_out_instead_of_showing_zero_cost(router, db, registry, profile, jd, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    router.overrides["claude-sonnet-5-5"] = Silent(FakeClient(registry.get("demo-concise")))
    result = tailor(router, db, profile, jd, model="claude-sonnet-5-5")
    assert result.status == "ok" and any("did not report token usage" in n for n in result.notes)
