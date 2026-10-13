"""Estimated metrics must look measured (23.4%, 21%), not like round guesses (35%, 40+, ~30%)."""
import asyncio

import pytest

from app.llm.base import LLMClient, ProviderError
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, tailor_system

from .test_validate import make_output


@pytest.mark.parametrize("token,expected", [
    ("35%", True), ("40+", True), ("$120K", True), ("1,500", True), ("10", True),
    ("23.4%", False), ("21%", False), ("17.8%", False), ("5%", False), ("38", False), ("2020", False), ("1.5", False),
])
def test_is_round_figure(token, expected):
    assert validate.is_round_figure(token) is expected


def test_style_problems_skip_the_users_numbers_and_the_computed_years(profile):
    out = make_output()
    out.sections.summary = "Data Engineer with 10+ years of experience."
    out.sections.experience[0].bullets = ["Cut runtime by 35% with Spark", "Saved approximately 23% of compute", "Served 50 analysts"]
    resume = validate.build_resume(profile, out)
    problems = validate.figure_style_problems(resume, "We served 50 analysts\nTotal professional experience: 10+ years")
    assert problems == ["35%", "approximately 23%"]  # 50 is the user's own number, 10+ is computed from the dates


def test_tilde_is_stripped_by_the_text_cleaner(profile):
    out = make_output()
    out.sections.experience[0].bullets = ["Saved ~23.4% of compute", "Built pipelines in Python"]
    assert validate.build_resume(profile, out).experience[0].bullets[0] == "Saved 23.4% of compute"


def test_prompt_asks_for_specific_figures():
    text = tailor_system("estimate")
    assert "never use numbers ending in 0 or 5" in text and "23.4%" in text and '"~"' in text


class Rounder(LLMClient):
    """Demo model that writes round figures first, then specific ones once told; can fail the retry."""

    def __init__(self, inner, fail_retry=False, round_figures=("35%", "40+")):
        self.inner, self.fail_retry, self.round_figures, self.calls = inner, fail_retry, round_figures, 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        retry = RETRY_MARKER in kwargs["user"]
        if retry and self.fail_retry:
            raise ProviderError("overloaded", 529)
        result = await self.inner.generate_structured(**kwargs)

        def bullet(figure):
            return (f"Improved pipeline throughput by {figure} by moving the nightly batch jobs to Spark and "
                    "partitioning the largest tables by date")

        if kwargs["schema_name"] == "tailor_revision":  # the fix: only the first bullet of the first two jobs
            edits = [e for e in result.data["bullets"] if not (e["bullet_index"] == 0 and e["company_index"] < 2)]
            result.data["bullets"] = edits + [{"company_index": i, "bullet_index": 0, "text": bullet(figure)}
                                              for i, figure in enumerate(("23.4%", "38"))]
            return result
        experience = result.data["sections"]["experience"]
        for exp, figure in zip(experience, self.round_figures, strict=False):  # 3 jobs, 2 figures: two change
            exp["bullets"][0] = bullet(figure)
        return result


def run(router, db, profile, jd):
    return asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))


def bullets(result):
    return [b for e in result.resume.experience for b in e.bullets]


def test_round_figures_trigger_one_refinement(router, db, registry, profile, jd):
    client = Rounder(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = run(router, db, profile, jd)
    assert result.status == "ok" and client.calls == 2 and result.retries == 1
    text = " ".join(bullets(result))
    assert "23.4%" in text and "35%" not in text and "40+" not in text
    assert not any("rounded" in w for w in result.ats_warnings)


def test_failed_refinement_keeps_the_valid_first_draft(router, db, registry, profile, jd):
    router.overrides["demo-concise"] = Rounder(FakeClient(registry.get("demo-concise")), fail_retry=True)
    result = run(router, db, profile, jd)
    assert result.status == "ok"  # the run does not fail over number style
    assert "35%" in " ".join(bullets(result))
    assert any("rounded" in w and "35%" in w for w in result.ats_warnings)


def test_a_single_round_figure_is_not_worth_a_second_call(router, db, registry, profile, jd):
    client = Rounder(FakeClient(registry.get("demo-concise")), round_figures=("35%", "38"))
    router.overrides["demo-concise"] = client
    result = run(router, db, profile, jd)
    assert client.calls == 1 and result.retries == 0
    assert any("rounded" in w for w in result.ats_warnings)  # still pointed out to the user
