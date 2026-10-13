"""The fix round: the model returns only what it changes (a TailorRevision), merged onto the first draft."""
import asyncio
import json

import pytest

from app.llm.base import InvalidOutputError, LLMClient, LLMResult, Usage
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import draft_text, run_tailor
from app.prompts import RETRY_MARKER, revise_user, tailor_user
from app.schemas import BulletChange, SkillChange, TailorRevision, TitleChange

from .test_posting_voice import CopiesThePosting
from .test_validate import make_output


@pytest.fixture
def first(profile):
    out = make_output()
    return out, validate.build_resume(profile, out)


def merged(profile, first, **changes):
    out, draft = first
    return validate.build_resume(profile, validate.apply_revision(out, draft, TailorRevision(**changes)))


def test_the_draft_shows_numbered_bullets(first):
    text = draft_text(first[1])
    assert "company_index 0 (Northwind Labs): Engineer 0\n[0] Built things a\n[1] Improved things" in text


def test_only_the_listed_parts_change(profile, first):
    out, draft = first
    resume = merged(profile, first, summary="A new summary about Python and SQL work.",
                    titles=[TitleChange(company_index=2, job_title="Data Engineer")],
                    bullets=[BulletChange(company_index=1, bullet_index=0, text="Rebuilt the SQL loaders for speed")])
    assert resume.summary == "A new summary about Python and SQL work." and resume.headline == draft.headline
    assert resume.experience[1].bullets == ["Rebuilt the SQL loaders for speed", "Improved things"]
    assert resume.experience[0].bullets == draft.experience[0].bullets
    assert resume.experience[2].job_title == "Data Engineer" and resume.experience[0].job_title == "Engineer 0"
    assert [len(e.bullets) for e in resume.experience] == [len(e.bullets) for e in draft.experience]
    revised = validate.apply_revision(out, draft, TailorRevision())
    assert revised.jd_analysis == out.jd_analysis and revised.career_plan == out.career_plan
    assert validate.build_resume(profile, revised) == draft  # an empty revision is the draft itself
    assert draft.experience[1].bullets[0] == "Built things b"  # the kept first draft is never changed


@pytest.mark.parametrize("changes,message", [
    ({"bullets": [BulletChange(company_index=0, bullet_index=2, text="x")]}, "no bullet"),
    ({"bullets": [BulletChange(company_index=0, bullet_index=-1, text="x")]}, "no bullet"),
    ({"bullets": [BulletChange(company_index=7, bullet_index=0, text="x")]}, "no bullet"),
    ({"bullets": [BulletChange(company_index=0, bullet_index=0, text="x"),
                  BulletChange(company_index=0, bullet_index=0, text="y")]}, "twice"),
    ({"titles": [TitleChange(company_index=7, job_title="Lead")]}, "no company_index"),
    # the new text is clearly bullet 1 rewritten, so the index is off by one
    ({"bullets": [BulletChange(company_index=0, bullet_index=0, text="Improved things quickly")]}, "instead"),
])
def test_an_edit_that_cannot_be_applied_as_meant_is_refused(first, changes, message):
    with pytest.raises(ValueError, match=message):
        validate.apply_revision(*first, TailorRevision(**changes))


def test_copied_labels_are_stripped_and_empty_texts_keep_the_draft(profile, first):
    resume = merged(profile, first, headline="Headline: Data Engineer | Python", summary="Summary: Short.",
                    bullets=[BulletChange(company_index=0, bullet_index=1, text="[1] Tuned the nightly SQL jobs"),
                             BulletChange(company_index=1, bullet_index=0, text="  ")])
    assert resume.headline == "Data Engineer | Python" and resume.summary == "Short."
    assert resume.experience[0].bullets[1] == "Tuned the nightly SQL jobs"
    assert resume.experience[1].bullets[0] == "Built things b"


def test_skills_are_swapped_or_removed_never_replaced_wholesale(profile, first):
    resume = merged(profile, first, skill_changes=[SkillChange(old="sql", new="PostgreSQL"),
                                                   SkillChange(old="Cobol", new="Kotlin")])
    assert resume.skills[0].skills == ["Python", "PostgreSQL"]  # an unknown "old" is ignored
    resume = merged(profile, first, skill_changes=[SkillChange(old="SQL", new="")])
    assert resume.skills[0].skills == ["Python"]
    with pytest.raises(ValueError, match="skills"):  # nothing left: the first draft is kept
        merged(profile, first, skill_changes=[SkillChange(old="SQL", new=""), SkillChange(old="Python", new="")])


def test_the_fix_prompt_shares_the_first_prompts_opening(profile, first, jd):
    out, draft = first
    prompt = revise_user(jd, profile.companies, "", "Total professional experience: 7 years.",
                         analysis=out.jd_analysis, plan=out.career_plan, draft=draft_text(draft),
                         problems="Too many bullets carry numbers.")
    opening = tailor_user(jd, profile.companies, "", "Total professional experience: 7 years.")
    head = opening[: opening.index("</candidate_hints>") + len("</candidate_hints>")]
    assert prompt.startswith(head)  # the provider can reuse its cache of this part
    assert f"{RETRY_MARKER}: Too many bullets carry numbers." in prompt and "<previous_draft>" in prompt
    assert "Must-have skills: Python, SQL" in prompt and "bullet_index" in prompt and "[0] Built things a" in prompt


class Recording(LLMClient):
    """Keeps every request and answer; `spoil` may change the fix answer."""

    def __init__(self, inner, spoil=None):
        self.inner, self.spoil, self.calls = inner, spoil, []

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_revision" and self.spoil:
            self.spoil(result)
        self.calls.append((kwargs, json.loads(json.dumps(result.data))))
        return result


def tailor(router, db, profile, jd, model="demo-concise"):
    return asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id=model))


def test_the_fix_round_asks_only_for_the_changes(router, db, registry, profile, jd):
    client = Recording(CopiesThePosting(FakeClient(registry.get("demo-concise"))))
    router.overrides["demo-concise"] = client
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and result.attempts == 2 and result.retries == 1
    (first_ask, first_answer), (fix_ask, fix_answer) = client.calls
    assert fix_ask["schema_name"] == "tailor_revision" and fix_ask["system"] == first_ask["system"]
    assert len(fix_answer["bullets"]) == 4 and fix_answer["headline"] == fix_answer["summary"] == ""
    assert fix_answer["skill_changes"] == [{"old": "High-Quality Code", "new": ""}]
    assert len(json.dumps(fix_answer)) < len(json.dumps(first_answer)) / 3  # far less to write
    assert "modern software development" in result.analysis.keywords  # the first draft's analysis is kept
    assert validate.posting_voice_problems(result.resume) == []


def test_a_fix_that_cannot_be_applied_keeps_the_first_draft(router, db, registry, profile, jd):
    def off_the_end(result):
        result.data["bullets"] = [{"company_index": 0, "bullet_index": 99, "text": "Somewhere else entirely"}]

    router.overrides["demo-concise"] = Recording(CopiesThePosting(FakeClient(registry.get("demo-concise"))),
                                                 spoil=off_the_end)
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and result.attempts == 2 and result.retries == 1
    assert any("Refine your craft" in b for e in result.resume.experience for b in e.bullets)  # the first draft


class BreaksTheFix(LLMClient):
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        if kwargs["schema_name"] == "tailor_revision":
            raise InvalidOutputError("not JSON")
        return await self.inner.generate_structured(**kwargs)


def test_an_invalid_fix_answer_keeps_the_first_draft_without_a_third_call(router, db, registry, profile, jd):
    client = BreaksTheFix(CopiesThePosting(FakeClient(registry.get("demo-concise"))))
    router.overrides["demo-concise"] = client
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and client.calls == 2
    assert any("Refine your craft" in b for e in result.resume.experience for b in e.bullets)


class Priced(LLMClient):
    def __init__(self, inner):
        self.inner = inner

    async def generate_structured(self, **kwargs):
        r = await self.inner.generate_structured(**kwargs)
        return LLMResult(data=r.data, usage=Usage(1000, 2000), served_model="claude-sonnet-5-5", raw_text=r.raw_text)


@pytest.mark.parametrize("spoil", [None, lambda r: r.data.update(bullets=[{"company_index": 9, "bullet_index": 0,
                                                                           "text": "x"}])])
def test_both_calls_are_paid_for_even_when_the_fix_is_refused(router, db, registry, profile, jd, monkeypatch, spoil):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    router.overrides["claude-sonnet-5-5"] = Priced(Recording(CopiesThePosting(FakeClient(registry.get("demo-concise"))),
                                                             spoil=spoil))
    result = tailor(router, db, profile, jd, model="claude-sonnet-5-5")
    assert result.status == "ok" and result.input_tokens == 2000 and result.output_tokens == 4000
    assert result.cost_usd == pytest.approx(2 * (1000 * 2 / 1e6 + 2000 * 10 / 1e6))


def test_the_demo_model_answers_a_clean_draft_with_an_empty_fix(registry, profile, jd, first):
    client = FakeClient(registry.get("demo-concise"))
    context = {"profile": profile.model_dump(), "jd": jd, "hints": "", "bullets": [2, 2, 2]}
    data = asyncio.run(client.generate_structured(system="", user="", schema={}, schema_name="tailor_output",
                                                  max_tokens=10, context=context)).data
    from app.schemas import TailorOutput

    draft = validate.build_resume(profile, TailorOutput.model_validate(data))
    fix = asyncio.run(client.generate_structured(
        system="", user="", schema={}, schema_name="tailor_revision", max_tokens=10,
        context={**context, "draft": draft.model_dump(exclude={"profile"})})).data
    assert fix == {"headline": "", "summary": "", "titles": [], "bullets": [], "skill_changes": []}
