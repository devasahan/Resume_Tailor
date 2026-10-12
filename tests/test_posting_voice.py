"""Text a recruiter reads as copied from the posting: its own voice, repeated phrases, qualities listed as skills."""
import asyncio

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, tailor_system
from app.schemas import JDAnalysis, SkillGroup

from .test_validate import make_output

LONG = "with the platform team so every release stays predictable for the people who depend on it"


def lean_analysis(**overrides) -> JDAnalysis:
    fields = {
        "target_role": "Java Engineer", "seniority": "Mid", "domain": "Consulting",
        "must_have_skills": ["Java", "test-driven development", "modern software development"],
        "nice_to_have_skills": ["Spring Boot"], "keywords": ["low-maintenance software solutions", "pair programming"],
        "key_responsibilities": ["Coach and guide client teams"],
    }
    fields.update(overrides)
    return JDAnalysis(**fields)


def test_posting_voice_is_found_in_the_summary_and_bullets(profile):
    resume = validate.build_resume(profile, make_output())
    resume.summary = "Java Engineer with 7+ years of experience building US-based payroll services."
    resume.experience[0].bullets = [
        f"Refine your craft by spearheading weekly architecture reviews {LONG}",
        f"Ran weekly architecture reviews and knowledge shares on concurrency patterns {LONG}",
    ]
    resume.experience[1].bullets = [f"Helped our clients adopt pairing {LONG}"]
    assert validate.posting_voice_problems(resume) == [resume.experience[0].bullets[0], resume.experience[1].bullets[0]]
    resume.summary = "We build Java services that you can trust."
    assert validate.posting_voice_problems(resume)[0] == resume.summary


def test_a_posting_phrase_used_more_than_twice_is_stuffing(profile):
    resume = validate.build_resume(profile, make_output())
    resume.headline = "Java Engineer | Modern Software Development"  # the headline and skills do not count
    resume.skills = [SkillGroup(category="Practices", skills=["Modern Software Development", "Spring Boot"])]
    resume.summary = f"Java Engineer who brings modern software development {LONG}."
    resume.experience[0].bullets = [f"Coached teams on modern software-development practices {LONG}",
                                    f"Built Spring Boot services and low-maintenance software solutions {LONG}"]
    resume.experience[1].bullets = [f"Introduced modern software development approaches {LONG}",
                                    f"Moved billing to Spring Boot {LONG}"]
    resume.experience[2].bullets = [f"Kept Spring Boot upgrades and low-maintenance software solutions on track {LONG}",
                                    f"Maintained Spring Boot services {LONG}"]
    # 3 times (a hyphen counts as the same phrase); 'low-maintenance software solutions' twice is fine;
    # Spring Boot four times is fine too: a tool recurs wherever it was used
    assert validate.repeated_phrase_problems(resume, lean_analysis()) == ["'modern software development' (3 times)"]


def test_qualities_are_not_skills(profile):
    resume = validate.build_resume(profile, make_output())
    resume.skills = [
        SkillGroup(category="Practices", skills=["Test-Driven Development", "Clean Architecture", "Site Reliability Engineering"]),
        SkillGroup(category="Quality", skills=["High-Quality Code", "Scalable and Maintainable Applications",
                                               "Low-Maintenance Software Solutions", "Robust APIs"]),
    ]
    assert validate.quality_skill_problems(resume) == [
        "High-Quality Code", "Scalable and Maintainable Applications", "Low-Maintenance Software Solutions", "Robust APIs"]


class CopiesThePosting(LLMClient):
    """First draft only: copies the posting's voice, repeats one of its phrases and lists a quality as a skill."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if RETRY_MARKER not in kwargs["user"] and kwargs["schema_name"] == "tailor_output":
            data = result.data
            data["jd_analysis"]["keywords"].append("modern software development")
            jobs = data["sections"]["experience"]
            jobs[0]["bullets"][0] = f"Refine your craft by leading weekly architecture reviews {LONG}"
            for job, index in ((jobs[0], 1), (jobs[1], 0), (jobs[1], 1)):
                job["bullets"][index] = f"Brought modern software development practices to the data platform {LONG}"
            data["sections"]["skills"][0]["skills"].append("High-Quality Code")
        return result


def test_the_one_rewrite_pass_fixes_copied_posting_text(router, db, registry, profile, jd):
    client = CopiesThePosting(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    feedback = client.prompts[1]
    assert "copy the posting's own voice" in feedback and "Refine your craft" in feedback
    assert "'modern software development' (3 times)" in feedback and "at most twice" in feedback
    assert "qualities, not skills: 'High-Quality Code'" in feedback
    assert validate.posting_voice_problems(result.resume) == []
    assert validate.quality_skill_problems(result.resume) == []
    assert validate.repeated_phrase_problems(result.resume, result.analysis) == []


def test_prompt_states_the_posting_copy_rules():
    text = tailor_system("estimate")
    assert "at most twice across the summary and bullets" in text and '"TDD"' in text
    assert "never a quality or outcome" in text and '"High-Quality Code"' in text
    assert "never copy its own setting into earlier jobs" in text and '"refine your craft"' in text
    assert 'Never write "you", "your", "we" or "our"' in text
