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


def test_standard_names_with_you_or_your_are_not_posting_voice(profile):
    resume = validate.build_resume(profile, make_output())
    resume.summary = "Fintech engineer. Our clients rely on it."  # a sentence that starts with Our still counts
    resume.experience[0].bullets = [f"Built Know Your Customer (KYC) document checks {LONG}",
                                    f"Moved hosting to pay-as-you-go cloud billing {LONG}"]
    resume.experience[1].bullets = [f"Rolled out a Bring Your Own Device (BYOD) policy {LONG}",
                                    f"Cut deploy time so we could ship daily {LONG}"]
    assert validate.posting_voice_problems(resume) == [resume.summary, resume.experience[1].bullets[1]]


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


def test_hyphenated_phrases_count_and_tool_names_may_recur(profile):
    resume = validate.build_resume(profile, make_output())
    resume.summary = f"Data engineer who applies test-driven development on Azure Data Factory {LONG}."
    resume.experience[0].bullets = [f"Brought test-driven development to Azure Data Factory pipelines {LONG}",
                                    f"Ran Ruby on Rails services next to Azure Data Factory {LONG}"]
    resume.experience[1].bullets = [f"Used test-driven development for Ruby on Rails jobs {LONG}",
                                    f"Moved batch loads to Azure Data Factory {LONG}"]
    resume.experience[2].bullets = [f"Kept Ruby on Rails releases steady {LONG}", f"Wrote Azure Data Factory alerts {LONG}"]
    analysis = lean_analysis(must_have_skills=["test-driven development", "Azure Data Factory"],
                             keywords=["Ruby on Rails", "Microsoft Power BI"])
    assert validate.repeated_phrase_problems(resume, analysis) == ["'test-driven development' (3 times)"]


def test_qualities_are_not_skills(profile):
    resume = validate.build_resume(profile, make_output())
    resume.skills = [
        SkillGroup(category="Practices", skills=["Test-Driven Development", "Clean Architecture", "Site Reliability Engineering",
                                                 "Scalable Vector Graphics (SVG)", "Robust Optimization", "Quality Assurance"]),
        SkillGroup(category="Quality", skills=["High-Quality Code", "Scalable and Maintainable Applications",
                                               "Low-Maintenance Software Solutions", "Robust APIs", "High Quality Code"]),
    ]
    assert validate.quality_skill_problems(resume) == [
        "High-Quality Code", "Scalable and Maintainable Applications", "Low-Maintenance Software Solutions", "Robust APIs",
        "High Quality Code"]


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
    assert 'Never write "you", "your", "we" or "our"' in text and "Know Your Customer" in text
    assert "even when the JD requires it" in text


DOUBLED = f"Constructed RESTful APIs with Redis caching layers to serve multi-tenant tenant data schemas {LONG}"


def test_a_word_repeated_by_mistake_is_found_and_dropped(profile):
    """Seen in a real result: 'multi-tenant tenant data schemas'."""
    assert validate.drop_repeated_words("to serve multi-tenant tenant data schemas.") == "to serve multi-tenant data schemas."
    assert validate.drop_repeated_words("Led the the migration") == "Led the migration"
    for fine in ("Built a plug-in in Java", "an add-on on Shopify", "what it had had", "Java, Java and Kotlin",
                 "built-in in-memory cache", "Kotlin\nKotlin"):
        assert validate.drop_repeated_words(fine) == fine
    resume = validate.build_resume(profile, make_output())
    resume.experience[1].bullets[0] = DOUBLED
    resume.summary = "Backend engineer who owns the the full delivery cycle."
    assert validate.repeated_word_problems(resume) == ["the the", "multi-tenant tenant"]
    assert validate.fix_repeated_words(resume) == ["the the", "multi-tenant tenant"]
    assert "owns the full" in resume.summary and "multi-tenant data schemas" in resume.experience[1].bullets[0]
    assert validate.repeated_word_problems(resume) == []
    _greeting, paragraphs, _closing = validate.clean_cover_letter("Dear Sam,", ["I built multi-tenant tenant APIs.", "Two."],
                                                                  "Sincerely,")
    assert paragraphs[0] == "I built multi-tenant APIs."


class RepeatsAWord(LLMClient):
    """Puts 'multi-tenant tenant' in a bullet: in the first draft only, or (always=True) in every draft."""

    def __init__(self, inner, always=False):
        self.inner, self.always, self.prompts = inner, always, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and (self.always or RETRY_MARKER not in kwargs["user"]):
            result.data["sections"]["experience"][1]["bullets"][0] = DOUBLED
        if kwargs["schema_name"] == "tailor_revision" and self.always:  # the fix writes the repeat again
            edits = [e for e in result.data["bullets"] if (e["company_index"], e["bullet_index"]) != (1, 0)]
            result.data["bullets"] = edits + [{"company_index": 1, "bullet_index": 0, "text": DOUBLED}]
        return result


def test_the_rewrite_pass_asks_to_fix_a_repeated_word(router, db, registry, profile, jd):
    client = RepeatsAWord(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and len(client.prompts) == 2
    assert "repeated by mistake: 'multi-tenant tenant'" in client.prompts[1]
    assert validate.repeated_word_problems(result.resume) == []
    assert not any(note.startswith("Removed a repeated word") for note in result.notes)  # the rewrite fixed it


def test_a_repeat_the_rewrite_left_is_removed_before_anyone_sees_it(router, db, registry, profile, jd):
    router.overrides["demo-concise"] = RepeatsAWord(FakeClient(registry.get("demo-concise")), always=True)
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and validate.repeated_word_problems(result.resume) == []
    assert "multi-tenant data schemas" in " ".join(b for exp in result.resume.experience for b in exp.bullets)
    assert "Removed a repeated word: 'multi-tenant tenant'." in result.notes
