"""Match the keywords (SQL, React, AWS), not the sentences: the job description's terms are used as keywords, and
its sentences are never copied into the resume. Shaped on a real result that listed "advanced SQL statements" and
"AWS-hosted systems" as skills and pasted the posting's lists of duties into the bullets."""
import asyncio

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import run_checks, run_tailor
from app.prompts import RETRY_MARKER, tailor_system
from app.schemas import JDAnalysis, SkillGroup, TailorResult

from .conftest import SAMPLE_JD
from .test_validate import make_output

POSTING = (
    "Senior Full-Stack Engineer at Litware. What you will do: ship features with React, TypeScript and PostgreSQL, "
    "taking ownership from problem definition to production rollout. Debug complex issues across frontend, backend, "
    "and data boundaries. Contribute to architecture discussions. Use AI tools like Copilot, Cursor, or Claude Code "
    "daily. What you bring: 5+ years of experience building web products, advanced SQL statements and AWS-hosted "
    "systems."
)
COPIED = [  # sentences of POSTING, as a draft pastes them
    ("Debug complex issues across frontend, backend, and data boundaries in the checkout service, so support closes "
     "payment tickets within a day."),
    "Use AI tools like Copilot, Cursor, or Claude Code daily to draft tests for the billing API.",
]


def analysis(**overrides) -> JDAnalysis:
    fields = {
        "target_role": "Senior Full-Stack Engineer", "seniority": "Senior", "domain": "SaaS",
        "must_have_skills": ["React", "TypeScript", "PostgreSQL", "SQL", "AWS"],
        "nice_to_have_skills": ["Copilot", "Cursor", "Claude Code"], "keywords": ["React", "SQL", "AWS"],
        "key_responsibilities": ["Debug complex issues across frontend, backend, and data boundaries"],
    }
    fields.update(overrides)
    return JDAnalysis(**fields)


# ------------------------------------------------------------------ the analysis keeps keywords only
def test_a_list_of_examples_becomes_the_tools_themselves():
    assert validate.clean_terms(["AI tools like Copilot, Cursor, or Claude Code"]) == ["Copilot", "Cursor", "Claude Code"]
    assert validate.clean_terms(["TypeScript or JavaScript", "cloud platforms such as AWS and GCP"]) == [
        "TypeScript", "JavaScript", "AWS", "GCP"]
    assert validate.clean_terms(["AWS (ECS, SQS)", "Amazon Web Services (AWS)", "SQL", "sql"]) == [
        "AWS (ECS, SQS)", "Amazon Web Services (AWS)", "SQL"]  # a list in brackets stays with its keyword
    assert validate.clean_terms(["C#, .NET", "Google OR-Tools", "Node.js."]) == ["C#", ".NET", "Google OR-Tools", "Node.js"]
    # the end of a list is not an item
    assert validate.clean_terms(["Python, Go, etc.", "AI tools like Copilot, Cursor, or similar",
                                 "AWS, GCP or other cloud platforms", "tools such as Jira, Confluence and more"]) == [
        "Python", "Go", "Copilot", "Cursor", "AWS", "GCP", "Jira", "Confluence"]


def test_pieces_of_the_posting_are_not_keywords():
    pieces = ["advanced SQL statements", "AWS-hosted systems", "high-scale production systems", "experience with Kafka",
              "ownership from problem definition to production rollout", "Strong Python skills"]
    assert validate.clean_terms([*pieces, "SQL", "AWS"]) == ["SQL", "AWS"]
    names = ["Deep Learning", "Visual Basic", "SOLID", "Solid.js", "self-hosted runners", "Good Manufacturing Practice (GMP)",
             "Strong Customer Authentication (SCA)", "Large-Scale Scrum (LeSS)", "Ruby on Rails", "C++", "CI/CD",
             "Continuous Integration and Continuous Delivery", "schema design", "User Experience", "Like-for-Like sales"]
    assert validate.clean_terms(names) == names
    assert validate.clean_terms(validate.clean_terms(names)) == names  # cleaning twice changes nothing


def test_the_analysis_keeps_keywords_only():
    jd = analysis(must_have_skills=["React", "advanced SQL statements", "AI tools like Copilot, Cursor, or Claude Code"],
                  nice_to_have_skills=["AWS-hosted systems", "Kotlin or Swift"],
                  keywords=["high-scale production systems", "SQL"])
    validate.clean_analysis(jd)
    assert jd.must_have_skills == ["React", "Copilot", "Cursor", "Claude Code"]
    assert jd.nice_to_have_skills == ["Kotlin", "Swift"] and jd.keywords == ["SQL"]
    assert jd.key_responsibilities == ["Debug complex issues across frontend, backend, and data boundaries"]


def test_skills_lists_are_split_into_their_tools(profile):
    out = make_output(skills=[SkillGroup(category="AI", skills=["AI tools like Copilot, Cursor, or Claude Code",
                                                                  "Claude Code", "AWS (ECS, SQS)", ".NET"])])
    assert validate.build_resume(profile, out).skills[0].skills == [
        "Copilot", "Cursor", "Claude Code", "AWS (ECS, SQS)", ".NET"]


def test_skills_written_as_phrases_are_found(profile):
    resume = validate.build_resume(profile, make_output())
    resume.skills = [SkillGroup(category="Core", skills=[
        "SQL", "advanced SQL statements", "AWS-hosted systems", "Amazon Web Services (AWS)", "AWS (ECS, SQS)",
        "Deep Learning", "high-scale production systems", "Test-Driven Development"])]
    assert validate.skill_phrase_problems(resume) == [
        "advanced SQL statements", "AWS-hosted systems", "high-scale production systems"]


# ------------------------------------------------------------------ copied sentences
def test_sentences_copied_from_the_posting_are_found_in_any_tense(profile):
    resume = validate.build_resume(profile, make_output())
    resume.summary = ("Senior Full-Stack Engineer with 8+ years of experience building SaaS billing tools, who takes "
                      "ownership from problem definition to production rollout.")
    resume.experience[0].bullets = list(COPIED)
    resume.experience[1].bullets = ["Contributed to architecture discussions for the reporting service."]
    assert validate.copied_posting_spans(resume, POSTING, analysis()) == [
        "'takes ownership from problem definition to production rollout' (summary)",  # the posting says 'taking'
        "'Debug complex issues across frontend, backend, and data boundaries' (Northwind Labs)",
        "'Use AI tools like Copilot' (Northwind Labs)",
        "'Contributed to architecture discussions' (Contoso Systems)",
    ]


def test_keywords_and_tool_lists_from_the_posting_are_fine(profile):
    resume = validate.build_resume(profile, make_output())
    resume.summary = "Senior Full-Stack Engineer with 8+ years of experience building SaaS billing tools."
    resume.experience[0].bullets = [
        "Moved the invoice screens to React, TypeScript and PostgreSQL, so finance closes the month two days sooner.",
        "Wrote SQL checks on AWS that catch broken imports before customers see them.",
        "Drafted tests with Copilot, Cursor, or Claude Code while keeping reviews human.",
    ]
    assert validate.copied_posting_spans(resume, POSTING, analysis()) == []


def test_run_checks_notes_what_is_left(profile):
    resume = validate.build_resume(profile, make_output())
    resume.experience[0].bullets = list(COPIED)
    resume.skills[0].skills.append("advanced SQL statements")
    result = TailorResult(requested_model="m", resume=resume, analysis=analysis())
    run_checks(result, POSTING)
    copied = [w for w in result.ats_warnings if w.startswith("Copied word for word from the job description")]
    assert len(copied) == 1 and "'Debug complex issues across frontend, backend, and data boundaries'" in copied[0]
    assert "Skills written as phrases, not keywords: advanced SQL statements." in " ".join(result.ats_warnings)


# ------------------------------------------------------------------ the one fix round
class PastesThePosting(LLMClient):
    """First draft only: pastes two of the posting's sentences into bullets, lists a phrase as a skill and as a
    must-have, the way the real result did."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if RETRY_MARKER not in kwargs["user"] and kwargs["schema_name"] == "tailor_output":
            data = result.data
            data["jd_analysis"]["must_have_skills"].append("advanced SQL statements")
            data["sections"]["experience"][0]["bullets"][:2] = COPIED
            data["sections"]["skills"][0]["skills"].append("AWS-hosted systems")
        return result


def test_the_one_fix_round_removes_copied_sentences_and_phrase_skills(router, db, registry, profile):
    plain = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=POSTING, model_id="demo-concise"))
    assert plain.status == "ok" and plain.retries == 0  # the demo's own draft needs no fix round
    client = PastesThePosting(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=POSTING, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    feedback = client.prompts[1]
    assert "copy the job description word for word" in feedback and "Match the keyword, not the sentence" in feedback
    assert "'Debug complex issues across frontend, backend, and data boundaries' (Northwind Labs)" in feedback
    assert "'Use AI tools like Copilot, Cursor, or Claude Code daily' (Northwind Labs)" in feedback
    assert "phrases from the job description, not keywords: 'AWS-hosted systems'" in feedback
    assert "advanced SQL statements" not in result.analysis.must_have_skills  # never pushed into a bullet
    assert validate.copied_posting_spans(result.resume, POSTING, result.analysis) == []
    assert validate.skill_phrase_problems(result.resume) == []


def test_a_normal_draft_needs_no_fix_round(router, db, profile):
    for jd in (SAMPLE_JD, POSTING):
        result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
        assert result.status == "ok" and result.retries == 0
        assert validate.copied_posting_spans(result.resume, jd, result.analysis) == []


def test_prompt_matches_keywords_not_sentences():
    text = tailor_system("estimate")
    assert "Match the keyword, not the sentence" in text and "a run of four or more words" in text
    assert '"SQL", not "advanced SQL statements"' in text and '"AWS", not "AWS-hosted systems"' in text
    assert "take the tools themselves" in text and "never a piece of a JD sentence" in text
    assert "never as a phrase from the JD" in text and "shortest exact phrase" not in text
