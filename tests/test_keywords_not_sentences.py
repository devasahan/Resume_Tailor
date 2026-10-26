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
    assert validate.clean_terms(["Python, Java, and Go", "React, Redux, & TypeScript", "SQL, including window functions"]) == [
        "Python", "Java", "Go", "React", "Redux", "TypeScript", "SQL", "window functions"]
    assert validate.clean_terms(["C#, .NET", "Google OR-Tools", "Node.js."]) == ["C#", ".NET", "Google OR-Tools", "Node.js"]
    # the end of a list is not an item
    assert validate.clean_terms(["Python, Go, etc.", "AI tools like Copilot, Cursor, or similar",
                                 "AWS, GCP or other cloud platforms", "tools such as Jira, Confluence and more"]) == [
        "Python", "Go", "Copilot", "Cursor", "AWS", "GCP", "Jira", "Confluence"]
    # an "and" inside a name stays: only the one before a list's last item splits
    assert validate.clean_terms(["statements such as Profit and Loss, Balance Sheet and Cash Flow",
                                 "units like Labor and Delivery or Med-Surg"]) == [
        "Profit and Loss", "Balance Sheet", "Cash Flow", "Labor and Delivery", "Med-Surg"]


def test_brackets_in_the_analysis_keep_only_the_keyword():
    assert validate.clean_terms(["AWS (e.g., EC2, S3, Lambda)", "SQL (including window functions)", "Java (8, 11, 17)",
                                 "Cloud platforms such as Azure (e.g., AKS)", "Amazon Web Services (AWS)", "sql"]) == [
        "AWS", "SQL", "Java", "Azure", "Amazon Web Services (AWS)"]  # an abbreviation in brackets stays


def test_the_keyword_inside_posting_prose_is_kept():
    assert validate.clean_terms([
        "advanced SQL", "Strong Python skills", "experience with Kafka", "Proficiency in Excel", "AWS-hosted systems",
        "deep understanding of distributed systems", "advanced SQL statements", "the ability to work independently",
        "ownership from problem definition to production rollout"]) == [
        "SQL", "Python", "Kafka", "Excel", "AWS", "distributed systems", "SQL statements"]


def test_real_names_stay_whole():
    names = ["Deep Learning", "Visual Basic", "SOLID", "SOLID principles", "Solid.js", "Solid Edge", "self-hosted runners",
             "Microsoft-hosted agents", "Strong typing", "Good Manufacturing Practice (GMP)",
             "Good Documentation Practices (GDP)", "Basic Life Support (BLS)", "Advanced Cardiac Life Support (ACLS)",
             "Advanced Message Queuing Protocol (AMQP)", "Strong Customer Authentication (SCA)",
             "Large-Scale Scrum (LeSS)", "Certified Information Systems Security Professional (CISSP)",
             "Payment Card Industry Data Security Standard (PCI DSS)", "Lean Six Sigma Green Belt",
             "AWS Certified Developer - Associate", "Extract, Transform, Load (ETL)",
             "Governance, Risk, and Compliance (GRC)", "Water, Sanitation and Hygiene (WASH)", "Research and Development",
             "Microsoft Office: Word, Excel", "Java 8, 11, 17", "Ruby on Rails", "C++", "CI/CD", "CAN", "IT", "Ph.D.",
             "Continuous Integration and Continuous Delivery", "schema design", "User Experience", "Like-for-Like sales"]
    assert validate.clean_terms(names) == names
    assert validate.clean_terms(validate.clean_terms(names)) == names  # cleaning twice changes nothing


def test_the_analysis_keeps_keywords_only():
    jd = analysis(must_have_skills=["React", "advanced SQL statements", "AI tools like Copilot, Cursor, or Claude Code"],
                  nice_to_have_skills=["AWS-hosted systems", "Kotlin or Swift"],
                  keywords=["high-scale production systems", "Cloud platforms such as AWS (e.g., EC2, S3)"])
    validate.clean_analysis(jd)
    assert jd.must_have_skills == ["React", "SQL statements", "Copilot", "Cursor", "Claude Code"]
    assert jd.nice_to_have_skills == ["AWS", "Kotlin", "Swift"] and jd.keywords == ["production systems", "AWS"]
    assert jd.key_responsibilities == ["Debug complex issues across frontend, backend, and data boundaries"]
    before = jd.model_dump()
    assert validate.clean_analysis(jd).model_dump() == before  # each draft cleans it again: nothing changes


def test_skills_lists_are_split_into_their_tools(profile):
    out = make_output(skills=[SkillGroup(category="AI", skills=[
        "AI tools like Copilot, Cursor, or Claude Code", "Claude Code", "AWS (ECS, SQS)", ".NET", "Python, Go, and Rust",
        "AWS (e.g., EC2, S3)", "AWS (ECS (Fargate), SQS)", "Extract, Transform, Load (ETL)", "-", "and"])])
    assert validate.build_resume(profile, out).skills[0].skills == [
        "Copilot", "Cursor", "Claude Code", "AWS (ECS, SQS)", ".NET", "Python", "Go", "Rust", "AWS (EC2, S3)",
        "AWS (ECS (Fargate), SQS)", "Extract, Transform, Load (ETL)"]


def test_skills_written_as_phrases_are_found(profile):
    resume = validate.build_resume(profile, make_output())
    resume.skills = [SkillGroup(category="Core", skills=[
        "SQL", "advanced SQL statements", "AWS-hosted systems", "Amazon Web Services (AWS)", "AWS (ECS, SQS)",
        "Deep Learning", "high-scale production systems", "Test-Driven Development", "Experience with Kafka",
        "Basic Life Support (BLS)", "CAN", "Solid Edge", "Certified Information Systems Security Professional (CISSP)",
        "Advanced SQL"])]
    assert validate.skill_phrase_problems(resume) == [
        "advanced SQL statements", "AWS-hosted systems", "high-scale production systems", "Experience with Kafka"]


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
        "'Use AI tools like Copilot, Cursor, or Claude Code daily' (Northwind Labs)",
        "'Contributed to architecture discussions' (Contoso Systems)",
    ]


def test_a_copy_is_found_however_it_is_spelled(profile):
    jd = ("Identify and resolve performance bottlenecks. Own features end-to-end from design to deployment. Improve "
          "the patient experience across all touchpoints. Mentor junior engineers through code reviews and design "
          "discussions.")
    resume = validate.build_resume(profile, make_output())
    resume.skills = [SkillGroup(category="Practices", skills=["Code Review", "System Design", "Mentoring"])]
    resume.experience[0].bullets = ["Identified and resolved performance bottlenecks in the billing API.",
                                    "Owned features end to end from design to deployment."]
    resume.experience[1].bullets = ["Improve the patient experience across all touchpoints.",  # mostly small words
                                    "Mentor junior engineers through code reviews and design discussions."]
    assert [span.split("' (")[0] + "'" for span in validate.copied_posting_spans(resume, jd, analysis())] == [
        "'Identified and resolved performance bottlenecks'", "'Owned features end to end from design to deployment'",
        "'Improve the patient experience across all touchpoints'",
        "'Mentor junior engineers through code reviews and design discussions'"]  # own skills hide no copy


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
    assert "SQL statements" in result.analysis.must_have_skills  # ... its keyword is
    assert all(validate.is_keyword(term) for term in result.analysis.must_have_skills)
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
