"""The tech line under each job ("Tech: Python, FastAPI, AWS (ECS, SQS)"): cleaned, checked, kept through fixes."""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient, tech_line
from app.llm.schema import strict_json_schema
from app.pipeline import validate
from app.pipeline.tailor import draft_text, run_tailor
from app.schemas import (
    ExperienceOut,
    TailorOutput,
    TailorResult,
    TailorRevision,
    TechChange,
)

from .test_posting_coverage import _with_titles
from .test_validate import make_output


def with_tech(profile, *lines):
    out = make_output()
    for exp, tech in zip(out.sections.experience, lines, strict=False):
        exp.tech = list(tech)
    return out, validate.build_resume(profile, out)


def test_the_model_must_write_a_tech_line_per_job_and_may_change_it_in_a_fix():
    schema = strict_json_schema(TailorOutput)
    job = schema["properties"]["sections"]["properties"]["experience"]["items"]
    assert "tech" in job["required"] and job["properties"]["tech"]["type"] == "array"
    revision = strict_json_schema(TailorRevision)
    assert "tech_changes" in revision["required"]


@pytest.mark.parametrize("raw,clean", [
    (["Tech: Python, FastAPI, AWS (ECS, SQS)"], ["Python", "FastAPI", "AWS (ECS, SQS)"]),  # one joined line, a label
    (["Python", "python", " Docker. ", ""], ["Python", "Docker"]),  # duplicates, a full stop, empty items
    ([".NET", "ASP.NET Core", "C#", "Node.js"], [".NET", "ASP.NET Core", "C#", "Node.js"]),
    (["Tech stack: Redis; Celery"], ["Redis", "Celery"]),
    ([f"Tool {i}" for i in range(20)], [f"Tool {i}" for i in range(validate.MAX_TECH_ITEMS)]),  # stays short
    ([], []),  # a non-technical role
])
def test_tech_lines_are_cleaned(profile, raw, clean):
    out = make_output()
    out.sections.experience[0].tech = raw
    assert validate.build_resume(profile, out).experience[0].tech == clean


def test_runs_saved_before_the_tech_line_still_load(router, db, profile, jd):
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    old = result.model_dump()
    for exp in old["resume"]["experience"]:
        del exp["tech"]
    assert [e.tech for e in TailorResult.model_validate(old).resume.experience] == [[], [], []]


# ------------------------------------------------------------------ the fix round
def test_the_draft_shows_each_tech_line_after_its_bullets(profile):
    _out, draft = with_tech(profile, ["Python", "dbt"])
    text = draft_text(draft)
    assert "[1] Improved things\nTech: Python, dbt\ncompany_index 1" in text and "Tech: (none)" in text


def test_a_fix_keeps_the_tech_lines_it_does_not_change(profile):
    out, draft = with_tech(profile, ["Python", "dbt"], ["Java"], ["SQL"])
    revised = validate.apply_revision(out, draft, TailorRevision(
        summary="A new summary about Python and SQL work.", tech_changes=[TechChange(company_index=1, tech=["Scala"])]))
    resume = validate.build_resume(profile, revised)
    assert [e.tech for e in resume.experience] == [["Python", "dbt"], ["Scala"], ["SQL"]]
    cleared = validate.build_resume(profile, validate.apply_revision(
        out, draft, TailorRevision(tech_changes=[TechChange(company_index=0, tech=[])])))  # [] removes that line
    assert [e.tech for e in cleared.experience] == [[], ["Java"], ["SQL"]]


@pytest.mark.parametrize("changes,message", [
    ([TechChange(company_index=7, tech=["Go"])], "no company_index"),
    ([TechChange(company_index=0, tech=["Go"]), TechChange(company_index=0, tech=["Rust"])], "twice"),
])
def test_a_tech_change_that_cannot_be_applied_is_refused(profile, changes, message):
    out, draft = with_tech(profile, ["Python"])
    with pytest.raises(ValueError, match=message):
        validate.apply_revision(out, draft, TailorRevision(tech_changes=changes))


class OldStackInOldJob(LLMClient):
    """Writes OpenTofu (common from 2024) into the tech line of a job that ended in 2019."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output":
            result.data["sections"]["experience"][2]["tech"].append("OpenTofu")
        return result


def test_a_tool_that_did_not_exist_yet_in_a_tech_line_is_fixed_in_one_round(router, db, registry, profile, jd):
    client = OldStackInOldJob(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and client.calls == 2 and result.retries == 1
    assert "OpenTofu" not in result.resume.experience[2].tech and result.resume.experience[2].tech
    assert all(e.tech for e in result.resume.experience)  # the other lines survived the fix


# ------------------------------------------------------------------ checks
def test_era_and_quality_checks_read_the_tech_lines():
    resume = _with_titles(["Backend Engineer", "Software Engineer II", "Software Engineer", "Software Engineer",
                           "Junior Software Engineer"])
    resume.experience[3].tech = ["Python", "OpenTofu", "LangChain"]  # Q2 Holdings ended May 2021
    problems = " ".join(validate.era_problems(resume))
    assert "OpenTofu, which only became common in 2024" in problems and "LangChain" in problems
    resume.experience[0].tech = ["Python", "Scalable Architecture"]
    assert validate.quality_skill_problems(resume) == ["Scalable Architecture"]


def test_tech_lines_count_for_the_ats_but_never_prove_a_must_have(profile):
    _out, resume = with_tech(profile, ["Airflow"])
    assert "Airflow" in validate.resume_text(resume) and "Airflow" not in validate.evidence_text(resume)


def test_tools_the_posting_does_not_name_are_flagged_once_per_job_and_never_as_figures(profile):
    _out, resume = with_tech(profile, ["Python", "AWS (S3, EC2)", "Java 17"], ["SQL"])
    flags = validate.claim_flags(resume, jd="Python, SQL and AWS S3", hints="")
    tech = [f for f in flags if f.text.startswith("Tech: ")]
    assert len(tech) == 1 and tech[0].section == "experience:0" and tech[0].level == "info"
    assert "AWS (S3, EC2), Java 17" in tech[0].reason  # EC2 and Java 17 are not in the posting; Python is
    assert not any(f.level == "verify" for f in flags)  # 'S3', 'EC2' and '17' are names, not invented numbers
    assert not [f for f in validate.claim_flags(resume, jd="Python, SQL, AWS, S3, EC2, Java 17", hints="")
                if f.text.startswith("Tech: ")]


def test_short_tool_names_are_matched_as_whole_words(profile):
    """'Go' is not in 'good', 'Java' not in 'JavaScript', 'Rust' not in 'trusted', 'R' not in 'React'."""
    _out, resume = with_tech(profile, ["Go", "Java", "Rust", "R", "Scala"])
    jd = "A good, trusted JavaScript and React developer for scalable systems."
    flag = next(f for f in validate.claim_flags(resume, jd=jd, hints="") if f.text.startswith("Tech: "))
    assert "(Go, Java, Rust, R, Scala)" in flag.reason
    assert not [f for f in validate.claim_flags(resume, jd="Go, Java, Rust, R and Scala", hints="")
                if f.text.startswith("Tech: ")]


def test_the_demo_bullets_only_name_tools_of_each_jobs_time(router, db, profile):
    jd = "Senior Platform Engineer. Kubernetes, Terraform and GitHub Actions on AWS, with Python."
    old = profile.model_copy(update={"companies": [*profile.companies[:2],
                                                    profile.companies[2].model_copy(update={"end": "Dec 2016"})]})
    result = asyncio.run(run_tailor(router, db, profile=old, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.attempts == 1 and validate.era_problems(result.resume) == []
    assert not any("Kubernetes" in b or "GitHub Actions" in b for b in result.resume.experience[2].bullets)


def test_the_demo_tech_line_only_has_tools_of_that_time():
    keywords = ["Python", "Kubernetes", "AWS", "Communication", "CI/CD"]
    assert tech_line(keywords, 0, "Present") == ["Python", "Kubernetes", "AWS", "FastAPI", "Pandas", "pytest",
                                                  "Helm", "Amazon S3"]
    old = tech_line(keywords, 3, "Dec 2016")
    assert "Kubernetes" not in old and "FastAPI" not in old and "Helm" not in old  # common from 2017-2021
    assert "Communication" not in old and "CI/CD" not in old  # practices are not tools
    assert tech_line(["Communication", "Salesforce"], 0, "Present") == ["Salesforce"]


def test_an_edit_that_leaves_out_the_tech_lines_keeps_them(client):
    from .test_api import save_profile, tailor

    save_profile(client)
    run = tailor(client).json()
    body = {"headline": run["resume"]["headline"], "summary": run["resume"]["summary"],
            "skills": run["resume"]["skills"],
            "experience": [{k: e[k] for k in ("company_index", "job_title", "bullets")} for e in run["resume"]["experience"]]}
    edited = client.put(f"/api/runs/{run['run_id']}/resume", json=body).json()
    assert [e["tech"] for e in edited["resume"]["experience"]] == [e["tech"] for e in run["resume"]["experience"]]
    body["experience"][1]["tech"] = []  # sent empty on purpose: the line goes
    edited = client.put(f"/api/runs/{run['run_id']}/resume", json=body).json()
    assert edited["resume"]["experience"][1]["tech"] == []


def test_experience_out_still_builds_without_a_tech_line():
    assert ExperienceOut(company_index=0, job_title="x", bullets=["y"]).tech == []
