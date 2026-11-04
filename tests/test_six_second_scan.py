"""What a recruiter reads in a six-second scan: the title line under the name (the role only), the company and title
of each job, and each job's first bullet, which opens with the job's biggest win, its size and its result."""
import asyncio
import io
from datetime import date

import pytest
from docx import Document
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.fit import bullet_budget, try_styles
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, tailor_system
from app.render.docx import resume_docx
from app.render.pdf import Sheet, render_resume_pdf, resume_layout
from app.schemas import Company, FitStyle, Profile

from .test_validate import make_output

NO_FIGURE = ("Rebuilt the order export behind the finance dashboards with Python and SQL, so month-end numbers matched "
             "the ledger without manual fixes.")
LEAD = ("Built the matchmaking service for 83K concurrent players in Java on AWS, cutting queue-to-lobby time by 22% "
        "with region-aware queues.")


# ------------------------------------------------------------------ the first bullet
def test_prompt_opens_each_job_with_its_biggest_win():
    estimate, words_only = tailor_system("estimate"), tailor_system("hints_only")
    for text in (estimate, words_only):
        assert "reads only the first bullet of each job" in text and "never a general duty" in text
        assert "Check that each job's first bullet is its strongest achievement" in text
    assert "Every job's first bullet is always one of them, and it gives two figures" in estimate
    assert "83K concurrent players" in estimate and "at most one figure in any other bullet" in estimate
    assert "first bullet shows its size and its result" in words_only and "83K" not in words_only


def test_jobs_that_open_without_a_figure_are_found(profile):
    resume = validate.build_resume(profile, make_output())
    resume.experience[0].bullets = [LEAD, NO_FIGURE]
    resume.experience[1].bullets = [NO_FIGURE, "Cut report runtime by 23.4% with incremental loads in Airflow."]
    resume.experience[2].bullets = ["Moved reporting to dbt in 2018, keeping every model tested.", LEAD]  # a year
    assert validate.first_bullet_problems(resume) == ["Contoso Systems", "Fabrikam"]


def test_one_flag_names_every_estimate_in_a_bullet(profile):
    resume = validate.build_resume(profile, make_output())
    resume.experience[0].bullets = [LEAD]
    verify = [f for f in validate.claim_flags(resume, "jd", "") if f.level == "verify"]
    assert len(verify) == 1 and "'83K' and '22%' are estimates" in verify[0].reason


class OpensWithoutFigures(LLMClient):
    """First draft only: every job opens with a bullet that has no figure."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            for job in result.data["sections"]["experience"]:
                job["bullets"][0] = NO_FIGURE
        return result


def test_the_one_fix_round_puts_the_figures_first(router, db, registry, profile, jd):
    client = OpensWithoutFigures(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    assert "These jobs open without a measured result: Northwind Labs, Contoso Systems, Fabrikam" in client.prompts[1]
    assert "at most 2 bullets with figures" in client.prompts[1]
    assert validate.first_bullet_problems(result.resume) == [] and result.metrics_mode == "estimate"
    assert not any("do not open with a measured result" in w for w in result.ats_warnings)


def test_every_demo_job_opens_with_its_size_and_result(router, db, profile, jd):
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 0
    assert all(len(validate._numbers_in(exp.bullets[0])) == 2 for exp in result.resume.experience)
    assert validate.metric_density_problems(result.resume) == []  # still at most 2 figure bullets per job


def test_no_figures_mode_asks_for_none(router, db, profile, jd):
    db.set_settings({"metrics_mode": "hints_only"})
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 0 and result.metrics_mode == "hints_only"
    assert not any(validate.has_figure(b) for exp in result.resume.experience for b in exp.bullets)
    assert not any("measured result" in w for w in result.ats_warnings)


# ------------------------------------------------------------------ company first, the title below
def test_the_company_comes_first_and_the_title_has_its_own_line(profile):
    resume = validate.build_resume(profile, make_output())
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(render_resume_pdf(
        resume, FitStyle(stacked=True))[0])).pages)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    row = lines.index("Northwind Labs | Remote Mar 2022 - Present")
    assert lines[row + 1] == "Engineer 0"
    one_line = PdfReader(io.BytesIO(render_resume_pdf(resume, FitStyle())[0])).pages[0].extract_text()
    assert "Northwind Labs | Engineer 0 | Remote" in " ".join(one_line.split())


def test_dates_stay_on_the_company_line_for_layout_readers(profile):
    pdfminer = pytest.importorskip("pdfminer.high_level")
    pdf, _ = render_resume_pdf(validate.build_resume(profile, make_output()), FitStyle(stacked=True))
    lines = [" ".join(line.split()) for line in pdfminer.extract_text(io.BytesIO(pdf)).splitlines()]
    for name, dates in (("Northwind Labs", "Mar 2022 - Present"), ("Contoso Systems", "Jun 2019 - Feb 2022")):
        assert next(line for line in lines if name in line).endswith(dates)


def test_the_two_line_header_is_used_only_when_it_costs_nothing(profile):
    resume = validate.build_resume(profile, make_output())
    style, pages, _pdf, notes = try_styles(resume)
    assert style.stacked and pages == 1 and style.font_pt == 10.5 and not notes
    sentence = ("Designed and operated reliable data pipelines with Python and SQL for several product teams, "
                "documenting every decision clearly.")
    for count in range(12, 40):  # more bullets until two pages are full
        for exp in resume.experience:
            exp.bullets = [sentence] * count
        style, pages, _pdf, _notes = try_styles(resume)
        if not style.stacked:
            break
    assert pages == 2 and not style.stacked  # the title moves onto the company's line ...
    assert resume_layout(resume, style.model_copy(update={"stacked": True}))[0] == 3  # ... as its own line needs a page
    assert FitStyle.model_validate({"font_pt": 10.5, "margin_in": 0.65, "paper": "Letter"}).stacked is False


def test_the_planned_bullets_are_the_same_as_before():
    jobs = Profile(name="Jordan Rivera", companies=[
        Company(name=f"Employer {i}", start=f"Jan {2023 - 3 * i}", end="Present" if i == 0 else f"Dec {2025 - 3 * i}")
        for i in range(9)])
    assert bullet_budget(jobs, today=date(2026, 10, 28)) == [8, 6, 5, 4, 3, 3, 3, 3, 3]


def test_a_long_career_keeps_every_bullet(router, db, jd):
    jobs = Profile(name="Jordan Rivera", companies=[
        Company(name=f"Employer {i}", start=f"Jan {2023 - 3 * i}", end="Present" if i == 0 else f"Dec {2025 - 3 * i}")
        for i in range(9)])
    result = asyncio.run(run_tailor(router, db, profile=jobs, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.fits and result.pages == 2 and not result.style.stacked  # two full pages
    assert [len(exp.bullets) for exp in result.resume.experience] == bullet_budget(jobs)


# ------------------------------------------------------------------ the title line under the name
def test_the_title_line_is_larger_but_stays_on_one_line(profile):
    sheet = Sheet("Letter", 0.65 * 72, 10.5, 1.27, "", "")
    assert sheet.headline_size("Senior Data Engineer | Python | SQL", 10.5) == 12.5
    long = "Senior Distributed Systems and Platform Infrastructure Engineer | Kubernetes | Go | AWS | Terraform | Kafka"
    assert 10.5 <= sheet.headline_size(long, 10.5) < 12.5
    resume = validate.build_resume(profile, make_output())
    paragraphs = Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs
    headline = next(p for p in paragraphs if p.text == resume.headline)
    assert headline.runs[0].bold and headline.runs[0].font.size.pt == 12.5


# ------------------------------------------------------------------ the headline: the role only
def test_prompt_asks_for_the_role_only():
    text = tailor_system("estimate")
    assert 'HEADLINE: the target role only, as the JD names it (for example "Full-Stack Engineer"' in text
    assert "no domain or areas of work and no skills" in text and "areas of work for this JD" not in text
    assert "without the hiring company's level number (II, III, L5)" in text  # the level rule stays


def test_whatever_a_draft_adds_after_the_role_goes(profile):
    resume = validate.build_resume(profile, make_output())
    cut = {}
    for headline in ("AI Engineer | AI Platforms & Applied Machine Learning", "Full-Stack Engineer - Payments",
                     "Senior Data Engineer | Python | SQL | Airflow", "Director, Engineering", "Full-Stack Engineer"):
        resume.headline = headline
        validate.role_only_headline(resume)
        cut[headline] = resume.headline
    assert cut == {
        "AI Engineer | AI Platforms & Applied Machine Learning": "AI Engineer",
        "Full-Stack Engineer - Payments": "Full-Stack Engineer",
        "Senior Data Engineer | Python | SQL | Airflow": "Senior Data Engineer",
        "Director, Engineering": "Director, Engineering",  # a comma is part of a role
        "Full-Stack Engineer": "Full-Stack Engineer",
    }


class AddsToTheRole(LLMClient):
    """First draft only: the role followed by three skills."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            result.data["sections"]["headline"] = "Senior Data Engineer | Python | SQL | Airflow"
        return result


def test_a_draft_headline_keeps_only_the_role_without_a_fix_call(router, db, registry, profile, jd):
    plain = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert plain.resume.headline == "Senior Data Engineer" and plain.retries == 0
    client = AddsToTheRole(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 0 and len(client.prompts) == 1  # fixed in code, for free
    assert result.resume.headline == "Senior Data Engineer"
