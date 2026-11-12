"""What a recruiter reads in a six-second scan: the title line under the name (the role only), the company and title
of each job, and each job's first bullet, which opens with the job's biggest win, its size and its result."""
import asyncio
import io
from datetime import date

from docx import Document
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.fit import bullet_budget
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, tailor_system
from app.render.docx import resume_docx
from app.render.pdf import TITLE_SIZE, Sheet, render_resume_pdf
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
    # its size and its result, the result with its base ('by 17.8% (from 101 to 83 a month)', 'by 21%, from 62 to 49
    # hours') or as the real numbers alone ('fell from 87 to 76 a week')
    assert [len(validate._numbers_in(exp.bullets[0])) for exp in result.resume.experience] == [4, 4, 3]
    assert validate.unbased_percentage_problems(result.resume) == validate.base_mismatch_problems(result.resume) == []
    assert validate.metric_density_problems(result.resume) == []  # still at most 2 figure bullets per job


def test_no_figures_mode_asks_for_none(router, db, profile, jd):
    db.set_settings({"metrics_mode": "hints_only"})
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 0 and result.metrics_mode == "hints_only"
    assert not any(validate.has_figure(b) for exp in result.resume.experience for b in exp.bullets)
    assert not any("measured result" in w for w in result.ats_warnings)


# ------------------------------------------------------------------ the title first, then the company
def test_the_title_comes_first_on_the_job_line(profile):
    resume = validate.build_resume(profile, make_output())
    text = PdfReader(io.BytesIO(render_resume_pdf(resume, FitStyle())[0])).pages[0].extract_text()
    lines = [" ".join(line.split()) for line in text.splitlines()]
    assert "Engineer 0 · Northwind Labs · Remote Mar 2022 – Present" in lines  # one line, the dates at its end
    assert "Engineer 1 · Contoso Systems · Dallas, TX · Onsite Jun 2019 – Feb 2022" in lines  # with its place


def test_runs_saved_with_the_title_on_its_own_line_still_load():
    """Runs saved before the restyle may hold 'stacked' (the title under the company): unused now, never an error."""
    old = FitStyle.model_validate({"font_pt": 10.5, "margin_in": 0.65, "paper": "Letter", "stacked": True})
    assert old == FitStyle(font_pt=10.5, margin_in=0.65) and "stacked" not in old.model_dump()


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
    assert result.status == "ok" and result.fits and result.pages == 2  # two full pages
    assert [len(exp.bullets) for exp in result.resume.experience] == bullet_budget(jobs)


# ------------------------------------------------------------------ the title line under the name
def test_the_title_line_is_larger_but_stays_on_one_line(profile):
    sheet = Sheet("Letter", 0.65 * 72, 10.5, 1.42, "", "")
    assert sheet.headline_size("Senior Data Engineer | Python | SQL", 10.5) == TITLE_SIZE * 10.5  # the design's 15/13.5
    long = "Senior Distributed Systems and Platform Infrastructure Engineer | Kubernetes | Go | AWS | Terraform | Kafka"
    assert 10.5 <= sheet.headline_size(long, 10.5) < TITLE_SIZE * 10.5
    resume = validate.build_resume(profile, make_output())
    paragraphs = Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs
    headline = next(p for p in paragraphs if p.text == resume.headline)
    size = round(TITLE_SIZE * FitStyle().font_pt * 2) / 2  # Word keeps half points
    assert not headline.runs[0].bold and headline.runs[0].font.size.pt == size  # Medium: not bold in Word


# ------------------------------------------------------------------ the headline: the role's short name
def test_prompt_asks_for_the_roles_short_name():
    text = tailor_system("estimate")
    assert "HEADLINE: `general_role`, the role's short, common name in 2-3 words, never the posting's full job title" in text
    assert '"Full-Stack Engineer" for "Senior Software Engineer, Full Stack (Payments Platform)"' in text
    assert "as the JD names it" not in text
    assert ("In the summary, write the role as `general_role` (with a level such as Senior when the candidate's most "
            "recent title in your plan has it), without the hiring company's level number (II, III, L5)") in text


def test_a_copied_posting_title_becomes_the_roles_short_name(profile):
    resume = validate.build_resume(profile, make_output())
    short = {}
    for headline in ("Senior Software Engineer, Full Stack (Payments Platform)", "AI Engineer II - Agentic Systems",
                     "Staff Data Engineer, Analytics Infrastructure", "Manager, Customer Success",
                     "Senior Full-Stack Engineer (Remote)", "Sr. Software Engineer 3", "Lead Data Engineer",
                     "Senior Data Engineer | Python | SQL | Airflow", "Tech Lead", "Full-Stack Engineer"):
        resume.headline = headline
        validate.short_role_headline(resume)
        short[headline] = resume.headline
    assert short == {
        "Senior Software Engineer, Full Stack (Payments Platform)": "Software Engineer",  # the model writes Full-Stack
        "AI Engineer II - Agentic Systems": "AI Engineer",
        "Staff Data Engineer, Analytics Infrastructure": "Data Engineer",
        "Manager, Customer Success": "Customer Success Manager",  # a bare role noun takes its function
        "Senior Full-Stack Engineer (Remote)": "Full-Stack Engineer",
        "Sr. Software Engineer 3": "Software Engineer",
        "Lead Data Engineer": "Data Engineer",
        "Senior Data Engineer | Python | SQL | Airflow": "Data Engineer",
        "Tech Lead": "Tech Lead",  # 'Lead' is the role here, not a level
        "Full-Stack Engineer": "Full-Stack Engineer",
    }


class AddsToTheRole(LLMClient):
    """First draft only: the posting's level, then three skills."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            result.data["sections"]["headline"] = "Senior Data Engineer | Python | SQL | Airflow"
        return result


def test_a_draft_headline_keeps_only_the_roles_short_name_without_a_fix_call(router, db, registry, profile, jd):
    plain = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert plain.resume.headline == "Data Engineer" and plain.retries == 0  # the posting asks for a Senior one
    client = AddsToTheRole(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 0 and len(client.prompts) == 1  # fixed in code, for free
    assert result.resume.headline == "Data Engineer"
