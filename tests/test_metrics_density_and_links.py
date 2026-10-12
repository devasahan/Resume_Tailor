import asyncio
import io

import pytest
from docx import Document
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, tailor_system
from app.render.docx import resume_docx
from app.render.pdf import display_url, render_resume_pdf
from app.schemas import FitStyle

from .test_validate import make_output


# ------------------------------------------------------------------ LinkedIn / website not clickable
@pytest.mark.parametrize("raw,shown", [
    ("https://www.linkedin.com/in/juan-ramirez-127360365/", "linkedin.com/in/juan-ramirez-127360365"),
    ("www.linkedin.com/in/juan/", "linkedin.com/in/juan"),
    ("HTTP://WWW.Example.com/me", "Example.com/me"),
    ("linkedin.com/in/juan", "linkedin.com/in/juan"),
    ("", ""),
])
def test_display_url(raw, shown):
    assert display_url(raw) == shown


def test_pdf_has_no_link_and_no_auto_link_prefix(profile):
    fancy = profile.model_copy(deep=True)
    fancy.contact.linkedin = "https://www.linkedin.com/in/juan-ramirez-127360365/"
    fancy.contact.website = "https://portfolio.example.com/"
    resume = validate.build_resume(fancy, make_output())
    pdf = render_resume_pdf(resume, FitStyle())[0]
    page = PdfReader(io.BytesIO(pdf)).pages[0]
    text = page.extract_text()
    assert page.get("/Annots") is None  # the file itself contains no links
    assert "linkedin.com/in/juan-ramirez-127360365" in text and "portfolio.example.com" in text
    assert "www." not in text and "http" not in text  # what PDF viewers turn into links on their own
    docx_text = "\n".join(p.text for p in Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs)
    assert "linkedin.com/in/juan-ramirez-127360365" in docx_text and "www." not in docx_text
    assert resume.profile.contact.linkedin == "https://www.linkedin.com/in/juan-ramirez-127360365/"  # stored value unchanged


# ------------------------------------------------------------------ fewer metrics
def test_has_figure_ignores_plain_years():
    assert validate.has_figure("Cut runtime by 23.4%") and validate.has_figure("Supported 38 analysts")
    assert not validate.has_figure("Migrated the warehouse in 2021") and not validate.has_figure("Built pipelines")


def test_density_limits_per_role_and_overall(profile):
    out = make_output()
    out.sections.experience[0].bullets = ["Cut runtime by 23.4%", "Saved 17.8% of spend", "Served 38 analysts", "Built pipelines"]
    problems = validate.metric_density_problems(validate.build_resume(profile, out))
    assert any("Northwind Labs: 3 of 4" in p for p in problems)
    out = make_output()
    out.sections.experience[0].bullets = ["Cut runtime by 23.4%", "Built pipelines"]
    assert validate.metric_density_problems(validate.build_resume(profile, out)) == []


def test_prompt_asks_for_fewer_figures():
    text = tailor_system("estimate")
    assert "about 2 of every 5 bullets" in text and "at most 2 bullets per employer" in text


class Dense(LLMClient):
    """Writes a figure in every bullet first; spreads them out once told."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        result = await self.inner.generate_structured(**kwargs)
        figures = iter(["23.4%", "17.8%", "38", "21%", "1.7TB", "12.6%", "47", "8.3%", "19%", "26.1%"])
        for exp in result.data["sections"]["experience"]:
            for i in range(len(exp["bullets"])):
                if RETRY_MARKER not in kwargs["user"] or i == 0:
                    exp["bullets"][i] = f"Improved delivery of item {'abcdefgh'[i]} by {next(figures)}"
        return result


def test_too_many_figures_trigger_one_rewrite(router, db, registry, profile, jd):
    client = Dense(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and client.calls == 2 and result.retries == 1
    assert validate.metric_density_problems(result.resume) == []
    assert not any("Many bullets carry figures" in w for w in result.ats_warnings)
