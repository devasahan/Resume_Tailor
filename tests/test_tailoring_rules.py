"""Rules added after the first real test: header order, years + domain in the summary, related skills, metrics."""
import asyncio
import io
from datetime import date

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.main import create_app
from app.pipeline import validate
from app.pipeline.facts import candidate_facts, experience_months, parse_point, years_phrase
from app.pipeline.tailor import run_tailor
from app.prompts import tailor_system, tailor_user
from app.render.docx import resume_docx
from app.render.pdf import render_resume_pdf
from app.schemas import Company, FitStyle, Profile
from app.storage.db import Database

from .test_validate import make_output

TODAY = date(2026, 10, 11)


# ------------------------------------------------------------------ years of experience
@pytest.mark.parametrize("text,is_end,expected", [
    ("Jan 2021", False, (2021, 1)), ("January 2021", False, (2021, 1)), ("Sept. 2019", False, (2019, 9)),
    ("sep 2019", False, (2019, 9)), ("03/2020", False, (2020, 3)), ("2020-07", False, (2020, 7)),
    ("2018", False, (2018, 1)), ("2018", True, (2018, 12)), ("Present", True, (2026, 10)), ("", True, (2026, 10)),
    ("Present", False, None), ("someday", False, None), ("13/2020", False, None),
])
def test_parse_point(text, is_end, expected):
    assert parse_point(text, is_end, today=TODAY) == expected


def test_overlaps_count_once_and_years_round_down():
    jobs = [
        Company(name="A", start="Jan 2020", end="Present"),  # 82 months to Oct 2026
        Company(name="B", start="Jun 2021", end="Dec 2022"),  # fully inside A
        Company(name="C", start="Jan 2016", end="Dec 2017"),  # 24 months, separate
    ]
    assert experience_months(jobs, TODAY) == 82 + 24
    assert years_phrase(106) == "8+ years" and years_phrase(11) == "" and years_phrase(None) == ""


def test_candidate_facts_text(profile):
    assert "10+ years" in candidate_facts(profile, TODAY)  # Jul 2016 - Oct 2026 with no gaps over a month
    unknown = Profile(name="X", companies=[Company(name="A", start="", end="")])
    assert "unknown" in candidate_facts(unknown, TODAY)


# ------------------------------------------------------------------ prompts
def test_system_prompt_rules_per_metrics_mode():
    estimate, strict = tailor_system("estimate"), tailor_system("hints_only")
    for text in (estimate, strict):
        assert "{{" not in text
        assert "years of experience EXACTLY as given in <candidate_facts>" in text
        assert "domain or industry" in text
        assert "closely related skills that SUPPORT the required ones" in text
    assert "realistic, conservative estimates" in estimate and "do NOT invent numbers" not in estimate
    assert "do NOT invent numbers" in strict and "realistic, conservative estimates" not in strict
    with pytest.raises(ValueError):
        tailor_system("anything")


def test_user_prompt_carries_the_computed_facts(profile):
    text = tailor_user("A job description", profile.companies, "", candidate_facts(profile, TODAY))
    assert "<candidate_facts>\nTotal professional experience: 10+ years" in text


# ------------------------------------------------------------------ checks
def test_computed_years_are_not_flagged_but_estimates_are(profile):
    out = make_output()
    out.sections.summary = "Senior Data Engineer with 10+ years of experience in fintech."
    out.sections.experience[0].bullets = ["Cut pipeline runtime by ~35% using Spark", "Built pipelines in Python"]
    resume = validate.build_resume(profile, out)
    flags = validate.claim_flags(resume, "jd text", "\n" + candidate_facts(profile, TODAY))
    verify = [f for f in flags if f.level == "verify"]
    assert not [f for f in verify if f.section == "summary"]  # 10+ comes from the dates, so it is a fact
    assert any("35" in f.reason and "estimate" in f.reason for f in verify)


# ------------------------------------------------------------------ layout
def test_header_order_is_name_headline_contact(profile):
    resume = validate.build_resume(profile, make_output())
    lines = PdfReader(io.BytesIO(render_resume_pdf(resume, FitStyle())[0])).pages[0].extract_text().splitlines()
    assert lines[0] == "Jordan Rivera"
    assert lines[1] == resume.headline
    assert lines[2] == "Austin, TX | (555) 010-2030 | jordan@example.com | linkedin.com/in/jordanrivera"
    paragraphs = [p.text for p in Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs]
    assert paragraphs[:3] == ["Jordan Rivera", resume.headline, lines[2]]


# ------------------------------------------------------------------ end to end: what the model is sent
class Capture(LLMClient):
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return await self.inner.generate_structured(**kwargs)


def test_run_sends_facts_and_the_chosen_metrics_rule(router, db, registry, profile, jd):
    capture = Capture(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = capture
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and "years" in result.facts
    assert "realistic, conservative estimates" in capture.calls[0]["system"]  # default: estimate
    assert "<candidate_facts>\nTotal professional experience:" in capture.calls[0]["user"]

    db.set_settings({"metrics_mode": "hints_only"})
    capture.calls.clear()
    asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert "do NOT invent numbers" in capture.calls[0]["system"]


def test_metrics_setting_is_validated(tmp_path):
    client = TestClient(create_app(db=Database(tmp_path / "s.db")))
    assert client.get("/api/settings").json()["settings"]["metrics_mode"] == "estimate"
    assert client.put("/api/settings", json={"metrics_mode": "always"}).status_code == 422
    assert client.put("/api/settings", json={"metrics_mode": "hints_only"}).json()["settings"]["metrics_mode"] == "hints_only"
    assert 'id="metrics_mode"' in client.get("/settings").text



def test_summary_rule_is_richer_but_bounded():
    text = tailor_system("estimate")
    assert "SUMMARY: 3-4 sentences, 60-85 words" in text
    assert "core technical strengths" in text and "what the candidate brings to this particular role" in text


def test_summary_length_warnings(profile):
    out = make_output()
    out.sections.summary = "Short summary."
    assert any("Summary is short" in w for w in validate.ats_lint(validate.build_resume(profile, out)))
    out.sections.summary = " ".join(["word"] * 70)
    assert not any("Summary is" in w for w in validate.ats_lint(validate.build_resume(profile, out)))
    out.sections.summary = " ".join(["word"] * 100)
    assert any("Summary is long" in w for w in validate.ats_lint(validate.build_resume(profile, out)))
