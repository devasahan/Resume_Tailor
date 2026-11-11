"""Settings > Certifications section: an on/off switch, off by default.

While it is off, no resume gets a CERTIFICATIONS section, the Tailor page gets no rows to tick, and the model is
neither shown the person's certifications nor asked to pick any (in the first draft, the fix round, the shorter
draft, the cover letter and Compare). The lists saved on the Profile page never change. Switched on, everything
works as before (test_certifications.py switches it on for its own tests).
"""
import asyncio
import io

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app import config, main
from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.main import create_app
from app.pipeline import validate
from app.pipeline.compare import compare
from app.pipeline.fit import bullet_budget
from app.pipeline.tailor import run_tailor
from app.prompts import (
    RETRY_MARKER,
    SHORTEN_NOTE,
    revise_user,
    tailor_system,
    tailor_user,
)
from app.render.docx import resume_docx
from app.render.pdf import render_resume_pdf
from app.schemas import Certification, Company, Profile
from app.storage.db import Database

from .conftest import ADMIN_EMAIL, ADMIN_PASSWORD, SAMPLE_JD, SAMPLE_RESUME_TEXT, log_in
from .test_api import save_profile, tailor
from .test_jobs import answer
from .test_validate import make_output

CKA = Certification(name="Certified Kubernetes Administrator (CKA)", issuer="The Linux Foundation", year="2024")
GCP = Certification(name="Google Cloud Professional Data Engineer", issuer="Google Cloud", year="2023")
LISTED_RESUME = SAMPLE_RESUME_TEXT + (
    "\nCERTIFICATIONS\nCertified Kubernetes Administrator (CKA), The Linux Foundation, 2024\n"
    "Google Cloud Professional Data Engineer, Google Cloud, 2023\n")
OFF_RULE = ("CERTIFICATIONS: return `certifications` as an empty list: no certification is picked or suggested for "
            "this resume. Never name a certification or license anywhere in the resume: not in the headline, the "
            "summary, the skills or the bullets.")
ON_RULE = "CERTIFICATIONS: `certifications` lists the 4 certifications that best support this application"
OFF_FIX = "Certifications are never named in this resume (the CERTIFICATIONS rule), so take every one out of the text."
SWITCH_TEXT = "This list is printed only when the Certifications section is switched on in Settings."


def pdf_text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


def docx_lines(data: bytes) -> list[str]:
    return [p.text for p in Document(io.BytesIO(data)).paragraphs]


@pytest.fixture
def certified(profile) -> Profile:
    """The shared profile plus two certifications the person has passed."""
    copy = profile.model_copy(deep=True)
    copy.certifications = [CKA.model_copy(), GCP.model_copy()]
    return copy


class Recording(LLMClient):
    """Keeps every call (its prompts and context) with the answer. The first draft names one of the person's
    certifications in a bullet, so the fix round runs; `picks` (optional) also puts certification picks and
    certification skills in it, as a model that ignores its instructions might."""

    def __init__(self, inner, picks=None):
        self.inner, self.picks, self.calls = inner, picks, []

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        first_draft = RETRY_MARKER not in kwargs["user"] and SHORTEN_NOTE not in kwargs["user"]
        if kwargs["schema_name"] == "tailor_output" and first_draft:
            job = result.data["sections"]["experience"][0]
            job["bullets"][0] = "Passed the Certified Kubernetes Administrator (CKA) exam " + job["bullets"][0]
            if self.picks is not None:
                result.data["certifications"] = list(self.picks)
                result.data["sections"]["skills"][0]["skills"] += ["Project Management Professional (PMP)", CKA.name]
        self.calls.append({**kwargs, "answer": result.data})
        return result


def kind(call: dict) -> str:
    if call["schema_name"] == "cover_letter":
        return "cover letter"
    return "fix" if RETRY_MARKER in call["user"] else "shorter" if SHORTEN_NOTE in call["user"] else "first"


def tailored(router, profile, model="demo-concise", **kwargs):
    return asyncio.run(run_tailor(router, router.db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id=model,
                                  **kwargs))


# ------------------------------------------------------------------ the setting
def test_the_switch_is_off_by_default(db, client):
    assert config.DEFAULT_SETTINGS["certifications_section"] is False
    assert db.get_settings()["certifications_section"] is False
    assert client.get("/api/settings").json()["settings"]["certifications_section"] is False


def test_the_settings_api_takes_only_true_or_false(client, user_client):
    on = client.put("/api/settings", json={"certifications_section": True})
    assert on.status_code == 200 and on.json()["settings"]["certifications_section"] is True
    assert client.get("/api/settings").json()["settings"]["certifications_section"] is True
    for bad in ("true", "false", "on", 1, 0, None, [], {}):  # never read as a choice the admin did not make
        refused = client.put("/api/settings", json={"certifications_section": bad})
        assert refused.status_code == 422, bad
        assert "certifications_section" in refused.json()["detail"]
    assert client.get("/api/settings").json()["settings"]["certifications_section"] is True  # nothing changed
    off = client.put("/api/settings", json={"certifications_section": False})
    assert off.status_code == 200 and off.json()["settings"]["certifications_section"] is False
    assert user_client.put("/api/settings", json={"certifications_section": True}).status_code == 403
    assert client.get("/api/settings").json()["settings"]["certifications_section"] is False


def test_the_settings_page_has_the_switch(client):
    page = client.get("/settings").text
    assert 'id="certifications_section"' in page and 'role="switch"' in page and "Certifications section" in page
    assert ("on: for each job, the AI picks the best 4 of your certifications from the Profile page and prints them "
            "after Education; off: no Certifications section is printed, and your saved list stays on the Profile "
            "page for later") in page
    # the saved state is shown when the page loads (and again after Save), and Save sends the switch
    assert "field('certifications_section').checked = st.certifications_section === true;" in page
    assert "certifications_section: field('certifications_section').checked," in page


def test_the_profile_page_says_when_the_list_is_printed(client):
    assert SWITCH_TEXT in client.get("/profile").text


# ------------------------------------------------------------------ prompts
def test_off_the_prompts_carry_no_certifications():
    system_on = tailor_system("estimate", "us")
    system_off = tailor_system("estimate", "us", certifications_section=False)
    assert ON_RULE in system_on and OFF_RULE not in system_on
    assert OFF_RULE in system_off and ON_RULE not in system_off and "<candidate_certifications>" not in system_off
    assert len(system_off) < len(system_on) - 1000  # a shorter prompt: a little faster
    rule_on = next(line for line in system_on.split("\n") if line.startswith(ON_RULE))
    assert system_off == system_on.replace(rule_on, OFF_RULE)  # only the rule differs
    companies = [Company(name="Northwind Labs", start="Mar 2022", end="Present")]
    on = tailor_user(SAMPLE_JD, companies, "", "facts", certifications=[CKA])
    off = tailor_user(SAMPLE_JD, companies, "", "facts", certifications=[CKA], certifications_section=False)
    assert "<candidate_certifications>\n0: Certified Kubernetes Administrator (CKA)" in on
    assert "candidate_certifications" not in off and CKA.name not in off
    assert "</candidate_hints>\n\nWrite the tailored sections" in off  # one blank line, as between the other blocks
    assert off == on.replace(on[on.index("<candidate_certifications>"):on.index("Write the tailored")], "")
    analysis, plan = make_output().jd_analysis, make_output().career_plan
    fix = revise_user(SAMPLE_JD, companies, "", "facts", analysis=analysis, plan=plan, draft="Headline: x",
                      problems="Fix it.", certifications=[CKA], certifications_section=False)
    assert "candidate_certifications" not in fix and "</candidate_hints>\n\n<jd_analysis>" in fix
    assert fix.startswith(off[: off.index("</candidate_hints>")])  # the same opening, so a provider's cache works


# ------------------------------------------------------------------ a whole run
def test_off_no_section_no_rows_and_no_certifications_in_any_prompt(router, registry, certified):
    before = certified.model_copy(deep=True)
    client = Recording(FakeClient(registry.get("demo-detailed")))
    router.overrides["demo-detailed"] = client
    result = tailored(router, certified, "demo-detailed", cover_letter=True)
    assert result.status == "ok"
    assert {"first", "fix", "shorter", "cover letter"} <= {kind(call) for call in client.calls}
    for call in client.calls:  # every prompt: no list, no rule asking for picks
        assert "candidate_certifications" not in call["user"] + call["system"], kind(call)
        if kind(call) != "fix":  # the fix round quotes the name the first draft wrote, to take it out
            assert CKA.name not in call["user"] and GCP.name not in call["user"], kind(call)
    for call in client.calls:
        if call["schema_name"] != "cover_letter":
            assert OFF_RULE in call["system"] and ON_RULE not in call["system"]
            assert call["context"]["certifications_section"] is False
        if call["schema_name"] == "tailor_output":
            assert call["answer"]["certifications"] == []  # the demo model, like a real one, picks none
    fix = next(call["user"] for call in client.calls if kind(call) == "fix")
    assert "These certification names appear in the headline, summary or bullets" in fix and OFF_FIX in fix
    # nothing printed, nothing to tick, and the name the draft wrote is out of the text
    assert result.resume.certifications == [] and result.certifications == []
    assert validate.certification_mentions(result.resume, [CKA.name, GCP.name]) == []
    text = pdf_text(render_resume_pdf(result.resume, result.style)[0])
    assert "CERTIFICATIONS" not in text and "Linux Foundation" not in text
    assert "CERTIFICATIONS" not in docx_lines(resume_docx(result.resume, result.style))
    assert certified == before and result.resume.profile.certifications == before.certifications  # the list stays


def test_off_a_model_that_still_picks_prints_nothing(router, registry, certified):
    picks = [{"name": CKA.name, "issuer": CKA.issuer, "candidate_index": 0},
             {"name": "Project Management Professional (PMP)", "issuer": "PMI", "candidate_index": -1}]
    router.overrides["demo-concise"] = Recording(FakeClient(registry.get("demo-concise")), picks=picks)
    result = tailored(router, certified)
    assert result.status == "ok" and result.resume.certifications == [] and result.certifications == []
    skills = [skill for group in result.resume.skills for skill in group.skills]
    assert "Project Management Professional (PMP)" not in skills and CKA.name not in skills  # names stay out
    assert "CERTIFICATIONS" not in pdf_text(render_resume_pdf(result.resume, result.style)[0])


def test_on_everything_works_as_before(router, registry, certified):
    router.db.set_settings({"certifications_section": True})
    client = Recording(FakeClient(registry.get("demo-detailed")))
    router.overrides["demo-detailed"] = client
    result = tailored(router, certified, "demo-detailed")
    assert result.status == "ok"
    listed = ("<candidate_certifications>\n0: Certified Kubernetes Administrator (CKA) | The Linux Foundation | "
              "earned 2024\n1: Google Cloud Professional Data Engineer | Google Cloud | earned 2023\n"
              "</candidate_certifications>")
    tailoring = [call for call in client.calls if call["schema_name"] != "cover_letter"]
    assert {"first", "fix", "shorter"} <= {kind(call) for call in tailoring}
    assert all(listed in call["user"] and ON_RULE in call["system"] for call in tailoring)
    fix = next(call["user"] for call in tailoring if kind(call) == "fix")
    assert "Certifications are shown only in their own section (the CERTIFICATIONS rule)" in fix
    assert [(row.name, row.have) for row in result.certifications][:2] == [(CKA.name, True), (GCP.name, True)]
    assert len(result.certifications) == 4 and result.resume.certifications == [CKA, GCP]
    text = pdf_text(render_resume_pdf(result.resume, result.style)[0])
    assert text.index("EDUCATION") < text.index("CERTIFICATIONS") < text.index("The Linux Foundation")


def test_off_the_page_plan_keeps_no_room_for_certifications(router, registry, profile):
    """Room for certifications that are never printed would cost bullets: with nine long jobs, 4 certifications
    take three bullets' room on two pages."""
    long_career = profile.model_copy(deep=True)
    long_career.companies = [Company(name=f"Employer {i}", start=f"Jan {2023 - 4 * i}",
                                     end="Present" if i == 0 else f"Dec {2026 - 4 * i}", location="Austin, TX")
                             for i in range(9)]
    many = long_career.model_copy(update={"certifications": [
        Certification(name=f"Certification number {i} with a long official name", issuer="Issuer", year="2020")
        for i in range(4)]})
    client = Recording(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    assert tailored(router, many).status == "ok"
    planned_off = client.calls[0]["context"]["bullets"]
    assert planned_off == bullet_budget(long_career)  # as if there were no certifications
    router.db.set_settings({"certifications_section": True})
    client.calls.clear()
    assert tailored(router, many).status == "ok"
    planned_on = client.calls[0]["context"]["bullets"]
    assert planned_on == bullet_budget(many) and sum(planned_on) < sum(planned_off)


def test_compare_follows_the_switch(router, certified):
    models = ["demo-concise", "demo-detailed"]
    results = asyncio.run(compare(router, router.db, profile=certified, profile_id=None, jd=SAMPLE_JD,
                                  model_ids=models, save=False))
    assert [r.status for r in results] == ["ok", "ok"]
    assert all(r.certifications == [] and r.resume.certifications == [] for r in results)
    router.db.set_settings({"certifications_section": True})
    results = asyncio.run(compare(router, router.db, profile=certified, profile_id=None, jd=SAMPLE_JD,
                                  model_ids=models, save=False))
    assert all(r.resume.certifications == [CKA, GCP] and len(r.certifications) == 4 for r in results)


# ------------------------------------------------------------------ the web app
def stored(client, resume_id) -> list[str]:
    entry = next(r for r in client.get("/api/resumes").json()["resumes"] if r["id"] == resume_id)
    return [c["name"] for c in entry["profile"]["certifications"]]


def test_off_in_the_web_app_and_on_again(client, user_client):
    resume_id = save_profile(client, text=LISTED_RESUME)["id"]
    assert stored(client, resume_id) == [CKA.name, GCP.name]  # read from the uploaded resume, switch or not
    first = tailor(client, profile_id=resume_id).json()
    assert first["status"] == "ok" and first["certifications"] == [] and first["resume"]["certifications"] == []
    pdf = client.get(f"/api/runs/{first['run_id']}/resume.pdf?download=1")
    assert pdf.status_code == 200 and "CERTIFICATIONS" not in pdf_text(pdf.content)
    assert client.put(f"/api/runs/{first['run_id']}/certifications", json={"have": []}).status_code == 409
    letter = client.post(f"/api/runs/{first['run_id']}/cover-letter", json={})
    assert letter.status_code == 200 and letter.json()["cover_letter"] and letter.json()["certifications"] == []
    assert "CERTIFICATIONS" not in pdf_text(client.get(f"/api/runs/{first['run_id']}/resume.pdf").content)
    page = client.get("/tailor").text  # no rows: the certifications box stays hidden
    assert 'class="certs hidden" id="certs"' in page and "$('#certs').classList.toggle('hidden', !rows.length);" in page
    assert client.put("/api/settings", json={"resume_model": "demo-concise"}).status_code == 200  # users get it
    save_profile(user_client, text=LISTED_RESUME)
    assert tailor(user_client).json()["certifications"] == []  # what a user's Tailor page gets
    assert stored(client, resume_id) == [CKA.name, GCP.name]  # the saved list is kept for later
    assert client.put("/api/settings", json={"certifications_section": True}).status_code == 200
    on = tailor(client, profile_id=resume_id).json()
    assert [c["name"] for c in on["resume"]["certifications"]] == [CKA.name, GCP.name]
    assert len(on["certifications"]) == 4 and "CERTIFICATIONS" in pdf_text(
        client.get(f"/api/runs/{on['run_id']}/resume.pdf").content)
    assert client.put("/api/settings", json={"certifications_section": False}).status_code == 200
    assert stored(client, resume_id) == [CKA.name, GCP.name]
    assert "CERTIFICATIONS" in pdf_text(client.get(f"/api/runs/{on['run_id']}/resume.pdf").content)  # made while on
    again = tailor(client, profile_id=resume_id).json()
    assert again["certifications"] == [] and again["resume"]["certifications"] == []


@pytest.fixture
def slow_app(tmp_path, monkeypatch):
    """The web app (logged in as the admin) with every job too slow to answer at once, as on a slow host."""
    monkeypatch.setattr(main, "JOB_WAIT_SECONDS", 0)
    monkeypatch.setattr(main, "JOB_POLL_WAIT_SECONDS", 0)
    db = Database(tmp_path / "jobs.db")
    db.set_settings({"parser_model": "rules"})
    with TestClient(create_app(db=db)) as admin:
        yield log_in(admin, ADMIN_EMAIL, ADMIN_PASSWORD)


def test_off_the_job_answer_has_no_certifications(slow_app):
    resume_id = save_profile(slow_app, text=LISTED_RESUME)["id"]
    started = slow_app.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "profile_id": resume_id})
    assert started.status_code == 202
    finished = answer(slow_app, started).json()
    assert finished["status"] == "ok" and finished["certifications"] == []
    assert finished["resume"]["certifications"] == []
