"""Certifications: the AI picks the 4 best for each job, and only the ones the person has are ever printed.

A pick that is one of the person's own (read from their resume, or ticked "I have it" on the Tailor page) is
printed under CERTIFICATIONS, after EDUCATION. Every other pick is only a suggestion on the Tailor page.
Certification names never appear in the headline, summary, skills or bullets, so unticking really takes one off.
"""
import asyncio
import io

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.parsing.heuristic import _is_header, _parse_certifications, parse_resume_text
from app.pipeline import validate
from app.pipeline.fit import bullet_budget
from app.pipeline.tailor import refit, run_tailor
from app.prompts import (
    PROMPT_VERSION,
    RETRY_MARKER,
    SHORTEN_NOTE,
    tailor_system,
    tailor_user,
)
from app.render.docx import resume_docx
from app.render.pdf import render_resume_pdf
from app.schemas import (
    Certification,
    CertificationChoice,
    CertificationPick,
    FitStyle,
    Profile,
    Resume,
    TailorOutput,
    TailorResult,
)

from .conftest import SAMPLE_JD, SAMPLE_RESUME_TEXT
from .test_api import docx_bytes, save_profile, tailor
from .test_render import long_resume
from .test_validate import make_output

AWS = Certification(name="AWS Certified Solutions Architect – Professional", issuer="Amazon Web Services", year="2023")
CSM = Certification(name="Certified ScrumMaster (CSM)", issuer="Scrum Alliance", year="2020")
CERT_RESUME = SAMPLE_RESUME_TEXT + (
    "\nCERTIFICATIONS\nAWS Certified Solutions Architect – Professional, Amazon Web Services, 2023\n"
    "Certified ScrumMaster (CSM), Scrum Alliance, 2020\n")


def text_of(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


@pytest.fixture
def certified(profile) -> Profile:
    """The shared profile plus two certifications (the shared one stays without, so page tests do not move)."""
    copy = profile.model_copy(deep=True)
    copy.certifications = [AWS.model_copy(), CSM.model_copy()]
    return copy


def pick(name, index=-1, issuer=""):
    return CertificationPick(name=name, issuer=issuer, candidate_index=index)


# ------------------------------------------------------------------ reading the uploaded resume
def test_certification_headings_are_whole_lines():
    for heading in ("Certifications", "CERTIFICATIONS:", "Licenses & Certifications", "Certifications and Licenses",
                    "Professional Certifications", "Certificates", "Licences", "Credentials"):
        assert _is_header(heading) == "certifications", heading
    assert _is_header("Certificate in Data Analytics, UC Irvine") is None  # an entry, not a heading
    assert _is_header("Education & Certifications") == "education_and_certifications"
    assert _is_header("Training") == "other" and _is_header("Training new hires") is None


def test_each_line_format_becomes_name_issuer_and_year():
    found = [(c.name, c.issuer, c.year) for c in _parse_certifications([
        "AWS Certified Solutions Architect – Professional, Amazon Web Services, 2023",  # a dash is part of a name
        "• Certified Kubernetes Administrator (CKA) | The Linux Foundation | Mar 2022",
        "Project Management Professional (PMP) (2021)",
        "MCSA: Windows Server 2016",  # a number in a name is no date
        "Google Cloud Professional Data Engineer",
        "Issued Jan 2024 · Expires Jan 2027",  # LinkedIn: the date of the one above
        "Credential ID ABC-123",
        "Lean Six Sigma Green Belt, Villanova University, 2019 - 2022",
        "AWS Certified Solutions Architect Professional (2024)",  # the first one again, in another spelling
    ])]
    assert found == [
        ("AWS Certified Solutions Architect – Professional", "Amazon Web Services", "2023"),
        ("Certified Kubernetes Administrator (CKA)", "The Linux Foundation", "Mar 2022"),
        ("Project Management Professional (PMP)", "", "2021"),
        ("MCSA: Windows Server 2016", "", ""),
        ("Google Cloud Professional Data Engineer", "", "Jan 2024"),
        ("Lean Six Sigma Green Belt", "Villanova University", "2019"),
    ]


def test_a_certifications_section_makes_no_fake_employer():
    """'Licenses & Certifications' was not a heading: its dated lines were read as employers."""
    text = SAMPLE_RESUME_TEXT.replace("EDUCATION", "LICENSES & CERTIFICATIONS\nAWS Certified Solutions Architect – "
                                      "Professional | Jan 2023 - Jan 2026\nCertified ScrumMaster (CSM), Scrum Alliance, "
                                      "2020\n\nEDUCATION") + "Certifications: AWS SAA, CKA\nAWARDS\nEmployee of the year 2020\n"
    profile = parse_resume_text(text)
    assert [c.name for c in profile.companies] == ["Northwind Labs", "Contoso Systems", "Fabrikam Inc."]
    assert [(c.name, c.year) for c in profile.certifications] == [
        ("AWS Certified Solutions Architect – Professional", "Jan 2023"), ("Certified ScrumMaster (CSM)", "2020"),
        ("AWS SAA", ""), ("CKA", "")]
    assert profile.education[0].school == "University of Texas at Austin"
    combined = parse_resume_text(SAMPLE_RESUME_TEXT.replace("EDUCATION", "EDUCATION & CERTIFICATIONS")
                                 + "PMP Certification, PMI, 2020\n")
    assert [c.name for c in combined.certifications] == ["PMP Certification"]
    assert combined.education[0].degree == "B.S." and len(combined.education) == 1
    assert parse_resume_text(SAMPLE_RESUME_TEXT).certifications == []


# ------------------------------------------------------------------ the picks
def test_one_certification_in_other_spellings_is_one():
    assert validate.same_certification("AWS Certified Solutions Architect – Associate",
                                       "AWS Certified Solutions Architect Associate")
    assert validate.same_certification("Project Management Professional (PMP)", "PMP")
    assert validate.same_certification("Certified ScrumMaster (CSM)", "Certified ScrumMaster")
    assert not validate.same_certification("AWS Certified Developer Associate", "AWS Certified Solutions Architect Associate")


def test_only_the_persons_own_picks_are_printed(certified):
    printed, rows = validate.resolve_certifications(certified, [
        pick("CSM", 1),  # their own, by number
        pick("Certified Information Systems Security Professional (CISSP)", issuer="ISC2"),  # a suggestion
        pick("AWS Certified Solutions Architect Professional"),  # their own, named but not numbered
        pick("Certified Kubernetes Administrator (CKA)", 7, "The Linux Foundation"),  # a number that is not theirs
        pick("PMP"),  # a fifth: only 4
    ])
    assert [(r.name, r.have) for r in rows] == [
        ("Certified ScrumMaster (CSM)", True), ("Certified Information Systems Security Professional (CISSP)", False),
        ("AWS Certified Solutions Architect – Professional", True), ("Certified Kubernetes Administrator (CKA)", False)]
    assert printed == [CSM, AWS]  # exactly as the person wrote them, never a suggestion
    assert rows[1].issuer == "ISC2" and rows[0].year == "2020"
    again, _rows = validate.resolve_certifications(certified, [pick("CSM", 1), pick("Certified ScrumMaster", -1)])
    assert again == [CSM] and len(_rows) == 1  # the same one twice counts once


def test_a_printed_certification_must_be_one_the_person_has(certified):
    resume = validate.build_resume(certified, make_output())
    resume.certifications = [AWS]
    validate.assert_certifications_held(resume)
    resume.certifications = [Certification(name="CISSP")]
    with pytest.raises(ValueError, match="not one of the person's certifications"):
        validate.assert_certifications_held(resume)


def test_a_certification_is_never_a_must_have_skill():
    out = make_output()
    analysis = out.jd_analysis
    analysis.must_have_skills = ["Python", "AWS", "CISSP", "PMP certification", "certificate management",
                                 "software licensing", "Certified ScrumMaster", "RN license"]
    analysis.nice_to_have_skills = ["AWS certifications", "Spark"]
    analysis.keywords = ["Python", "CISSP", "AWS"]
    picks = [pick("Certified Information Systems Security Professional (CISSP)")]
    validate.clean_analysis(analysis, picks)
    assert analysis.must_have_skills == ["Python", "AWS", "certificate management", "software licensing"]
    assert analysis.nice_to_have_skills == ["Spark"] and analysis.keywords == ["Python", "CISSP", "AWS"]
    before = analysis.model_dump()
    assert validate.clean_analysis(analysis, picks).model_dump() == before  # cleaning again changes nothing


def test_certifications_leave_the_skills_and_are_found_in_the_text(certified):
    resume = validate.build_resume(certified, make_output())
    resume.skills[0].skills = ["Python", "AWS Certified Developer", "PMP", "SQL", "Certified Ethical Hacker", "AWS"]
    removed = validate.remove_certification_skills(resume, ["Project Management Professional (PMP)"])
    assert removed == ["AWS Certified Developer", "PMP", "Certified Ethical Hacker"]
    assert resume.skills[0].skills == ["Python", "SQL", "AWS"]
    resume.summary = "Data engineer who holds the CISSP and builds pipelines."
    resume.experience[1].bullets[0] = "Earned the Project Management Professional credential while leading releases."
    names = ["Certified Information Systems Security Professional (CISSP)", "Project Management Professional (PMP)"]
    assert validate.certification_mentions(resume, names) == [
        ("summary", "CISSP"), ("experience:1", "Project Management Professional")]


def test_a_printed_certification_the_posting_names_counts_as_found(certified):
    out = make_output()
    out.jd_analysis.keywords = ["Python", "Certified ScrumMaster"]
    resume = validate.build_resume(certified, out)
    assert "Certified ScrumMaster" in validate.keyword_coverage(resume, out.jd_analysis).missing
    resume.certifications = [CSM]
    assert "Certified ScrumMaster" in validate.keyword_coverage(resume, out.jd_analysis).found


# ------------------------------------------------------------------ a whole run
class Recording(LLMClient):
    """Keeps every prompt; the first draft names a certification in a bullet, so the fix round runs."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            job = result.data["sections"]["experience"][0]
            job["bullets"][0] = "Passed the Certified ScrumMaster (CSM) exam " + job["bullets"][0]
        return result


def test_every_prompt_carries_the_persons_certifications(router, db, registry, certified):
    """The first draft, the fix round and the shorter draft all see the numbered list (a missed one would lose the
    person's own picks)."""
    client = Recording(FakeClient(registry.get("demo-detailed")))
    router.overrides["demo-detailed"] = client
    result = asyncio.run(run_tailor(router, db, profile=certified, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-detailed"))
    assert result.status == "ok"
    kinds = ["fix" if RETRY_MARKER in p else "shorter" if SHORTEN_NOTE in p else "first" for p in client.prompts]
    assert {"first", "fix", "shorter"} <= set(kinds), kinds
    listed = ("<candidate_certifications>\n0: AWS Certified Solutions Architect – Professional | Amazon Web Services"
              " | 2023\n1: Certified ScrumMaster (CSM) | Scrum Alliance | 2020\n</candidate_certifications>")
    assert all(listed in prompt for prompt in client.prompts)
    fix = next(p for p in client.prompts if RETRY_MARKER in p)
    assert "These certification names appear in the headline, summary or bullets: 'Certified ScrumMaster'" in fix


def test_a_demo_run_prints_only_the_persons_own(router, db, certified):
    result = asyncio.run(run_tailor(router, db, profile=certified, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise"))
    assert result.status == "ok" and len(result.certifications) == 4
    assert [(r.name, r.have) for r in result.certifications] == [
        (AWS.name, True), (CSM.name, True), ("Databricks Certified Data Engineer Professional", False),
        ("Microsoft Certified: Azure Solutions Architect Expert", False)]
    assert result.resume.certifications == [AWS, CSM] and result.resume.profile == certified
    text = text_of(render_resume_pdf(result.resume, result.style)[0])
    assert text.index("EDUCATION") < text.index("CERTIFICATIONS") < text.index("Scrum Alliance")
    assert "Databricks" not in text and "Azure Solutions Architect" not in text
    words = [p.text for p in Document(io.BytesIO(resume_docx(result.resume, result.style))).paragraphs]
    assert words[words.index("CERTIFICATIONS") + 1] == f"{AWS.name} | Amazon Web Services\t2023"


def test_no_certification_no_section(router, db, profile):
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise"))
    assert result.resume.certifications == [] and not any(r.have for r in result.certifications)
    assert "CERTIFICATIONS" not in text_of(render_resume_pdf(result.resume, FitStyle())[0])


def test_the_page_plan_leaves_room_for_the_persons_certifications(profile):
    many = profile.model_copy(deep=True)
    many.certifications = [Certification(name=f"Certification number {i}", issuer="Issuer", year="2020")
                           for i in range(4)]
    assert sum(bullet_budget(many)) <= sum(bullet_budget(profile))


def test_a_tick_that_needs_room_removes_the_fewest_bullets(certified):
    resume = long_resume(certified, bullets=24)
    resume.certifications = [AWS, CSM]
    fitted, _style, pages, fits, _notes, removed = asyncio.run(refit(resume, "Letter"))
    assert fits and pages <= 2 and removed and sum(removed.values()) == (
        sum(len(e.bullets) for e in resume.experience) - sum(len(e.bullets) for e in fitted.experience))
    assert fitted.certifications == [AWS, CSM] and len(resume.experience[0].bullets) == 24  # a copy was trimmed


# ------------------------------------------------------------------ prompts
def test_the_prompts_list_the_persons_certifications_and_the_rule():
    assert PROMPT_VERSION == "v29"
    system = tailor_system()
    assert "CERTIFICATIONS: `certifications` lists the 4 certifications that best support this application" in system
    assert "never name one in the headline, the summary, the skills or the bullets" in system
    assert "Do NOT invent client or product names, certifications" in system
    user = tailor_user(SAMPLE_JD, [], "", certifications=[AWS, Certification(name="CKA")])
    assert ("<candidate_certifications>\n0: AWS Certified Solutions Architect – Professional | Amazon Web Services"
            " | 2023\n1: CKA\n</candidate_certifications>") in user
    assert "<candidate_certifications>\n(none)\n</candidate_certifications>" in tailor_user(SAMPLE_JD, [], "")


# ------------------------------------------------------------------ the Tailor page ticks
def upload(client, text=CERT_RESUME):
    return save_profile(client, text=text)


def stored_certifications(client, resume_id):
    entry = next(r for r in client.get("/api/resumes").json()["resumes"] if r["id"] == resume_id)
    return [c["name"] for c in entry["profile"]["certifications"]]


def ticks(client, run_id, have):
    return client.put(f"/api/runs/{run_id}/certifications", json={"have": have})


def test_ticking_prints_it_and_remembers_it_for_the_next_job(client):
    resume_id = upload(client)["id"]
    assert stored_certifications(client, resume_id) == [AWS.name, CSM.name]  # read from the uploaded resume
    run = tailor(client, profile_id=resume_id).json()
    assert [(r["name"], r["have"]) for r in run["certifications"]] == [
        (AWS.name, True), (CSM.name, True), ("Databricks Certified Data Engineer Professional", False),
        ("Microsoft Certified: Azure Solutions Architect Expert", False)]
    ticked = ticks(client, run["run_id"], [True, True, True, False])
    assert ticked.status_code == 200, ticked.text
    assert [c["name"] for c in ticked.json()["resume"]["certifications"]] == [
        AWS.name, CSM.name, "Databricks Certified Data Engineer Professional"]
    pdf = text_of(client.get(f"/api/runs/{run['run_id']}/resume.pdf").content)
    assert "Databricks Certified Data Engineer Professional" in pdf and "Azure Solutions" not in pdf
    assert stored_certifications(client, resume_id)[-1] == "Databricks Certified Data Engineer Professional"
    nxt = tailor(client, profile_id=resume_id).json()
    assert [r["name"] for r in nxt["certifications"] if r["have"]][:3] == [
        AWS.name, CSM.name, "Databricks Certified Data Engineer Professional"]  # remembered: ticked from the start
    # unticking takes it off this resume and off the base resume
    rows = [r["have"] for r in nxt["certifications"]]
    rows[2] = False
    off = ticks(client, nxt["run_id"], rows).json()
    assert "Databricks Certified Data Engineer Professional" not in [c["name"] for c in off["resume"]["certifications"]]
    assert "Databricks Certified Data Engineer Professional" not in stored_certifications(client, resume_id)
    assert "Databricks" not in text_of(client.get(f"/api/runs/{nxt['run_id']}/resume.pdf").content)


def test_an_older_runs_rows_never_undo_a_newer_tick(client):
    resume_id = upload(client)["id"]
    old = tailor(client, profile_id=resume_id).json()
    new = tailor(client, profile_id=resume_id).json()
    assert ticks(client, new["run_id"], [True, True, True, False]).status_code == 200  # Databricks, on the new run
    # on the old run Databricks still shows unticked; ticking Azure there must not drop Databricks
    assert ticks(client, old["run_id"], [True, True, False, True]).status_code == 200
    assert stored_certifications(client, resume_id) == [
        AWS.name, CSM.name, "Databricks Certified Data Engineer Professional",
        "Microsoft Certified: Azure Solutions Architect Expert"]


def test_wrong_ticks_are_refused(client, user_client):
    resume_id = upload(client)["id"]
    run = tailor(client, profile_id=resume_id).json()
    assert ticks(client, run["run_id"], [True]).status_code == 422
    assert ticks(client, run["run_id"], [True] * 5).status_code == 422
    assert ticks(user_client, run["run_id"], [True] * 4).status_code == 404  # someone else's resume
    other_site = client.put(f"/api/runs/{run['run_id']}/certifications", json={"have": [True] * 4},
                            headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert other_site.status_code == 403
    assert TestClient(client.app).put(f"/api/runs/{run['run_id']}/certifications",
                                      json={"have": [True] * 4}).status_code == 401
    db = client.app.state.db
    older = db.get_run(run["run_id"])
    older.certifications = []  # a run made before certifications existed
    db.update_run(older)
    assert ticks(client, run["run_id"], []).status_code == 409
    assert stored_certifications(client, resume_id) == [AWS.name, CSM.name]  # nothing changed


def test_a_user_sees_the_rows_but_never_the_model(client, user_client):
    upload(user_client)
    run = tailor(user_client).json()
    assert set(run["certifications"][0]) == {"name", "issuer", "year", "have"}
    assert "served_model" not in run and "cost_usd" not in run and "resume" not in run
    ticked = ticks(user_client, run["run_id"], [False, True, False, False])
    assert ticked.status_code == 200 and [r["have"] for r in ticked.json()["certifications"]] == [False, True, False, False]
    assert "AWS Certified" not in text_of(user_client.get(f"/api/runs/{run['run_id']}/resume.pdf").content)


def test_a_tick_keeps_a_cover_letter_written_meanwhile_and_says_when_bullets_went(client, monkeypatch):
    from app import main

    resume_id = upload(client)["id"]
    run = tailor(client, profile_id=resume_id).json()
    real_refit = main.refit

    async def letter_lands_meanwhile(resume, paper):
        assert client.post(f"/api/runs/{run['run_id']}/cover-letter", json={}).status_code == 200
        fitted = await real_refit(resume, paper)
        return (*fitted[:5], {"Northwind Labs": 1})  # as if two pages needed one bullet less

    monkeypatch.setattr(main, "refit", letter_lands_meanwhile)
    ticked = ticks(client, run["run_id"], [True, True, True, False]).json()
    assert ticked["cover_letter"] is not None
    assert ticked["message"] == "To keep two pages, 1 bullet from Northwind Labs was removed."


def test_an_admin_edit_keeps_the_printed_certifications(client):
    resume_id = upload(client)["id"]
    run = tailor(client, profile_id=resume_id).json()
    exp = [{"company_index": e["company_index"], "job_title": e["job_title"], "bullets": e["bullets"]}
           for e in run["resume"]["experience"]]
    edited = client.put(f"/api/runs/{run['run_id']}/resume", json={
        "headline": "Edited", "summary": run["resume"]["summary"], "skills": run["resume"]["skills"], "experience": exp})
    assert edited.status_code == 200
    assert [c["name"] for c in edited.json()["resume"]["certifications"]] == [AWS.name, CSM.name]


# ------------------------------------------------------------------ older data
def test_runs_and_resumes_saved_before_certifications_still_load(router, db, profile):
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", save=False))
    data = result.model_dump()
    for holder in (data, data["resume"], data["resume"]["profile"]):
        del holder["certifications"]
    old = TailorResult.model_validate(data)
    assert old.certifications == [] and old.resume.certifications == [] and old.resume.profile.certifications == []
    assert Resume.model_validate(data["resume"]).certifications == []
    assert TailorOutput.model_validate({k: v for k, v in make_output().model_dump().items()
                                        if k != "certifications"}).certifications == []
    assert CertificationChoice(name="X").have is False


def test_a_resume_upload_with_certifications_is_read(client):
    response = client.post("/api/resumes", files={"file": ("cv.docx", docx_bytes(CERT_RESUME))})
    assert response.status_code == 200
    assert [c["name"] for c in response.json()["resume"]["profile"]["certifications"]] == [AWS.name, CSM.name]
    assert "'Certifications: '" in client.get("/profile").text  # each card lists them
