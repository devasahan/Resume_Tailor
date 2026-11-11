"""Certifications: the AI picks the 4 best for each job, and only the ones the person has are ever printed.

A pick that is one of the person's own (read from their resume, or ticked "I have it" on the Tailor page) is
printed under CERTIFICATIONS, after EDUCATION. Every other pick is only a suggestion on the Tailor page.
Certification names never appear in the headline, summary, skills or bullets, so unticking really takes one off.
All of this needs Settings > Certifications section on (it is off by default), so these tests switch it on;
test_certification_switch.py covers the switch itself.
"""
import asyncio
import io

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.parsing.heuristic import (
    _is_header,
    _parse_certifications,
    parse_resume_text,
    read_certification_list,
)
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
    SkillGroup,
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


@pytest.fixture
def db(db):
    """The shared database with the Certifications section switched on (Settings; it is off by default)."""
    db.set_settings({"certifications_section": True})
    return db


@pytest.fixture
def client(client):
    """The shared web app with the Certifications section switched on, like `db` above."""
    client.app.state.db.set_settings({"certifications_section": True})
    return client


def pick(name, index=-1, issuer=""):
    return CertificationPick(name=name, issuer=issuer, candidate_index=index)


# ------------------------------------------------------------------ reading the uploaded resume
def test_certification_headings_are_whole_lines():
    for heading in ("Certifications", "CERTIFICATIONS:", "Licenses & Certifications", "Certifications and Licenses",
                    "Professional Certifications", "Certificates", "Licences", "Credentials"):
        assert _is_header(heading) == "certifications", heading
    assert _is_header("Certificate in Data Analytics, UC Irvine") is None  # an entry, not a heading
    assert _is_header("Education & Certifications") == "education_and_certifications"
    for joined in ("Certifications & Training", "Certifications / Licenses", "Certifications (Selected)"):
        assert _is_header(joined) == "certifications_and_more", joined
    assert _is_header("Training") is None  # only after the certifications does it start a section


UK_CV = """Alex Morgan
London | alex@example.com

PROFESSIONAL EXPERIENCE
Senior Data Engineer, Acme Ltd, London    Jan 2020 - Present
Responsibilities:
- Own the data platform
Achievements:
- Cut costs
Data Engineer, Globex plc, Manchester    Jun 2016 - Dec 2019
Training:
- Spark
Graduate Analyst, Initech, Leeds    Sep 2014 - May 2016

EDUCATION
University of Leeds
BSc in Mathematics, 2014
Coursework:
Statistics
"""


def test_sub_headings_inside_a_job_never_end_experience():
    """'Achievements:' under a job or 'Coursework:' under a degree is no section heading (a resume without any
    certification kept all its employers before certifications were read, and still does)."""
    profile = parse_resume_text(UK_CV)
    assert len(profile.companies) == 3 and profile.education[0].degree == "BSc" and profile.certifications == []


def test_the_section_after_the_certifications_is_not_read_as_certifications():
    after = parse_resume_text(UK_CV + "\nCERTIFICATIONS\nAWS Certified Solutions Architect – Associate, Amazon Web "
                              "Services, 2021\nVOLUNTEER EXPERIENCE\nHabitat for Humanity | Volunteer, 2015 - 2018\n")
    assert [c.name for c in after.certifications] == ["AWS Certified Solutions Architect – Associate"]
    mixed = parse_resume_text(UK_CV.replace("EDUCATION", "CERTIFICATIONS & TRAINING\nAWS Certified Solutions "
                                            "Architect – Associate | Jan 2023 - Jan 2026\nAgile coaching workshop, "
                                            "2019\n\nEDUCATION"))
    assert [c.name for c in mixed.certifications] == ["AWS Certified Solutions Architect – Associate"]
    assert len(mixed.companies) == 3
    skills = parse_resume_text(UK_CV.replace("EDUCATION", "SKILLS\nCertificates: TLS, PKI, X.509\n\nEDUCATION"))
    assert skills.certifications == []  # TLS certificates are a skill, not a credential


def test_linkedin_bullets_and_dates_first_are_read():
    found = [(c.name, c.issuer, c.year) for c in _parse_certifications([
        "AWS Certified Solutions Architect – Associate", "Amazon Web Services (AWS)",
        "Issued Jan 2023 · Expires Jan 2026", "Credential ID ABC123",  # LinkedIn: the issuer has its own line
        "2021 | AWS Certified Developer – Associate | Amazon Web Services",
        "Certified ScrumMaster (Scrum Alliance, Mar 2020)",
        "● Google Cloud Professional Data Engineer, Google Cloud, 2022", "\uf0b7 PMP"])]
    assert found == [
        ("AWS Certified Solutions Architect – Associate", "Amazon Web Services (AWS)", "Jan 2023"),
        ("AWS Certified Developer – Associate", "Amazon Web Services", "2021"),
        ("Certified ScrumMaster", "Scrum Alliance", "Mar 2020"),
        ("Google Cloud Professional Data Engineer", "Google Cloud", "2022"), ("PMP", "", "")]


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
    # a bracket counts only when it stands for the name
    assert validate.same_certification("Google Cloud Professional Cloud Architect (PCA)", "PCA")
    assert not validate.same_certification("Solutions Architect Associate (AWS)", "AWS")  # a platform, not this one
    assert not validate.same_certification("Kubernetes (CKA)", "Kubernetes")  # one word is a skill
    assert not validate.same_certification("AWS Certified Solutions Architect (SAA-C03)",
                                           "AWS Certified Solutions Architect (SAP-C02)")  # two exam codes, two levels


def test_a_certification_named_like_a_tool_never_takes_the_tool_away():
    """'GitHub Copilot' and 'GitHub Actions' are certifications and tools; 'Python' or 'AWS' typed as a certification
    is a skill. Such names are printed when picked, but never remove a skill or mark a sentence."""
    own = ["GitHub Copilot", "GitHub Actions", "Python", "AWS", "Kubernetes (CKA)"]
    assert validate.certification_terms(own) == []
    for skill in ("GitHub Copilot", "GitHub Actions", "Python", "AWS", "Kubernetes"):
        assert not validate.names_certification(skill, own), skill


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
    analysis.must_have_skills += ["AWS Certified Solutions Architect", "FAA certification", "SOC 2 certification",
                                  "GitHub Actions"]
    names = ["Certified Information Systems Security Professional (CISSP)", "GitHub Actions"]
    validate.clean_analysis(analysis, names)
    assert analysis.must_have_skills == ["Python", "AWS", "certificate management", "software licensing",
                                         "FAA certification", "SOC 2 certification", "GitHub Actions"]
    assert analysis.nice_to_have_skills == ["Spark"] and analysis.keywords == ["Python", "CISSP", "AWS"]
    before = analysis.model_dump()
    assert validate.clean_analysis(analysis, names).model_dump() == before  # cleaning again changes nothing


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
    # any credential claim is found, not only the picks; work to a standard is not a claim
    resume.summary = "Senior Cloud Engineer and AWS Certified Solutions Architect who builds AWS-certified pipelines."
    resume.experience[0].bullets[0] = "Led the SOC 2 certification and HIPAA-certified hosting for the billing API."
    resume.experience[2].bullets[0] = "Rolled out GitHub Actions and GitHub Copilot to every team."
    assert validate.certification_mentions(resume, names + ["GitHub Actions", "GitHub Copilot"]) == [
        ("summary", "AWS Certified Solutions Architect"), ("summary", "AWS-certified"),
        ("experience:1", "Project Management Professional")]  # still there from above


def test_a_group_of_certifications_leaves_the_skills(certified):
    resume = validate.build_resume(certified, make_output())
    resume.skills = [SkillGroup(category="Core", skills=["Python", "GitHub Actions", "FAA certification"]),
                     SkillGroup(category="Certifications", skills=["CISSP", "OSCP", "CompTIA Security+"])]
    assert validate.remove_certification_skills(resume, ["GitHub Actions"]) == ["CISSP", "OSCP", "CompTIA Security+"]
    assert [(g.category, g.skills) for g in resume.skills] == [("Core", ["Python", "GitHub Actions", "FAA certification"])]


def test_one_bad_pick_never_costs_the_whole_draft():
    base = make_output().model_dump()
    base["certifications"] = [
        "Certified ScrumMaster (CSM)",  # a bare name
        {"name": "CISSP", "issuer": ["ISC2"]},  # a list where text belongs: this one goes
        {"name": "", "candidate_index": 0},  # no name, but one of the person's own
        {"name": "PMP", "candidate_index": float("inf")},  # a number that is not one
        {"name": True},  # not a name: this one goes
    ]
    picks = TailorOutput.model_validate(base).certifications
    assert [(p.name, p.candidate_index) for p in picks] == [
        ("Certified ScrumMaster (CSM)", -1), ("", 0), ("PMP", -1)]


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
              " | earned 2023\n1: Certified ScrumMaster (CSM) | Scrum Alliance | earned 2020\n</candidate_certifications>")
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
    assert PROMPT_VERSION == "v33"
    system = tailor_system()
    assert "CERTIFICATIONS: `certifications` lists the 4 certifications that best support this application" in system
    assert "never name one in the headline, the summary, the skills or the bullets" in system
    assert "Do NOT invent client or product names, certifications" in system
    user = tailor_user(SAMPLE_JD, [], "", certifications=[AWS, Certification(name="CKA")])
    assert ("<candidate_certifications>\n0: AWS Certified Solutions Architect – Professional | Amazon Web Services"
            " | earned 2023\n1: CKA\n</candidate_certifications>") in user
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
    assert set(run["certifications"][0]) == {"name", "issuer", "year", "expires", "have"}
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
    assert ticked["message"] == ("To keep two pages, 1 bullet from Northwind Labs was removed. Untick a "
                                 "certification to bring them back.") and ticked["message_kind"] == "warn"


def test_unticking_brings_back_the_bullets_a_tick_removed(client, monkeypatch):
    from app import main

    resume_id = upload(client)["id"]
    run = tailor(client, profile_id=resume_id).json()
    whole = [len(e["bullets"]) for e in run["resume"]["experience"]]
    given = []
    real_refit = main.refit

    async def tight_refit(resume, paper):  # as if 4 certifications needed one bullet less on two pages
        given.append([len(e.bullets) for e in resume.experience])
        fitted = await real_refit(resume, paper)
        if len(resume.certifications) < 4:
            return fitted
        trimmed = fitted[0].model_copy(deep=True)
        trimmed.experience[-1].bullets.pop()
        name = trimmed.profile.companies[trimmed.experience[-1].company_index].name
        return (trimmed, *fitted[1:5], {name: 1})

    monkeypatch.setattr(main, "refit", tight_refit)
    ticked = ticks(client, run["run_id"], [True, True, True, True]).json()
    assert ticked["message_kind"] == "warn"
    assert sum(len(e["bullets"]) for e in ticked["resume"]["experience"]) == sum(whole) - 1
    back = ticks(client, run["run_id"], [True, True, False, False]).json()
    assert given[-1] == whole  # the next tick starts again from the whole text
    assert [len(e["bullets"]) for e in back["resume"]["experience"]] == whole
    assert back["message"] == "The bullets removed earlier are back." and back["message_kind"] == "ok"


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
    assert "Certifications you have passed" in client.get("/profile").text  # each card lists them, to change


# ------------------------------------------------------------------ certifications pasted on the Profile page
TABLE = ("1. Software Engineer\n"
         "S.No.\tCertification\tExam code\n"
         "1\tAWS Cloud Practitioner\tCLF-C02\n"
         "2\tAWS Developer – Associate\tDVA-C02 → DVA-C03\n"
         "2. Backend Engineer\n"
         "16\tAWS Developer – Associate\tDVA-C02 → DVA-C03\n"
         "21\tCertified Kubernetes Administrator\tCKA\n"
         "31\tMeta Front-End Developer (Coursera)\t—\n"
         "32\tMeta Back-End Developer (Coursera)\t—\n"
         "54\tNVIDIA Generative AI LLMs (Associate)\tNCA-GENL\n"
         "55\tNVIDIA Generative AI LLMs (Professional)\tNCP-GENL\n")


def test_a_pasted_table_keeps_the_names_once():
    """Rows copied from a table: the number and exam-code columns and the titles between rows are left out, and a
    certification listed under two roles is kept once. A level or a course site in brackets keeps two apart."""
    assert [c.name for c in read_certification_list(TABLE)] == [
        "AWS Cloud Practitioner", "AWS Developer – Associate", "Certified Kubernetes Administrator",
        "Meta Front-End Developer (Coursera)", "Meta Back-End Developer (Coursera)",
        "NVIDIA Generative AI LLMs (Associate)", "NVIDIA Generative AI LLMs (Professional)"]
    lines = "1. AWS Certified Developer – Associate, Amazon Web Services, 2024\n- PMP | PMI | 2021\n• CKAD\nCertification\n"
    assert [(c.name, c.issuer, c.year) for c in read_certification_list(lines)] == [
        ("AWS Certified Developer – Associate", "Amazon Web Services", "2024"), ("PMP", "PMI", "2021"), ("CKAD", "", "")]
    assert not validate.same_certification("Certified Secure Software Lifecycle Professional (ISC2)",
                                           "Certified Information Systems Security Professional (ISC2)")
    # a bulleted list copied from Word, a table with its own header, names with a comma
    assert [(c.name, c.issuer) for c in read_certification_list(
        "•\tAWS Certified Developer – Associate, Amazon Web Services, 2022\n•\tCertified Kubernetes Administrator (CKA)")] == [
        ("AWS Certified Developer – Associate", "Amazon Web Services"), ("Certified Kubernetes Administrator (CKA)", "")]
    assert [(c.name, c.issuer, c.year) for c in read_certification_list(
        "Certification\tIssuer\tYear\nCKAD\tThe Linux Foundation\t2024\nPMP\tPMI\t2021")] == [
        ("CKAD", "The Linux Foundation", "2024"), ("PMP", "PMI", "2021")]
    oracle = ("Oracle Certified Professional, Java SE 11 Developer | Oracle | 2020\n"
              "Oracle Certified Professional, Java SE 17 Developer | Oracle | 2023")  # as the box shows them
    assert [c.name for c in read_certification_list(oracle)] == [
        "Oracle Certified Professional, Java SE 11 Developer", "Oracle Certified Professional, Java SE 17 Developer"]


def test_pasted_certifications_are_printed_without_asking(client):
    resume_id = save_profile(client)["id"]  # a resume that lists none
    saved = client.put(f"/api/resumes/{resume_id}/certifications", json={"text": TABLE})
    assert saved.status_code == 200, saved.text
    assert len(saved.json()["resume"]["profile"]["certifications"]) == 7
    run = tailor(client, profile_id=resume_id).json()
    printed = [c["name"] for c in run["resume"]["certifications"]]
    assert len(printed) == 4 and all(r["have"] for r in run["certifications"])  # nothing to tick
    assert set(printed) <= {c.name for c in read_certification_list(TABLE)}  # only the person's own
    # pasting again replaces the list; an empty box clears it
    assert client.put(f"/api/resumes/{resume_id}/certifications", json={"text": "CKAD"}).json()[
        "resume"]["profile"]["certifications"] == [{"name": "CKAD", "issuer": "", "year": "", "expires": ""}]
    assert client.put(f"/api/resumes/{resume_id}/certifications", json={"text": ""}).json()[
        "resume"]["profile"]["certifications"] == []


def test_only_the_owner_can_change_a_resumes_certifications(client, user_client):
    resume_id = save_profile(client)["id"]
    assert user_client.put(f"/api/resumes/{resume_id}/certifications", json={"text": "CKAD"}).status_code == 404
    other_site = client.put(f"/api/resumes/{resume_id}/certifications", json={"text": "CKAD"},
                            headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert other_site.status_code == 403
    unreadable = client.put(f"/api/resumes/{resume_id}/certifications", json={"text": "1.\n2.\n---"})
    assert unreadable.status_code == 422  # never wipes the saved list by mistake
    too_many = "\n".join(f"Certification number {i}" for i in range(101))
    assert client.put(f"/api/resumes/{resume_id}/certifications", json={"text": too_many}).status_code == 422
    assert stored_certifications(client, resume_id) == []


def test_a_save_from_an_old_page_never_undoes_a_newer_tick(client):
    resume_id = upload(client)["id"]
    shown = [{"name": AWS.name, "issuer": AWS.issuer, "year": AWS.year},
             {"name": CSM.name, "issuer": CSM.issuer, "year": CSM.year}]  # what the Profile page showed
    run = tailor(client, profile_id=resume_id).json()
    assert ticks(client, run["run_id"], [True, True, True, False]).status_code == 200  # meanwhile, in a Tailor tab
    stale = client.put(f"/api/resumes/{resume_id}/certifications",
                       json={"text": f"{AWS.name}\n{CSM.name}", "loaded": shown})
    assert stale.status_code == 409 and "Reload this page" in stale.json()["detail"]
    assert len(stored_certifications(client, resume_id)) == 3  # the tick is still there


def test_no_name_in_the_checks_module_is_defined_twice():
    """A second helper with the same name silently replaces the first (it broke the keyword and dash rules once)."""
    import ast
    import collections
    from pathlib import Path

    tree = ast.parse(Path(validate.__file__).read_text(encoding="utf-8"))
    names = collections.Counter()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] += 1
    assert [name for name, count in names.items() if count > 1] == []



# ------------------------------------------------------------------ dates: earned and expires
def test_both_dates_are_read_in_every_usual_form():
    found = [(c.name, c.year, c.expires) for c in _parse_certifications([
        "AWS Certified Developer – Associate, Amazon Web Services, Mar 2023 - Mar 2026",
        "PMP, PMI, 2021, expires 2027",
        "CompTIA Security+ (valid until Dec 2025)",
        "Google Cloud Professional Data Engineer", "Google Cloud", "Issued Jan 2024 · Expires Jan 2026",
        "Azure Fundamentals", "Issued Feb 2023 · No Expiration Date",
        "HashiCorp Terraform Associate", "Issued Mar 2024", "Expires Mar 2026",  # LinkedIn, on lines of their own
        "Lean Six Sigma Green Belt, 2019 - Present"])]
    assert found == [
        ("AWS Certified Developer – Associate", "Mar 2023", "Mar 2026"), ("PMP", "2021", "2027"),
        ("CompTIA Security+", "", "Dec 2025"), ("Google Cloud Professional Data Engineer", "Jan 2024", "Jan 2026"),
        ("Azure Fundamentals", "Feb 2023", ""), ("HashiCorp Terraform Associate", "Mar 2024", "Mar 2026"),
        ("Lean Six Sigma Green Belt", "2019", "")]
    table = "Certification\tIssuer\tIssued\tExpires\nCKAD\tThe Linux Foundation\tMar 2024\t03/2026\nPMP\tPMI\t2021\tNo Expiration"
    assert [(c.name, c.year, c.expires) for c in read_certification_list(table)] == [
        ("CKAD", "Mar 2024", "03/2026"), ("PMP", "2021", "")]
    box = "AWS Certified Developer – Associate | Amazon Web Services | Mar 2024 - Mar 2027\nCSM | Scrum Alliance | Expires Feb 2026"
    assert [(c.year, c.expires) for c in read_certification_list(box)] == [("Mar 2024", "Mar 2027"), ("", "Feb 2026")]


def test_a_date_has_expired_only_once_its_month_is_over():
    from datetime import date

    from app.pipeline.facts import is_expired

    today = date(2026, 11, 11)
    assert [is_expired(d, today) for d in ("Oct 2026", "03/2026", "2025")] == [True, True, True]
    assert [is_expired(d, today) for d in ("Nov 2026", "2026", "No Expiration", "", "soon")] == [False] * 5


def test_an_expired_certification_is_never_picked_or_printed(certified):
    from datetime import date

    from app.prompts import describe_certifications

    today = date(2026, 11, 11)
    certified.certifications[0].expires = "Jan 2025"  # AWS has expired; CSM has no expiry
    certified.certifications[1].expires = "Feb 2027"
    printed, rows = validate.resolve_certifications(
        certified, [pick("AWS", 0), pick("AWS Certified Solutions Architect Professional"), pick("CSM", 1)], today)
    assert [r.name for r in rows] == [CSM.name] and [c.name for c in printed] == [CSM.name]
    assert printed[0].expires == "Feb 2027" and rows[0].expires == "Feb 2027"
    # the model is never shown it; the others keep their numbers
    assert describe_certifications(certified.certifications, today) == (
        "1: Certified ScrumMaster (CSM) | Scrum Alliance | earned 2020, expires Feb 2027")


def test_the_resume_shows_both_dates(certified):
    certified.certifications = [
        Certification(name="AWS Certified Developer – Associate", issuer="Amazon Web Services", year="Mar 2023",
                      expires="Mar 2026"),
        Certification(name="Certified ScrumMaster (CSM)", issuer="Scrum Alliance", year="Feb 2020"),
        Certification(name="PMP", expires="Jun 2027")]
    resume = validate.build_resume(certified, make_output())
    resume.certifications = list(certified.certifications)
    text = " ".join(text_of(render_resume_pdf(resume, FitStyle())[0]).split())
    assert "Amazon Web Services Mar 2023 - Mar 2026" in text and "Scrum Alliance Feb 2020" in text
    assert "PMP Expires Jun 2027" in text
    words = [p.text for p in Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs]
    assert "AWS Certified Developer – Associate | Amazon Web Services\tMar 2023 - Mar 2026" in words


def test_the_profile_page_names_the_expired_ones(client):
    resume_id = save_profile(client)["id"]
    saved = client.put(f"/api/resumes/{resume_id}/certifications", json={
        "text": "CKAD | The Linux Foundation | Mar 2020 - Mar 2023\nPMP | PMI | Jun 2021 - Jun 2099"}).json()["resume"]
    assert saved["expired"] == ["CKAD"]
    assert [(c["year"], c["expires"]) for c in saved["profile"]["certifications"]] == [
        ("Mar 2020", "Mar 2023"), ("Jun 2021", "Jun 2099")]
    run = tailor(client, profile_id=resume_id).json()
    assert [c["name"] for c in run["resume"]["certifications"]] == ["PMP"]  # the expired one is left off
    assert "Expired, so never printed" in client.get("/profile").text
