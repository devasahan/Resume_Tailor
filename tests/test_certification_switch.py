"""Settings > Certifications section: an on/off switch, off by default.

While it is off, no resume gets a CERTIFICATIONS section, the Tailor page gets no rows to tick, and the model is
not shown the person's certifications: it only lists the ones the job post names, which are never printed and only
keep their names out of the text (in the first draft, the fix round, the shorter draft, the cover letter and
Compare). In both modes, the credential abbreviations a job post names (CISSP, PMP) stay out of the skills and the
text whatever the model lists. The lists saved on the Profile page never change. Switched on, everything works as
before (test_certifications.py switches it on for its own tests).
"""
import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app import config, main
from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.llm.schema import strict_json_schema
from app.main import create_app
from app.pipeline import validate
from app.pipeline.compare import compare
from app.pipeline.fit import bullet_budget
from app.pipeline.tailor import run_tailor, write_cover_letter
from app.prompts import (
    RETRY_MARKER,
    SHORTEN_NOTE,
    revise_user,
    tailor_system,
    tailor_user,
)
from app.render.docx import resume_docx
from app.render.pdf import render_resume_pdf
from app.schemas import Certification, Company, Profile, TailorOutput
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
OFF_RULE = ("CERTIFICATIONS: this resume has no Certifications section. `certifications` lists only the certifications "
            "the job description itself names, at most 4 (an empty list when it names none), each with its official "
            "full name, its abbreviation in brackets when it has one (\"Project Management Professional (PMP)\"), its "
            "issuer and `candidate_index` -1. They are never printed: they only keep those names out of the resume. "
            "Never name a certification or license anywhere in the resume: not in the headline, the summary, the "
            "skills or the bullets.")
ON_RULE = "CERTIFICATIONS: `certifications` lists the 4 certifications that best support this application"
OFF_FIX = "Certifications are never named in this resume (the CERTIFICATIONS rule), so take every one out of the text."
CERT_FLAG = ("Names a certification. Certification names stay out of the headline, summary and bullets; remove it from "
             "the text.")


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


def tailored(router, profile, model="demo-concise", jd=SAMPLE_JD, **kwargs):
    return asyncio.run(run_tailor(router, router.db, profile=profile, profile_id=None, jd=jd, model_id=model,
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


def profile_script(viewer) -> str:
    """The Profile page's own script, as the server sends it to this viewer."""
    page = viewer.get("/profile")
    assert page.status_code == 200
    return page.text.split('<script src="/static/app.js', 1)[1]


def test_the_profile_page_says_whether_the_list_is_printed_now(client, user_client):
    """The page gets the switch from the server, as a boolean only, and words the box and the save status by it:
    while off, saving 2 said 'Your next resumes use the best of them for each job', which was false. Settings is
    the admin's: a user is told that their administrator switches the section on, never sent to Settings."""
    for viewer in (client, user_client):  # off by default
        assert "const CERTIFICATIONS_ON = false;" in profile_script(viewer)
    script = profile_script(client)
    assert ("status.textContent = CERTIFICATIONS_ON ? `Saved ${saved.length}. Your next resumes use the best of them "
            "for each job.`\n        : `Saved ${saved.length}. The Certifications section is switched off, so they "
            "are kept here for later.`;") in script
    assert ("+ (CERTIFICATIONS_ON ? 'For each job, the best ones that fit, up to 4, are put on the resume "
            "automatically, after Education, with their dates. ' : '')") in script
    note = script.split("function switchNote() {", 1)[1].split("\n}\n", 1)[0]
    for_users, for_admin = note.split("if (!ME.admin) {", 1)[1].split("\n  }\n", 1)
    assert "Settings" not in for_users and "settings" not in for_users  # a user cannot open it
    assert ("'This list is printed only while your administrator keeps the Certifications section switched on.'"
            in for_users)
    assert ("'The Certifications section is switched off, so this list is not printed on resumes for now: it is kept "
            "here for later. Your administrator can switch it on.'") in for_users
    assert "'This list is printed only when the Certifications section is switched on in Settings.'" in for_admin
    assert "el('a', { href: '/settings', text: 'Switch it on in Settings' })" in for_admin
    assert "switchNote()," in script.split("function certificationsBox(r) {", 1)[1]  # shown in every box
    assert client.put("/api/settings", json={"certifications_section": True}).status_code == 200
    for viewer in (client, user_client):
        assert "const CERTIFICATIONS_ON = true;" in profile_script(viewer)


# ------------------------------------------------------------------ prompts
def test_off_the_prompts_carry_no_certifications():
    system_on = tailor_system("estimate", "us")
    system_off = tailor_system("estimate", "us", certifications_section=False)
    assert ON_RULE in system_on and OFF_RULE not in system_on
    assert OFF_RULE in system_off and ON_RULE not in system_off and "<candidate_certifications>" not in system_off
    assert len(system_off) < len(system_on) - 800  # a shorter prompt: a little faster
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
            assert call["answer"]["certifications"] == []  # the job post names none, so the demo model lists none
    fix = next(call["user"] for call in client.calls if kind(call) == "fix")
    assert "These certification names appear in the headline, summary or bullets" in fix and OFF_FIX in fix
    # nothing printed, nothing to tick, and the name the draft wrote is out of the text
    assert result.resume.certifications == [] and result.certifications == [] and result.posting_certifications == []
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


class Unfixed(Recording):
    """Recording, with a fix round that changes nothing: the name the first draft wrote stays in the text."""

    async def generate_structured(self, **kwargs):
        result = await super().generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_revision":
            result.data.update(headline="", summary="", titles=[], bullets=[], skill_changes=[])
        return result


def certification_flags(flags) -> list[tuple[str, str, str]]:
    return [(f.section, f.text, f.reason) for f in flags if f.reason.startswith("Names a certification")]


def test_off_a_name_left_in_the_text_is_flagged_with_a_reason_that_is_true(router, registry, certified):
    """While off, a name left in a bullet was flagged 'Certifications belong only in their own section', a section
    this resume does not have."""
    router.overrides["demo-concise"] = Unfixed(FakeClient(registry.get("demo-concise")))
    result = tailored(router, certified)
    assert result.status == "ok" and result.resume.certifications == []
    assert certification_flags(result.flags) == [
        ("experience:0", "Certified Kubernetes Administrator", CERT_FLAG), ("experience:0", "CKA", CERT_FLAG)]


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


# ------------------------------------------------------------------ certifications the job post names
SECURITY_JD = ("Senior Security Engineer at Harrowgate Signal. You will harden cloud workloads on AWS with Python, "
               "Terraform and Kubernetes, run incident response and threat modeling, and mentor engineers. "
               "{credential} required. 5+ years of experience required.")


class Slips(LLMClient):
    """A model that breaks the rules on a certification the job post requires and the person does not hold: each
    draft lists it as a must-have, a keyword and a skill, names it in a bullet, and lists no certification at all
    (the checks must not depend on its list). Its fix round does what it is asked, or (`complies` False) changes
    nothing."""

    def __init__(self, inner, credential, complies=True):
        self.inner, self.credential, self.complies, self.calls = inner, credential, complies, []

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        data = result.data
        if kwargs["schema_name"] == "tailor_output":
            data["certifications"] = []
            data["jd_analysis"]["must_have_skills"].insert(0, self.credential)
            data["jd_analysis"]["keywords"].insert(0, self.credential)
            data["sections"]["skills"][0]["skills"].append(self.credential)
            job = data["sections"]["experience"][0]
            job["bullets"][1] = job["bullets"][1].rstrip(".") + f", applying {self.credential} practices."
        elif kwargs["schema_name"] == "tailor_revision" and not self.complies:
            data.update(headline="", summary="", titles=[], bullets=[], skill_changes=[])
        self.calls.append({**kwargs, "answer": data})
        return result


def problems_of(fix_prompt: str) -> str:
    """What the fix round was asked to fix."""
    return fix_prompt.split(f"{RETRY_MARKER}: ", 1)[1].split("\n\nRevise the previous draft", 1)[0]


def test_the_credential_abbreviations_a_job_post_names_are_found():
    found = validate.credential_abbreviations
    assert found("Senior Security Engineer. CISSP required; CCNA or CCNP a plus.") == ["CISSP", "CCNA", "CCNP"]
    assert found("PMP preferred. PMPs welcome, PMP® holders too") == ["PMP"]
    assert found("CKA/CKAD, CISSP-certified") == ["CISSP", "CKA", "CKAD"]  # whole words, in the list's order
    assert found("CKAD only") == ["CKAD"] and found("RHCE/RHCSA") == ["RHCSA", "RHCE"]
    assert found("cissp, xPMP, PMP2, OCSP stapling") == []  # capitals and whole words only
    # left out on purpose: they also name a pricing model, a government agency, a job and a marketing figure
    assert found("Build CAPM valuation models, follow CISA advisories, work with our CSM and CPA targets") == []
    names = found("CISSP required")
    assert validate.names_certification("CISSP", names) and validate.certification_terms(names) == ["CISSP"]


@pytest.mark.parametrize("post", [
    "You will run PMP and programmatic guaranteed deals in our demand-side platform.",  # a private marketplace
    "Prepare PMP and PMF studies for dam safety reviews.",  # probable maximum precipitation
    "Check the state PMP before dispensing controlled substances.",  # a prescription monitoring program
    "Deliver the Project Management Plan (PMP) for each federal contract.",
    "Join the regional CISM team after critical incidents.",  # critical incident stress management
    "Complete 24 CEH each year to keep your license.",  # continuing education hours
])
def test_an_abbreviation_with_another_meaning_is_no_credential_there(post):
    assert validate.credential_abbreviations(post) == []


def test_an_abbreviation_with_another_meaning_counts_where_it_is_a_credential():
    found = validate.credential_abbreviations
    assert found("PMP required.") == ["PMP"] and found("PMP-certified project lead") == ["PMP"]
    assert found("CISM certification a plus") == ["CISM"] and found("Certifications: PMP, CSM") == ["PMP", "CSM"]
    assert found("PMP or PRINCE2") == ["PMP"] and found("CISA or CISSP certification preferred") == ["CISSP", "CISA"]
    # a '+', an exam version and a plural in capitals are the same credential
    assert found("OSCP+ required") == ["OSCP"] and found("CEHv12 required") == ["CEH"]
    assert found("MUST HAVE: CISSPS") == ["CISSP"]
    assert found("Study the PMP's exam outline") == ["PMP"] and found("Must have PMPs") == ["PMP"]
    assert found("You must have run PMP deals before") == []


def test_a_credential_with_a_plus_or_an_exam_version_leaves_the_skills():
    resume = SimpleNamespace(skills=[SimpleNamespace(category="Security", skills=["OSCP+", "CEHv12", "Burp Suite"])])
    assert validate.remove_certification_skills(resume, ["OSCP", "CEH"]) == ["OSCP+", "CEHv12"]
    assert resume.skills[0].skills == ["Burp Suite"] and validate.names_certification("CEHv12", ["CEH"])


def test_a_listed_name_with_another_meaning_counts_in_the_text_only_as_a_credential():
    """'CISA' taken from the model's 'Certified Information Systems Auditor (CISA)' is the agency in 'Tracked CISA KEV
    advisories'; named on its own (the job post asked for it as a credential), every mention counts."""
    agency = "Tracked CISA KEV advisories against 4,313 hosts with a weekly patch report."
    exam = "Passed the CISA certification exam while leading 3 audits."

    def resume(bullet):
        return SimpleNamespace(headline="Security Engineer", summary="Security engineer for cloud platforms.",
                               experience=[SimpleNamespace(company_index=0, bullets=[bullet])])

    listed = ["Certified Information Systems Auditor (CISA)"]
    assert validate.certification_mentions(resume(agency), listed) == []
    assert ("experience:0", "CISA") in validate.certification_mentions(resume(exam), listed)
    assert validate.certification_mentions(resume(agency), ["CISA"]) == [("experience:0", "CISA")]


ADTECH_JD = ("Programmatic Trader at Larkspur Media. You will run PMP and programmatic guaranteed deals in our "
             "demand-side platform, report on pacing with SQL and Python, and work with publishers. 3+ years of "
             "experience required.")


def test_an_ad_tech_post_keeps_its_pmp_deals(router, registry, profile):
    """'PMP deals' is a private marketplace: the word stays in the skills and the bullets, with no fix round and no
    flag about certifications."""
    client = Slips(FakeClient(registry.get("demo-concise")), "PMP", complies=False)
    router.overrides["demo-concise"] = client
    result = tailored(router, profile, jd=ADTECH_JD)
    assert result.status == "ok"
    assert "PMP" in [skill for group in result.resume.skills for skill in group.skills]
    assert "PMP" in " ".join(result.resume.experience[0].bullets)
    fixes = [problems_of(call["user"]) for call in client.calls if kind(call) == "fix"]
    assert not any("certification names" in problems for problems in fixes)
    assert not [flag for flag in result.flags if flag.text == "PMP"]


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("credential", ["CISSP", "PMP"])
def test_a_required_credential_the_person_lacks_stays_out_of_the_resume(router, registry, profile, on, credential):
    """'CISSP required' or 'PMP required' and no certification saved: while the switch was off, a model that listed
    the abbreviation as a must-have and a skill kept it in the printed Skills line, and the fix round asked for it word
    for word in a bullet. Now it is a certification name in both modes, whatever the model lists."""
    router.db.set_settings({"certifications_section": on})
    client = Slips(FakeClient(registry.get("demo-concise")), credential, complies=False)
    router.overrides["demo-concise"] = client
    result = tailored(router, profile, jd=SECURITY_JD.format(credential=credential))
    assert result.status == "ok"
    assert credential in client.calls[0]["answer"]["sections"]["skills"][0]["skills"]  # the slip
    assert credential not in [skill for group in result.resume.skills for skill in group.skills]  # removed
    assert credential not in result.analysis.must_have_skills + result.analysis.nice_to_have_skills
    fix = next(call["user"] for call in client.calls if kind(call) == "fix")
    problems = problems_of(fix)
    assert f"These certification names appear in the headline, summary or bullets: '{credential}'." in problems
    assert f'"{credential}"' not in problems  # never a must-have phrase to write in word for word
    assert credential not in fix.split("<jd_analysis>", 1)[1].split("</jd_analysis>", 1)[0]  # nor a keyword to use
    assert not any(credential in warning for warning in result.ats_warnings)  # no 'Required by the posting' warning
    # the fix round changed nothing here: the name left in the bullet is flagged, with a reason true in both modes
    assert [(f.section, f.reason) for f in result.flags if f.text == credential] == [("experience:0", CERT_FLAG)]
    assert result.resume.certifications == [] and result.certifications == [] and result.posting_certifications == []
    assert "CERTIFICATIONS" not in pdf_text(render_resume_pdf(result.resume, result.style)[0])


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_a_fix_round_that_does_what_it_is_asked_takes_the_credential_out(router, registry, profile, on):
    """The same slip, with a fix round that does what it is asked: the abbreviation is gone from the whole resume."""
    router.db.set_settings({"certifications_section": on})
    router.overrides["demo-concise"] = Slips(FakeClient(registry.get("demo-concise")), "CISSP")
    result = tailored(router, profile, jd=SECURITY_JD.format(credential="CISSP"))
    assert result.status == "ok" and not [f for f in result.flags if f.text == "CISSP"]
    text = pdf_text(render_resume_pdf(result.resume, result.style)[0])
    assert "CISSP" not in text and "CERTIFICATIONS" not in text


POSTED_PICK = {"name": "Project Management Professional (PMP)", "issuer": "Project Management Institute",
               "candidate_index": -1}


class ListsThePostings(LLMClient):
    """A model that lists the certification the job post names, as the off rule asks (the demo lists none while off),
    or picks it as a suggestion while the switch is on, and still names it in a bullet; its fix round changes
    nothing."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        data = result.data
        if kwargs["schema_name"] == "tailor_output":
            data["certifications"] = [dict(POSTED_PICK)]
            job = data["sections"]["experience"][0]
            job["bullets"][1] = job["bullets"][1].rstrip(".") + ", following Project Management Professional rules."
        elif kwargs["schema_name"] == "tailor_revision":
            data.update(headline="", summary="", titles=[], bullets=[], skill_changes=[])
        self.calls.append({**kwargs, "answer": data})
        return result


DELIVERY_JD = ("Delivery Lead at Corvane Freight. You will plan releases for a logistics platform with Python, SQL and "
               "AWS, run Agile ceremonies and report to stakeholders. Project Management Professional certification "
               "required. 5+ years of experience required.")


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_the_certifications_the_job_post_names_are_kept_out_of_the_text(router, registry, profile, on):
    """While off, the model lists the job post's own certifications (the one-line rule asks it to); they are never
    printed and never rows, but the checks use their names, also when a new cover letter checks the run again. The
    post names no abbreviation here, so only the model's list knows 'Project Management Professional'."""
    router.db.set_settings({"certifications_section": on})
    client = ListsThePostings(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = tailored(router, profile, jd=DELIVERY_JD)
    assert result.status == "ok" and validate.credential_abbreviations(DELIVERY_JD) == []
    first = next(call for call in client.calls if kind(call) == "first")
    assert (OFF_RULE in first["system"]) is not on and (ON_RULE in first["system"]) is on
    fix = problems_of(next(call["user"] for call in client.calls if kind(call) == "fix"))
    assert ("These certification names appear in the headline, summary or bullets: 'Project Management "
            "Professional'") in fix
    assert (OFF_FIX in fix) is not on
    assert result.resume.certifications == []  # never printed: the person does not have it
    if on:  # a suggestion to tick, as before
        assert [(row.name, row.have) for row in result.certifications] == [(POSTED_PICK["name"], False)]
        assert result.posting_certifications == []
    else:  # no rows; the name is kept for the checks only
        assert result.certifications == []
        assert result.posting_certifications == [POSTED_PICK["name"]]
    flagged = [(f.section, f.text, f.reason) for f in result.flags if f.reason == CERT_FLAG]
    assert flagged == [("experience:0", "Project Management Professional", CERT_FLAG)]
    assert validate.certification_mentions(result.resume, []) == []  # no generic claim: only the list finds it
    again, error = asyncio.run(write_cover_letter(router, router.db, result.run_id, jd=DELIVERY_JD,
                                                  model_id="demo-concise"))
    assert error is None and again.cover_letter is not None
    assert [(f.section, f.text, f.reason) for f in again.flags if f.reason == CERT_FLAG] == flagged
    assert "CERTIFICATIONS" not in pdf_text(render_resume_pdf(again.resume, again.style)[0])


def test_the_schema_reads_true_in_both_modes():
    """The schema goes with every draft, in both modes (pasted into the system message for some providers): it no
    longer asks for 'the 4 certifications' or points at a <candidate_certifications> block the off prompt lacks."""
    schema = json.dumps(strict_json_schema(TailorOutput))
    for stale in ("the 4 certifications", "<candidate_certifications>", "suggestions to tick", "picks for the job"):
        assert stale not in schema, stale
    picks = strict_json_schema(TailorOutput)["properties"]["certifications"]
    assert picks["description"] == "the certifications the CERTIFICATIONS rule asks for, at most 4, best first"
    assert picks["items"]["properties"]["candidate_index"]["description"] == (
        "the number of the candidate's own certification when the prompt lists their certifications and this is one "
        "of them, otherwise -1")
    for section in (True, False):  # the Honesty rule no longer says the -1 ones are shown as ideas
        system = tailor_system("estimate", "us", certifications_section=section)
        honesty = "The certifications you list with `candidate_index` -1 are never presented as the candidate's own"
        assert honesty in system and "shown to the candidate as ideas" not in system


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


def test_off_a_name_typed_into_the_text_is_flagged_the_same_way(client):
    resume_id = save_profile(client, text=LISTED_RESUME)["id"]
    run = tailor(client, profile_id=resume_id).json()
    exp = [{"company_index": e["company_index"], "job_title": e["job_title"], "bullets": list(e["bullets"])}
           for e in run["resume"]["experience"]]
    exp[0]["bullets"][1] = ("Earned the Certified Kubernetes Administrator (CKA) while moving 38 internal teams onto "
                            "managed Kubernetes clusters with Terraform modules.")
    edited = client.put(f"/api/runs/{run['run_id']}/resume", json={
        "headline": run["resume"]["headline"], "summary": run["resume"]["summary"], "skills": run["resume"]["skills"],
        "experience": exp})
    assert edited.status_code == 200 and edited.json()["resume"]["certifications"] == []
    flags = [(f["section"], f["text"], f["reason"]) for f in edited.json()["flags"]
             if f["reason"].startswith("Names a certification")]
    assert flags == [
        ("experience:0", "Certified Kubernetes Administrator", CERT_FLAG), ("experience:0", "CKA", CERT_FLAG)]


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
