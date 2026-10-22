import io

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader
from fpdf import FPDF


from .conftest import SAMPLE_JD, SAMPLE_RESUME_TEXT


def docx_bytes(text=SAMPLE_RESUME_TEXT) -> bytes:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def pdf_bytes(text=SAMPLE_RESUME_TEXT) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in text.splitlines():
        pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def save_profile(client, name="base.docx", text=SAMPLE_RESUME_TEXT):
    """Upload a base resume the way the Profile page does; returns the saved list entry."""
    response = client.post("/api/resumes", files={"file": (name, docx_bytes(text))})
    assert response.status_code == 200, response.text
    return response.json()["resume"]


def tailor(client, **overrides):
    body = {"jd": SAMPLE_JD, "model": "demo-concise", **overrides}
    return client.post("/api/tailor", json=body)


def test_pages_render(client):
    for path in ("/", "/profile", "/tailor", "/history", "/settings"):
        assert client.get(path).status_code == 200


def test_tailor_needs_a_profile_first(client):
    r = tailor(client)
    assert r.status_code == 400 and "profile" in r.json()["detail"].lower()


@pytest.mark.parametrize("make,name", [(docx_bytes, "base.docx"), (pdf_bytes, "base.pdf")])
def test_upload_parsing(client, make, name):
    parsed = client.post("/api/profile/parse", files={"file": (name, make())}, data={"model": "rules"}).json()
    profile = parsed["profile"]
    assert profile["name"] == "Jordan Rivera" and profile["contact"]["email"] == "jordan.rivera@example.com"
    assert [c["name"] for c in profile["companies"]][:2] == ["Northwind Labs", "Contoso Systems"]
    assert parsed["method"] == "rules"


def test_upload_rejects_bad_files(client):
    assert client.post("/api/profile/parse", files={"file": ("x.exe", b"MZ....")}, data={"model": "rules"}).status_code == 422
    assert client.post("/api/profile/parse", files={"file": ("x.txt", b"hi")}, data={"model": "rules"}).status_code == 422
    r = client.post("/api/profile/parse", files={"file": ("x.docx", docx_bytes())}, data={"model": "claude-sonnet-5-5"})
    assert r.status_code == 422  # no API key in the test environment


def test_profile_validation(client):
    assert client.post("/api/profile", json={"name": " ", "companies": [{"name": "A"}]}).status_code == 422
    assert client.post("/api/profile", json={"name": "A B", "companies": []}).status_code == 422


def test_full_flow_pdf_docx_and_edit(client):
    save_profile(client)
    run = tailor(client, cover_letter=True, company_name="Acme").json()
    assert run["status"] == "ok" and run["fits"] and run["pages"] == 2  # three long-stay jobs: a full two pages
    run_id = run["run_id"]

    pdf = client.get(f"/api/runs/{run_id}/resume.pdf")
    reader = PdfReader(io.BytesIO(pdf.content))
    assert pdf.headers["content-type"] == "application/pdf" and len(reader.pages) == run["pages"]
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Jordan Rivera" in text and "Northwind Labs" in text and "SUMMARY" in text and "EXPERIENCE" in text
    assert "Tech:" not in text  # no tech line under the jobs: the tools are in the bullets and the skills

    docx = client.get(f"/api/runs/{run_id}/resume.docx")
    assert "Jordan Rivera" in "\n".join(p.text for p in Document(io.BytesIO(docx.content)).paragraphs)
    assert client.get(f"/api/runs/{run_id}/cover.pdf").status_code == 200
    assert client.get(f"/api/runs/{run_id}/cover.docx").status_code == 200

    edit = {
        "headline": "Data Engineer | Python", "summary": "Edited summary text.",
        "skills": run["resume"]["skills"],
        "experience": [{"company_index": e["company_index"], "job_title": e["job_title"], "bullets": e["bullets"][:2]} for e in run["resume"]["experience"]],
        "profile": {"name": "Hacker"},  # must be ignored: fixed facts never come from the request
    }
    edited = client.put(f"/api/runs/{run_id}/resume", json=edit).json()
    assert edited["resume"]["headline"] == "Data Engineer | Python"
    assert edited["resume"]["profile"]["name"] == "Jordan Rivera"
    assert all(len(e["bullets"]) == 2 for e in edited["resume"]["experience"])
    assert client.get(f"/api/runs/{run_id}").json()["resume"]["summary"] == "Edited summary text."

    bad = dict(edit, experience=edit["experience"][:1])
    assert client.put(f"/api/runs/{run_id}/resume", json=bad).status_code == 422

    assert client.get(f"/run/{run_id}", follow_redirects=False).headers["location"] == "/tailor"  # page removed
    assert client.get("/api/runs/9999").status_code == 404


def test_history_and_costs(client):
    save_profile(client)
    tailor(client)
    tailor(client, model="demo-detailed", compare=True)
    runs = client.get("/api/runs").json()
    assert len(runs["runs"]) == 2
    assert {r["requested_model"] for r in runs["runs"]} == {"demo-concise", "demo-detailed"}
    assert {r["task"] for r in runs["runs"]} == {"resume"}


def test_unavailable_model_is_rejected(client):
    save_profile(client)
    r = tailor(client, model="claude-sonnet-5-5")
    assert r.status_code == 422 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_jd_must_be_substantial(client):
    save_profile(client)
    assert tailor(client, jd="too short").status_code == 422


def test_settings_validation_and_effect(client):
    assert client.put("/api/settings", json={"resume_model": "nope"}).status_code == 422
    assert client.put("/api/settings", json={"paper_size": "Legal"}).status_code == 422
    assert client.put("/api/settings", json={"allowed_regions": ["MARS"]}).status_code == 422
    assert client.put("/api/settings", json={"unknown": 1}).status_code == 422
    r = client.put("/api/settings", json={"resume_model": "demo-detailed", "paper_size": "A4", "daily_cost_cap_usd": 2}).json()
    assert r["resolved"]["resume"] == "demo-detailed" and r["settings"]["paper_size"] == "A4"
    save_profile(client)
    run = tailor(client, model="demo-concise").json()
    assert run["style"]["paper"] == "A4"


def test_cost_cap_returns_failed_run(client):
    save_profile(client)
    client.put("/api/settings", json={"daily_cost_cap_usd": 0.0001})
    client.app.state.db.save_run(__import__("app.schemas", fromlist=["TailorResult"]).TailorResult(requested_model="m", cost_usd=1), None, "")
    run = tailor(client).json()
    assert run["status"] == "error" and "cap" in run["error"].lower()


def test_only_the_admin_changes_settings_or_wipes_data(client, user_client):
    anonymous = TestClient(client.app)
    assert anonymous.put("/api/settings", json={"paper_size": "A4"}).status_code == 401
    assert user_client.put("/api/settings", json={"paper_size": "A4"}).status_code == 403
    assert user_client.delete("/api/data").status_code == 403
    assert client.put("/api/settings", json={"paper_size": "A4"}).status_code == 200


def test_model_check_and_data_deletion(client):
    r = client.post("/api/models/check", json={"models": ["demo-concise", "claude-sonnet-5-5"]}).json()["results"]
    assert r[0]["ok"] is True and r[1]["ok"] is False and "no API key" in r[1]["error"]
    save_profile(client)
    tailor(client)
    assert client.delete("/api/data").json() == {"deleted": True}
    assert client.get("/api/profile").json()["profile"] is None and client.get("/api/runs").json()["runs"] == []


def test_models_endpoint_never_leaks_keys(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret")
    body = client.get("/api/models").text
    assert "sk-ant-very-secret" not in body
    models = {m["id"]: m for m in client.get("/api/models").json()["models"]}
    assert models["claude-sonnet-5-5"]["available"] is True and models["deepseek-v4-flash"]["available"] is False


def test_security_headers_on_pages_but_not_on_pdfs(client):
    page = client.get("/tailor")
    assert page.headers["x-content-type-options"] == "nosniff" and page.headers["x-frame-options"] == "SAMEORIGIN"
    assert "default-src 'self'" in page.headers["content-security-policy"] and "object-src 'none'" in page.headers["content-security-policy"]
    save_profile(client)
    run_id = tailor(client).json()["run_id"]
    pdf = client.get(f"/api/runs/{run_id}/resume.pdf")
    assert "content-security-policy" not in pdf.headers and pdf.headers["x-content-type-options"] == "nosniff"


def test_cross_site_style_requests_cannot_use_a_non_json_content_type(client):
    save_profile(client)
    r = client.post("/api/tailor", content='{"jd": "x", "model": "demo-concise"}', headers={"content-type": "text/plain"})
    assert r.status_code == 422


@pytest.mark.parametrize("text,expected", [
    ("JUAN DANIEL RAMIREZ", "JUAN_DANIEL_RAMIREZ"),
    ("José Álvarez-Łukasz", "Jose_Alvarez_ukasz"),
    ("王小明", ""),
    ("O'Brien, Ann", "O_Brien_Ann"),
    ("Senior Data Engineer (Remote)", "Senior_Data_Engineer_Remote"),
])
def test_file_name_parts(text, expected):
    from app.main import file_part

    assert file_part(text) == expected


def _result(name="Jordan Rivera", company_typed="", company_in_jd="", role="Senior Data Engineer"):
    from app.schemas import (
        Contact,
        JDAnalysis,
        Profile,
        Resume,
        ResumeExperience,
        SkillGroup,
        TailorResult,
    )

    profile = Profile(name=name, contact=Contact(), companies=[])
    resume = Resume(profile=profile, headline="h", summary="s", skills=[SkillGroup(category="c", skills=["x"])],
                    experience=[ResumeExperience(company_index=0, job_title="Data Engineer", bullets=["b"])])
    analysis = JDAnalysis(company=company_in_jd, target_role=role, seniority="", domain="", must_have_skills=[],
                          nice_to_have_skills=[], keywords=[], key_responsibilities=[])
    return TailorResult(requested_model="m", resume=resume, analysis=analysis, company_name=company_typed)


def test_download_name_is_the_usual_name_then_document_pattern():
    from app.main import download_name

    # the company and the role are not part of it, wherever they came from
    typed = _result(company_typed="Société Générale", company_in_jd="SG")
    assert download_name(typed, "pdf") == "Jordan_Rivera_Resume.pdf"
    assert download_name(typed, "docx") == "Jordan_Rivera_Resume.docx"
    assert download_name(typed, "pdf", cover=True) == "Jordan_Rivera_Cover_Letter.pdf"
    # a name written in capitals gets normal capitals; accents and symbols as in file_part
    assert download_name(_result(name="JUAN DANIEL RAMIREZ"), "pdf") == "Juan_Daniel_Ramirez_Resume.pdf"
    assert download_name(_result(name="josé o'brien-smith"), "pdf") == "Jose_O_Brien_Smith_Resume.pdf"
    assert download_name(_result(name="Ann McDonald"), "pdf") == "Ann_McDonald_Resume.pdf"
    # no usable name: only the document
    assert download_name(_result(name="王小明"), "pdf", cover=True) == "Cover_Letter.pdf"


def test_runs_saved_before_the_company_field_still_load():
    from app.schemas import TailorResult

    old = _result().model_dump()
    del old["company_name"]
    del old["analysis"]["company"]
    loaded = TailorResult.model_validate(old)
    assert loaded.company_name == "" and loaded.analysis.company == ""


def test_download_headers(client):
    save_profile(client)
    run_id = tailor(client, cover_letter=True).json()["run_id"]
    pdf = client.get(f"/api/runs/{run_id}/resume.pdf?download=1")
    assert pdf.headers["content-disposition"] == 'attachment; filename="Jordan_Rivera_Resume.pdf"'
    assert client.get(f"/api/runs/{run_id}/resume.pdf").headers["content-disposition"].startswith("inline;")
    docx = client.get(f"/api/runs/{run_id}/resume.docx")
    assert docx.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert docx.headers["content-disposition"] == 'attachment; filename="Jordan_Rivera_Resume.docx"'
    assert (client.get(f"/api/runs/{run_id}/cover.docx").headers["content-disposition"]
            == 'attachment; filename="Jordan_Rivera_Cover_Letter.docx"')
    typed = tailor(client, company_name="Acme Corp").json()["run_id"]
    assert (client.get(f"/api/runs/{typed}/resume.pdf?download=1").headers["content-disposition"]
            == 'attachment; filename="Jordan_Rivera_Resume.pdf"')  # the company typed on the form is not in it
    from_jd = tailor(client, jd="About Globex Corporation\n" + SAMPLE_JD).json()["run_id"]
    assert (client.get(f"/api/runs/{from_jd}/resume.pdf?download=1").headers["content-disposition"]
            == 'attachment; filename="Jordan_Rivera_Resume.pdf"')
    no_cover = tailor(client).json()["run_id"]
    assert client.get(f"/api/runs/{no_cover}/cover.pdf").status_code == 404
