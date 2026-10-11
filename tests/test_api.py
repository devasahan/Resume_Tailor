import io

import pytest
from docx import Document
from fastapi.testclient import TestClient
from pypdf import PdfReader
from weasyprint import HTML

from app.main import create_app
from app.storage.db import Database

from .conftest import SAMPLE_JD, SAMPLE_RESUME_TEXT


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    return TestClient(create_app(db=Database(tmp_path / "api.db")))


def docx_bytes(text=SAMPLE_RESUME_TEXT) -> bytes:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def pdf_bytes(text=SAMPLE_RESUME_TEXT) -> bytes:
    html = "<html><body>" + "".join(f"<p>{line}</p>" for line in text.splitlines()) + "</body></html>"
    return HTML(string=html).write_pdf()


def save_profile(client):
    parsed = client.post("/api/profile/parse", files={"file": ("base.docx", docx_bytes())}, data={"model": "rules"}).json()
    assert client.post("/api/profile", json=parsed["profile"]).status_code == 200
    return parsed


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
    assert run["status"] == "ok" and run["fits"] and run["pages"] == 1
    run_id = run["run_id"]

    pdf = client.get(f"/api/runs/{run_id}/resume.pdf")
    reader = PdfReader(io.BytesIO(pdf.content))
    assert pdf.headers["content-type"] == "application/pdf" and len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "Jordan Rivera" in text and "Northwind Labs" in text and "SUMMARY" in text and "EXPERIENCE" in text

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

    assert client.get(f"/run/{run_id}").status_code == 200
    assert client.get("/api/runs/9999").status_code == 404


def test_history_and_costs(client):
    save_profile(client)
    tailor(client)
    tailor(client, model="demo-detailed", compare=True)
    runs = client.get("/api/runs").json()
    assert len(runs["runs"]) == 2
    assert {r["requested_model"] for r in runs["runs"]} == {"demo-concise", "demo-detailed"}


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


def test_admin_token_protects_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    client = TestClient(create_app(db=Database(tmp_path / "a.db")))
    assert client.put("/api/settings", json={"paper_size": "A4"}).status_code == 401
    assert client.put("/api/settings", json={"paper_size": "A4"}, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.put("/api/settings", json={"paper_size": "A4"}, headers={"X-Admin-Token": "s3cret"}).status_code == 200
    assert client.delete("/api/data").status_code == 401
    assert client.get("/api/settings").json()["admin_token_required"] is True


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
