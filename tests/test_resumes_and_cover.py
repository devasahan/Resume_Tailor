"""Upload-only resume list, picking a resume on the Tailor page, and the cover letter as its own step."""
import asyncio
import io
import sqlite3

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.llm.registry import Registry
from app.llm.router import Router
from app.main import create_app
from app.pipeline.tailor import run_tailor, write_cover_letter
from app.schemas import Profile, TailorResult
from app.storage.db import Database

from .conftest import ADMIN_EMAIL, ADMIN_PASSWORD, SAMPLE_JD, SAMPLE_RESUME_TEXT, log_in
from .test_api import docx_bytes, pdf_bytes, save_profile, tailor
from .test_pipeline import Priced

OTHER_RESUME = SAMPLE_RESUME_TEXT.replace("Jordan Rivera", "Casey Morgan").replace("jordan.rivera", "casey.morgan")
GLOBEX_JD = "About Globex Corporation\n" + SAMPLE_JD


@pytest.fixture
def app_db(tmp_path):
    """A test client logged in as the admin, plus direct access to its database and router."""
    db = Database(tmp_path / "app.db")
    db.set_settings({"parser_model": "rules"})
    router = Router(Registry.load(), db)
    return log_in(TestClient(create_app(db=db, router=router)), ADMIN_EMAIL, ADMIN_PASSWORD), db, router


# ------------------------------------------------------------------ uploading and listing
@pytest.mark.parametrize("name,make", [
    ("base.docx", lambda: docx_bytes()), ("base.pdf", lambda: pdf_bytes()), ("base.txt", lambda: SAMPLE_RESUME_TEXT.encode()),
])
def test_upload_saves_and_lists_the_resume(client, name, make):
    response = client.post("/api/resumes", files={"file": (name, make())})
    assert response.status_code == 200, response.text
    entry = response.json()["resume"]
    assert entry["profile"]["name"] == "Jordan Rivera" and entry["filename"] == name
    assert entry["read_with"] == "built-in rules" and entry["method"] == "rules"
    listed = client.get("/api/resumes").json()["resumes"]
    assert [r["id"] for r in listed] == [entry["id"]]
    assert [c["name"] for c in listed[0]["profile"]["companies"]][:2] == ["Northwind Labs", "Contoso Systems"]


def test_unreadable_uploads_are_rejected_and_not_saved(client):
    no_jobs = client.post("/api/resumes", files={"file": ("cv.txt", b"Jane Doe\njane@example.com\nI like turtles a lot.")})
    assert no_jobs.status_code == 422 and "employer" in no_jobs.json()["detail"] and "not saved" in no_jobs.json()["detail"]
    assert client.post("/api/resumes", files={"file": ("x.exe", b"MZ....")}).status_code == 422
    huge = b"x" * (5 * 1024 * 1024 + 10)
    assert client.post("/api/resumes", files={"file": ("big.txt", huge)}).status_code == 422
    assert client.get("/api/resumes").json()["resumes"] == []


def test_two_resumes_and_choosing_one_to_tailor(client):
    first = save_profile(client)
    second = save_profile(client, name="casey.docx", text=OTHER_RESUME)
    listed = client.get("/api/resumes").json()["resumes"]
    assert [r["id"] for r in listed] == [second["id"], first["id"]]  # newest first
    run = tailor(client, profile_id=first["id"]).json()
    assert run["resume"]["profile"]["name"] == "Jordan Rivera"
    assert tailor(client).json()["resume"]["profile"]["name"] == "Casey Morgan"  # no id: the newest
    assert tailor(client, profile_id=99999).status_code == 404
    runs = client.get("/api/runs").json()["runs"]
    assert {r["candidate"] for r in runs if r["task"] == "resume"} == {"Jordan Rivera", "Casey Morgan"}


def test_delete_keeps_old_runs_working(client, user_client):
    entry = save_profile(client)
    run_id = tailor(client, profile_id=entry["id"]).json()["run_id"]
    assert user_client.delete(f"/api/resumes/{entry['id']}").status_code == 404  # only its owner can delete it
    assert client.delete(f"/api/resumes/{entry['id']}").status_code == 200
    assert client.get("/api/resumes").json()["resumes"] == []
    assert client.get(f"/api/runs/{run_id}/resume.pdf").status_code == 200  # the run keeps its own copy
    assert client.delete(f"/api/resumes/{entry['id']}").status_code == 404
    assert tailor(client, profile_id=entry["id"]).status_code == 404
    assert tailor(client).status_code == 400  # nothing left to tailor from


def test_a_profile_saved_through_the_old_form_is_listed(client):
    profile = client.post("/api/profile/parse", files={"file": ("b.docx", docx_bytes())}, data={"model": "rules"}).json()
    assert client.post("/api/profile", json=profile["profile"]).status_code == 200
    (entry,) = client.get("/api/resumes").json()["resumes"]
    assert entry["read_with"] == "typed in" and entry["filename"] == ""


def test_reader_setting(client):
    assert client.put("/api/settings", json={"parser_model": "rules"}).status_code == 200
    assert client.get("/api/settings").json()["resolved"]["parser"] is None
    assert client.put("/api/settings", json={"parser_model": "nope"}).status_code == 422
    assert client.put("/api/settings", json={"parser_model": "demo-concise"}).status_code == 200
    entry = save_profile(client)
    assert entry["method"] == "model:demo-concise" and entry["read_with"]  # an explicit choice is used


def test_upload_under_the_cost_cap_falls_back_to_the_rules(app_db):
    client, db, _router = app_db
    db.set_settings({"parser_model": "demo-concise", "daily_cost_cap_usd": 0.5})
    db.save_run(TailorResult(requested_model="m", cost_usd=0.6), None, "")
    entry = save_profile(client)
    assert entry["method"] == "rules" and any("cap" in w.lower() for w in entry["warnings"])


# ------------------------------------------------------------------ storage
def test_migration_shows_only_the_newest_legacy_profile_and_keeps_all_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, data TEXT NOT NULL)")
    for name in ("Old One", "Old Two", "Newest"):
        conn.execute("INSERT INTO profiles(created_at, data) VALUES('2026-01-01T00:00:00Z', ?)",
                     (Profile(name=name).model_dump_json(),))
    conn.commit()
    conn.close()
    db = Database(path)
    assert [e["profile"].name for e in db.list_profiles(user_id=None)] == ["Newest"]  # no owner yet
    Database(path)  # opening again changes nothing
    assert [e["profile"].name for e in db.list_profiles(user_id=None)] == ["Newest"]
    with sqlite3.connect(path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 3  # hidden, not deleted


def test_retention_keeps_uploaded_resumes(app_db):
    client, db, _router = app_db
    entry = save_profile(client)
    with sqlite3.connect(db.path) as raw:
        raw.execute("UPDATE profiles SET created_at = '2020-01-01T00:00:00Z'")
    db.purge(30)
    admin_id = db.user_by_email(ADMIN_EMAIL)["id"]
    assert [e["id"] for e in db.list_profiles(user_id=admin_id)] == [entry["id"]]


# ------------------------------------------------------------------ tailoring with the Settings model
def test_tailor_without_a_model_uses_the_settings_default(client):
    save_profile(client)
    assert client.put("/api/settings", json={"resume_model": "demo-detailed"}).status_code == 200
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD}).json()
    assert run["requested_model"] == "demo-detailed" and run["status"] == "ok"


def test_runs_store_the_job_description_and_old_runs_still_load(client):
    save_profile(client)
    run = tailor(client).json()
    assert run["jd"] == SAMPLE_JD
    old = dict(run)
    del old["jd"]
    assert TailorResult.model_validate(old).jd == ""


# ------------------------------------------------------------------ cover letter as its own step
def test_cover_letter_for_an_existing_run(client):
    save_profile(client)
    run = tailor(client, jd=GLOBEX_JD).json()
    assert run["cover_letter"] is None
    response = client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})
    assert response.status_code == 200, response.text
    letter = response.json()["cover_letter"]
    assert len(letter["paragraphs"]) >= 2 and letter["company_name"] == "Globex Corporation" and letter["hiring_manager"] == ""
    pdf = client.get(f"/api/runs/{run['run_id']}/cover.pdf?download=1")
    assert pdf.headers["content-disposition"] == (
        'attachment; filename="Jordan_Rivera_Cover_Letter.pdf"')
    assert "Re: Senior Data Engineer" in "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf.content)).pages)
    runs = client.get("/api/runs").json()["runs"]
    row = next(r for r in runs if r["id"] == run["run_id"])
    assert row["cover_letter"] is True
    assert row["input_tokens"] == run["input_tokens"]  # the run's own columns never change...
    spend = [r for r in runs if r["task"] == "cover_letter"]
    assert len(spend) == 1 and spend[0]["input_tokens"] > 0  # ...the letter has its own dated row


def test_rewriting_the_cover_letter_does_not_duplicate_flags(client):
    save_profile(client)
    run_id = tailor(client).json()["run_id"]
    counts = []
    for _ in range(2):
        flags = client.post(f"/api/runs/{run_id}/cover-letter", json={}).json()["flags"]
        counts.append(sum(f["section"] == "cover_letter" for f in flags))
    assert counts[0] == counts[1]


def test_cover_letter_errors(app_db):
    client, db, _router = app_db
    save_profile(client)
    assert client.post("/api/runs/9999/cover-letter", json={}).status_code == 404
    failed = db.save_run(TailorResult(requested_model="m", status="error", error="boom"), None, "")
    assert client.post(f"/api/runs/{failed}/cover-letter", json={}).status_code == 409
    good = tailor(client).json()
    legacy = TailorResult.model_validate({**good, "jd": "", "run_id": None})
    legacy_id = db.save_run(legacy, None, "")  # a run saved before job descriptions were kept
    assert client.post(f"/api/runs/{legacy_id}/cover-letter", json={}).status_code == 409
    assert client.post(f"/api/runs/{legacy_id}/cover-letter", json={"jd": SAMPLE_JD}).status_code == 200
    plain = client.post(f"/api/runs/{good['run_id']}/cover-letter", content="{}", headers={"content-type": "text/plain"})
    assert plain.status_code == 422  # a cross-site form cannot trigger a paid call


def test_cover_letter_cost_counts_today_even_for_an_older_run(router, db, registry, profile, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    router.overrides["claude-sonnet-5-5"] = Priced(FakeClient(registry.get("demo-concise")))
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, model_id="demo-concise"))
    with sqlite3.connect(db.path) as raw:  # the run was made yesterday
        raw.execute("UPDATE runs SET created_at = '2020-01-01T00:00:00Z' WHERE id = ?", (result.run_id,))
    for _ in range(2):
        fresh, error = asyncio.run(write_cover_letter(router, db, result.run_id, jd=SAMPLE_JD, model_id="claude-sonnet-5-5"))
        assert error is None and fresh.cover_letter is not None
    assert fresh.cover_cost_usd == pytest.approx(0.044) and fresh.cost_usd == 0  # 2 x (1000 in at $2 + 2000 out at $10 per 1M)
    assert db.cost_today() == pytest.approx(0.044)  # the daily cap sees it
    db.set_settings({"daily_cost_cap_usd": 0.04})
    _fresh, error = asyncio.run(write_cover_letter(router, db, result.run_id, jd=SAMPLE_JD, model_id="claude-sonnet-5-5"))
    assert error and "cap" in error.lower()


def test_cover_letter_stops_at_the_cost_cap(app_db):
    client, db, _router = app_db
    save_profile(client)
    run_id = tailor(client).json()["run_id"]
    db.set_settings({"daily_cost_cap_usd": 0.5})
    db.save_run(TailorResult(requested_model="m", cost_usd=0.6), None, "")
    response = client.post(f"/api/runs/{run_id}/cover-letter", json={})
    assert response.status_code == 502 and "cap" in response.json()["detail"].lower()
    assert client.get(f"/api/runs/{run_id}").json()["cover_letter"] is None


class SpendsDuringResume(LLMClient):
    """Demo model that, while writing the resume, records spending that pushes the day over the cap."""

    def __init__(self, inner, db):
        self.inner, self.db = inner, db

    async def generate_structured(self, **kwargs):
        if kwargs["schema_name"] == "tailor_output":
            self.db.save_run(TailorResult(requested_model="other", cost_usd=0.6), None, "")
        return await self.inner.generate_structured(**kwargs)


def test_cost_cap_between_resume_and_cover_letter_keeps_the_resume(router, db, registry, profile):
    db.set_settings({"daily_cost_cap_usd": 0.5})
    router.overrides["demo-concise"] = SpendsDuringResume(FakeClient(registry.get("demo-concise")), db)
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", cover_letter=True))
    assert result.status == "ok" and result.resume is not None and result.cover_letter is None
    assert any(n.startswith("Cover letter not written") for n in result.notes)


# ------------------------------------------------------------------ pages
def test_pages_show_only_the_simple_controls(client):
    tailor_page = client.get("/tailor").text
    for gone in ("Real facts about you", "Compare several", "Company applying to", "Hiring manager"):
        assert gone not in tailor_page
    assert "Create resume &amp; cover letter" in tailor_page and "Resume only" in tailor_page and 'id="resume"' in tailor_page
    assert "Make cover letter" not in tailor_page  # one main button now does both
    profile_page = client.get("/profile").text
    assert "Save profile" not in profile_page and "Read file" not in profile_page and "Drop resumes here" in profile_page


# ------------------------------------------------------------------ fixes from review
def test_cross_site_requests_are_refused(client):
    files = {"file": ("base.docx", docx_bytes())}
    assert client.post("/api/resumes", files=files, headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/resumes", files=files, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post("/api/resumes", files=files, headers={"Sec-Fetch-Site": "same-site"}).status_code == 403
    assert client.delete("/api/data", headers={"Origin": "null"}).status_code == 403
    same = {"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"}
    assert client.post("/api/resumes", files=files, headers=same).status_code == 200
    assert client.get("/api/resumes", headers={"Origin": "https://evil.example"}).status_code == 200  # reads are fine


def test_long_file_names_keep_their_type(client):
    name = "Jordan_Rivera_" + "x" * 140 + ".docx"
    response = client.post("/api/resumes", files={"file": (name, docx_bytes())})
    assert response.status_code == 200, response.text
    stored = response.json()["resume"]["filename"]
    assert len(stored) <= 120 and stored.endswith(".docx")


def test_an_edit_saved_while_the_letter_is_written_keeps_the_letter(client, monkeypatch):
    from app import main

    save_profile(client)
    run = tailor(client).json()
    real_fit = main.fit_report

    async def slow_fit(resume, paper):  # the letter lands while the edit is being checked
        assert client.post(f"/api/runs/{run['run_id']}/cover-letter", json={}).status_code == 200
        return await real_fit(resume, paper)

    monkeypatch.setattr(main, "fit_report", slow_fit)
    exp = [{"company_index": e["company_index"], "job_title": e["job_title"], "bullets": e["bullets"]}
           for e in run["resume"]["experience"]]
    edited = client.put(f"/api/runs/{run['run_id']}/resume", json={
        "headline": "Edited", "summary": run["resume"]["summary"], "skills": run["resume"]["skills"], "experience": exp})
    assert edited.status_code == 200 and edited.json()["cover_letter"] is not None
    assert client.get(f"/api/runs/{run['run_id']}/cover.pdf").status_code == 200


def test_automatic_label_ignores_the_saved_choice(router, db, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    db.set_settings({"parser_model": "rules"})
    assert router.default_model("parser") is None
    assert router.default_model("parser", automatic=True) == "claude-sonnet-5-5"  # what Automatic would send it to
