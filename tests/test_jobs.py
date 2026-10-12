"""Slow tailoring and cover letters: the request answers within JOB_WAIT_SECONDS and the page asks again, so a
hosting proxy's time limit ("Gateway Time-out" on PythonAnywhere) never cuts the work off."""
import sqlite3
import threading
from wsgiref.simple_server import make_server

import httpx
import pytest
from a2wsgi import ASGIMiddleware
from fastapi.testclient import TestClient

from app import main
from app.main import create_app
from app.storage.db import Database

from .conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    SAMPLE_JD,
    USER_PASSWORD,
    add_user,
    log_in,
)
from .test_api import docx_bytes, save_profile
from .test_deploy import Quiet


@pytest.fixture
def browsers(tmp_path, monkeypatch):
    """The admin's and a user's browser on one running app (the work goes on between requests), with every job
    too slow to answer at once."""
    monkeypatch.setattr(main, "JOB_WAIT_SECONDS", 0)
    db = Database(tmp_path / "jobs.db")
    db.set_settings({"parser_model": "rules"})
    with TestClient(create_app(db=db)) as admin:
        log_in(admin, ADMIN_EMAIL, ADMIN_PASSWORD)
        add_user(admin)
        with TestClient(admin.app) as user:
            yield admin, log_in(user, "user@example.com", USER_PASSWORD), db


def answer(browser, response):
    """What the page ends up with: it asks GET /api/jobs/{job} again while the answer is 'running'."""
    for _ in range(500):
        if response.status_code != 202:
            return response
        assert response.json()["status"] == "running"
        threading.Event().wait(0.02)
        response = browser.get(f"/api/jobs/{response.json()['job']}")
    raise AssertionError("the job never ended")


def test_slow_tailoring_answers_at_once_and_the_page_gets_the_run_later(browsers):
    admin, user, _db = browsers
    save_profile(admin)
    started = admin.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"})
    assert started.status_code == 202 and set(started.json()) == {"job", "status"}
    run = answer(admin, started)
    assert run.status_code == 200 and run.json()["status"] == "ok"
    assert run.json() == admin.get(f"/api/runs/{run.json()['run_id']}").json()
    assert user.get(f"/api/jobs/{started.json()['job']}").status_code == 404  # only its owner sees a job
    assert TestClient(admin.app).get(f"/api/jobs/{started.json()['job']}").status_code == 401
    assert admin.get("/api/jobs/" + "x" * 65).status_code == 404


def test_a_user_gets_the_short_view_and_a_slow_cover_letter(browsers):
    _admin, user, _db = browsers
    save_profile(user)
    run = answer(user, user.post("/api/tailor", json={"jd": SAMPLE_JD})).json()
    assert run["status"] == "ok" and "served_model" not in run and "cost_usd" not in run
    letter = answer(user, user.post(f"/api/runs/{run['run_id']}/cover-letter", json={}))
    assert letter.status_code == 200 and letter.json()["cover_letter"]["paragraphs"]


def test_a_failed_slow_cover_letter_tells_the_user_only_to_try_again(browsers, monkeypatch):
    admin, user, _db = browsers

    async def busy_model(*args, **kwargs):
        return None, "Cover letter failed: Gemini 3.8 Flash: HTTP 503: high demand"

    save_profile(admin)
    save_profile(user)
    admin_run = answer(admin, admin.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"})).json()
    user_run = answer(user, user.post("/api/tailor", json={"jd": SAMPLE_JD})).json()
    monkeypatch.setattr(main, "write_cover_letter", busy_model)
    failed = answer(admin, admin.post(f"/api/runs/{admin_run['run_id']}/cover-letter", json={}))
    assert failed.status_code == 502 and "HTTP 503" in failed.json()["detail"]  # the admin sees why
    failed = answer(user, user.post(f"/api/runs/{user_run['run_id']}/cover-letter", json={}))
    assert failed.status_code == 502 and failed.json()["detail"] == main.BUSY_MESSAGE


def test_a_job_stopped_by_a_restart_ends_instead_of_waiting_forever(browsers):
    admin, user, db = browsers
    user_id = db.user_by_email("user@example.com")["id"]
    db.create_job("old", user_id, "tailor")
    db.create_job("new", user_id, "tailor")
    with sqlite3.connect(db.path) as raw:
        raw.execute("UPDATE jobs SET created_at = '2020-01-01T00:00:00Z' WHERE id = 'old'")
    assert user.get("/api/jobs/new").status_code == 202  # still within the time a job can take
    lost = user.get("/api/jobs/old")
    assert lost.status_code == 502 and lost.json()["detail"] == main.BUSY_MESSAGE
    db.create_job("admin-old", db.user_by_email(ADMIN_EMAIL)["id"], "tailor")
    with sqlite3.connect(db.path) as raw:
        raw.execute("UPDATE jobs SET created_at = '2020-01-01T00:00:00Z' WHERE id = 'admin-old'")
    assert admin.get("/api/jobs/admin-old").json()["detail"] == main.JOB_LOST
    db.purge(0)  # old jobs are deleted after a day
    assert db.get_job("old", user_id) is None and db.get_job("new", user_id) is not None


def test_fast_work_still_answers_in_the_same_request(client):
    save_profile(client)
    response = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"})
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_slow_work_goes_on_behind_the_web_tab_bridge_too(tmp_path, monkeypatch):
    """Path B on PythonAnywhere (WSGI through a2wsgi): the job keeps running after its request has answered."""
    monkeypatch.setattr(main, "JOB_WAIT_SECONDS", 0)
    db = Database(tmp_path / "w.db")
    db.set_settings({"parser_model": "rules"})
    server = make_server("127.0.0.1", 0, ASGIMiddleware(create_app(db=db)), handler_class=Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as http:
            log_in(http, ADMIN_EMAIL, ADMIN_PASSWORD)
            assert http.post("/api/resumes", files={"file": ("base.docx", docx_bytes())}).status_code == 200
            started = http.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"})
            assert started.status_code == 202
            run = answer(http, started)
            assert run.status_code == 200 and run.json()["status"] == "ok"
    finally:
        server.shutdown()


def test_pages_ask_for_the_current_script_so_an_update_never_mixes_old_and_new(client, tmp_path):
    """A browser that cached the old app.js broke the new Tailor page ("finished is not defined")."""
    version = main.asset_version(main.WEB / "static")
    page = client.get("/tailor").text
    assert f'/static/app.js?v={version}"' in page and f'/static/style.css?v={version}"' in page
    assert client.get(f"/static/app.js?v={version}").status_code == 200
    (tmp_path / "app.js").write_text("one")
    (tmp_path / "style.css").write_text("css")
    first = main.asset_version(tmp_path)
    (tmp_path / "app.js").write_text("two")
    assert main.asset_version(tmp_path) != first
