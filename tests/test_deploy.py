"""Running behind a host's proxy (PythonAnywhere): same-origin check, HTTPS only, the WSGI entry point, and the
setup script."""
import os
import shutil
import subprocess
import threading
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

import httpx
import pytest
from a2wsgi import ASGIMiddleware
from fastapi.testclient import TestClient

from app.main import create_app
from app.storage.db import Database

from .conftest import ADMIN_EMAIL, ADMIN_PASSWORD

ROOT = Path(__file__).resolve().parent.parent
LOGIN = {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}


def test_the_site_name_from_the_proxy_counts_as_same_origin(client):
    stranger = TestClient(client.app)
    proxied = {"Origin": "https://jdoe.pythonanywhere.com", "Host": "localhost",
               "X-Forwarded-Host": "jdoe.pythonanywhere.com"}
    assert stranger.post("/api/login", json=LOGIN, headers=proxied).status_code == 200
    other_site = {**proxied, "Origin": "https://evil.example"}
    assert stranger.post("/api/login", json=LOGIN, headers=other_site).status_code == 403
    assert stranger.post("/api/login", json=LOGIN, headers={"Origin": "https://jdoe.pythonanywhere.com",
                                                             "Host": "localhost"}).status_code == 403


def test_force_https_sends_plain_http_visitors_to_https(client, monkeypatch):
    monkeypatch.setenv("FORCE_HTTPS", "1")
    page = client.get("/login?next=/tailor", headers={"X-Forwarded-Proto": "http"}, follow_redirects=False)
    assert page.status_code == 308 and page.headers["location"] == "https://testserver/login?next=/tailor"
    assert client.post("/api/login", json=LOGIN, headers={"X-Forwarded-Proto": "http"}).status_code == 403
    assert client.get("/api/me", headers={"X-Forwarded-Proto": "https"}).status_code == 200
    assert client.get("/api/me").status_code == 200  # no proxy header (local use): nothing changes
    monkeypatch.delenv("FORCE_HTTPS")
    assert client.get("/api/me", headers={"X-Forwarded-Proto": "http"}).status_code == 200


class Quiet(WSGIRequestHandler):
    def log_message(self, *args):
        pass


def test_the_wsgi_entry_point_serves_the_app(tmp_path):
    """The Web-tab route: a real WSGI server, the login page, a login and a page that needs it."""
    db = Database(tmp_path / "w.db")
    server = make_server("127.0.0.1", 0, ASGIMiddleware(create_app(db=db)), handler_class=Quiet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as http:
            assert http.get("/login").status_code == 200
            assert http.post("/api/login", json=LOGIN).status_code == 200
            assert http.get("/api/me").json()["email"] == ADMIN_EMAIL
            assert "Log out" in http.get("/tailor").text
    finally:
        server.shutdown()
    import app.wsgi  # the module the WSGI file imports
    assert callable(app.wsgi.application)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_setup_script_makes_env_once_and_prints_the_commands(tmp_path):
    project = tmp_path / "Resume_Tailor"
    (project / "deploy").mkdir(parents=True)
    shutil.copy(ROOT / ".env.example", project / ".env.example")
    shutil.copy(ROOT / "deploy" / "pythonanywhere_setup.sh", project / "deploy" / "pythonanywhere_setup.sh")
    env = {**os.environ, "HOME": str(tmp_path), "USER": "jdoe", "RESUME_TAILOR_SKIP_INSTALL": "1"}
    script = str(project / "deploy" / "pythonanywhere_setup.sh")
    assert subprocess.run(["bash", "-n", script], check=False).returncode == 0
    first = subprocess.run(["bash", script], env=env, capture_output=True, text=True, check=True).stdout
    dotenv = (project / ".env").read_text()
    assert "COOKIE_SECURE=1" in dotenv and "FORCE_HTTPS=1" in dotenv and "ADMIN_EMAIL=\n" in dotenv
    assert oct((project / ".env").stat().st_mode)[-3:] == "600"
    assert "--domain jdoe.pythonanywhere.com" in first and "--uds ${DOMAIN_SOCKET}" in first
    assert "from app.wsgi import application" in first and "https://jdoe.pythonanywhere.com" in first
    (project / ".env").write_text("ADMIN_EMAIL=kept@example.com\n")
    again = subprocess.run(["bash", script], env=env, capture_output=True, text=True, check=True).stdout
    assert "left as it is" in again and (project / ".env").read_text() == "ADMIN_EMAIL=kept@example.com\n"
