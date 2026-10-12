"""Running behind a host's proxy (PythonAnywhere): same-origin check, HTTPS only, the WSGI entry point, and the
setup script."""
import io
import json
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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_the_wsgi_file_works_in_workers_forked_after_loading(tmp_path, monkeypatch):
    """WSGI servers often import the app once and then fork workers: each worker needs its own bridge thread."""
    from wsgiref.util import setup_testing_defaults

    monkeypatch.setenv("DATA_DIR", str(tmp_path))  # importing app.wsgi builds the real app
    from app import wsgi

    def status_of(path, body=b""):
        environ, seen = {"PATH_INFO": path}, []
        if body:
            environ.update(REQUEST_METHOD="POST", CONTENT_TYPE="application/json", CONTENT_LENGTH=str(len(body)),
                           **{"wsgi.input": io.BytesIO(body)})
        setup_testing_defaults(environ)
        answer = b"".join(wsgi.application(environ, lambda status, headers, exc=None: seen.append(status)))
        return seen[0], answer

    login = json.dumps(LOGIN).encode()
    # the parent made its bridge and hashed the admin password: both started threads
    assert status_of("/login")[0].startswith("200") and status_of("/api/login", login)[0].startswith("200")
    pid = os.fork()
    if pid == 0:  # the worker: must answer (page and login) instead of waiting forever on the parent's threads
        ok = status_of("/login")[0].startswith("200") and status_of("/api/login", login)[0].startswith("200")
        os._exit(0 if ok else 1)
    deadline_ok = False
    for _ in range(100):
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            deadline_ok = True
            break
        threading.Event().wait(0.1)
    if not deadline_ok:
        os.kill(pid, 9)
        os.waitpid(pid, 0)
    assert deadline_ok and os.waitstatus_to_exitcode(status) == 0


def test_hsts_header_when_https_is_forced(client, monkeypatch):
    assert "strict-transport-security" not in client.get("/login").headers  # local use: never
    monkeypatch.setenv("FORCE_HTTPS", "1")
    assert client.get("/login").headers["strict-transport-security"] == "max-age=31536000"


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
    mixed = subprocess.run(["bash", script], env={**env, "USER": "ResumeBuilder"}, capture_output=True, text=True,
                           check=True).stdout  # the address is lowercase, the home folder keeps its capitals
    assert "--domain resumebuilder.pythonanywhere.com" in mixed and "https://resumebuilder.pythonanywhere.com" in mixed
    (project / ".env").write_text("ADMIN_EMAIL=kept@example.com\n")
    again = subprocess.run(["bash", script], env=env, capture_output=True, text=True, check=True).stdout
    assert "left as it is" in again and (project / ".env").read_text() == "ADMIN_EMAIL=kept@example.com\n"
