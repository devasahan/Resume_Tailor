"""Accounts through the web app: sign-up and approval, login and sessions, what each role can reach,
private data per account, what users are told, and the download history the admin sees."""
import pytest
from fastapi.testclient import TestClient

from app import auth
from app.llm.base import InvalidOutputError, LLMClient
from app.main import BUSY_MESSAGE, public_error, safe_next
from app.schemas import TailorResult

from .conftest import ADMIN_EMAIL, USER_PASSWORD, add_user, log_in
from .test_api import save_profile, tailor

NEW_PASSWORD = "a long new password"


def browser(client) -> TestClient:
    """Another browser on the same app, not logged in."""
    return TestClient(client.app)


def sign_up(client, email="new@example.com", password=NEW_PASSWORD, confirm=None):
    return browser(client).post("/api/signup", json={"email": email, "password": password,
                                                     "confirm": password if confirm is None else confirm})


def user_id_of(client, email) -> int:
    return client.app.state.db.user_by_email(email)["id"]


# ------------------------------------------------------------------ sign up, approval, login
def test_sign_up_waits_for_approval_then_logs_in(client):
    response = sign_up(client)
    assert response.status_code == 200 and "admin approves" in response.json()["message"]
    stranger = browser(client)
    refused = stranger.post("/api/login", json={"email": "new@example.com", "password": NEW_PASSWORD})
    assert refused.status_code == 403 and "waiting" in refused.json()["detail"]
    pending = client.get("/api/users").json()
    assert pending["pending"] == 1 and pending["users"][0]["status"] == "pending"
    assert 'class="badge"' in client.get("/tailor").text  # the admin sees how many wait
    assert client.post(f"/api/users/{user_id_of(client, 'new@example.com')}/approve").status_code == 200
    assert stranger.post("/api/login", json={"email": "New@Example.com", "password": NEW_PASSWORD}).status_code == 200
    assert stranger.get("/api/me").json()["role"] == "user"


def test_sign_up_answers_the_same_for_an_existing_email(client):
    first, again = sign_up(client), sign_up(client, password="another long password")
    assert first.json() == again.json() and again.status_code == 200
    assert sign_up(client, email=ADMIN_EMAIL).json() == first.json()  # nothing reveals the admin either
    assert client.app.state.db.count_users("pending") == 1


def test_sign_up_rules(client, monkeypatch):
    assert sign_up(client, email="nope").status_code == 422
    assert sign_up(client, password="short").status_code == 422
    assert sign_up(client, confirm="something else").status_code == 422
    monkeypatch.setattr(auth, "MAX_PENDING", 1)
    assert sign_up(client, email="one@example.com").status_code == 200
    assert sign_up(client, email="two@example.com").status_code == 429


def test_reject_disable_enable_and_delete(client, user_client):
    db = client.app.state.db
    sign_up(client)
    pending = user_id_of(client, "new@example.com")
    assert client.delete(f"/api/users/{pending}").status_code == 200 and db.get_user(pending) is None  # rejected
    active = user_id_of(client, "user@example.com")
    assert client.post(f"/api/users/{active}/disable").status_code == 200
    assert user_client.get("/api/resumes").status_code == 401  # the open session ends at once
    assert client.post(f"/api/users/{active}/approve").status_code == 409  # not pending
    assert client.post(f"/api/users/{active}/enable").status_code == 200
    log_in(user_client, "user@example.com", USER_PASSWORD)
    save_profile(user_client)
    assert client.delete(f"/api/users/{active}").status_code == 200
    assert db.list_profiles(user_id=active) == [] and user_client.get("/api/resumes").status_code == 401


def test_the_admin_account_cannot_be_changed_from_the_users_page(client):
    admin = user_id_of(client, ADMIN_EMAIL)
    assert client.post(f"/api/users/{admin}/disable").status_code == 409
    assert client.delete(f"/api/users/{admin}").status_code == 404
    assert client.post(f"/api/users/{admin}/reset-password").status_code == 404
    assert client.get("/api/me").status_code == 200


def test_reset_password_and_change_it(client, user_client):
    user = user_id_of(client, "user@example.com")
    temporary = client.post(f"/api/users/{user}/reset-password").json()["password"]
    assert user_client.get("/api/resumes").status_code == 401  # the old session ended
    log_in(user_client, "user@example.com", temporary)
    wrong = user_client.post("/api/account/password", json={"current": "not it at all", "new": NEW_PASSWORD})
    assert wrong.status_code == 403
    assert user_client.post("/api/account/password", json={"current": temporary, "new": "short"}).status_code == 422
    assert user_client.post("/api/account/password", json={"current": temporary, "new": NEW_PASSWORD}).status_code == 200
    assert user_client.get("/api/resumes").status_code == 200  # this browser got a fresh session
    log_in(browser(client), "user@example.com", NEW_PASSWORD)
    assert client.post("/api/account/password", json={"current": "x", "new": NEW_PASSWORD}).status_code == 409  # admin: .env


def test_login_messages_and_lockout(client):
    add_user(client)
    stranger = browser(client)
    bad = stranger.post("/api/login", json={"email": "user@example.com", "password": "wrong password!"})
    unknown = stranger.post("/api/login", json={"email": "nobody@example.com", "password": "wrong password!"})
    assert bad.status_code == unknown.status_code == 401 and bad.json() == unknown.json()
    for _ in range(auth.MAX_FAILURES - 1):
        stranger.post("/api/login", json={"email": "user@example.com", "password": "wrong password!"})
    locked = stranger.post("/api/login", json={"email": "user@example.com", "password": USER_PASSWORD})
    assert locked.status_code == 401 and locked.json() == unknown.json()  # a pause looks like any wrong password
    assert client.post(f"/api/users/{user_id_of(client, 'user@example.com')}/enable").status_code == 409  # still active
    assert any(u["locked_until"] for u in client.get("/api/users").json()["users"])


def test_cookie_flags_new_session_at_login_and_logout(client):
    db = client.app.state.db
    user = add_user(client)
    planted = auth.start_session(db, add_user(client, "attacker@example.com"))  # a session id someone else knows
    stranger = browser(client)
    stranger.cookies.set(auth.COOKIE, planted)
    response = stranger.post("/api/login", json={"email": "user@example.com", "password": USER_PASSWORD})
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie  # plain http in tests
    fresh = response.cookies.get(auth.COOKIE)
    assert fresh and fresh != planted and auth.session_user(db, planted) is None  # the old id is gone
    assert auth.session_user(db, fresh).id == user
    stranger.cookies.clear()
    stranger.cookies.set(auth.COOKIE, fresh)
    assert stranger.post("/api/logout").status_code == 200
    assert stranger.get("/api/resumes").status_code == 401
    behind_proxy = browser(client).post("/api/login", json={"email": "user@example.com", "password": USER_PASSWORD},
                                        headers={"X-Forwarded-Proto": "https"})
    assert "secure" in behind_proxy.headers["set-cookie"].lower()


def test_login_from_another_website_is_refused(client):
    add_user(client)
    response = browser(client).post("/api/login", json={"email": "user@example.com", "password": USER_PASSWORD},
                                    headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


@pytest.mark.parametrize("target,expected", [
    ("/tailor", "/tailor"), ("/history?x=1", "/history?x=1"), ("", ""), ("//evil.example", ""),
    ("/\\evil.example", ""), ("https://evil.example/", ""), ("/api/data", ""), ("/login", ""), ("tailor", ""),
])
def test_safe_next(target, expected):
    assert safe_next(target) == expected


# ------------------------------------------------------------------ who can reach what
def test_anonymous_visitors_only_see_login_and_sign_up(client):
    stranger = browser(client)
    for path in ("/login", "/signup"):
        assert stranger.get(path).status_code == 200
    page = stranger.get("/history", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/login?next=/history"
    assert stranger.get("/tailor", follow_redirects=False).headers["location"].startswith("/login")
    for path in ("/api/resumes", "/api/runs/1", "/api/runs/1/resume.pdf", "/api/settings", "/api/downloads"):
        assert stranger.get(path).status_code == 401, path
    for path in ("/docs", "/openapi.json", "/redoc"):
        assert stranger.get(path, follow_redirects=False).status_code in (401, 303), path
        assert client.get(path).status_code == 404, path  # turned off, even for the admin
    assert stranger.get("/static/app.js").status_code == 200


def test_users_get_profile_and_tailor_only(client, user_client):
    nav = user_client.get("/tailor").text
    assert 'href="/profile"' in nav and 'href="/tailor"' in nav
    assert 'href="/history"' not in nav and 'href="/settings"' not in nav and 'href="/users"' not in nav
    for path in ("/history", "/settings", "/users"):
        page = user_client.get(path, follow_redirects=False)
        assert page.status_code == 303 and page.headers["location"] == "/tailor", path
    assert user_client.get("/account").status_code == 200
    for method, path in (("GET", "/api/settings"), ("GET", "/api/models"), ("GET", "/api/runs"), ("GET", "/api/users"),
                         ("GET", "/api/downloads"), ("PUT", "/api/settings"), ("POST", "/api/models/check"),
                         ("DELETE", "/api/data"), ("GET", "/api/profile"), ("POST", "/api/users/1/approve")):
        assert user_client.request(method, path, json={}).status_code == 403, path
    admin_nav = client.get("/tailor").text
    assert all(f'href="{p}"' in admin_nav for p in ("/history", "/users", "/settings"))


def test_each_account_sees_only_its_own_resumes_and_runs(client, user_client):
    mine = save_profile(user_client)
    theirs = save_profile(client)
    assert [r["id"] for r in user_client.get("/api/resumes").json()["resumes"]] == [mine["id"]]
    assert [r["id"] for r in client.get("/api/resumes").json()["resumes"]] == [theirs["id"]]
    assert tailor(user_client, profile_id=theirs["id"]).status_code == 404
    assert user_client.delete(f"/api/resumes/{theirs['id']}").status_code == 404
    admin_run = tailor(client, profile_id=theirs["id"]).json()["run_id"]
    for path in (f"/api/runs/{admin_run}", f"/api/runs/{admin_run}/resume.pdf", f"/api/runs/{admin_run}/cover.pdf"):
        assert user_client.get(path).status_code == 404, path
    assert user_client.post(f"/api/runs/{admin_run}/cover-letter", json={}).status_code == 404
    user_run = tailor(user_client).json()["run_id"]  # no profile_id: the user's own newest resume
    assert client.get(f"/api/runs/{user_run}/resume.pdf").status_code == 200  # the admin can view it (History)
    assert client.post(f"/api/runs/{user_run}/cover-letter", json={}).status_code == 404  # but not change it
    assert user_client.post(f"/api/runs/{user_run}/cover-letter", json={}).status_code == 200
    for file in ("resume.docx", "cover.docx"):  # users download PDFs only; Word files are the admin's
        assert user_client.get(f"/api/runs/{user_run}/{file}").status_code == 403, file
        assert client.get(f"/api/runs/{user_run}/{file}").status_code == 200, file
    another = log_in(browser(client), add_user(client, "other@example.com") and "other@example.com", USER_PASSWORD)
    assert another.get(f"/api/runs/{user_run}/resume.pdf").status_code == 404
    assert another.get("/api/resumes").json()["resumes"] == []


# ------------------------------------------------------------------ what users are told
def test_users_never_choose_the_model_or_see_its_name_or_cost(client, user_client):
    save_profile(user_client)
    assert client.put("/api/settings", json={"resume_model": "demo-detailed"}).status_code == 200
    run = tailor(user_client, model="demo-concise", compare=True).json()
    assert set(run) == {"run_id", "status", "error", "pages", "fits", "coverage", "cover_letter", "created_at"}
    stored = client.get(f"/api/runs/{run['run_id']}").json()
    assert stored["requested_model"] == "demo-detailed"  # the Settings model, not the one asked for
    assert user_client.post("/api/profile/parse", files={"file": ("b.txt", b"x")}, data={"model": "demo-concise"}).status_code == 403
    upload = user_client.get("/api/resumes").json()["resumes"][0]
    assert upload["read_with"] == "" and upload["method"] == ""


def test_users_get_plain_errors_and_the_admin_the_details(client, user_client, monkeypatch):
    save_profile(user_client)
    save_profile(client)
    client.put("/api/settings", json={"daily_cost_cap_usd": 0.0001})
    client.app.state.db.save_run(TailorResult(requested_model="m", cost_usd=1), None, "")
    user_run = tailor(user_client).json()
    assert user_run["status"] == "error" and user_run["error"] == "Today's limit for this site has been reached. Try again tomorrow."
    admin_run = tailor(client).json()
    assert "Settings" in admin_run["error"]
    assert public_error("Gemini 3.8 Flash: HTTP 400: API key not valid. In use: AQ.-...0N9w (from .env)").startswith("The service is busy")
    assert public_error("Upload a resume first (Profile page).") == "Upload a resume first (Profile page)."
    client.put("/api/settings", json={"daily_cost_cap_usd": 0})
    router = client.app.state.router  # a Settings model whose key is missing (there are no keys in tests)
    monkeypatch.setattr(router, "default_model", lambda task, automatic=False: "gemini-3.8-flash")
    refused = tailor(user_client)
    assert refused.status_code == 422 and refused.json()["detail"].startswith("The service is busy")
    assert "GEMINI" in tailor(client, model=None).json()["detail"]  # the admin sees which key is missing


# ------------------------------------------------------------------ download history
def test_download_buttons_are_recorded_and_previews_are_not(client, user_client):
    save_profile(user_client)
    run_id = tailor(user_client).json()["run_id"]
    user_client.post(f"/api/runs/{run_id}/cover-letter", json={})
    assert user_client.get(f"/api/runs/{run_id}/resume.pdf").status_code == 200  # the preview
    resume = user_client.get(f"/api/runs/{run_id}/resume.pdf?download=1")
    cover = user_client.get(f"/api/runs/{run_id}/cover.pdf?download=1")
    assert resume.headers["cache-control"] == "private, no-store" and cover.status_code == 200
    client.get(f"/api/runs/{run_id}/resume.pdf")  # the admin's View link from History
    data = client.get("/api/downloads").json()
    rows = data["downloads"]
    assert [(r["email"], r["kind"], r["format"]) for r in rows] == [
        ("user@example.com", "cover_letter", "pdf"), ("user@example.com", "resume", "pdf")]
    assert rows[1]["candidate"] == "Jordan Rivera" and rows[1]["role"] and rows[1]["model"] and rows[1]["run_kept"]
    assert rows[1]["filename"].endswith(".pdf")
    assert data["accounts"] == [{"user_id": user_id_of(client, "user@example.com"), "email": "user@example.com", "downloads": 2}]


def test_history_filters_by_account_and_outlives_its_runs(client, user_client):
    save_profile(user_client)
    save_profile(client)
    user_run = tailor(user_client).json()["run_id"]
    admin_run = tailor(client).json()["run_id"]
    user_client.get(f"/api/runs/{user_run}/resume.pdf?download=1")
    client.get(f"/api/runs/{admin_run}/resume.pdf?download=1")  # the admin's own downloads count too
    everyone = client.get("/api/downloads").json()["downloads"]
    assert {r["email"] for r in everyone} == {"user@example.com", ADMIN_EMAIL}
    only_user = client.get(f"/api/downloads?user_id={user_id_of(client, 'user@example.com')}").json()["downloads"]
    assert [r["email"] for r in only_user] == ["user@example.com"]
    client.app.state.db.delete_user(user_id_of(client, "user@example.com"))  # the run goes, the history row stays
    kept = client.get("/api/downloads").json()["downloads"]
    assert [r["run_kept"] for r in kept if r["email"] == "user@example.com"] == [0]


def test_delete_all_data_keeps_accounts(client, user_client):
    save_profile(user_client)
    run_id = tailor(user_client).json()["run_id"]
    user_client.get(f"/api/runs/{run_id}/resume.pdf?download=1")
    assert client.delete("/api/data").status_code == 200
    assert client.get("/api/downloads").json()["downloads"] == []
    assert user_client.get("/api/resumes").json()["resumes"] == []  # still logged in, data gone
    assert len(client.get("/api/users").json()["users"]) == 2


def test_pages_render_for_each_role(client, user_client):
    for path in ("/profile", "/tailor", "/history", "/users", "/settings", "/account"):
        assert client.get(path).status_code == 200, path
    for path in ("/profile", "/tailor", "/account"):
        page = user_client.get(path)
        assert page.status_code == 200 and "user@example.com" in page.text, path
    assert client.get("/login", follow_redirects=False).status_code == 303  # already logged in


class BrokenModel(LLMClient):
    """A model that answers with nothing usable, naming itself in the error the way real adapters do."""

    async def generate_structured(self, **kwargs):
        raise InvalidOutputError("Demo Concise returned an empty answer")


def test_a_failing_model_is_never_named_to_users(client, user_client):
    save_profile(user_client)
    ok_run = tailor(user_client).json()["run_id"]  # made before the model breaks
    client.app.state.router.overrides["demo-concise"] = BrokenModel()
    run = tailor(user_client).json()
    assert run["status"] == "error" and run["error"] == BUSY_MESSAGE
    assert user_client.get(f"/api/runs/{run['run_id']}").json()["error"] == BUSY_MESSAGE
    letter = user_client.post(f"/api/runs/{ok_run}/cover-letter", json={})
    assert letter.status_code == 502 and letter.json()["detail"] == BUSY_MESSAGE
    assert "demo-concise" in client.get(f"/api/runs/{run['run_id']}").json()["error"]  # the admin sees which one
    assert public_error("Gemini 3.8 Flash returned text that is not valid JSON") == BUSY_MESSAGE


def test_a_later_sign_up_replaces_a_pending_one_with_the_same_email(client):
    sign_up(client, email="real@example.com", password="someone else's guess")  # signed up first by a stranger
    sign_up(client, email="real@example.com", password=NEW_PASSWORD)  # the real person
    assert client.app.state.db.count_users("pending") == 1
    client.post(f"/api/users/{user_id_of(client, 'real@example.com')}/approve")
    person = browser(client)
    assert person.post("/api/login", json={"email": "real@example.com", "password": "someone else's guess"}).status_code == 401
    assert person.post("/api/login", json={"email": "real@example.com", "password": NEW_PASSWORD}).status_code == 200
    sign_up(client, email="real@example.com", password="third try password")  # an approved account never changes
    assert browser(client).post("/api/login", json={"email": "real@example.com", "password": NEW_PASSWORD}).status_code == 200


def test_change_password_has_the_login_limit(client, user_client):
    for _ in range(auth.MAX_FAILURES):
        user_client.post("/api/account/password", json={"current": "a wrong guess!", "new": NEW_PASSWORD})
    paused = user_client.post("/api/account/password", json={"current": USER_PASSWORD, "new": NEW_PASSWORD})
    assert paused.status_code == 429
    assert browser(client).post("/api/login", json={"email": "user@example.com", "password": NEW_PASSWORD}).status_code == 401


def test_upload_errors_keep_the_file_name_even_with_technical_words(user_client):
    response = user_client.post("/api/resumes", files={"file": ("Regional_Provider_Token_Resume.txt", b"Jordan Rivera\nAustin, TX | jordan@example.com\n\nSKILLS\nPython, SQL, Airflow and dbt for data teams\n")})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail.startswith("Could not find") and "Regional_Provider_Token_Resume.txt" in detail and "Settings" not in detail
