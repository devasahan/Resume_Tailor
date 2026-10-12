"""Password hashing, tokens and the admin account from .env."""
import sqlite3

from app import auth
from app.storage.db import Database

from .conftest import ADMIN_EMAIL, ADMIN_PASSWORD


def test_password_hash_round_trip_and_no_plain_text():
    encoded = auth.hash_password("correct horse battery")
    assert encoded.startswith("scrypt$") and "correct horse" not in encoded
    assert auth.verify_password("correct horse battery", encoded)
    assert not auth.verify_password("correct horse batterY", encoded)
    assert auth.hash_password("correct horse battery") != encoded  # a fresh salt every time


def test_malformed_hashes_never_verify():
    for bad in ("", "plain", "md5$x$y", "scrypt$16$8$1$!!$!!", "scrypt$99999999$8$1$AA==$AA==", "scrypt$a$b$c$d$e"):
        assert not auth.verify_password("anything", bad)


def test_tokens_are_random_and_only_their_hash_is_stored():
    (raw1, hash1), (raw2, _hash2) = auth.new_token(), auth.new_token()
    assert raw1 != raw2 and len(raw1) >= 40 and hash1 == auth.token_hash(raw1) and raw1 not in hash1


def test_email_and_password_rules():
    assert auth.normalize_email("  Jo@Example.COM ") == "jo@example.com"
    assert auth.normalize_email("not-an-email") is None and auth.normalize_email("a@b") is None
    assert auth.password_problem("short") and auth.password_problem("x" * 257)
    assert auth.password_problem("ten chars!") is None


def test_secure_cookie_follows_https_or_the_proxy(monkeypatch):
    assert auth.cookie_secure({}, "https") and not auth.cookie_secure({}, "http")
    assert auth.cookie_secure({"x-forwarded-proto": "https"}, "http")  # PythonAnywhere's proxy
    monkeypatch.setenv("COOKIE_SECURE", "1")
    assert auth.cookie_secure({}, "http")
    monkeypatch.setenv("COOKIE_SECURE", "0")
    assert not auth.cookie_secure({"x-forwarded-proto": "https"}, "https")


def test_bootstrap_creates_the_admin_and_gives_it_old_data(tmp_path, profile):
    db = Database(tmp_path / "a.db")
    old_profile = db.save_profile(profile)  # saved before accounts existed: no owner
    admin_id = auth.bootstrap_admin(db)
    row = db.user_by_email(ADMIN_EMAIL)
    assert row["id"] == admin_id and row["role"] == "admin" and row["status"] == "active"
    assert auth.verify_password(ADMIN_PASSWORD, row["password_hash"])
    assert [e["id"] for e in db.list_profiles(user_id=admin_id)] == [old_profile]
    with sqlite3.connect(db.path) as raw:  # the password itself is nowhere in the database file
        dump = "\n".join(raw.iterdump())
    assert ADMIN_PASSWORD not in dump


def test_a_new_admin_password_replaces_the_old_and_ends_sessions(tmp_path, monkeypatch):
    db = Database(tmp_path / "a.db")
    admin_id = auth.bootstrap_admin(db)
    raw = auth.start_session(db, admin_id)
    assert auth.session_user(db, raw) is not None
    assert auth.bootstrap_admin(db) == admin_id and auth.session_user(db, raw) is not None  # same password: nothing ends
    monkeypatch.setenv("ADMIN_PASSWORD", "a brand new password")
    auth.bootstrap_admin(db)
    assert auth.session_user(db, raw) is None
    assert auth.login(db, ADMIN_EMAIL, ADMIN_PASSWORD)[0] is None
    assert auth.login(db, ADMIN_EMAIL, "a brand new password")[0].is_admin


def test_only_the_env_account_is_admin(tmp_path, monkeypatch):
    db = Database(tmp_path / "a.db")
    first = auth.bootstrap_admin(db)
    monkeypatch.setenv("ADMIN_EMAIL", "new-admin@example.com")
    second = auth.bootstrap_admin(db)
    assert db.get_user(second)["role"] == "admin" and db.get_user(first)["role"] == "user"


def test_a_sign_up_with_the_admin_email_cannot_take_over(tmp_path, monkeypatch):
    """Someone signed up early with ADMIN_EMAIL (or the admin row was made without a password): .env still wins."""
    db = Database(tmp_path / "a.db")
    db.create_user(ADMIN_EMAIL, auth.hash_password("attacker password"), role="user", status="pending")
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    assert auth.bootstrap_admin(db) is None  # no .env password: the sign-up is not promoted
    assert db.user_by_email(ADMIN_EMAIL)["role"] == "user"
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN_PASSWORD)
    auth.bootstrap_admin(db)
    assert auth.login(db, ADMIN_EMAIL, "attacker password")[0] is None
    assert auth.login(db, ADMIN_EMAIL, ADMIN_PASSWORD)[0].is_admin


def test_no_admin_email_means_no_admin(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_EMAIL", "")
    assert auth.bootstrap_admin(Database(tmp_path / "a.db")) is None


def test_lockout_after_five_failures_and_restart_clears_the_admin(tmp_path):
    db = Database(tmp_path / "a.db")
    auth.bootstrap_admin(db)
    for _ in range(auth.MAX_FAILURES):
        assert auth.login(db, ADMIN_EMAIL, "wrong password!")[1] == auth.BAD_LOGIN
    # even the right password waits, and the answer is the one an unknown email gets
    assert auth.login(db, ADMIN_EMAIL, ADMIN_PASSWORD) == (None, auth.BAD_LOGIN)
    assert auth.login(db, "nobody@example.com", ADMIN_PASSWORD) == (None, auth.BAD_LOGIN)
    auth.bootstrap_admin(db)  # restarting the app (PythonAnywhere: Reload) unlocks the admin
    assert auth.login(db, ADMIN_EMAIL, ADMIN_PASSWORD)[0] is not None


def test_a_pause_that_ran_out_starts_the_count_again(tmp_path):
    """One wrong guess after a pause must not pause the account again: that would keep someone out forever."""
    db = Database(tmp_path / "a.db")
    user_id = db.create_user("u@example.com", auth.hash_password("right password"), status="active")
    for _ in range(auth.MAX_FAILURES):
        auth.login(db, "u@example.com", "wrong password!")
    assert not auth.claim_attempt(db, user_id)  # paused
    with sqlite3.connect(db.path) as raw:
        raw.execute("UPDATE users SET locked_until = '2020-01-01T00:00:00Z'")  # the pause ran out
    assert auth.login(db, "u@example.com", "wrong password!")[1] == auth.BAD_LOGIN
    row = db.get_user(user_id)
    assert row["failed_logins"] == 1 and row["locked_until"] == ""
    assert auth.login(db, "u@example.com", "right password")[0] is not None


def test_only_five_tries_per_pause_even_all_at_once(tmp_path):
    """The try is counted before the password is checked, in one write, so a burst gets no extra guesses."""
    from concurrent.futures import ThreadPoolExecutor

    db = Database(tmp_path / "a.db")
    user_id = db.create_user("u@example.com", auth.hash_password("right password"), status="active")
    with ThreadPoolExecutor(max_workers=16) as pool:
        allowed = list(pool.map(lambda _: auth.claim_attempt(db, user_id), range(40)))
    assert sum(allowed) == auth.MAX_FAILURES


def test_pending_and_disabled_are_named_only_after_the_right_password(tmp_path):
    db = Database(tmp_path / "a.db")
    user_id = db.create_user("p@example.com", auth.hash_password("pending password"), status="pending")
    assert auth.login(db, "p@example.com", "wrong password!") == (None, auth.BAD_LOGIN)
    assert auth.login(db, "p@example.com", "pending password") == (None, auth.PENDING)
    assert db.set_user_status(user_id, "active", only_from="pending")
    assert auth.login(db, "P@Example.com ", "pending password")[0].email == "p@example.com"
    assert db.set_user_status(user_id, "disabled", only_from="active")
    assert auth.login(db, "p@example.com", "pending password") == (None, auth.DISABLED)
    assert auth.login(db, "nobody@example.com", "pending password") == (None, auth.BAD_LOGIN)
