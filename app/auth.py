"""Accounts: password hashing, session tokens, the admin account from .env, and the login check.

Standard library only. Passwords are stored as scrypt hashes. The session cookie holds a random token and the
database keeps only its sha256, so a copy of the database is not enough to log in as anyone.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import config

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1  # tests lower SCRYPT_N to stay fast
_MAXMEM = 64 * 1024 * 1024

COOKIE = "rt_session"
SESSION_DAYS = 30
MIN_PASSWORD, MAX_PASSWORD = 10, 256
MAX_FAILURES, LOCK_MINUTES = 5, 15
MAX_PENDING = 50  # sign-ups waiting for approval; more are refused until the admin catches up

# One answer for an unknown email, a wrong password and a paused account, so a stranger cannot tell which
# emails have accounts. It still tells a real user why logging in may not work for a while.
BAD_LOGIN = (f"Email or password is incorrect. After {MAX_FAILURES} wrong tries in a row, that email cannot log in "
             f"for {LOCK_MINUTES} minutes.")
LOCKED = f"Too many wrong tries. Try again in {LOCK_MINUTES} minutes."
PENDING = "Your account is waiting for the admin's approval."
DISABLED = "This account is disabled. Contact the admin."

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class User:
    id: int
    email: str
    role: str
    status: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @classmethod
    def from_row(cls, row: dict) -> User:
        return cls(id=row["id"], email=row["email"], role=row["role"], status=row["status"])


def utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------ passwords
def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


# Each scrypt call needs 16 MiB. Running them all on two threads caps that memory (and the CPU) however many
# login or sign-up requests arrive at once; the memory stays with these two threads. Threads do not survive a
# fork (a WSGI server that loads the app and then forks its workers), so each process makes its own pool.
_hasher: ThreadPoolExecutor | None = None
_hasher_pid = 0
_hasher_lock = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _hasher, _hasher_pid
    if _hasher is None or _hasher_pid != os.getpid():
        with _hasher_lock:
            if _hasher is None or _hasher_pid != os.getpid():
                _hasher, _hasher_pid = ThreadPoolExecutor(max_workers=2, thread_name_prefix="scrypt"), os.getpid()
    return _hasher


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return _pool().submit(hashlib.scrypt, password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM,
                          dklen=dklen).result()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, 32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """False for a wrong password and for anything that is not a hash this module wrote."""
    try:
        scheme, n, r, p, salt, digest = encoded.split("$")
        n, r, p = int(n), int(r), int(p)
        if scheme != "scrypt" or not (1 < n <= 2**20 and 0 < r <= 32 and 0 < p <= 16):
            return False
        expected = base64.b64decode(digest, validate=True)
        actual = _scrypt(password, base64.b64decode(salt, validate=True), n, r, p, len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


_dummy: dict[int, str] = {}


def _waste_time(password: str) -> None:
    """Check a password against a throwaway hash, so an unknown email takes as long as a known one."""
    if SCRYPT_N not in _dummy:
        _dummy[SCRYPT_N] = hash_password(secrets.token_hex(8))
    verify_password(password, _dummy[SCRYPT_N])


def password_problem(password: str) -> str | None:
    if len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters for the password."
    if len(password) > MAX_PASSWORD:
        return f"Use at most {MAX_PASSWORD} characters for the password."
    return None


def normalize_email(text: str) -> str | None:
    email = (text or "").strip().lower()
    return email if len(email) <= 254 and _EMAIL.match(email) else None


def temporary_password() -> str:
    return secrets.token_urlsafe(9)  # 12 characters, shown once to the admin


# ------------------------------------------------------------------ sessions
def token_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_token() -> tuple[str, str]:
    """(raw token for the cookie, its hash for the database)."""
    raw = secrets.token_urlsafe(32)
    return raw, token_hash(raw)


def start_session(db, user_id: int) -> str:
    raw, hashed = new_token()
    db.create_session(hashed, user_id, utc(now_utc() + timedelta(days=SESSION_DAYS)))
    return raw


def session_user(db, raw: str | None) -> User | None:
    if not raw or len(raw) > 200:
        return None
    row = db.session_user(token_hash(raw), utc(now_utc()))
    return User.from_row(row) if row else None


def end_session(db, raw: str | None) -> None:
    if raw and len(raw) <= 200:
        db.delete_session(token_hash(raw))


def cookie_secure(headers, scheme: str) -> bool:
    """COOKIE_SECURE=1/0 decides; unset means: HTTPS, directly or behind a proxy such as PythonAnywhere's."""
    forced = config.cookie_secure()
    if forced is not None:
        return forced
    forwarded = (headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return scheme == "https" or forwarded == "https"


def cookie_options(headers, scheme: str) -> dict:
    return {"httponly": True, "samesite": "lax", "path": "/", "max_age": SESSION_DAYS * 86400,
            "secure": cookie_secure(headers, scheme)}


# ------------------------------------------------------------------ login and the admin account
def claim_attempt(db, user_id: int) -> bool:
    """Count one password try before checking it. False while the account is paused. The count and the pause
    are one database write, so tries sent all at once (or to several web workers) still get only
    MAX_FAILURES checks per pause, and a pause that has run out starts the count again."""
    now = now_utc()
    return db.claim_login_attempt(user_id, MAX_FAILURES, utc(now), utc(now + timedelta(minutes=LOCK_MINUTES)))


def login(db, email: str, password: str) -> tuple[User | None, str | None]:
    """(user, None) on success, else (None, the message to show). Unknown emails, wrong passwords and paused
    accounts get the same answer in about the same time. Pending and disabled accounts are only named after
    the right password."""
    address = normalize_email(email)
    row = db.user_by_email(address) if address else None
    if row is None or not claim_attempt(db, row["id"]):
        _waste_time(password)
        return None, BAD_LOGIN
    if not verify_password(password, row["password_hash"]):
        return None, BAD_LOGIN
    if row["status"] == "pending":
        db.clear_login_failures(row["id"])
        return None, PENDING
    if row["status"] != "active":
        db.clear_login_failures(row["id"])
        return None, DISABLED
    db.record_login_success(row["id"], utc(now_utc()))
    return User.from_row(row), None


def bootstrap_admin(db) -> int | None:
    """Make the ADMIN_EMAIL account the one active admin, every start. .env is the source of truth: a new
    ADMIN_PASSWORD replaces the stored one (and ends old sessions), a lockout is cleared, any other admin
    becomes a normal user, and resumes and runs saved before accounts existed are given to the admin."""
    email, password = config.admin_credentials()
    address = normalize_email(email)
    if not address:
        return None
    row = db.user_by_email(address)
    if row is None:
        if not password:
            return None
        db.create_user(address, hash_password(password), role="admin", status="active")
        row = db.user_by_email(address)  # another worker may have created it first: either way it exists now
        if row is None:
            return None
    elif not password:
        if row["role"] != "admin":  # promoting an account needs the .env password, so a sign-up never becomes admin
            return None
    elif not verify_password(password, row["password_hash"]):
        db.set_password_hash(row["id"], hash_password(password))
    db.make_admin(row["id"])
    db.demote_other_admins(row["id"])
    db.assign_unowned(row["id"])
    return row["id"]
