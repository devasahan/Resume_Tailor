"""SQLite storage: settings, accounts and sessions, the uploaded base resumes (profiles), one row per run,
and the download history the admin sees.

Privacy: uploaded resumes stay until their owner deletes them. Runs (including the job description they
were made for) and download history are deleted after the `retention_days` setting. Full resume text is
never written to logs. Every change is a single statement, so several web workers can share the file.
"""
from __future__ import annotations

import json
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import config
from ..schemas import Profile, TailorResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    method TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '[]',
    hidden INTEGER NOT NULL DEFAULT 0,
    user_id INTEGER
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    profile_id INTEGER,
    user_id INTEGER,
    jd_hash TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT 'resume',
    requested_model TEXT NOT NULL,
    served_model TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    result TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    approved_at TEXT NOT NULL DEFAULT '',
    last_login_at TEXT NOT NULL DEFAULT '',
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS downloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    user_id INTEGER,
    email TEXT NOT NULL DEFAULT '',
    run_id INTEGER,
    kind TEXT NOT NULL,
    format TEXT NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    candidate TEXT NOT NULL DEFAULT '',
    company TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    cost_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS downloads_created ON downloads(created_at);
CREATE INDEX IF NOT EXISTS downloads_user ON downloads(user_id);
"""


# Columns added after the first release, in the order they are added (profiles.hidden after its siblings).
_PROFILE_COLUMNS = [
    ("filename", "TEXT NOT NULL DEFAULT ''"),
    ("method", "TEXT NOT NULL DEFAULT ''"),
    ("notes", "TEXT NOT NULL DEFAULT '[]'"),
    ("hidden", "INTEGER NOT NULL DEFAULT 0"),
    ("user_id", "INTEGER"),
]
_RUN_COLUMNS = [("user_id", "INTEGER")]


def _add_column(conn: sqlite3.Connection, table: str, name: str, ddl: str) -> bool:
    """Add a column unless it is there; True only when this call added it."""
    if name in {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}:
        return False
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
    except sqlite3.OperationalError as exc:  # another process added it first
        if "duplicate column" not in str(exc).lower():
            raise
        return False
    return True


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database up to date. Safe to run on every start and from two processes at once.

    Older versions saved a new profile row on every "Save profile", so a database can hold many copies of
    one person. When the resume list arrives, only the newest of those stays visible; the others are
    hidden, not deleted (runs keep their own copy of the profile, so nothing depends on them).
    Resumes and runs saved before accounts existed have no owner; the admin account receives them at start.
    """
    for name, ddl in _PROFILE_COLUMNS:
        if _add_column(conn, "profiles", name, ddl) and name == "hidden":
            conn.execute("UPDATE profiles SET hidden = 1 WHERE id <> (SELECT MAX(id) FROM profiles)")
    for name, ddl in _RUN_COLUMNS:
        _add_column(conn, "runs", name, ddl)
    # only now that the columns exist on an older database too
    conn.execute("CREATE INDEX IF NOT EXISTS profiles_user ON profiles(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS runs_user ON runs(user_id)")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    def __init__(self, path: Path | None = None):
        self.path = path or (config.data_dir() / "resume_tailor.db")
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            _migrate(conn)
        try:
            os.chmod(self.path, 0o600)  # profiles and resumes are personal data
        except OSError:
            pass

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ settings
    def get_settings(self) -> dict:
        merged = dict(config.DEFAULT_SETTINGS)
        with self._conn() as conn:
            for row in conn.execute("SELECT key, value FROM settings"):
                if row["key"] in merged:
                    merged[row["key"]] = json.loads(row["value"])
        return merged

    def set_settings(self, values: dict) -> dict:
        unknown = set(values) - set(config.DEFAULT_SETTINGS)
        if unknown:
            raise KeyError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
        with self._conn() as conn:
            for key, value in values.items():
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, json.dumps(value)),
                )
        return self.get_settings()

    # ------------------------------------------------------------------ profiles (uploaded base resumes)
    # Reads and deletes always name the owner (user_id, required): a resume is only ever seen by whoever
    # uploaded it. `user_id IS ?` also matches NULL, the owner of rows saved before accounts existed.
    def save_profile(
        self, profile: Profile, *, user_id: int | None = None, filename: str = "", method: str = "",
        notes: list[str] | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO profiles(created_at, data, filename, method, notes, user_id) VALUES(?, ?, ?, ?, ?, ?)",
                (now_iso(), profile.model_dump_json(), filename, method, json.dumps(notes or []), user_id),
            )
            return int(cur.lastrowid)

    def latest_profile(self, *, user_id: int | None) -> tuple[int, Profile] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, data FROM profiles WHERE hidden = 0 AND user_id IS ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return (row["id"], Profile.model_validate_json(row["data"])) if row else None

    def get_profile(self, profile_id: int, *, user_id: int | None) -> Profile | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM profiles WHERE id = ? AND hidden = 0 AND user_id IS ?", (profile_id, user_id)
            ).fetchone()
        return Profile.model_validate_json(row["data"]) if row else None

    @staticmethod
    def _entry(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "created_at": row["created_at"], "filename": row["filename"], "method": row["method"],
            "notes": json.loads(row["notes"] or "[]"), "profile": Profile.model_validate_json(row["data"]),
        }

    def list_profiles(self, *, user_id: int | None) -> list[dict]:
        """One owner's visible uploaded resumes, newest first: id, created_at, filename, method, notes, profile."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, data, filename, method, notes FROM profiles"
                " WHERE hidden = 0 AND user_id IS ? ORDER BY id DESC",
                (user_id,),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def profile_entry(self, profile_id: int, *, user_id: int | None) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, created_at, data, filename, method, notes FROM profiles"
                " WHERE id = ? AND hidden = 0 AND user_id IS ?",
                (profile_id, user_id),
            ).fetchone()
        return self._entry(row) if row else None

    def delete_profile(self, profile_id: int, *, user_id: int | None) -> bool:
        """Remove an uploaded resume. Runs made from it keep working: each run stores its own copy."""
        with self._conn() as conn:
            return conn.execute(
                "DELETE FROM profiles WHERE id = ? AND hidden = 0 AND user_id IS ?", (profile_id, user_id)
            ).rowcount > 0

    # ------------------------------------------------------------------ runs
    def save_run(
        self, result: TailorResult, profile_id: int | None, jd_hash: str, task: str = "resume", *,
        user_id: int | None = None,
    ) -> int:
        result.created_at = result.created_at or now_iso()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO runs(created_at, profile_id, user_id, jd_hash, task, requested_model, served_model,"
                " status, cost_usd, input_tokens, output_tokens, latency_ms, result)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.created_at, profile_id, user_id, jd_hash, task, result.requested_model,
                    result.served_model, result.status, result.cost_usd, result.input_tokens, result.output_tokens,
                    result.latency_ms, "{}",
                ),
            )
            run_id = int(cur.lastrowid)
            result.run_id = run_id
            conn.execute("UPDATE runs SET result = ? WHERE id = ?", (result.model_dump_json(), run_id))
        return run_id

    def update_run(self, result: TailorResult) -> None:
        """Rewrite a run's stored result (after an edit or a cover letter). Its cost columns never change:
        a cover letter written later is recorded as its own row, dated the day it was written."""
        assert result.run_id is not None
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET result = ?, status = ? WHERE id = ?",
                (result.model_dump_json(), result.status, result.run_id),
            )

    def get_run(self, run_id: int) -> TailorResult | None:
        with self._conn() as conn:
            row = conn.execute("SELECT result FROM runs WHERE id = ?", (run_id,)).fetchone()
        return TailorResult.model_validate_json(row["result"]) if row else None

    def get_run_owned(self, run_id: int) -> tuple[int | None, TailorResult] | None:
        """(owner's user id, result), or None."""
        with self._conn() as conn:
            row = conn.execute("SELECT user_id, result FROM runs WHERE id = ?", (run_id,)).fetchone()
        return (row["user_id"], TailorResult.model_validate_json(row["result"])) if row else None

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT r.id, r.created_at, r.task, r.requested_model, r.served_model, r.status, r.cost_usd,"
                " r.input_tokens, r.output_tokens, r.latency_ms, r.result FROM runs r ORDER BY r.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            item = {k: row[k] for k in row.keys() if k != "result"}
            result = json.loads(row["result"])
            analysis = result.get("analysis") or {}
            item["target_role"] = analysis.get("target_role", "")
            item["fits"] = result.get("fits", False)
            item["pages"] = result.get("pages", 0)
            item["error"] = result.get("error", "")
            item["flag_count"] = len(result.get("flags", []))
            item["candidate"] = ((result.get("resume") or {}).get("profile") or {}).get("name", "")
            item["cover_letter"] = bool(result.get("cover_letter"))
            out.append(item)
        return out

    def cost_today(self) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM runs WHERE substr(created_at, 1, 10) = ?",
                (today,),
            ).fetchone()
        return float(row["total"])

    # ------------------------------------------------------------------ accounts
    _USER_COLUMNS = "id, email, password_hash, role, status, created_at, approved_at, last_login_at, failed_logins, locked_until"

    def create_user(self, email: str, password_hash: str, *, role: str = "user", status: str = "pending") -> int | None:
        """The new user's id, or None when the email already has an account."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO users(email, password_hash, role, status, created_at, approved_at)"
                " VALUES(?, ?, ?, ?, ?, ?) ON CONFLICT(email) DO NOTHING",
                (email, password_hash, role, status, now_iso(), now_iso() if status == "active" else ""),
            )
            return int(cur.lastrowid) if cur.rowcount else None

    def user_by_email(self, email: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(f"SELECT {self._USER_COLUMNS} FROM users WHERE email = ?", (email,)).fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(f"SELECT {self._USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        """Every account for the Users page (never the password hash), pending first, then newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT u.id, u.email, u.role, u.status, u.created_at, u.approved_at, u.last_login_at,"
                " u.locked_until,"
                " (SELECT COUNT(*) FROM profiles p WHERE p.user_id = u.id AND p.hidden = 0) AS resumes,"
                " (SELECT COUNT(*) FROM downloads d WHERE d.user_id = u.id) AS downloads"
                " FROM users u ORDER BY (u.status = 'pending') DESC, u.id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def count_users(self, status: str) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users WHERE status = ?", (status,)).fetchone()[0])

    def set_user_status(self, user_id: int, status: str, *, only_from: str | None = None) -> bool:
        """Change a normal user's status (never an admin's). Leaving 'active' ends their sessions at once."""
        with self._conn() as conn:
            changed = conn.execute(
                "UPDATE users SET status = ?, failed_logins = 0, locked_until = '',"
                " approved_at = CASE WHEN ? = 'active' AND approved_at = '' THEN ? ELSE approved_at END"
                " WHERE id = ? AND role = 'user' AND (? IS NULL OR status = ?)",
                (status, status, now_iso(), user_id, only_from, only_from),
            ).rowcount > 0
            if changed and status != "active":
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return changed

    def set_password_hash(self, user_id: int, password_hash: str) -> None:
        """New password: old sessions end and any lockout is lifted."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, failed_logins = 0, locked_until = '' WHERE id = ?",
                (password_hash, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def delete_user(self, user_id: int) -> bool:
        """Remove a normal user with their sessions, resumes and runs. Download history keeps its own copy."""
        with self._conn() as conn:
            if not conn.execute("DELETE FROM users WHERE id = ? AND role = 'user'", (user_id,)).rowcount:
                return False
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM runs WHERE user_id = ?", (user_id,))
        return True

    def record_login_failure(self, user_id: int, max_failures: int, lock_until: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET failed_logins = failed_logins + 1,"
                " locked_until = CASE WHEN failed_logins + 1 >= ? THEN ? ELSE locked_until END WHERE id = ?",
                (max_failures, lock_until, user_id),
            )

    def record_login_success(self, user_id: int, when: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET failed_logins = 0, locked_until = '', last_login_at = ? WHERE id = ?",
                (when, user_id),
            )

    def make_admin(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET role = 'admin', status = 'active', failed_logins = 0, locked_until = '',"
                " approved_at = CASE WHEN approved_at = '' THEN ? ELSE approved_at END WHERE id = ?",
                (now_iso(), user_id),
            )

    def demote_other_admins(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE users SET role = 'user' WHERE role = 'admin' AND id <> ?", (user_id,))

    def assign_unowned(self, user_id: int) -> None:
        """Resumes and runs saved before accounts existed belong to the admin."""
        with self._conn() as conn:
            conn.execute("UPDATE profiles SET user_id = ? WHERE user_id IS NULL", (user_id,))
            conn.execute("UPDATE runs SET user_id = ? WHERE user_id IS NULL", (user_id,))

    # ------------------------------------------------------------------ sessions
    def create_session(self, token_hash: str, user_id: int, expires_at: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES(?, ?, ?, ?)",
                (token_hash, user_id, now_iso(), expires_at),
            )

    def session_user(self, token_hash: str, now: str) -> dict | None:
        """The active account behind a session that has not expired."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT u.id, u.email, u.role, u.status FROM sessions s JOIN users u ON u.id = s.user_id"
                " WHERE s.token_hash = ? AND s.expires_at > ? AND u.status = 'active'",
                (token_hash, now),
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    # ------------------------------------------------------------------ download history
    def record_download(
        self, *, user_id: int, email: str, run_id: int, kind: str, format: str, filename: str, candidate: str,
        company: str, role: str, model: str, cost_usd: float,
    ) -> int:
        """One row per click on a Download button. It copies what History shows, so the row stays readable
        after its run passes the retention period."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO downloads(created_at, user_id, email, run_id, kind, format, filename, candidate,"
                " company, role, model, cost_usd) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (now_iso(), user_id, email, run_id, kind, format, filename, candidate, company, role, model,
                 round(cost_usd, 6)),
            )
            return int(cur.lastrowid)

    def list_downloads(self, *, user_id: int | None = None, limit: int = 200) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, user_id, email, run_id, kind, format, filename, candidate, company, role,"
                " model, cost_usd, EXISTS(SELECT 1 FROM runs r WHERE r.id = downloads.run_id) AS run_kept"
                " FROM downloads WHERE (? IS NULL OR user_id = ?) ORDER BY id DESC LIMIT ?",
                (user_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def download_accounts(self) -> list[dict]:
        """Accounts that appear in the download history, for the History filter."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT user_id, email, COUNT(*) AS downloads FROM downloads GROUP BY user_id, email ORDER BY email"
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ retention
    def purge_all(self) -> None:
        """Every resume, run and download for everyone. Accounts, sessions and settings stay."""
        with self._conn() as conn:
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM profiles")
            conn.execute("DELETE FROM downloads")

    def purge(self, retention_days: int) -> int:
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now_iso(),))
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._conn() as conn:
            runs = conn.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,)).rowcount
            # uploaded resumes stay until the user deletes them; only hidden old copies expire
            profiles = conn.execute(
                "DELETE FROM profiles WHERE hidden = 1 AND created_at < ?", (cutoff,)
            ).rowcount
            conn.execute("DELETE FROM downloads WHERE created_at < ?", (cutoff,))
        return runs + profiles
