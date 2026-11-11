"""SQLite storage: settings, accounts and sessions, the uploaded base resumes (profiles), one row per run,
the download history the admin sees, and the slow jobs (tailoring, cover letters) a page is waiting for.

Privacy: uploaded resumes stay until their owner deletes them. Runs (including the job description they
were made for) and download history are deleted after the `retention_days` setting. Full resume text is
never written to logs. Every change is a single statement, so several web workers can share the file.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import os
import sys
from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone, tzinfo
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .. import config
from ..schemas import Profile, TailorResult
from ..timing import DATABASE

log = logging.getLogger(__name__)  # under "app": its INFO lines reach the server log (main.show_app_log)
SLOW_DB_SECONDS = 0.5  # a database step this slow is named in the server log (network storage, a lock held elsewhere)

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
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    run_id INTEGER,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    finished_at TEXT NOT NULL DEFAULT '',
    beat_at TEXT NOT NULL DEFAULT ''
);
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
_JOB_COLUMNS = [("beat_at", "TEXT NOT NULL DEFAULT ''")]


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
    for name, ddl in _JOB_COLUMNS:
        _add_column(conn, "jobs", name, ddl)
    # only now that the columns exist on an older database too
    conn.execute("CREATE INDEX IF NOT EXISTS profiles_user ON profiles(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS runs_user ON runs(user_id)")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def local_zone() -> tzinfo:
    """The owner's time zone (config.LOCAL_ZONE). Without time-zone data (Windows without the tzdata package) the
    day is the UTC day, rather than the app failing."""
    try:
        return ZoneInfo(config.LOCAL_ZONE)
    except ZoneInfoNotFoundError:
        logging.getLogger(__name__).warning("No time-zone data for %s: days are counted in UTC", config.LOCAL_ZONE)
        return timezone.utc


def day_bounds(now: datetime) -> tuple[str, str]:
    """The local day around `now` as [start, end) in the stored UTC format. Built from the local date, so the
    23- and 25-hour daylight-saving days come out right."""
    zone = local_zone()
    day = now.astimezone(zone).date()
    start = datetime.combine(day, time(), zone).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time(), zone).astimezone(timezone.utc)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


# A stored run is TailorResult.model_dump_json() (compact): '"cover_letter":{' only when it has a letter,
# '"cover_letter":null' otherwise. Quotes inside its text are escaped, so a job post or a letter cannot fake it.
COVER_MARK = '"cover_letter":{'


def like_contains(text: str) -> str:
    """A LIKE pattern for `text` anywhere in a name ('acme' -> '%acme%'). Its own \\, % and _ are matched as
    themselves (\\ escaped first); '' matches every name. Use with ESCAPE '\\'."""
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


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
        """One connection per step. Every step is counted for the job lines (app/timing.py), and one that takes
        SLOW_DB_SECONDS or more is named in the server log, by the method that asked for it (the frame behind
        contextmanager's __enter__)."""
        began = perf_counter()
        try:
            caller = sys._getframe(2).f_code.co_name
        except ValueError:  # called from the top of the stack
            caller = "?"
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
            took = perf_counter() - began
            DATABASE.add(took)
            if took >= SLOW_DB_SECONDS:
                log.info("slow database step %s: %.1fs", caller, took)

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

    def delete_run(self, run_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

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

    def cost_today(self, now: datetime | None = None) -> float:
        """Spent since midnight in Arlington, Texas (config.LOCAL_ZONE, daylight saving included). created_at is
        always now_iso()'s UTC format, so a text range finds the day."""
        start, end = day_bounds(now or utc_now())
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM runs WHERE created_at >= ? AND created_at < ?",
                (start, end),
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
            conn.execute("DELETE FROM jobs WHERE user_id = ?", (user_id,))  # before runs: see finish_job
            conn.execute("DELETE FROM runs WHERE user_id = ?", (user_id,))
        return True

    def claim_login_attempt(self, user_id: int, max_failures: int, now: str, lock_until: str) -> bool:
        """Count one password try in a single write; False while the account is paused. SET reads the row as it
        was, so a pause that has run out (locked_until set but past) starts the count again at 1."""
        with self._conn() as conn:
            return conn.execute(
                "UPDATE users SET"
                " failed_logins = CASE WHEN locked_until <> '' THEN 1 ELSE failed_logins + 1 END,"
                " locked_until = CASE WHEN (CASE WHEN locked_until <> '' THEN 1 ELSE failed_logins + 1 END) >= ?"
                " THEN ? ELSE '' END"
                " WHERE id = ? AND (locked_until = '' OR locked_until <= ?)",
                (max_failures, lock_until, user_id, now),
            ).rowcount > 0

    def clear_login_failures(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE users SET failed_logins = 0, locked_until = '' WHERE id = ?", (user_id,))

    def renew_pending_password(self, email: str, password_hash: str) -> bool:
        """A second sign-up for an address still waiting for approval replaces the first one's password, so
        someone who signed up with another person's email first cannot keep that person out."""
        with self._conn() as conn:
            return conn.execute(
                "UPDATE users SET password_hash = ?, failed_logins = 0, locked_until = ''"
                " WHERE email = ? AND status = 'pending' AND role = 'user'",
                (password_hash, email),
            ).rowcount > 0

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

    # History's filters, the same in every query so rows, totals and count agree: one account (or all), and part of
    # the company and of the job title in any case (SQLite's LIKE folds ASCII letters only)
    _DOWNLOAD_FILTER = "(? IS NULL OR d.user_id = ?) AND d.company LIKE ? ESCAPE '\\' AND d.role LIKE ? ESCAPE '\\'"

    @staticmethod
    def _download_args(user_id: int | None, company: str, role: str) -> tuple:
        return user_id, user_id, like_contains(company), like_contains(role)

    def list_downloads(self, *, user_id: int | None = None, company: str = "", role: str = "",
                       limit: int = 200) -> list[dict]:
        """History's matching rows, newest first, with run_kept, has_cover (a text search in the stored result,
        never parsed here; a cover-letter download always has one, as a letter is only ever replaced) and
        account_status: the account's status now, or 'removed' once it is deleted."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT d.id, d.created_at, d.user_id, d.email, d.run_id, d.kind, d.format, d.filename, d.candidate,"
                " d.company, d.role, d.model, d.cost_usd, r.id IS NOT NULL AS run_kept,"
                " CASE WHEN r.id IS NULL THEN 0 WHEN d.kind = 'cover_letter' THEN 1"
                " ELSE instr(r.result, ?) > 0 END AS has_cover,"
                " COALESCE(u.status, 'removed') AS account_status"
                " FROM downloads d LEFT JOIN runs r ON r.id = d.run_id LEFT JOIN users u ON u.id = d.user_id"
                " WHERE " + self._DOWNLOAD_FILTER + " ORDER BY d.id DESC LIMIT ?",
                (COVER_MARK, *self._download_args(user_id, company, role), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def download_totals(self, *, user_id: int | None = None, company: str = "", role: str = "") -> dict[str, int]:
        """For the matching downloads: resumes and cover letters (each tailored job counted once, however often it
        was downloaded) and how many downloads match in all (History's "newest N of M")."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT d.kind, COUNT(DISTINCT d.run_id) AS jobs, COUNT(*) AS clicks FROM downloads d"
                " WHERE " + self._DOWNLOAD_FILTER + " GROUP BY d.kind",
                self._download_args(user_id, company, role),
            ).fetchall()
        return ({"resume": 0, "cover_letter": 0} | {row["kind"]: row["jobs"] for row in rows}
                | {"downloads": sum(row["clicks"] for row in rows)})

    def download_accounts(self) -> list[dict]:
        """Active accounts that appear in the download history, by their email now, for History's Account filter.
        Disabled and removed accounts are not offered; their rows stay in History, tagged."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT u.id AS user_id, u.email, COUNT(*) AS downloads FROM users u"
                " JOIN downloads d ON d.user_id = u.id WHERE u.status = 'active'"
                " GROUP BY u.id, u.email ORDER BY u.email"
            ).fetchall()
        return [dict(row) for row in rows]

    def download_names(self, limit: int = 500) -> dict[str, list[str]]:
        """The companies and job titles in the download history, A to Z, for the suggestions under History's boxes."""
        with self._conn() as conn:
            companies = conn.execute("SELECT DISTINCT company FROM downloads WHERE company <> ''"
                                     " ORDER BY company COLLATE NOCASE LIMIT ?", (limit,)).fetchall()
            roles = conn.execute("SELECT DISTINCT role FROM downloads WHERE role <> ''"
                                 " ORDER BY role COLLATE NOCASE LIMIT ?", (limit,)).fetchall()
        return {"companies": [row[0] for row in companies], "roles": [row[0] for row in roles]}

    # ------------------------------------------------------------------ background work (tailoring, cover letters)
    def create_job(self, job_id: str, user_id: int, kind: str) -> None:
        now = now_iso()
        with self._conn() as conn:
            conn.execute("INSERT INTO jobs(id, user_id, kind, created_at, beat_at) VALUES(?, ?, ?, ?, ?)",
                         (job_id, user_id, kind, now, now))

    def touch_job(self, job_id: str) -> None:
        """The job's process is alive and still working on it."""
        with self._conn() as conn:
            conn.execute("UPDATE jobs SET beat_at = ? WHERE id = ? AND status = 'running'", (now_iso(), job_id))

    def finish_job(self, job_id: str, *, run_id: int | None, error: str = "") -> bool:
        """False when the job is gone: its account or all data was deleted while it ran (jobs are deleted before
        runs, so a run the job saved afterwards is still there for the caller to delete)."""
        with self._conn() as conn:
            return conn.execute(
                "UPDATE jobs SET status = ?, run_id = ?, error = ?, finished_at = ? WHERE id = ?",
                ("failed" if error else "done", run_id, error, now_iso(), job_id),
            ).rowcount > 0

    def get_job(self, job_id: str, user_id: int) -> dict | None:
        """A job only its owner can see."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ retention
    def purge_all(self) -> None:
        """Every resume, run and download for everyone. Accounts, sessions and settings stay."""
        with self._conn() as conn:
            conn.execute("DELETE FROM jobs")  # before runs: see finish_job
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM profiles")
            conn.execute("DELETE FROM downloads")

    def purge(self, retention_days: int) -> int:
        day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now_iso(),))
            conn.execute("DELETE FROM jobs WHERE created_at < ?", (day_ago,))  # the page stops asking long before
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
