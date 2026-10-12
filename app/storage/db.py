"""SQLite storage: settings, the uploaded base resumes (profiles), and one row per run.

Privacy: uploaded resumes stay until the user deletes them. Runs (including the job description they
were made for) are deleted after the `retention_days` setting. Full resume text is never written to logs.
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
    hidden INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    profile_id INTEGER,
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
"""


# Columns added to `profiles` after the first release, in the order they are added (hidden last).
_PROFILE_COLUMNS = [
    ("filename", "TEXT NOT NULL DEFAULT ''"),
    ("method", "TEXT NOT NULL DEFAULT ''"),
    ("notes", "TEXT NOT NULL DEFAULT '[]'"),
    ("hidden", "INTEGER NOT NULL DEFAULT 0"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database up to date. Safe to run on every start and from two processes at once.

    Older versions saved a new profile row on every "Save profile", so a database can hold many copies of
    one person. When the resume list arrives, only the newest of those stays visible; the others are
    hidden, not deleted (runs keep their own copy of the profile, so nothing depends on them).
    """
    have = {row["name"] for row in conn.execute("PRAGMA table_info(profiles)")}
    for name, ddl in _PROFILE_COLUMNS:
        if name in have:
            continue
        try:
            conn.execute(f"ALTER TABLE profiles ADD COLUMN {name} {ddl}")
        except sqlite3.OperationalError as exc:  # another process added it first
            if "duplicate column" not in str(exc).lower():
                raise
            continue
        if name == "hidden":
            conn.execute("UPDATE profiles SET hidden = 1 WHERE id <> (SELECT MAX(id) FROM profiles)")


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
    def save_profile(
        self, profile: Profile, *, filename: str = "", method: str = "", notes: list[str] | None = None
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO profiles(created_at, data, filename, method, notes) VALUES(?, ?, ?, ?, ?)",
                (now_iso(), profile.model_dump_json(), filename, method, json.dumps(notes or [])),
            )
            return int(cur.lastrowid)

    def latest_profile(self) -> tuple[int, Profile] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT id, data FROM profiles WHERE hidden = 0 ORDER BY id DESC LIMIT 1").fetchone()
        return (row["id"], Profile.model_validate_json(row["data"])) if row else None

    def get_profile(self, profile_id: int) -> Profile | None:
        with self._conn() as conn:
            row = conn.execute("SELECT data FROM profiles WHERE id = ? AND hidden = 0", (profile_id,)).fetchone()
        return Profile.model_validate_json(row["data"]) if row else None

    @staticmethod
    def _entry(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "created_at": row["created_at"], "filename": row["filename"], "method": row["method"],
            "notes": json.loads(row["notes"] or "[]"), "profile": Profile.model_validate_json(row["data"]),
        }

    def list_profiles(self) -> list[dict]:
        """Visible uploaded resumes, newest first: id, created_at, filename, method, notes, profile."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, data, filename, method, notes FROM profiles WHERE hidden = 0 ORDER BY id DESC"
            ).fetchall()
        return [self._entry(row) for row in rows]

    def profile_entry(self, profile_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, created_at, data, filename, method, notes FROM profiles WHERE id = ? AND hidden = 0",
                (profile_id,),
            ).fetchone()
        return self._entry(row) if row else None

    def delete_profile(self, profile_id: int) -> bool:
        """Remove an uploaded resume. Runs made from it keep working: each run stores its own copy."""
        with self._conn() as conn:
            return conn.execute("DELETE FROM profiles WHERE id = ? AND hidden = 0", (profile_id,)).rowcount > 0

    # ------------------------------------------------------------------ runs
    def save_run(self, result: TailorResult, profile_id: int | None, jd_hash: str, task: str = "resume") -> int:
        result.created_at = result.created_at or now_iso()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO runs(created_at, profile_id, jd_hash, task, requested_model, served_model, status,"
                " cost_usd, input_tokens, output_tokens, latency_ms, result) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.created_at, profile_id, jd_hash, task, result.requested_model, result.served_model,
                    result.status, result.cost_usd, result.input_tokens, result.output_tokens, result.latency_ms,
                    "{}",
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

    # ------------------------------------------------------------------ retention
    def purge_all(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM profiles")

    def purge(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._conn() as conn:
            runs = conn.execute("DELETE FROM runs WHERE created_at < ?", (cutoff,)).rowcount
            # uploaded resumes stay until the user deletes them; only hidden old copies expire
            profiles = conn.execute(
                "DELETE FROM profiles WHERE hidden = 1 AND created_at < ?", (cutoff,)
            ).rowcount
        return runs + profiles
