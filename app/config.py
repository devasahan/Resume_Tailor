"""Paths, environment and the tiny .env loader (no extra dependency)."""
from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LOCAL_ZONE = "America/Chicago"  # Arlington, Texas: the owner's days (History, Spent today, the daily cost cap)


FROM_DOTENV: set[str] = set()  # variable names whose value came from .env
_INHERITED: dict[str, str] = {}  # a system variable's own value, from before .env replaced it
_ENV_STAMP: tuple[int, int, int] | None = None  # (file id, modified time, size) of .env when it was last read
_ENV_LOCK = threading.Lock()  # one reader or writer of .env at a time: requests run in several threads


def env_file() -> Path:
    """The .env file the app reads and the Settings page writes keys to (ENV_FILE points elsewhere, in tests)."""
    return Path(os.environ.get("ENV_FILE") or ROOT / ".env")


def _stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_size  # a save replaces the file, so its id always changes


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ.

    A non-empty value in .env wins over a variable inherited from the system (for example an old
    ANTHROPIC_API_KEY set in Windows settings): the README tells people to edit .env, so that is
    what has to take effect. Empty values in .env are ignored.
    """
    global _ENV_STAMP
    path = path or env_file()
    if path == env_file():
        _ENV_STAMP = _stamp(path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: Notepad may add a BOM
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'").strip()
        if key and value:
            if key not in FROM_DOTENV and key in os.environ:
                _INHERITED.setdefault(key, os.environ[key])
            os.environ[key] = value
            FROM_DOTENV.add(key)


def _unset_from_dotenv(name: str) -> None:
    """A variable .env no longer sets: back to the system's own value, or gone."""
    FROM_DOTENV.discard(name)
    if name in _INHERITED:
        os.environ[name] = _INHERITED[name]
    else:
        os.environ.pop(name, None)


def refresh_dotenv() -> bool:
    """Read .env again when it changed since it was last read: a key saved on the Settings page by another worker
    process, or the file edited by hand. A variable that left .env leaves the environment too. True if it changed.
    One stat() per call, so it is cheap enough for every request."""
    path = env_file()
    if _stamp(path) == _ENV_STAMP:
        return False
    with _ENV_LOCK:
        if _stamp(path) == _ENV_STAMP:  # read (or saved) by another thread meanwhile
            return False
        before = set(FROM_DOTENV)
        FROM_DOTENV.clear()
        load_dotenv(path)
        for name in before - FROM_DOTENV:
            _unset_from_dotenv(name)
    return True


def set_env_value(name: str, value: str) -> None:
    """Write NAME=value into .env, in place of its line or as a new last line ('' removes the line), keeping every
    other line as it was, then use it at once. The file is replaced in one step, so it is never half written."""
    global _ENV_STAMP
    path = env_file()
    with _ENV_LOCK:
        lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
        out: list[str] = []
        written = False
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("#") and "=" in stripped and stripped.partition("=")[0].strip() == name:
                if value and not written:
                    out.append(f"{name}={value}")
                    written = True
                continue  # the old value (and any repeat of the name) goes
            out.append(line)
        if value and not written:
            out.append(f"{name}={value}")
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o600  # a new .env: readable by its owner only
        # A new file of its own (two app processes saving at once never write into the same one), created readable
        # by its owner only, so the key is never open to others, not even for a moment.
        handle, name_of_copy = tempfile.mkstemp(prefix=path.name + ".saving-", dir=path.parent)
        temporary = Path(name_of_copy)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write("\n".join(out) + "\n")
                file.flush()
                os.fsync(file.fileno())  # on disk before it takes the old file's place
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)  # never leave a copy of the keys behind
            raise
        _ENV_STAMP = _stamp(path)
        if value:
            if name not in FROM_DOTENV and name in os.environ:
                _INHERITED.setdefault(name, os.environ[name])
            os.environ[name] = value
            FROM_DOTENV.add(name)
        else:
            _unset_from_dotenv(name)


_HINT_WORD = re.compile(r"[a-z0-9]{1,6}")  # 'sk', 'ant', 'api03', 'proj': a provider's marker, never the secret


def key_hint(name: str) -> str:
    """Safe description of a secret: its provider's marker ('sk-ant-api03', 'sk-proj', 'AIza'), its last 4
    characters and where it came from. Never a piece of the random part."""
    value = os.environ.get(name, "")
    if not value:
        return ""
    words = []
    for word in value.split("-")[:3] if value.count("-") >= 2 else []:
        if not _HINT_WORD.fullmatch(word):
            break
        words.append(word)
    prefix = ("-".join(words) or value[:4]).rstrip("-")
    source = ".env" if name in FROM_DOTENV else "system environment variable"
    return f"{prefix}-...{value[-4:]} (from {source})"


load_dotenv()


def data_dir() -> Path:
    path = Path(os.environ.get("DATA_DIR", ROOT / "data"))
    if not path.is_absolute():  # "./data" in .env means inside the project, wherever the app was started from
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_file() -> Path:
    return Path(os.environ.get("MODELS_FILE", ROOT / "models.yaml"))


def admin_credentials() -> tuple[str, str]:
    """The admin account's email and password, from .env (never stored in code or in the database)."""
    return os.environ.get("ADMIN_EMAIL", "").strip(), os.environ.get("ADMIN_PASSWORD", "")


def force_https() -> bool:
    """FORCE_HTTPS=1 (on a server): send plain http:// visitors to https://, as reported by the proxy."""
    return os.environ.get("FORCE_HTTPS", "").strip().lower() in ("1", "true", "yes")


def cookie_secure() -> bool | None:
    """COOKIE_SECURE=1 or 0 forces the session cookie's Secure flag; unset (None) decides per request."""
    value = os.environ.get("COOKIE_SECURE", "").strip().lower()
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    return None


# Defaults for settings the user can change on the Settings page.
DEFAULT_SETTINGS: dict = {
    # model ids per task. null = pick the first available model by DEFAULT_PRIORITY.
    "resume_model": None,
    "cover_letter_model": None,
    "parser_model": None,
    "fallback_models": [],  # tried in order when the chosen model fails (not used by Compare)
    "daily_cost_cap_usd": 5.0,  # 0 = no cap
    "allowed_regions": ["US", "EU", "SG", "DEMO"],  # CN / UNKNOWN endpoints are opt-in
    "retention_days": 30,  # stored profiles and runs are deleted after this; 0 = keep
    "paper_size": "Letter",  # Letter | A4
    "metrics_mode": "estimate",  # estimate: realistic figures, flagged for the user | hints_only: only the user's numbers
    # A resume is ready within about this many seconds: the extra rewrite pass and the shorter-draft call run only when
    # they can finish in time (they are stopped at the limit). 0 = no limit: always run them (best quality, slower).
    "resume_time_limit_s": 30,
}

# Used only when a task has no model chosen in Settings. Real models first, demo last.
DEFAULT_PRIORITY = [  # Automatic resume model: the first one with a key (the Settings choice always wins)
    "gemini-3.8-flash",
    "claude-sonnet-5-5",
    "gpt-6.1-sol",
    "grok-4.3",
    "mistral-large",
    "gemini-3.5-flash-lite",
    "grok-4.7",
    "qwen-3.8-max",
    "deepseek-v4-flash",
    "glm-5.3",
    "kimi-k2.6",
    "demo-concise",
]
# Cheap models preferred for the (simple) base-resume parsing task.
RULES_READER = "rules"  # parser_model value: read uploads with the built-in rules (free, nothing is sent)

PARSER_PRIORITY = [
    "gemini-3.5-flash-lite",
    "mistral-large",
    "gemini-3.8-flash",
    "claude-sonnet-5-5",
    "grok-4.3",
    "gpt-6.1-sol",
    "demo-concise",
]
