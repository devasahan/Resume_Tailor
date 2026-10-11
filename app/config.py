"""Paths, environment and the tiny .env loader (no extra dependency)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


FROM_DOTENV: set[str] = set()  # variable names whose value came from .env


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env into os.environ.

    A non-empty value in .env wins over a variable inherited from the system (for example an old
    ANTHROPIC_API_KEY set in Windows settings): the README tells people to edit .env, so that is
    what has to take effect. Empty values in .env are ignored.
    """
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: Notepad may add a BOM
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'").strip()
        if key and value:
            os.environ[key] = value
            FROM_DOTENV.add(key)


def key_hint(name: str) -> str:
    """Safe description of a secret: its prefix, last 4 characters and where it came from."""
    value = os.environ.get(name, "")
    if not value:
        return ""
    prefix = "-".join(value.split("-")[:3]) if value.count("-") >= 2 else value[:4]
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


def admin_token() -> str:
    return os.environ.get("ADMIN_TOKEN", "")


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
}

# Used only when a task has no model chosen in Settings. Real models first, demo last.
DEFAULT_PRIORITY = [
    "claude-sonnet-5-5",
    "gemini-3.8-flash",
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
PARSER_PRIORITY = [
    "gemini-3.5-flash-lite",
    "mistral-large",
    "gemini-3.8-flash",
    "claude-sonnet-5-5",
    "grok-4.3",
    "demo-concise",
]
