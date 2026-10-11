"""Prompt files plus the small functions that fill them in.

Bump PROMPT_VERSION whenever the wording changes; every run records the version it used.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..schemas import Company, Resume

PROMPT_VERSION = "v2"
RETRY_MARKER = "Your previous answer was rejected"

_DIR = Path(__file__).parent
_TOKEN = re.compile(r"\{\{(\w+)\}\}")

SHORTEN_NOTE = (
    " The previous draft did not fit on one page. Make every bullet shorter (at most 18 words), use at most "
    "4 bullets for the most recent employer and 3 for the others, and keep the summary under 45 words."
)


def _read(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


def _fill(template: str, **values: str) -> str:
    # One pass, so text inside a value (a job description with braces) is never re-interpreted.
    return _TOKEN.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def is_current(company: Company) -> bool:
    return company.end.strip().lower() in ("", "present", "current", "now", "today")


def describe_companies(companies: list[Company]) -> str:
    lines = []
    for index, c in enumerate(companies):
        end = "Present (current role)" if is_current(c) else c.end
        parts = [c.name, f"{c.start or '?'} - {end}", c.mode or "work mode not stated"]
        if c.location:
            parts.append(c.location)
        lines.append(f"{index}: " + " | ".join(parts))
    return "\n".join(lines) or "(none)"


METRICS_MODES = ("estimate", "hints_only")


def tailor_system(metrics_mode: str = "estimate") -> str:
    if metrics_mode not in METRICS_MODES:
        raise ValueError(f"metrics_mode must be one of {METRICS_MODES}")
    return _fill(_read("system.md"), metrics_rule=_read(f"metrics_{metrics_mode}.md").strip())


def tailor_user(jd: str, companies: list[Company], hints: str, facts: str = "", shorten: bool = False) -> str:
    return _fill(
        _read("tailor.md"),
        jd=jd.strip(),
        count=str(len(companies)),
        companies=describe_companies(companies),
        facts=facts.strip() or "Total professional experience: unknown (do not state a number of years).",
        hints=hints.strip() or "(none given)",
        extra=SHORTEN_NOTE if shorten else "",
    )


def cover_system() -> str:
    return _read("cover_system.md")


def cover_user(jd: str, resume: Resume, role: str, company: str, manager: str, hints: str) -> str:
    experience = []
    for exp in resume.experience:
        company_name = resume.profile.companies[exp.company_index].name
        top = "; ".join(exp.bullets[:2])
        experience.append(f"- {exp.job_title} at {company_name}: {top}")
    return _fill(
        _read("cover.md"),
        jd=jd.strip(),
        company=company or "(not given)",
        manager=manager or "(not given)",
        role=role,
        headline=resume.headline,
        summary=resume.summary,
        experience="\n".join(experience),
        hints=hints.strip() or "(none given)",
    )


def parse_system() -> str:
    return _read("parse_system.md")


def parse_user(resume_text: str) -> str:
    return f"<resume_text>\n{resume_text.strip()}\n</resume_text>"


def retry_suffix(error: str) -> str:
    return f"\n\n{RETRY_MARKER}: {error}\nReturn a corrected JSON object that follows the schema exactly."
