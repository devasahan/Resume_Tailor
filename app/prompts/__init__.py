"""Prompt files plus the small functions that fill them in.

Bump PROMPT_VERSION whenever the wording changes; every run records the version it used.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from ..pipeline.facts import bullet_caps, experience_before, months_phrase, stay_months
from ..schemas import CareerStep, Company, JDAnalysis, Resume

PROMPT_VERSION = "v25"
RETRY_MARKER = "Your previous answer was rejected"

_DIR = Path(__file__).parent
_TOKEN = re.compile(r"\{\{(\w+)\}\}")

SHORTEN_NOTE = (
    " The previous draft did not fit on two pages, so some employers above now get one bullet fewer. Keep every "
    "bullet at most 28 words and the summary at 60-75 words, but keep the same quality: each bullet still says "
    "what was done, how, and what changed, and every must-have phrase from the job description still appears word "
    "for word in the summary or a bullet."
)


def _read(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


def _fill(template: str, **values: str) -> str:
    # One pass, so text inside a value (a job description with braces) is never re-interpreted.
    return _TOKEN.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def is_current(company: Company) -> bool:
    return company.end.strip().lower() in ("", "present", "current", "now", "today")


def describe_companies(
    companies: list[Company], bullets: list[int] | None = None, today: date | None = None
) -> str:
    """One line per employer: dates, work mode, how long the stay was and how much experience came before it
    (computed, so the model never does date arithmetic when choosing titles), and how many bullets to write
    (computed so the resume fits on two pages at most)."""
    bullets = bullets or bullet_caps(companies, today)
    lines = []
    for index, c in enumerate(companies):
        end = "Present (current role)" if is_current(c) else c.end
        parts = [c.name, f"{c.start or '?'} - {end}", c.mode or "work mode not stated"]
        if c.location:
            parts.append(c.location)
        stay = months_phrase(stay_months(c, today))
        before = experience_before(companies, index, today)
        if stay:
            history = "first job" if before == 0 else f"{months_phrase(before)} of experience before it"
            parts.append(f"{stay} stay, {history}")
        parts.append(f"write exactly {bullets[index]} bullets")
        lines.append(f"{index}: " + " | ".join(parts))
    return "\n".join(lines) or "(none)"


METRICS_MODES = ("estimate", "hints_only")


def tailor_system(metrics_mode: str = "estimate") -> str:
    if metrics_mode not in METRICS_MODES:
        raise ValueError(f"metrics_mode must be one of {METRICS_MODES}")
    return _fill(_read("system.md"), metrics_rule=_read(f"metrics_{metrics_mode}.md").strip())


def tailor_user(
    jd: str, companies: list[Company], hints: str, facts: str = "", shorten: bool = False,
    bullets: list[int] | None = None, today: date | None = None,
) -> str:
    return _fill(
        _read("tailor.md"),
        jd=jd.strip(),
        count=str(len(companies)),
        companies=describe_companies(companies, bullets, today),
        facts=facts.strip() or "Total professional experience: unknown (do not state a number of years).",
        hints=hints.strip() or "(none given)",
        extra=SHORTEN_NOTE if shorten else "",
    )


def revise_user(
    jd: str, companies: list[Company], hints: str, facts: str = "", *, bullets: list[int] | None = None,
    today: date | None = None, analysis: JDAnalysis, plan: list[CareerStep], draft: str, problems: str,
) -> str:
    """The fix round: the same opening as tailor_user (so the provider can reuse its cache of it), then the final
    analysis and plan, the numbered draft and what to fix. The answer is a TailorRevision: only what changes."""
    steps = [f"company_index {s.company_index}: {s.product or 'product not known'} | {s.scope} | {', '.join(s.focus)}"
             for s in plan]
    return _fill(
        _read("revise.md"),
        jd=jd.strip(),
        count=str(len(companies)),
        companies=describe_companies(companies, bullets, today),
        facts=facts.strip() or "Total professional experience: unknown (do not state a number of years).",
        hints=hints.strip() or "(none given)",
        must_haves=", ".join(analysis.must_have_skills) or "(none)",
        keywords=", ".join(analysis.keywords) or "(none)",
        plan="\n".join(steps) or "(none)",
        draft=draft,
        marker=RETRY_MARKER,
        problems=problems,
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


def job_facts_system() -> str:
    return _read("job_facts_system.md")


def job_facts_user(jd: str) -> str:
    return f"<job_description>\n{jd.strip()}\n</job_description>"


def retry_suffix(error: str) -> str:
    return f"\n\n{RETRY_MARKER}: {error}\nReturn a corrected JSON object that follows the schema exactly."
