"""Prompt files plus the small functions that fill them in.

Bump PROMPT_VERSION whenever the wording changes; every run records the version it used.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from functools import cache
from pathlib import Path

from ..pipeline.facts import (
    bullet_caps,
    experience_before,
    is_expired,
    months_phrase,
    stay_months,
)
from ..pipeline.notes import clean_notes
from ..schemas import CareerStep, Certification, Company, JDAnalysis, Resume

PROMPT_VERSION = "v36"
RETRY_MARKER = "Your previous answer was rejected"

_DIR = Path(__file__).parent
_TOKEN = re.compile(r"\{\{(\w+)\}\}")

SHORTEN_NOTE = (
    " The previous draft did not fit on two pages, so some employers above now get one bullet fewer. Keep every "
    "bullet at most 28 words and the summary at 60-75 words, but keep the same quality: each bullet still says "
    "what was done, how, and what changed, each percentage keeps its base, and every must-have phrase from the job "
    "description still appears word for word in the summary or a bullet."
)


@cache
def _read(name: str) -> str:
    """A prompt file, read once per process: the files change only with a deploy, which restarts the app, and on a
    host's network storage each read can be slow."""
    return (_DIR / name).read_text(encoding="utf-8")


def read_all() -> int:
    """Read every prompt file now (when the server starts, see main.warm_up). Returns how many there are."""
    names = sorted(path.name for path in _DIR.glob("*.md"))
    for name in names:
        _read(name)
    return len(names)


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


def describe_certifications(certifications: Sequence[Certification], today: date | None = None) -> str:
    """One numbered line per certification the candidate has, with its dates: the model picks the ones that fit the job
    by number. An expired one is left out (the others keep their numbers)."""
    lines = []
    for index, c in enumerate(certifications):
        if is_expired(c.expires, today):
            continue
        dates = ", ".join(part for part in (f"earned {c.year}" if c.year else "",
                                            f"expires {c.expires}" if c.expires else "") if part)
        lines.append(f"{index}: " + " | ".join(part for part in (c.name, c.issuer, dates) if part))
    return "\n".join(lines) or "(none)"


def certifications_block(
    certifications: Sequence[Certification], today: date | None = None, section: bool = True
) -> str:
    """The <candidate_certifications> block of a user prompt, with the blank line before it (the prompt files put
    {{certifications}} right under the block above it). "" while the Certifications section is off (Settings): the
    model is then shown no certifications at all."""
    if not section:
        return ""
    listed = describe_certifications(certifications, today)
    return f"\n<candidate_certifications>\n{listed}\n</candidate_certifications>\n"


METRICS_MODES = ("estimate", "hints_only")
SPELLING_STYLES = ("us", "eu")  # the Tailor page's US style / EU style (British spelling) buttons
SIGN_OFF = {"us": '"Sincerely,"', "eu": '"Kind regards,"'}


def _spelling_rule(spelling: str) -> str:
    if spelling not in SPELLING_STYLES:
        raise ValueError(f"spelling must be one of {SPELLING_STYLES}")
    return _read(f"spelling_{spelling}.md").strip()


def tailor_system(metrics_mode: str = "estimate", spelling: str = "us", certifications_section: bool = True) -> str:
    """`certifications_section` False (Settings): the CERTIFICATIONS rule is one line: list only the job post's own
    certifications (never printed, they only keep those names out of the text) and name none in the resume."""
    if metrics_mode not in METRICS_MODES:
        raise ValueError(f"metrics_mode must be one of {METRICS_MODES}")
    rule = "certifications_on.md" if certifications_section else "certifications_off.md"
    return _fill(_read("system.md"), metrics_rule=_read(f"metrics_{metrics_mode}.md").strip(),
                 spelling_rule=_spelling_rule(spelling), certifications_rule=_read(rule).strip())


def prompt_notes(text: str) -> str:
    """One employer's notes as the model reads them (clean_notes: no links or source lists), with angle brackets as
    their single-quote kin, so a note can never close the block or open another."""
    return clean_notes(text).replace("<", "‹").replace(">", "›")


def projects_block(companies: list[Company]) -> str:
    """The person's own notes on the products and projects they worked on, per employer, numbered as in <employers>
    ("" when no employer has any; see prompt_notes)."""
    entries = []
    for index, company in enumerate(companies):
        notes = prompt_notes(company.projects)
        if notes:
            entries.append(f"{index}: {company.name}\n{notes}")
    return "<employer_projects>\n" + "\n\n".join(entries) + "\n</employer_projects>\n" if entries else ""


def tailor_user(
    jd: str, companies: list[Company], hints: str, facts: str = "", shorten: bool = False,
    bullets: list[int] | None = None, today: date | None = None, certifications: Sequence[Certification] = (),
    certifications_section: bool = True,
) -> str:
    """`certifications_section` False (Settings): no <candidate_certifications> block at all."""
    return _fill(
        _read("tailor.md"),
        jd=jd.strip(),
        count=str(len(companies)),
        companies=describe_companies(companies, bullets, today),
        projects=projects_block(companies),
        facts=facts.strip() or "Total professional experience: unknown (do not state a number of years).",
        hints=hints.strip() or "(none given)",
        certifications=certifications_block(certifications, today, certifications_section),
        extra=SHORTEN_NOTE if shorten else "",
    )


def revise_user(
    jd: str, companies: list[Company], hints: str, facts: str = "", *, bullets: list[int] | None = None,
    today: date | None = None, analysis: JDAnalysis, plan: list[CareerStep], draft: str, problems: str,
    certifications: Sequence[Certification] = (), certifications_section: bool = True,
) -> str:
    """The fix round: the same opening as tailor_user (so the provider can reuse its cache of it), then the final
    analysis and plan, the numbered draft and what to fix. The answer is a TailorRevision: only what changes.
    `certifications_section` False (Settings): no <candidate_certifications> block, as in tailor_user."""
    steps = [f"company_index {s.company_index}: {s.product or 'product not known'} | {s.scope} | {', '.join(s.focus)}"
             for s in plan]
    return _fill(
        _read("revise.md"),
        jd=jd.strip(),
        count=str(len(companies)),
        companies=describe_companies(companies, bullets, today),
        projects=projects_block(companies),
        facts=facts.strip() or "Total professional experience: unknown (do not state a number of years).",
        hints=hints.strip() or "(none given)",
        certifications=certifications_block(certifications, today, certifications_section),
        must_haves=", ".join(analysis.must_have_skills) or "(none)",
        keywords=", ".join(analysis.keywords) or "(none)",
        plan="\n".join(steps) or "(none)",
        draft=draft,
        marker=RETRY_MARKER,
        problems=problems,
    )


def cover_system(spelling: str = "us") -> str:
    return _fill(_read("cover_system.md"), spelling_rule=_spelling_rule(spelling), sign_off=SIGN_OFF[spelling])


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
