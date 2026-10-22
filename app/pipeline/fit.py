"""Make the resume fit on at most two pages (one for a short career), in the least destructive order:

  0. before the model writes anything, plan how many full-length bullets fit (bullet_budget), so the
     first draft fits and nothing has to be cut
  1. pick the largest font and margins that fit (never below 9.5 pt / 0.5 in); a second page that would hold
     only a few lines is avoided by a smaller step that fits everything on one page
  2. (done by the caller) ask the model once more with a smaller bullet budget
  3. drop the last bullet of the roles with the most bullets, oldest first, keeping 2 per role
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

from ..render.pdf import render_resume_pdf, resume_layout
from ..schemas import FitStyle, Profile, Resume, ResumeExperience, SkillGroup
from .facts import bullet_caps, stay_months

LADDER = [(10.5, 0.65), (10.25, 0.6), (10.0, 0.55), (9.75, 0.5), (9.5, 0.5)]
MAX_PAGES = 2
MIN_LAST_PAGE_FILL = 0.25  # a second page less full than this looks unfinished: one page at a smaller step is better


def styles(paper: str) -> list[FitStyle]:
    return [FitStyle(font_pt=f, margin_in=m, paper=paper) for f, m in LADDER]  # type: ignore[arg-type]


# Placeholder text at the full length the prompt asks for (32-word bullets, an 85-word summary, 32 skills, a
# 10-item tech line per job).
BUDGET_STYLE = 3  # plan at 9.75 pt, keeping the last ladder step (9.5 pt) as headroom for the longest allowed text
_PLAN_BULLET = (
    "Designed and rolled out the event-driven order service behind the public REST API, adding retries, idempotency "
    "keys and rate limiting so partner integrations stayed stable during seasonal traffic spikes"
)
_PLAN_SUMMARY = " ".join((
    "Senior Full-Stack Engineer with many years of experience building high-traffic e-commerce and SaaS platforms "
    "for growing product teams, specializing in backend services, integrations and modern front ends on cloud "
    "infrastructure. " * 3
).split()[:85])
_PLAN_SKILLS = [
    SkillGroup(category=category, skills=skills.split(", "))
    for category, skills in [
        ("Languages and Frameworks", "TypeScript, JavaScript, React, Next.js, Node.js, Python, FastAPI, PostgreSQL"),
        ("Backend and Integrations", "REST APIs, GraphQL, Webhooks, Message Queues, Caching, Microservices, gRPC, Kafka"),
        ("Cloud and Infrastructure", "AWS, Docker, Kubernetes, Terraform, CloudWatch, GitHub Actions, Redis, Helm"),
        ("Engineering Practices", "CI/CD, Observability, Code Review, Unit Testing, Agile, System Design, TDD, DDD"),
    ]
]
_PLAN_TECH = ["Python", "FastAPI", "PostgreSQL", "Redis", "Celery", "Docker", "Kubernetes", "AWS (ECS, SQS, Lambda)",
              "Terraform", "GitHub Actions"]


def _plan_resume(profile: Profile, counts: list[int]) -> Resume:
    return Resume(
        profile=profile.model_copy(deep=True), headline="Senior Full-Stack Engineer | TypeScript | React | AWS",
        summary=_PLAN_SUMMARY, skills=_PLAN_SKILLS,
        experience=[
            ResumeExperience(company_index=i, job_title="Senior Full-Stack Engineer", bullets=[_PLAN_BULLET] * n,
                             tech=list(_PLAN_TECH))
            for i, n in enumerate(counts)
        ],
    )


def bullet_budget(profile: Profile, paper: str = "Letter", today: date | None = None, max_pages: int = MAX_PAGES) -> list[int]:
    """Bullets per employer (most recent first) that fit on `max_pages` pages when every bullet is full length.

    Each employer gets up to its cap from bullet_caps (recent roles and long stays get the most). Only when that
    is too long for the pages: every employer starts with 2 and extras are handed out in order while the
    placeholder resume still fits. Recent roles are favoured and so are long stays: a role held for 2-3 years gets
    its 3rd bullet before a role held for 8 months, even when that one is newer.
    Cached per profile, paper, page limit and day, because compare mode asks once per model for the same profile.
    """
    return list(_budget(profile.model_dump_json(), paper, today or date.today(), max_pages))


@lru_cache(maxsize=64)
def _budget(profile_json: str, paper: str, today: date, max_pages: int) -> tuple[int, ...]:
    profile = Profile.model_validate_json(profile_json)
    caps = bullet_caps(profile.companies, today)
    style = styles(paper)[BUDGET_STYLE]

    def fits(counts: list[int]) -> bool:
        return resume_layout(_plan_resume(profile, counts), style)[0] <= max_pages

    if not caps or fits(caps):  # the usual case on two pages: every employer gets its cap
        return tuple(caps)
    base = [min(2, cap) for cap in caps]
    if not fits(base):
        return tuple(base)
    extras = extra_order(profile, caps, today)

    def with_first(n: int) -> list[int]:
        counts = list(base)
        for index, level in extras[:n]:
            counts[index] = level
        return counts

    low, high = 0, len(extras)  # with_first(low) fits, with_first(len(extras)) == caps does not
    while high - low > 1:  # more bullets never take less room, so the longest prefix that fits is found by halving
        middle = (low + high) // 2
        low, high = (middle, high) if fits(with_first(middle)) else (low, middle)
    return tuple(with_first(low))


def extra_order(profile: Profile, caps: list[int], today: date) -> list[tuple[int, int]]:
    """(employer, bullet number) for every bullet above 2, in the order they are handed out when the pages run
    out: recent roles first, and long stays before short ones (a role held 3 years gets its 3rd bullet before a
    newer role held 8 months)."""
    years = [min((stay_months(c, today) or 12) / 12, 3) for c in profile.companies]  # unreadable dates: 1 year
    return [(index, level) for _score, index, level in sorted(
        (level + 0.75 * index - 0.5 * years[index], index, level)
        for index, cap in enumerate(caps)
        for level in range(3, cap + 1)
    )]


def smaller_budget(counts: list[int]) -> list[int]:
    """The budget for a second try after a draft did not fit: one bullet fewer for every employer above 3 (and at
    least two employers above 2, most bullets first, then the oldest)."""
    smaller = list(counts)
    above = [i for i, n in enumerate(smaller) if n > 3]
    if len(above) < 2:
        above += sorted((i for i, n in enumerate(smaller) if n > 2 and i not in above),
                        key=lambda i: (-smaller[i], -i))[: 2 - len(above)]
    for index in above:
        smaller[index] -= 1
    return smaller


def _pages_note(pages: int) -> str:
    return {1: "one page", 2: "two pages"}.get(pages, f"{pages} pages")


def try_styles(resume: Resume, paper: str = "Letter") -> tuple[FitStyle, int, bytes, list[str]] | None:
    """The largest style that fits on at most MAX_PAGES pages, or None. A second page that would hold only a few
    lines is avoided when a smaller step fits everything on one page."""
    ladder = styles(paper)
    layouts = []
    for index, style in enumerate(ladder):
        pages, fill = resume_layout(resume, style)
        layouts.append((index, style, pages))
        if pages == 1 or (pages <= MAX_PAGES and fill >= MIN_LAST_PAGE_FILL):
            break
    else:  # no step is both within the limit and well filled: the largest one within the limit
        layouts = [layout for layout in layouts if layout[2] <= MAX_PAGES][:1]
        if not layouts:
            return None
    index, style, pages = layouts[-1]
    pdf, pages = render_resume_pdf(resume, style)
    notes = []
    if index:
        notes.append(f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in to fit "
                     f"{_pages_note(pages)}.")
    return style, pages, pdf, notes


def render_smallest(resume: Resume, paper: str = "Letter") -> tuple[FitStyle, int, bytes]:
    style = styles(paper)[-1]
    pdf, pages = render_resume_pdf(resume, style)
    return style, pages, pdf


def trim_to_fit(resume: Resume, paper: str = "Letter") -> tuple[Resume, FitStyle, int, bytes, list[str]]:
    """Remove bullets until the resume fits on MAX_PAGES pages (or nothing more can be removed). Works on a copy."""
    work = resume.model_copy(deep=True)
    removed: list[str] = []
    while True:
        pages, _fill = resume_layout(work, styles(paper)[-1])
        if pages <= MAX_PAGES:
            break
        candidates = [e for e in work.experience if len(e.bullets) > 2]
        if not candidates:
            break
        most = max(len(e.bullets) for e in candidates)
        target = [e for e in candidates if len(e.bullets) == most][-1]  # the oldest among the longest
        company = work.profile.companies[target.company_index].name
        removed.append(f"{company}: {target.bullets.pop()}")
    found = try_styles(work, paper)  # after trimming, the largest font that fits
    if found:
        style, pages, pdf, _notes = found
    else:
        style, pages, pdf = render_smallest(work, paper)
    notes = [f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in."] if style != styles(paper)[0] else []
    if removed:
        notes.append(f"Removed {len(removed)} bullet(s) to fit {_pages_note(MAX_PAGES)}: "
                     + " / ".join(r[:60] + "..." for r in removed))
    if pages > MAX_PAGES:
        notes.append(f"Still more than {_pages_note(MAX_PAGES)} after trimming. Shorten the text in the editor.")
    return work, style, pages, pdf, notes
