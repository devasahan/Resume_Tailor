"""Make the resume fit on exactly one page, in the least destructive order:

  0. before the model writes anything, plan how many full-length bullets fit (bullet_budget), so the
     first draft fits and nothing has to be cut
  1. shrink font and margins down a ladder (never below 9.5 pt / 0.5 in)
  2. (done by the caller) ask the model once more with a smaller bullet budget
  3. drop the last bullet of the roles with the most bullets, oldest first, keeping 2 per role
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

from ..render.pdf import render_resume_pdf
from ..schemas import FitStyle, Profile, Resume, ResumeExperience, SkillGroup
from .facts import bullet_caps

LADDER = [(10.5, 0.65), (10.25, 0.6), (10.0, 0.55), (9.75, 0.5), (9.5, 0.5)]


def styles(paper: str) -> list[FitStyle]:
    return [FitStyle(font_pt=f, margin_in=m, paper=paper) for f, m in LADDER]  # type: ignore[arg-type]


# Placeholder text at the full length the prompt asks for (26-word bullets, an 80-word summary, 26 skills).
BUDGET_STYLE = 3  # plan at 9.75 pt, keeping the last ladder step (9.5 pt) as headroom for the longest allowed text
_PLAN_BULLET = (
    "Designed and rolled out the event-driven order service behind the public REST API, adding retries and rate "
    "limiting so partner integrations stayed stable during traffic spikes"
)
_PLAN_SUMMARY = " ".join((
    "Senior Full-Stack Engineer with many years of experience building high-traffic e-commerce and SaaS platforms "
    "for growing product teams, specializing in backend services, integrations and modern front ends on cloud "
    "infrastructure. " * 3
).split()[:80])
_PLAN_SKILLS = [
    SkillGroup(category=category, skills=skills.split(", "))
    for category, skills in [
        ("Languages and Frameworks", "TypeScript, JavaScript, React, Next.js, Node.js, Python, PostgreSQL"),
        ("Backend and Integrations", "REST APIs, GraphQL, Webhooks, Message Queues, Caching, Microservices"),
        ("Cloud and Infrastructure", "AWS, Docker, Kubernetes, Terraform, CloudWatch, GitHub Actions, Redis"),
        ("Engineering Practices", "CI/CD, Observability, Code Review, Unit Testing, Agile, System Design"),
    ]
]


def _plan_resume(profile: Profile, counts: list[int]) -> Resume:
    return Resume(
        profile=profile.model_copy(deep=True), headline="Senior Full-Stack Engineer | TypeScript | React | AWS",
        summary=_PLAN_SUMMARY, skills=_PLAN_SKILLS,
        experience=[
            ResumeExperience(company_index=i, job_title="Senior Full-Stack Engineer", bullets=[_PLAN_BULLET] * n)
            for i, n in enumerate(counts)
        ],
    )


def bullet_budget(profile: Profile, paper: str = "Letter", today: date | None = None) -> list[int]:
    """Bullets per employer (most recent first) that fit on one page when every bullet is full length.

    Every employer starts with 2. Extra bullets are then handed out one at a time (up to each employer's cap
    from bullet_caps) while the placeholder resume still fits; recent roles go first but older ones are not
    starved: the newest role's 4th bullet comes after the second role's 3rd, its 5th after the third role's 3rd.
    Cached per profile, paper and day, because compare mode asks once per model for the same profile.
    """
    return list(_budget(profile.model_dump_json(), paper, today or date.today()))


@lru_cache(maxsize=64)
def _budget(profile_json: str, paper: str, today: date) -> tuple[int, ...]:
    profile = Profile.model_validate_json(profile_json)
    caps = bullet_caps(profile.companies, today)
    counts = [min(2, cap) for cap in caps]
    style = styles(paper)[BUDGET_STYLE]

    def fits() -> bool:
        return render_resume_pdf(_plan_resume(profile, counts), style)[1] == 1

    if not counts or not fits():
        return tuple(counts)
    extras = sorted(
        (level + 0.75 * index, index, level) for index, cap in enumerate(caps) for level in range(3, cap + 1)
    )
    for _score, index, level in extras:
        counts[index] = level
        if not fits():
            counts[index] = level - 1
            break
    return tuple(counts)


def smaller_budget(counts: list[int]) -> list[int]:
    """One bullet fewer for two different employers: the one with the most, then the oldest with the most."""
    smaller = list(counts)
    candidates = sorted((i for i, n in enumerate(smaller) if n > 2), key=lambda i: (-smaller[i], -i))
    for index in candidates[:2]:
        smaller[index] -= 1
    return smaller


def try_styles(resume: Resume, paper: str = "Letter") -> tuple[FitStyle, int, bytes, list[str]] | None:
    """Return the first style that renders on one page, or None."""
    ladder = styles(paper)
    for index, style in enumerate(ladder):
        pdf, pages = render_resume_pdf(resume, style)
        if pages == 1:
            notes = []
            if index:
                notes.append(f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in to fit one page.")
            return style, pages, pdf, notes
    return None


def render_smallest(resume: Resume, paper: str = "Letter") -> tuple[FitStyle, int, bytes]:
    style = styles(paper)[-1]
    pdf, pages = render_resume_pdf(resume, style)
    return style, pages, pdf


def trim_to_fit(resume: Resume, paper: str = "Letter") -> tuple[Resume, FitStyle, int, bytes, list[str]]:
    """Remove bullets until one page (or nothing more can be removed). Works on a copy."""
    work = resume.model_copy(deep=True)
    removed: list[str] = []
    while True:
        style, pages, pdf = render_smallest(work, paper)
        if pages == 1:
            break
        candidates = [e for e in work.experience if len(e.bullets) > 2]
        if not candidates:
            break
        most = max(len(e.bullets) for e in candidates)
        target = [e for e in candidates if len(e.bullets) == most][-1]  # the oldest among the longest
        company = work.profile.companies[target.company_index].name
        removed.append(f"{company}: {target.bullets.pop()}")
    notes = [f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in."]
    if removed:
        notes.append(f"Removed {len(removed)} bullet(s) to fit one page: " + " / ".join(r[:60] + "..." for r in removed))
    if pages != 1:
        notes.append("Still more than one page after trimming. Shorten the text in the editor.")
    return work, style, pages, pdf, notes
