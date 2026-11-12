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

from ..render.pdf import render_resume_pdf, resume_layout, resume_sheet
from ..schemas import (
    MAX_CERTIFICATIONS,
    FitStyle,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
)
from .facts import bullet_caps, is_expired, stay_months

LADDER = [(10.5, 0.65), (10.25, 0.6), (10.0, 0.55), (9.75, 0.5), (9.5, 0.5)]
MAX_PAGES = 2
MIN_LAST_PAGE_FILL = 0.25  # a second page less full than this looks unfinished: one page at a smaller step is better


def styles(paper: str) -> list[FitStyle]:
    return [FitStyle(font_pt=f, margin_in=m, paper=paper) for f, m in LADDER]  # type: ignore[arg-type]


# Placeholder text at the full length the prompt asks for (32-word bullets, an 85-word summary, 32 skills).
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


def _plan_resume(profile: Profile, counts: list[int]) -> Resume:
    return Resume(
        profile=profile.model_copy(deep=True), headline="Senior Full-Stack Engineer | TypeScript | React | AWS",
        summary=_PLAN_SUMMARY, skills=_PLAN_SKILLS,
        experience=[
            ResumeExperience(company_index=i, job_title="Senior Full-Stack Engineer", bullets=[_PLAN_BULLET] * n)
            for i, n in enumerate(counts)
        ],
        # room for the certifications the person has: the ones that fit the job are printed
        certifications=[cert.model_copy() for cert in profile.certifications
                        if not is_expired(cert.expires)][:MAX_CERTIFICATIONS],
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


def try_styles(resume: Resume, paper: str = "Letter",
               document: str = "Resume") -> tuple[FitStyle, int, bytes, list[str]] | None:
    """The largest style that fits on at most MAX_PAGES pages, or None. A second page that would hold only a few
    lines is avoided when a smaller step fits everything on one page. Each job's title then goes on its own line
    under the company when that still fits the same pages, so it never costs a bullet, a font step or a page.
    `document` ("Resume", or "CV" in EU style) names the PDF file in its title and language, as the preview does."""
    layouts = []  # (ladder step, style, pages, the laid-out sheet, kept to build the file from)
    for index, style in enumerate(styles(paper)):
        sheet = resume_sheet(resume, style, document)
        pages, fill = sheet.pdf.page, sheet.last_page_fill()
        layouts.append((index, style, pages, sheet))
        if pages == 1 or (pages <= MAX_PAGES and fill >= MIN_LAST_PAGE_FILL):
            break
    else:  # no step is both within the limit and well filled: the largest one within the limit
        layouts = [layout for layout in layouts if layout[2] <= MAX_PAGES][:1]
        if not layouts:
            return None
    index, style, pages, sheet = layouts[-1]
    stacked = style.model_copy(update={"stacked": True})
    stacked_sheet = resume_sheet(resume, stacked, document)
    if stacked_sheet.pdf.page == pages:
        style, sheet = stacked, stacked_sheet
    pdf, pages = sheet.finish()
    notes = []
    if index:
        notes.append(f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in to fit "
                     f"{_pages_note(pages)}.")
    return style, pages, pdf, notes


def render_smallest(resume: Resume, paper: str = "Letter", document: str = "Resume") -> tuple[FitStyle, int, bytes]:
    style = styles(paper)[-1]
    pdf, pages = render_resume_pdf(resume, style, document)
    return style, pages, pdf


def _removal_order(resume: Resume) -> list[int]:
    """Positions in resume.experience, one per bullet trim_to_fit may remove, in order: always the last bullet of
    the oldest among the employers with the most bullets, keeping 2 each. Needs no layout."""
    counts = [len(e.bullets) for e in resume.experience]
    order = []
    while any(n > 2 for n in counts):
        most = max(counts)
        position = max(i for i, n in enumerate(counts) if n == most)
        counts[position] -= 1
        order.append(position)
    return order


def _without(resume: Resume, order: list[int], k: int) -> Resume:
    work = resume.model_copy(deep=True)
    for position in order[:k]:
        work.experience[position].bullets.pop()
    return work


MAX_REMOVED_IN_NOTE = 5


def trim_to_fit(resume: Resume, paper: str = "Letter",
                document: str = "Resume") -> tuple[Resume, FitStyle, int, bytes, list[str]]:
    """Remove bullets until the resume fits on MAX_PAGES pages (or nothing more can be removed). Works on a copy.
    Fewer bullets never take more room, so the fewest removals that fit are found by halving (a few layouts, not
    one per removed bullet). Each job keeps its first 2 bullets, and the headline, summary, skills and titles stay
    as they are (a cover letter written from them meanwhile stays true). `document`: as in try_styles."""
    order, smallest, seen = _removal_order(resume), styles(paper)[-1], {}

    def fits(k: int) -> bool:
        if k not in seen:
            seen[k] = resume_layout(_without(resume, order, k), smallest)[0] <= MAX_PAGES
        return seen[k]

    low, high = 0, 1  # look at 1, 2, 4, 8 ... removals (a small overflow costs few layouts), then halve
    if fits(0):
        high = 0
    else:
        while high < len(order) and not fits(high):
            low, high = high, min(len(order), high * 2)
        while high - low > 1 and fits(high):  # not fits(low), fits(high)
            middle = (low + high) // 2
            low, high = (low, middle) if fits(middle) else (middle, high)
    k = min(high, len(order))
    work = _without(resume, order, k)
    removed, popped = [], {}
    for position in order[:k]:  # what each removal took, in order: the last bullet left at that employer
        exp = resume.experience[position]
        popped[position] = popped.get(position, 0) + 1
        company = resume.profile.companies[exp.company_index].name
        removed.append(f"{company}: {exp.bullets[len(exp.bullets) - popped[position]]}")
    found = try_styles(work, paper, document)  # after trimming, the largest font that fits
    if found:
        style, pages, pdf, _notes = found
    else:
        style, pages, pdf = render_smallest(work, paper, document)
    notes = [f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in."] if style != styles(paper)[0] else []
    if removed:
        shown = " / ".join(r[:60] + "..." for r in removed[:MAX_REMOVED_IN_NOTE])
        more = f" and {len(removed) - MAX_REMOVED_IN_NOTE} more" if len(removed) > MAX_REMOVED_IN_NOTE else ""
        notes.append(f"Removed {len(removed)} bullet(s) to fit {_pages_note(MAX_PAGES)}: {shown}{more}")
    if pages > MAX_PAGES:
        notes.append(f"Still more than {_pages_note(MAX_PAGES)} after trimming. Shorten the text in the editor.")
    return work, style, pages, pdf, notes
