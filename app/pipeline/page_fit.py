"""One page for a new run (layout 1): the owner's rule since prompt v38. Words are cut, never the font or the margins.

  0. before the model writes anything, plan how many bullets fit on one page at the design's own sizes (plan_pages):
     the most recent job gets 2 and every other job 1, then the most recent job's extras until it holds about half of
     all the bullets that fit (the owner's choice: about half the page, 5-6 bullets for a career of 5 jobs), then the
     older jobs' extras (recent jobs, long stays and the jobs the person's project notes show most relevant to the
     posting; the most recent job always keeps the most) while the placeholder resume fits, or runs over by less than
     one of its two-line bullets: real bullets are mostly shorter, so the page ends full (a draft as long as its plan
     loses that one bullet by design, planned_removals, which never counts toward asking the model for a shorter
     draft); only when even that minimum overflows one page is the plan made for two (without that allowance)
  1. after the draft, measure it at the design's own sizes (a few milliseconds, no file is made)
  2. when it is over, remove bullets (trim_to_page): the most recent job's last bullet while it holds more than half of
     them, or more than 8 (the most any job gets, a two-page plan's cap), and keeps the most; else the last bullet of
     the older job with the most bullets, the less relevant job first (its bullets' must-have and keyword mentions),
     then the older; every job keeps its first bullet, and a bullet that holds a must-have's only mention goes last
     (but before the most recent job's lead), as does the current job's last bullet naming the post's programming
     language (the owner's rule since prompt v39, read with the post's text as the check reads it)
  3. when one page cannot be reached even then, two pages from the untrimmed draft, trimmed only as far as two pages
     need; only when two pages cannot be reached at the design's sizes either, the old size ladder (fit.py), whose
     notes say the font was reduced: on the untrimmed draft when a smaller step fits it on two pages, else with only
     the bullets removed that its smallest steps need for two pages

A whole job always stays on one page (FitStyle.layout 1: its title line and all its bullets), so two pages never split
a job. Runs saved before keep layout 0 and the old fitting (fit.py): they look exactly as they did.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from ..render.pdf import resume_layout
from ..schemas import (
    MAX_CERTIFICATIONS,
    Company,
    FitStyle,
    JDAnalysis,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
)
from .facts import RECENCY_CAPS, bullet_caps, is_expired, stay_months
from .fit import LADDER, render_smallest, styles, try_styles

LAYOUT = 1  # whole jobs on one page, 'B.S. in Computer Science · School · Town' (app/render/pdf.py)
ONE_PAGE, TWO_PAGES = 1, 2
MAX_REMOVED_IN_NOTE = 5
# How the notes of a fit start (fit_new, fit_report, and fit.py for older runs), see refitted_notes: the notes on how
# the text was laid out (FIT_NOTE) give way to a new fit's; a note naming the bullets removed (REMOVED_NOTE) stays while
# they are off the resume
FIT_NOTE = re.compile(r"Kept two pages|Font reduced|Still ")
REMOVED_NOTE = re.compile(r"Removed \d+ bullet\(s\) to fit ")


def design_style(paper: str = "Letter") -> FitStyle:
    """The design's own sizes in the new layout: every new run is laid out at it (the font is never reduced, except
    as the very last resort, see fit_new)."""
    return styles(paper, LAYOUT)[0]


# Placeholder text at the longest the prompt asks for: a 22-word bullet (two lines, as every bullet of 16-30 words), a
# 45-word summary (three lines) and 3 skill rows of 6 items (one line each).
_PLAN_BULLET = ("Built the order sync service behind the partner REST API in Go, adding retries and rate limits so "
                "integrations stayed stable at peak")
_PLAN_SUMMARY = ("Senior Full-Stack Engineer with 7+ years of experience building e-commerce and SaaS platforms for "
                 "growing product teams. Focuses on backend services, partner integrations and React front ends on "
                 "AWS. Cut checkout latency from 912 to 684ms for 4.6K merchants at the current job.")
_PLAN_SKILLS = [
    SkillGroup(category="Languages", skills=["TypeScript", "Python", "Go", "SQL", "React", "Node.js"]),
    SkillGroup(category="Cloud and Data", skills=["AWS", "PostgreSQL", "Redis", "Docker", "Terraform", "Kafka"]),
    SkillGroup(category="Practices", skills=["REST APIs", "CI/CD", "pytest", "Observability", "Code Review", "TDD"]),
]


def _plan_resume(profile: Profile, counts: list[int]) -> Resume:
    return Resume(
        profile=profile.model_copy(deep=True), headline="Full-Stack Engineer", summary=_PLAN_SUMMARY,
        skills=_PLAN_SKILLS,
        experience=[ResumeExperience(company_index=i, job_title="Senior Full-Stack Engineer",
                                     bullets=[_PLAN_BULLET] * n) for i, n in enumerate(counts)],
        # room for the certifications the person has: the ones that fit the job are printed
        certifications=[cert.model_copy() for cert in profile.certifications
                        if not is_expired(cert.expires)][:MAX_CERTIFICATIONS],
    )


# ---------------------------------------------------------------------- relevance
_WORD = re.compile(r"[a-z][a-z0-9+#.]*[a-z0-9+#]|[a-z]")
_COMMON = frozenset("""
a about above across after again all also an and any are as at be been being both but by can could did do does each
for from had has have having how if in into is it its just more most no not of on once only or other our out over own
same should so some such than that the their them then there these they this those through to too under until up very
was we were what when where which while who why will with within without would you your
ability able add build building built work working experience experienced years year strong team teams new using use
used including include across help role roles job jobs company candidate candidates skills skill knowledge required
preferred plus etc well looking join our us
""".split())  # noqa: SIM905 (a word list reads better as text)


def _words(text: str) -> set[str]:
    return {w.strip(".") for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _COMMON}


def job_relevance(profile: Profile, jd: str) -> tuple[float, ...]:
    """How much each job's project notes share with the posting, 0..1 (the most relevant job 1), before the draft: the
    profile holds no job titles, so a job without notes scores 0. Only a tie-breaker after recency and stay length."""
    posting = _words(jd)
    shared = [len(_words(company.projects) & posting) if company.projects.strip() else 0
              for company in profile.companies]
    top = max(shared, default=0)
    return tuple(round(n / top, 3) if top else 0.0 for n in shared)


def _terms(analysis: JDAnalysis | None) -> tuple[list[str], list[str]]:
    if analysis is None:
        return [], []
    return list(analysis.must_have_skills), list(analysis.nice_to_have_skills) + list(analysis.keywords)


def draft_relevance(resume: Resume, analysis: JDAnalysis | None) -> list[float]:
    """Each job's relevance to the posting in the draft, 0..1 (the most relevant 1): its bullets' mentions of the
    posting's must-haves (counted twice) and of its other skills and keywords."""
    from .validate import (
        mentions,  # validate imports this module's callers; kept local to avoid a cycle
    )

    must, other = _terms(analysis)
    scores = []
    for exp in resume.experience:
        text = " ".join(exp.bullets)
        scores.append(2 * sum(mentions(text, t) for t in must) + sum(mentions(text, t) for t in other))
    top = max(scores, default=0)
    return [round(s / top, 3) if top else 0.0 for s in scores]


# ---------------------------------------------------------------------- the plan
# The most recent job holds about half of a plan's bullets (the owner's choice: about half the page, 5-6 bullets that
# cover the posting's main requirements, 1-2 for each older job). Once it has lasted CURRENT_MONTHS, it may hold
# CURRENT_CAP however short the older stays are (bullet_caps alone gives an 11-month job 3, or one more than any
# other job: 4 beside an 11-month one); a shorter stay keeps bullet_caps (its cap is not raised), though like any
# current job it still takes about half of the bullets that fit, up to that cap.
CURRENT_CAP, CURRENT_MONTHS = 6, 6


def plan_caps(companies: list[Company], today: date | None = None) -> list[int]:
    """Most bullets each employer can get in a new run's plan: bullet_caps, except that the most recent job (the
    first) of a career of two or more may hold CURRENT_CAP once its stay is CURRENT_MONTHS or longer (or its dates
    cannot be read): 11 months and 11 months -> [6, 3], not [4, 3]. A career of one job keeps its own cap."""
    caps = bullet_caps(companies, today)
    if len(caps) > 1 and (stay_months(companies[0], today) or CURRENT_MONTHS) >= CURRENT_MONTHS:
        caps[0] = max(caps[0], CURRENT_CAP)
    return caps


def plan_pages(profile: Profile, paper: str = "Letter", today: date | None = None,
               relevance: tuple[float, ...] = ()) -> tuple[int, list[int]]:
    """(pages, bullets per employer, most recent first) at the design's own sizes, when every bullet is two lines
    (see module docstring): on one page about half of them for the most recent job, and as many as fill the page (the
    placeholder may run over by less than one bullet); two pages only when even 2 bullets for the most recent job and
    1 for every other one do not fit on one. A short career gets every job's own cap (plan_caps). `relevance`
    (job_relevance) decides between older jobs otherwise equal. Cached per profile, paper, day and relevance (compare
    mode asks once per model for the same profile and posting)."""
    key = profile.model_dump_json(exclude={"companies": {"__all__": {"projects"}}})  # notes take no room
    pages, counts = _plan(key, paper, today or date.today(), tuple(relevance))
    return pages, list(counts)


@lru_cache(maxsize=64)
def _plan(profile_json: str, paper: str, today: date, relevance: tuple[float, ...]) -> tuple[int, tuple[int, ...]]:
    profile = Profile.model_validate_json(profile_json)
    caps = plan_caps(profile.companies, today)
    if not caps:
        return ONE_PAGE, ()
    style = design_style(paper)

    def pages(counts: list[int]) -> int:
        return resume_layout(_plan_resume(profile, counts), style)[0]

    if pages(caps) <= ONE_PAGE:
        return ONE_PAGE, tuple(caps)
    least = [min(2, caps[0])] + [min(1, cap) for cap in caps[1:]]
    if pages(least) <= ONE_PAGE:  # one page, filled: one bullet fewer must fit (the current job has 2 or more)
        return ONE_PAGE, tuple(_most_that_fit(profile, caps, least, today, relevance,
                                              lambda counts: pages([counts[0] - 1, *counts[1:]]) <= ONE_PAGE))
    # two pages, without that allowance: whole jobs move to the next page, so a bullet too many there can cost more
    # than one removed bullet
    two = [min(2, cap) for cap in caps]
    start = two if pages(two) <= TWO_PAGES else least
    return TWO_PAGES, tuple(_most_that_fit(profile, caps, start, today, relevance,
                                           lambda counts: pages(counts) <= TWO_PAGES))


def extra_order(profile: Profile, caps: list[int], start: list[int], today: date,
                relevance: tuple[float, ...] = ()) -> list[tuple[int, int]]:
    """(employer, bullet number) for every bullet above `start`, in the order they are handed out: recent roles
    first, long stays before short ones, the more relevant job before an otherwise equal one; the most recent role
    stays one ahead (its bullet N+1 comes before any other role's bullet N)."""
    years = [min((stay_months(c, today) or 12) / 12, 3) for c in profile.companies]  # unreadable dates: 1 year

    def score(index: int, level: int) -> float:
        weight = relevance[index] if index < len(relevance) else 0.0
        return level + 0.75 * index - 0.5 * years[index] - weight

    others = [(score(index, level), index, level) for index, cap in enumerate(caps) if index
              for level in range(start[index] + 1, cap + 1)]
    lead: dict[int, float] = {}  # the most recent role's bullet N+1 goes before any other role's bullet N
    for value, _index, level in others:
        lead[level + 1] = min(lead.get(level + 1, math.inf), value)
    first, best = [], math.inf
    for level in range(caps[0], start[0], -1):  # from its last bullet down, so its own stay in order
        best = min(best, score(0, level), lead.get(level, math.inf))
        first.append((best, 0, level))
    return [(index, level) for _value, index, level in sorted(others + first)]  # a tie: the most recent role first


def _most_that_fit(profile: Profile, caps: list[int], start: list[int], today: date, relevance: tuple[float, ...],
                   fits) -> list[int]:
    """The most bullets above `start` that `fits` allows, the most recent job first: its extras until it holds about
    half of them all (ceil of half, up to its cap), then extra_order for the rest (the older jobs' extras, and its own
    above that half where it must stay ahead). How many fit in all is found first, with its extras all first: on one
    page every placeholder bullet takes the same room, so the total does not depend on who gets them (on two pages
    whole jobs move between pages, so it is close). For example 10 that fit for 5 jobs: [5, 2, 1, 1, 1]."""

    def longest(extras: list[tuple[int, int]]) -> list[int]:
        """`start` with the longest run of `extras` that fits, found by halving: more bullets never take less room."""

        def with_first(n: int) -> list[int]:
            counts = list(start)
            for index, level in extras[:n]:
                counts[index] = max(counts[index], level)
            return counts

        if fits(with_first(len(extras))):
            return with_first(len(extras))
        low, high = 0, len(extras)  # with_first(low) fits, with_first(high) does not
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (middle, high) if fits(with_first(middle)) else (low, middle)
        return with_first(low)

    def first_to(level: int) -> list[tuple[int, int]]:  # the most recent job's extras up to `level`, then the rest
        own = [(0, n) for n in range(start[0] + 1, level + 1)]
        return own + extra_order(profile, caps, [max(start[0], level), *start[1:]], today, relevance)

    room = sum(longest(first_to(caps[0])))
    return longest(first_to(min(caps[0], math.ceil(room / 2))))


def planned_removals(profile: Profile, counts: list[int], pages: int, paper: str = "Letter") -> int:
    """Bullets a draft as long as its plan (plan_pages: `pages`, `counts`) loses by design: 1 when a one-page plan
    fills its page, its placeholder running over by less than one of its bullets (_plan), else 0 (a career of one job
    keeps room to spare, and a two-page plan has no such allowance). One layout, a few milliseconds."""
    return int(pages == ONE_PAGE and measure(_plan_resume(profile, counts), paper) > ONE_PAGE)


# ---------------------------------------------------------------------- after the draft
def measure(resume: Resume, paper: str = "Letter") -> int:
    """Pages at the design's own sizes in the new layout (no file is made)."""
    return resume_layout(resume, design_style(paper))[0]


def _only_mentions(resume: Resume, analysis: JDAnalysis | None) -> set[tuple[int, int]]:
    """(job position, bullet index) of each bullet that holds the only mention of a must-have in the summary and the
    bullets: removing it would leave the posting's requirement unshown."""
    from .validate import mentions

    must, _other = _terms(analysis)
    held: set[tuple[int, int]] = set()
    for term in must:
        if mentions(resume.summary, term):
            continue
        places = [(i, j) for i, exp in enumerate(resume.experience) for j, b in enumerate(exp.bullets)
                  if mentions(b, term)]
        if len(places) == 1:
            held.add(places[0])
    return held


def _language_places(resume: Resume, analysis: JDAnalysis | None, jd: str = "") -> tuple[int, set[int]] | None:
    """(the current job's position, its bullets that each name the post's programming language), read by
    validate.missing_language itself with the post's text `jd`, as the fix round and the final check read it: the
    owner's rule since prompt v39, which trimming must not undo. None when the rule does not apply (the current job
    passes the check without any bullet: work not built in code whose post names no language)."""
    from .validate import current_job, missing_language

    job = current_job(resume)
    if job is None:
        return None

    def missing(bullets: list[str]) -> bool:  # the check, with only these bullets at the current job
        trial = [exp.model_copy(update={"bullets": bullets}) if exp is job else exp for exp in resume.experience]
        return missing_language(resume.model_copy(update={"experience": trial}), analysis, jd) is not None

    if not missing([]):
        return None
    position = next(i for i, exp in enumerate(resume.experience) if exp is job)
    return position, {j for j, bullet in enumerate(job.bullets) if not missing([bullet])}


def removal_order(resume: Resume, analysis: JDAnalysis | None = None, relevance: list[float] | None = None,
                  jd: str = "") -> list[tuple[int, int]]:
    """(job position, bullet index) in the order trim_to_page removes bullets, without any layout: from the most recent
    job only while it holds more than half of the bullets left, or more than 8 (the most any job gets: on two pages
    half of them can be more), and keeps the most after it ([6, 2, 1, 1, 1] -> [5, 2, 1, 1, 1]; [11, 3, 2, ...] -> [8,
    3, 2, ...]), else from the older job with the most bullets, the less relevant one and then the older one first
    ([5, 2, 1, 1, 1] -> [5, 1, 1, 1, 1]); within a job its last bullet. A bullet holding a must-have's only mention goes
    only when no other bullet can (so the most recent job gives one of its others first, even at half), or when only
    the most recent job's are left and taking one would end its lead; so does the current job's last bullet still
    naming the post's programming language (_language_places, with the post's text `jd`), counting its first bullet.
    Every job keeps its first bullet."""
    relevance = relevance if relevance is not None else draft_relevance(resume, analysis)
    held = _only_mentions(resume, analysis)
    language = _language_places(resume, analysis, jd)
    alive = [list(range(len(exp.bullets))) for exp in resume.experience]
    order: list[tuple[int, int]] = []

    def removable(position: int, protect: bool) -> list[int]:
        rest = alive[position][1:]  # never the first bullet: the job's strongest
        if not protect:
            return rest
        kept = [j for j in rest if (position, j) not in held]
        if language is not None and position == language[0]:
            naming = [j for j in alive[position] if j in language[1]]  # with its first bullet, which never goes
            if len(naming) == 1:  # the last one left that names the post's language
                kept = [j for j in kept if j != naming[0]]
        return kept

    def weight(i: int) -> float:
        return relevance[i] if i < len(relevance) else 0.0

    for protect in (True, False):
        while True:
            jobs = [i for i in range(len(alive)) if removable(i, protect)]
            if not jobs:
                break
            guarded = protect
            older = [i for i in jobs if i]
            lead_kept = len(alive[0]) - 1 > max((len(bullets) for bullets in alive[1:]), default=0)
            # more than half of the bullets left, or more than any job gets (RECENCY_CAPS: 8, a two-page plan's cap)
            over = 2 * len(alive[0]) > sum(map(len, alive)) or len(alive[0]) > RECENCY_CAPS[0]
            if 0 in jobs and over and lead_kept:
                jobs = [0]  # the most recent job gives
            elif older:
                jobs = older
            elif protect and not lead_kept:
                # only the most recent job could give, and that would end its lead: first an older job's bullet
                # holding a must-have's only mention
                held_older = [i for i in range(1, len(alive)) if removable(i, False)]
                if held_older:
                    jobs, guarded = held_older, False
            position = max(jobs, key=lambda i: (len(alive[i]) - weight(i), i))
            index = removable(position, guarded)[-1]
            alive[position].remove(index)
            order.append((position, index))
    return order


def _without(resume: Resume, order: list[tuple[int, int]], k: int) -> Resume:
    work = resume.model_copy(deep=True)
    gone: dict[int, set[int]] = {}
    for position, index in order[:k]:
        gone.setdefault(position, set()).add(index)
    for position, indexes in gone.items():
        exp = work.experience[position]
        exp.bullets = [b for j, b in enumerate(exp.bullets) if j not in indexes]
    return work


@dataclass
class Trim:
    resume: Resume
    pages: int
    removed: list[str] = field(default_factory=list)  # 'Company: bullet' for each bullet removed, in order


def trim_to_page(resume: Resume, paper: str, target: int, analysis: JDAnalysis | None = None,
                 relevance: list[float] | None = None, at: list[FitStyle] | None = None, jd: str = "") -> Trim:
    """The fewest bullets removed (removal_order, `jd` the post's text) for the resume to fit `target` pages at the
    design's own sizes (or at whichever of the styles `at` takes the fewest pages), found by halving (a few layouts of
    ~3 ms); with every removable bullet gone when even that does not fit. Works on a copy."""
    order = removal_order(resume, analysis, relevance, jd)
    seen: dict[int, int] = {}

    def pages(k: int) -> int:
        if k not in seen:
            work = _without(resume, order, k)
            seen[k] = measure(work, paper) if at is None else min(resume_layout(work, style)[0] for style in at)
        return seen[k]

    if pages(0) <= target:
        return Trim(resume.model_copy(deep=True), seen[0])
    if pages(len(order)) > target:
        k = len(order)
    else:
        low, high = 0, len(order)  # pages(low) > target, pages(high) <= target: fewer removals never fit better
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (low, middle) if pages(middle) <= target else (middle, high)
        k = high
    names = resume.profile.companies
    removed = [f"{names[resume.experience[p].company_index].name}: {resume.experience[p].bullets[j]}"
               for p, j in order[:k]]
    return Trim(_without(resume, order, k), pages(k), removed)


def pages_word(pages: int) -> str:
    return {1: "one page", 2: "two pages"}.get(pages, f"{pages} pages")


def removed_note(removed: list[str], pages: int) -> str:
    shown = " / ".join(r[:60] + "..." for r in removed[:MAX_REMOVED_IN_NOTE])
    more = f" and {len(removed) - MAX_REMOVED_IN_NOTE} more" if len(removed) > MAX_REMOVED_IN_NOTE else ""
    return f"Removed {len(removed)} bullet(s) to fit {pages_word(pages)}: {shown}{more}"


def refitted_notes(notes: list[str], fit: list[str], restored: bool = False) -> list[str]:
    """A run's notes after it is fitted again (a hand edit, a certification tick), each note once: the earlier fit's
    notes on the layout (FIT_NOTE: two pages kept, the font reduced, still over) give way to the new fit's `fit`. A note
    naming removed bullets (REMOVED_NOTE: when the run was made, or by a tick) stays while they are off the resume; only
    `restored` (a tick that started again from the bullets an earlier tick removed, TailorResult.untrimmed) drops the
    last such note, that tick's, with any copy of it."""
    removed = [note for note in notes if REMOVED_NOTE.match(note)]
    back = removed[-1] if restored and removed else None
    return list(dict.fromkeys([note for note in notes if note != back and not FIT_NOTE.match(note)] + fit))


def shortest_over_two(resume: Resume, paper: str = "Letter") -> bool:
    """Whether the text is over two pages at the design's own sizes even with every removable bullet gone (each job
    with only its first): only then does fit_new lay it out on the size ladder."""
    order = removal_order(resume)
    return measure(_without(resume, order, len(order)), paper) > TWO_PAGES


def on_the_ladder(resume: Resume, paper: str, document: str = "Resume") -> tuple[FitStyle, int, bytes, list[str]]:
    """Text over two pages at the design's own sizes, as it is, on the old size ladder (fit.py): the largest step that
    fits two pages (try_styles), else the smallest step (render_smallest); a note says the font was reduced (and to fit
    how many pages, only when it does)."""
    found = try_styles(resume, paper, document, layout=LAYOUT)
    if found:
        small, pages, pdf, notes = found
    else:
        small, pages, pdf = render_smallest(resume, paper, document, layout=LAYOUT)
        notes = []
    if not any(note.startswith("Font reduced") for note in notes) and small != design_style(paper):
        to_fit = f" to fit {pages_word(pages)}" if pages <= TWO_PAGES else ""
        notes.append(f"Font reduced to {small.font_pt:g} pt{to_fit}.")
    return small, pages, pdf, notes


@dataclass
class Fitted:
    resume: Resume
    style: FitStyle
    pages: int
    notes: list[str] = field(default_factory=list)
    removed: int = 0  # bullets removed
    pdf: bytes | None = None  # made only by the size ladder (the last resort); else made by the caller


def fit_new(resume: Resume, paper: str, target: int, analysis: JDAnalysis | None = None,
            document: str = "Resume", jd: str = "") -> Fitted:
    """A new run's resume on `target` pages (plan_pages) at the design's own sizes: as it is, or with the fewest
    bullets removed (trim_to_page; `analysis` and the post's text `jd` keep the must-haves and the post's language);
    when one page cannot be reached even then, two pages from the untrimmed draft; only when two pages cannot be
    reached at the design's sizes either, the old size ladder (on_the_ladder, its note says the font was reduced) on
    the untrimmed draft when one of its steps fits it on two pages, else with the fewest bullets removed for two pages
    at its smallest steps, else on the shortest version. Never changes `resume`."""
    style = design_style(paper)
    relevance = draft_relevance(resume, analysis)
    first = trim_to_page(resume, paper, target, analysis, relevance, jd=jd)
    if first.pages <= target:
        notes = [removed_note(first.removed, target)] if first.removed else []
        return Fitted(first.resume, style, first.pages, notes, len(first.removed))
    if target < TWO_PAGES:
        two = trim_to_page(resume, paper, TWO_PAGES, analysis, relevance, jd=jd)
        if two.pages <= TWO_PAGES:
            notes = [("Kept two pages: even one bullet for each older job does not fit on one page at the design's "
                      "own sizes.")]
            if two.removed:
                notes.append(removed_note(two.removed, TWO_PAGES))
            return Fitted(two.resume, style, two.pages, notes, len(two.removed))
    # the size ladder: the untrimmed draft when its smallest steps fit it on two pages (then try_styles finds the
    # largest step that does), else trimmed only as far as they need, else the shortest version (every removable bullet
    # gone) at the smallest step
    kept = trim_to_page(resume, paper, TWO_PAGES, analysis, relevance, styles(paper, LAYOUT)[len(LADDER):], jd=jd)
    small, pages, pdf, notes = on_the_ladder(kept.resume, paper, document)
    if kept.removed:
        notes.append(removed_note(kept.removed, min(pages, TWO_PAGES)))
    if pages > TWO_PAGES:
        notes.append(f"Still more than {pages_word(TWO_PAGES)} after trimming. Shorten the text in the editor.")
    return Fitted(kept.resume, small, pages, notes, len(kept.removed), pdf)
