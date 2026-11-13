"""Facts computed from the locked profile, given to the model so it never has to guess them.

Years of experience come from the employment dates: overlapping jobs are counted once, a current
role runs to today, and the total is rounded DOWN ("7+ years" for 7 years 10 months).
"""
from __future__ import annotations

import re
from datetime import date

from ..schemas import Company, Profile

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
CURRENT = {"", "present", "current", "now", "today"}


def parse_point(text: str, is_end: bool, today: date | None = None) -> tuple[int, int] | None:
    """'Jan 2021', 'January 2021', '01/2021', '2021-01', '2021', 'Present' -> (year, month)."""
    today = today or date.today()
    value = text.strip().lower().rstrip(".")
    if value in CURRENT:
        return (today.year, today.month) if is_end else None
    match = re.fullmatch(r"([a-z]+)\.?,?\s+(\d{4})", value)
    if match:
        name = match.group(1)
        month = MONTHS.get(name[:4]) if name[:4] in MONTHS else MONTHS.get(name[:3])
        if month:
            return int(match.group(2)), month
    match = re.fullmatch(r"(\d{1,2})[/.-](\d{4})", value)
    if match and 1 <= int(match.group(1)) <= 12:
        return int(match.group(2)), int(match.group(1))
    match = re.fullmatch(r"(\d{4})[/.-](\d{1,2})", value)
    if match and 1 <= int(match.group(2)) <= 12:
        return int(match.group(1)), int(match.group(2))
    match = re.fullmatch(r"(\d{4})", value)
    if match:
        return int(match.group(1)), (12 if is_end else 1)
    return None


def is_expired(expires: str, today: date | None = None) -> bool:
    """True when this expiry date ('Mar 2026', '03/2026', '2026': the end of that year) is before this month. An empty
    or unreadable one ('No Expiration') is not expired."""
    today = today or date.today()
    point = parse_point(expires, is_end=True, today=today) if expires and expires.strip() else None
    return point is not None and point < (today.year, today.month)


def _index(point: tuple[int, int]) -> int:
    return point[0] * 12 + (point[1] - 1)


def stay_months(company: Company, today: date | None = None) -> int | None:
    """Months at one employer (end month included), or None when the dates cannot be read."""
    start = parse_point(company.start, is_end=False, today=today)
    end = parse_point(company.end, is_end=True, today=today)
    if not start or not end or _index(end) < _index(start):
        return None
    return _index(end) - _index(start) + 1


RECENCY_CAPS = (8, 6, 5, 4)  # the most recent employer first; 3 for every older one


def stay_caps(companies: list[Company], today: date | None = None) -> list[int]:
    """Most bullets each employer's own stay deserves: 8 / 6 / 5 / 4 / 3 by recency, and by the length of the stay:
    2 under 6 months, 3 under a year, 5 under two years. A short career is not padded out."""
    caps = []
    for index, company in enumerate(companies):
        cap = RECENCY_CAPS[index] if index < len(RECENCY_CAPS) else 3
        months = stay_months(company, today)
        if months is not None:
            cap = min(cap, 2 if months < 6 else 3 if months < 12 else 5 if months < 24 else cap)
        caps.append(cap)
    return caps


def bullet_caps(companies: list[Company], today: date | None = None) -> list[int]:
    """Most bullets each employer deserves on a resume of up to two pages: its stay_caps, except that the most recent
    employer (the first) is always the richest: one bullet more than any other (8 at most), however short its stay;
    a career of one job keeps its own cap."""
    caps = stay_caps(companies, today)
    if len(caps) > 1:
        caps[0] = min(RECENCY_CAPS[0], max(caps[0], max(caps[1:]) + 1))
    return caps


def _span(company: Company, today: date | None = None) -> tuple[int, int] | None:
    """(first month, month after the last) worked at one employer, or None when the dates cannot be read."""
    start = parse_point(company.start, is_end=False, today=today)
    end = parse_point(company.end, is_end=True, today=today)
    if start and end and _index(end) >= _index(start):
        return _index(start), _index(end) + 1  # end month counts as worked
    return None


def _merged_months(spans: list[tuple[int, int]]) -> int | None:
    """Months covered by the spans, overlaps counted once. None for no spans."""
    if not spans:
        return None
    spans = sorted(spans)
    total, current_start, current_end = 0, *spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + (current_end - current_start)


def experience_months(companies: list[Company], today: date | None = None) -> int | None:
    """Total months worked, counting overlaps once. None when no dates can be read."""
    return _merged_months([s for s in (_span(c, today) for c in companies) if s])


def experience_before(companies: list[Company], index: int, today: date | None = None) -> int | None:
    """Months worked, at any employer, before the job at `index` started (overlaps counted once).

    None when that job's start date is unreadable; 0 when nothing came before it.
    """
    own = _span(companies[index], today)
    if own is None:
        return None
    clipped = []
    for other, company in enumerate(companies):
        span = _span(company, today) if other != index else None
        if span and span[0] < own[0]:
            clipped.append((span[0], min(span[1], own[0])))
    return _merged_months(clipped) or 0


def months_phrase(months: int | None) -> str:
    """'2 yrs 3 mo', '1 yr', '7 mo'; '' when unknown."""
    if months is None:
        return ""
    years, rest = divmod(months, 12)
    parts = []
    if years:
        parts.append(f"{years} yr" + ("s" if years > 1 else ""))
    if rest or not years:
        parts.append(f"{rest} mo")
    return " ".join(parts)


def years_phrase(months: int | None) -> str:
    """'7+ years' (rounded down), or '' when under a year or unknown."""
    if not months or months < 12:
        return ""
    return f"{months // 12}+ years"


def candidate_facts(profile: Profile, today: date | None = None) -> str:
    """Text for the <candidate_facts> block of the prompt (also used to accept these figures in checks)."""
    years = years_phrase(experience_months(profile.companies, today))
    if years:
        return f"Total professional experience: {years} (computed from the employment dates; use exactly this figure)."
    return "Total professional experience: unknown (do not state a number of years)."
