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


def _index(point: tuple[int, int]) -> int:
    return point[0] * 12 + (point[1] - 1)


def stay_months(company: Company, today: date | None = None) -> int | None:
    """Months at one employer (end month included), or None when the dates cannot be read."""
    start = parse_point(company.start, is_end=False, today=today)
    end = parse_point(company.end, is_end=True, today=today)
    if not start or not end or _index(end) < _index(start):
        return None
    return _index(end) - _index(start) + 1


def bullet_caps(companies: list[Company], today: date | None = None) -> list[int]:
    """Most bullets each employer deserves: 5 / 4 / 3 by recency, 3 for a stay under a year, 2 under 6 months."""
    caps = []
    for index, company in enumerate(companies):
        cap = 5 if index == 0 else 4 if index == 1 else 3
        months = stay_months(company, today)
        if months is not None:
            cap = min(cap, 2 if months < 6 else 3 if months < 12 else cap)
        caps.append(cap)
    return caps


def experience_months(companies: list[Company], today: date | None = None) -> int | None:
    """Total months worked, counting overlaps once. None when no dates can be read."""
    spans: list[tuple[int, int]] = []
    for company in companies:
        start = parse_point(company.start, is_end=False, today=today)
        end = parse_point(company.end, is_end=True, today=today)
        if start and end and _index(end) >= _index(start):
            spans.append((_index(start), _index(end) + 1))  # end month counts as worked
    if not spans:
        return None
    spans.sort()
    total, current_start, current_end = 0, *spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + (current_end - current_start)


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
