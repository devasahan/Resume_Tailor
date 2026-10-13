"""Rule-based extraction of the fixed facts from resume text.

Used when no real model is configured for parsing (and by the offline demo models). It is
best-effort: the Profile page always shows the result for the user to correct before saving.
"""
from __future__ import annotations

import re

from ..schemas import Company, Contact, Education, Profile

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[\w\-/%.]+", re.IGNORECASE)
LOCATION_RE = re.compile(r"\b([A-Z][A-Za-z.' -]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)?))\b")

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_POINT = rf"(?:{_MONTH}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
DATE_RANGE_RE = re.compile(
    rf"({_POINT})\s*(?:-|–|—|to)\s*({_POINT}|Present|Current|Now|Today)", re.IGNORECASE
)
POINT_RE = re.compile(rf"(?<![\w/]){_POINT}(?![\w/])", re.IGNORECASE)
_PRESENT = ("present", "current", "now", "today")
MODE_RE = re.compile(r"\b(Remote|On-?site|Hybrid)\b", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"\s*[|•·—–]\s*|\s+-\s+")

EXPERIENCE_HEAD = re.compile(
    r"^(?:work |professional |employment )?(?:experience|history)|^employment", re.IGNORECASE
)
EDUCATION_HEAD = re.compile(r"^education", re.IGNORECASE)
OTHER_HEAD = re.compile(
    r"^(skills|summary|profile|objective|projects|certifications?|awards|languages|"
    r"publications|references|interests|technical skills|core competencies)\b",
    re.IGNORECASE,
)
SCHOOL_RE = re.compile(r"\b(University|College|Institute|School|Academy|Polytechnic)\b", re.IGNORECASE)
DEGREE_RE = re.compile(
    r"\b(Bachelor|Master|B\.?\s?S\.?c?|M\.?\s?S\.?c?|B\.?A\.?|M\.?A\.?|MBA|Ph\.?D|Associate|Diploma|Doctorate)\b",
    re.IGNORECASE,
)


def _normalize_mode(text: str) -> str:
    low = text.lower().replace("-", "")
    return {"remote": "Remote", "onsite": "Onsite", "hybrid": "Hybrid"}.get(low, "")


def _is_header(line: str) -> str | None:
    bare = line.strip(" :").strip()
    if len(bare) > 40:
        return None
    if EXPERIENCE_HEAD.match(bare):
        return "experience"
    if EDUCATION_HEAD.match(bare):
        return "education"
    if OTHER_HEAD.match(bare):
        return "other"
    return None


def _split_sections(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    head: list[str] = []
    sections: dict[str, list[str]] = {"experience": [], "education": [], "other": []}
    current: str | None = None
    for line in lines:
        kind = _is_header(line)
        if kind:
            current = kind
            continue
        (sections[current] if current else head).append(line)
    return head, sections


def _parse_company_line(line: str, previous: str | None) -> Company | None:
    match = DATE_RANGE_RE.search(line)
    if not match:
        return None
    remainder = (line[: match.start()] + " " + line[match.end() :]).strip(" ,;|-–—•·()")
    if not remainder and previous:  # the date sits on its own line under the company name
        remainder = previous
    mode = ""
    location = ""
    kept: list[str] = []
    for part in (p.strip(" ,()") for p in SEPARATOR_RE.split(remainder)):
        if not part:
            continue
        found = MODE_RE.search(part)
        if found:
            mode = _normalize_mode(found.group(1))
            part = re.sub(r"\s*\(?\b(?:Remote|On-?site|Hybrid)\b\)?", "", part, flags=re.IGNORECASE).strip(" ,-()")
            if not part:
                continue
        if LOCATION_RE.fullmatch(part) and kept:
            location = part
        else:
            kept.append(part)
    if not kept:
        return None
    end = match.group(2).strip()
    return Company(
        name=kept[0],
        start=match.group(1).strip(),
        end="Present" if end.lower() in ("present", "current", "now", "today") else end,
        mode=mode,
        location=location,
    )


def _take_dates(line: str) -> tuple[str, str, str]:
    """(the line without its dates, start, end), months kept: 'Fontbonne University Aug 2016 – Aug 2020' ->
    ('Fontbonne University', 'Aug 2016', 'Aug 2020'). One date is the end ('May 2020', graduation)."""
    match = DATE_RANGE_RE.search(line)
    if match:
        end = match.group(2).strip()
        rest = line[: match.start()] + " " + line[match.end() :]
        return rest.strip(" ,;|-–—•·()"), match.group(1).strip(), "Present" if end.lower() in _PRESENT else end
    points = [m.group(0).strip() for m in POINT_RE.finditer(line)]
    if not points:
        return line, "", ""
    rest = POINT_RE.sub(" ", line)
    return rest.strip(" ,;|-–—•·()"), points[0] if len(points) > 1 else "", points[-1]


def _parse_education(lines: list[str]) -> list[Education]:
    entries: list[Education] = []
    for line in lines:
        text, start, end = _take_dates(line)
        if SCHOOL_RE.search(text):
            school = SEPARATOR_RE.split(text)[0].strip(" ,-–—")
            entries.append(Education(school=school or text, start=start, end=end))
        elif DEGREE_RE.search(text):
            if not entries:
                entries.append(Education(school=""))
            degree_line = re.sub(r"\s{2,}", " ", text).strip(" ,-–—")
            entry = entries[-1]
            if not entry.degree:
                if " in " in degree_line:
                    degree, _, fld = degree_line.partition(" in ")
                    entry.degree, entry.field = degree.strip(" ,"), fld.strip(" ,")
                else:
                    entry.degree = degree_line
            if end and not entry.end:
                entry.start, entry.end = start, end
        elif end and not text and entries and not entries[-1].end:  # the dates on their own line under the school
            entries[-1].start, entries[-1].end = start, end
    return [e for e in entries if e.school or e.degree]


def parse_resume_text(text: str) -> Profile:
    lines = [ln.strip() for ln in text.replace("\r", "").splitlines() if ln.strip()]
    head, sections = _split_sections(lines)
    top = head[:10] if head else lines[:10]

    email_match = EMAIL_RE.search(text)
    email = email_match.group(0) if email_match else ""
    phone_match = PHONE_RE.search(" ".join(top)) or PHONE_RE.search(text)
    phone = phone_match.group(0).strip() if phone_match else ""
    linkedin_match = LINKEDIN_RE.search(text)
    linkedin = linkedin_match.group(0) if linkedin_match else ""

    name = ""
    for line in top:
        if EMAIL_RE.search(line) or PHONE_RE.search(line) or "linkedin" in line.lower():
            continue
        if 1 <= len(line.split()) <= 5 and not any(ch.isdigit() for ch in line):
            name = line.strip()
            break

    location = ""
    for line in top:
        cleaned = EMAIL_RE.sub("", PHONE_RE.sub("", LINKEDIN_RE.sub("", line)))
        for chunk in SEPARATOR_RE.split(cleaned):
            chunk = chunk.strip(" ,")
            if chunk != name and LOCATION_RE.fullmatch(chunk):
                location = chunk
                break
        if location:
            break

    companies: list[Company] = []
    previous: str | None = None
    for line in sections["experience"]:
        company = _parse_company_line(line, previous)
        if company:
            companies.append(company)
            previous = None
        elif not MODE_RE.fullmatch(line.strip()):
            previous = line.strip(" ,|-–—•·")

    return Profile(
        name=name,
        contact=Contact(email=email, phone=phone, linkedin=linkedin, location=location),
        companies=companies,
        education=_parse_education(sections["education"]),
    )
