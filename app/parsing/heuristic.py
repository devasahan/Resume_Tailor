"""Rule-based extraction of the fixed facts from resume text.

Used when no real model is configured for parsing (and by the offline demo models). It is
best-effort: the Profile page always shows the result for the user to correct before saving.
"""
from __future__ import annotations

import re

from ..pipeline.facts import parse_point
from ..pipeline.validate import same_certification
from ..schemas import Certification, Company, Contact, Education, Profile

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
    r"^(skills|summary|profile|objective|projects|awards|languages|"
    r"publications|references|interests|technical skills|core competencies)\b",
    re.IGNORECASE,
)
# Sections that often follow the certifications, so they end that section (only that one: inside a job,
# "Achievements:" is a sub-heading, never the end of Experience). The whole line must match.
OTHER_FULL = re.compile(
    r"^(?:honou?rs(?:\s+(?:&|and)\s+awards)?|awards\s+(?:&|and)\s+honou?rs|achievements|accomplishments|"
    r"(?:extracurricular\s+)?activities|affiliations|(?:professional\s+)?memberships|courses|coursework|training|"
    r"professional\s+development|volunteer(?:ing)?(?:\s+(?:experience|work))?|leadership(?:\s+experience)?|"
    r"additional\s+information|hobbies|patents|presentations|military\s+service|"
    r"community\s+(?:involvement|service))$",
    re.IGNORECASE,
)
_CERTS = r"(?:(?:professional\s+)?certifications?|certificates?|credentials|licen[cs]es?)"
_CERT_PAIR = rf"{_CERTS}(?:\s*(?:&|and)\s*{_CERTS})?"
CERT_HEAD = re.compile(rf"^{_CERT_PAIR}$", re.IGNORECASE)  # the whole line: "Certificate in Data Analytics" is none
# 'Certifications & Training', 'Certifications / Licenses', 'Certifications (Selected)', 'Certifications and Awards'
CERT_HEAD_JOINED = re.compile(rf"^{_CERTS}\s*(?:[&/,(+]|and\b)", re.IGNORECASE)
# "Certifications: AWS SAA, CKA" (not "Certificates:", which in a skills list means TLS certificates)
CERT_INLINE = re.compile(
    rf"^(?:(?:professional\s+)?certifications?|credentials|licen[cs]es?)(?:\s*(?:&|and)\s*{_CERTS})?\s*:\s*(\S.*)$",
    re.IGNORECASE)
CERT_WORD = re.compile(r"\bcertif|\blicen[cs]e|\bcredential", re.IGNORECASE)
# A line that reads as a credential, in a section that mixes certifications with something else
_CREDENTIAL_LINE = re.compile(
    r"\b(?:certif\w*|licen[cs]\w*|credential|associate|professional|expert|practitioner|specialist|fundamentals|"
    r"foundations?|administrator|architect|developer|engineer|master|belt|auditor)\b", re.IGNORECASE)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9+]{1,6}\b")
# Only these words make a numbered line between a table's rows a certification, not a role title
_TITLE_IS_CREDENTIAL = re.compile(
    r"\b(?:certif\w*|licen[cs]\w*|credential|associate|professional|expert|practitioner|specialist|fundamentals|"
    r"foundations?)\b", re.IGNORECASE)
# A certification's dates: the date it was earned, after a separator or a word such as "Issued" ("Mar 2023", "2021"),
# and the end of a range, the date it expires ("- Mar 2026"; "- Present" or "No Expiration" is none). A number inside
# a name ("Windows Server 2016", "ISO 9001") is not a date.
_CERT_YEAR = rf"(?:{_MONTH}\s+)?(?:19|20)\d{{2}}"
_RANGE_END = rf"(?:\s*(?:-|–|—|to)\s*({_CERT_YEAR}|present|current|no\s+expiration(?:\s+date)?))?"
CERT_DATE_RE = re.compile(
    rf"(?:\s*[,|(]|\s+[-–—]|\s*\b(?:issued|earned|obtained|awarded|completed|since)\b:?)\s*({_CERT_YEAR})"
    rf"{_RANGE_END}\s*\)?",
    re.IGNORECASE,
)
# The date it expires, said in words: 'Expires Jan 2026', 'expiry: 2026', 'valid until Dec 2025', '(exp. 2027)'
_EXPIRES_RE = re.compile(
    rf"(?:\s*[,|(·]\s*|\s+)?\b(?:expires?|expiry|expiration(?:\s+date)?|exp\.|valid\s+(?:until|through|thru|till))"
    rf"(?:\s+on)?\s*:?\s*({_CERT_YEAR})\s*\)?",
    re.IGNORECASE,
)


def _expiry(end: str | None) -> str:
    """The end of a date range, or an Expires cell, as an expiry date: a date ('Mar 2026', '03/2026', '2026'), never
    'Present' or 'No Expiration'."""
    end = (end or "").strip()
    if not end or end.lower() in ("present", "current", "now", "today"):
        return ""
    return end if parse_point(end, is_end=True) is not None else ""
_ISSUED_LINE = re.compile(r"^(?:issued|earned|obtained)\b", re.IGNORECASE)
# A date that comes first: '2021 | AWS Certified Developer', 'Mar 2023 - Mar 2026 · Certified Kubernetes Administrator'
_LEAD_DATE = re.compile(
    rf"^\(?({_CERT_YEAR})\)?{_RANGE_END}\s*(?:[|,·:]|\s[-–—])\s*",
    re.IGNORECASE)
_SKIP_CERT_LINE = re.compile(
    r"^(?:expired|no\s+expiration|credential\s+(?:id|url)|(?:see|show)\s+credential)\b",
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
    if CERT_HEAD.match(bare):
        return "certifications"
    if CERT_HEAD_JOINED.match(bare):  # 'Certifications & Training': each line goes where it belongs
        return "education_and_certifications" if re.search(r"\beducation\b", bare, re.IGNORECASE) \
            else "certifications_and_more"
    if EXPERIENCE_HEAD.match(bare):
        return "experience"
    if EDUCATION_HEAD.match(bare):  # "Education & Certifications": each line goes where it belongs
        return "education_and_certifications" if CERT_WORD.search(bare) else "education"
    if OTHER_HEAD.match(bare):
        return "other"
    return None


def _reads_as_credential(line: str) -> bool:
    return bool(_CREDENTIAL_LINE.search(line) or _ACRONYM.search(line))


def _split_sections(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    head: list[str] = []
    sections: dict[str, list[str]] = {"experience": [], "education": [], "certifications": [], "other": []}
    current: str | None = None
    for line in lines:
        inline = CERT_INLINE.match(line.strip()) if current != "experience" else None
        if inline:  # a list on the heading's own line belongs to it; the section around it goes on
            sections["certifications"] += [part for part in re.split(r"\s*[,;]\s*", inline.group(1)) if part]
            continue
        kind = _is_header(line)
        if kind is None and current in ("certifications", "certifications_and_more", "education_and_certifications") \
                and OTHER_FULL.match(line.strip(" :").strip()):
            kind = "other"  # the section after the certifications
        if kind:
            current = kind
            continue
        if current == "education_and_certifications":
            sections["certifications" if CERT_WORD.search(line) else "education"].append(line)
        elif current == "certifications_and_more":
            sections["certifications" if _reads_as_credential(line) else "other"].append(line)
        else:
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


# '• AWS', '●AWS', '- AWS' (and Word's own bullet, U+F0B7); never the '-' of a name
_LIST_MARK = re.compile("^(?:[•·▪◦‣*>●○■□◆◇♦❖➢➤►▸✓✔\uf0b7\uf0a7]+|[-–—]+(?=\\s))\\s*")


def _balanced(part: str) -> str:
    """A piece without a stray bracket left by the split: '(CKA)' stays, 'Foundation)' loses its ')'."""
    while part.endswith(")") and part.count(")") > part.count("("):
        part = part[:-1].rstrip()
    while part.startswith("(") and part.count("(") > part.count(")"):
        part = part[1:].lstrip()
    return part


def _parse_certifications(lines: list[str]) -> list[Certification]:
    """One certification per line: 'Name, Issuer, 2023', 'Name | Issuer | Mar 2023 - Mar 2026', '• Name (2021,
    expires 2024)'. A name is never split at a dash ('AWS Certified Solutions Architect – Professional'). LinkedIn's own
    lines ('Issued Jan 2023 · Expires Jan 2026', 'Expires Jan 2026', 'Credential ID ...') are not certifications: their
    dates are the one above's."""
    certifications: list[Certification] = []
    for raw in lines:
        line = _LIST_MARK.sub("", raw.strip())
        if not line or _SKIP_CERT_LINE.match(line):
            continue
        expiring = _EXPIRES_RE.match(line)
        if expiring and not line[expiring.end():].strip(" ·|,"):  # a line of its own: 'Expires Jan 2026'
            if certifications and not certifications[-1].expires:
                certifications[-1].expires = expiring.group(1)
            continue
        if _ISSUED_LINE.match(line):
            dated = CERT_DATE_RE.search(" " + line)
            expiring = _EXPIRES_RE.search(line)
            if (len(certifications) >= 2 and not certifications[-1].issuer and not certifications[-1].year
                    and not certifications[-2].issuer and not _CREDENTIAL_LINE.search(certifications[-1].name)):
                issuer = certifications.pop().name  # LinkedIn: name / issuer / Issued ...
                certifications[-1].issuer = issuer
            if dated and certifications and not certifications[-1].year:
                certifications[-1].year = dated.group(1)
            if certifications and not certifications[-1].expires:
                certifications[-1].expires = (expiring.group(1) if expiring else "") or _expiry(
                    dated.group(2) if dated else None)
            continue
        year = expires = ""
        expiring = _EXPIRES_RE.search(line)
        if expiring:  # 'Name, Issuer, 2023, expires 2026'
            expires, line = expiring.group(1), (line[: expiring.start()] + line[expiring.end():]).strip()
        lead = _LEAD_DATE.match(line)
        if lead:
            year, expires, line = lead.group(1), expires or _expiry(lead.group(2)), line[lead.end():]
        line = re.sub(r"\s*\(([^()]*,[^()]*)\)\s*$", r", \1", line)  # 'Name (Issuer, 2020)' -> fields
        dated = CERT_DATE_RE.search(" " + line)
        if dated:
            year, expires = year or dated.group(1), expires or _expiry(dated.group(2))
            spaced = " " + line
            line = (spaced[: dated.start()] + spaced[dated.end():]).strip()
        # ' | ' is the box's own separator, so a name may keep a comma ('Oracle Certified Professional, Java SE 11')
        pieces = re.split(r"\s+\|\s+", line) if " | " in line else re.split(r"\s*[|·]\s*|,\s+", line)
        parts = [_balanced(part.strip(" ,;|")) for part in pieces]
        parts = [part for part in parts if part]
        if not parts or not re.search(r"[A-Za-z]", parts[0]):
            continue
        name, issuer = parts[0], parts[1] if len(parts) > 1 else ""
        if any(same_certification(name, c.name) for c in certifications):
            continue
        certifications.append(Certification(name=name, issuer=issuer, year=year, expires=expires))
    return certifications


MAX_OWN_CERTIFICATIONS = 100  # a pasted list longer than this is refused
_ROW_NUMBER = re.compile(r"\(?\d{1,3}[.)]?")  # a table's number cell: '1', '16.', '(3)'
_LEAD_NUMBER = re.compile(r"^\s*\(?\d{1,3}[.)]\s+")  # '1. AWS ...', '3) AWS ...'
_HEADER_CELLS = frozenset({"s.no.", "s.no", "no.", "no", "#", "certification", "certifications", "name",
                           "exam code", "code"})
# A table's header row names its columns: only these are read, so an exam code is never taken for an issuer
_COLUMNS = {
    "name": {"certification", "certifications", "certificate", "name", "credential", "license", "licence"},
    "issuer": {"issuer", "issued by", "organization", "organisation", "provider", "vendor", "authority", "body"},
    "year": {"year", "date", "issued", "earned", "obtained", "date earned", "issue date", "issued on"},
    "expires": {"expires", "expiry", "expiration", "expiration date", "expiry date", "expires on", "valid until",
                "valid through", "valid till"},
}


def _table_header(cells: list[str]) -> dict[int, str] | None:
    roles = {index: role for index, cell in enumerate(cells) for role, words in _COLUMNS.items()
             if cell.strip().lower().rstrip(":") in words}
    return roles if "name" in roles.values() else None


def read_certification_list(text: str) -> list[Certification]:
    """The certifications a person pastes on the Profile page: one per line ('Name', 'Name, Issuer, 2023',
    'Name | Issuer | Mar 2023'), or rows copied from a table, whose cells are split by tabs. A header row says which
    column is the name (and the issuer, the year earned and the expiry date); without one, a row's first cell is the
    name, its first date the year earned and a second date the expiry date. Number and exam-code columns and numbered titles between the rows ('2. Backend Engineer') are left
    out, and so is a bullet copied from Word ('•<tab>Name'). The same certification twice is kept once."""
    lines = [line for line in text.replace("\r", "").splitlines() if line.strip()]
    table = any("\t" in line for line in lines)
    header: dict[int, str] | None = None
    found: list[Certification] = []
    for line in lines:
        if "\t" in line:
            raw = [cell.strip() for cell in line.split("\t")]
            columns = _table_header(raw)
            if columns:
                header = columns
                continue
            if header:
                pick = {role: raw[index] for index, role in header.items() if index < len(raw)}
                name, issuer, year = pick.get("name", ""), pick.get("issuer", ""), pick.get("year", "")
                expires = _expiry(pick.get("expires", ""))
            else:
                cells = [c for c in raw if c and not _ROW_NUMBER.fullmatch(c) and _LIST_MARK.sub("", c).strip()]
                if not cells or cells[0].lower() in _HEADER_CELLS:
                    continue
                if len(cells) == 1:  # a bullet and a whole line ('•<tab>Name, Issuer, 2022'): read the line
                    found += _parse_certifications(cells)
                    continue
                name, issuer = cells[0], ""
                dates = [c for c in cells[1:] if re.fullmatch(_CERT_YEAR, c, re.IGNORECASE)]
                year, expires = (dates + ["", ""])[:2]
            name = _LIST_MARK.sub("", name).strip()
            if name and re.search(r"[A-Za-z]", name):
                found.append(Certification(name=name, issuer=issuer.strip(), year=year.strip(),
                                           expires=_expiry(expires)))
            continue
        if table and _LEAD_NUMBER.match(line) and not _TITLE_IS_CREDENTIAL.search(line):
            continue  # a numbered title between a table's rows ('2. Backend Engineer', '4. AI Engineer')
        line = _LEAD_NUMBER.sub("", line).strip()
        if line.lower().rstrip(":") in _HEADER_CELLS:
            continue
        found += _parse_certifications([line])
    unique: list[Certification] = []
    for cert in found:
        if not any(same_certification(cert.name, other.name) for other in unique):
            unique.append(cert)
    return unique


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
        certifications=_parse_certifications(sections["certifications"]),
    )
