"""Rule-based extraction of the fixed facts from resume text.

Used when no real model is configured for parsing (and by the offline demo models). It is
best-effort: the Profile page always shows the result for the user to correct before saving.
It also reads the app's own resumes back: job lines title first ("Data Engineer · Northwind Labs ·
Remote   Mar 2022 – Present") and, in resumes made before the restyle, company first ("Northwind
Labs | Data Engineer | Remote   Mar 2022 - Present").
"""
from __future__ import annotations

import re
from typing import NamedTuple

from ..pipeline.facts import parse_point
from ..pipeline.validate import same_certification
from ..schemas import Certification, Company, Contact, Education, Profile

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[\w\-/%.]+", re.IGNORECASE)
LOCATION_RE = re.compile(r"\b([A-Z][A-Za-z.' -]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)?))\b")
_TOWN_AND_STATE = re.compile(r",\s*[A-Z]{2}$")  # a place no company is named like: 'Austin, TX' (not 'Rivera, Quinn')

# A month, cut short or in full ('Sep', 'Sept.', 'September'): at most six letters after its first three, as a longer
# run makes the date patterns slow ('janjanjan...': their time grows with the square of the run)
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]{0,6}\.?"
_POINT = rf"(?:{_MONTH}\s+\d{{4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
DATE_RANGE_RE = re.compile(
    rf"({_POINT})\s*(?:-|–|—|to)\s*({_POINT}|Present|Current|Now|Today)", re.IGNORECASE
)
POINT_RE = re.compile(rf"(?<![\w/]){_POINT}(?![\w/])", re.IGNORECASE)
_PRESENT = ("present", "current", "now", "today")
MODE_RE = re.compile(r"\b(Remote|On-?site|Hybrid)\b", re.IGNORECASE)
_MODE_WORD = re.compile(r"\s*\(?\b(?:Remote|On-?site|Hybrid)\b\)?", re.IGNORECASE)  # to take it out of a part
SEPARATOR_RE = re.compile(r"\s*[|•·—–]\s*|\s+-\s+")
_DOT = re.compile(r"\s+·\s+")  # the app's own separator between a job line's parts: 'Title · Company · Place · Mode'
# A run of tabs (and the spaces around it), or of more than three spaces: one gap. Long runs of white space make some
# of the patterns below slow (their time grows with the square of the run).
_GAPS = re.compile(r"\s*\t\s*|\s{4,}")
# The gap a tab leaves between two items on a line, and a narrower one ('Austin,  TX' keeps its spaces)
_WIDE_GAP = re.compile(r"(?<!,)\s{3,}")
_GAP = re.compile(r"(?<!,)\s{2,}")

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
_CERT_DATE = rf"(?:{_CERT_YEAR}|\d{{1,2}}/(?:19|20)\d{{2}})"  # also '03/2026', as a pasted table may give it
_RANGE_END = rf"(?:\s*(?:-|–|—|to)\s*({_CERT_YEAR}|present|current|no\s+expiration(?:\s+date)?))?"
CERT_DATE_RE = re.compile(
    rf"(?:\s*[,|(]|\s+[-–—]|\s*\b(?:issued|earned|obtained|awarded|completed|since)\b:?)\s*({_CERT_YEAR})"
    rf"{_RANGE_END}\s*\)?",
    re.IGNORECASE,
)
# The date it expires, said in words: 'Expires Jan 2026', 'expiry: 2026', 'valid until Dec 2025', '(exp. 2027)'
_EXPIRES_RE = re.compile(
    rf"(?:\s*[,|(·]\s*|\s+)?\b(?:expires?|expiry|expiration(?:\s+date)?|exp\.|valid\s+(?:until|through|thru|till))"
    rf"(?:\s+on)?\s*:?\s*({_CERT_DATE})\s*\)?",
    re.IGNORECASE,
)
# A month by its name, in full or cut short ('Mar', 'March', 'Sept.'), never a word that starts like one ('Marketing')
_MONTH_NAME = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
               r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_ROW_DATE = rf"(?:(?:{_MONTH_NAME}\s+|\d{{1,2}}/)?(?:19|20)\d{{2}})"
# A row's own dates at its end, after the issuer: 'Name · Issuer   Mar 2023 – Mar 2026', 'Name | Issuer   2020' (the
# app's rows; see _row_dates for when they count)
_TRAILING_DATES = re.compile(
    rf"\s({_ROW_DATE})(?:\s*(?:-|–|—|to)\s*({_ROW_DATE}|present|current|no\s+expiration(?:\s+date)?))?\s*$",
    re.IGNORECASE,
)
_BARE_YEAR = re.compile(r"(?:19|20)\d{2}")
_DATE_WORD = re.compile(r"\b(?:issued|earned|obtained|awarded|completed|since)\b:?$", re.IGNORECASE)  # before a date
# The separators between a row's fields, with a space on each side: a dash or a dot inside a name never counts
_ROW_SEPARATOR = re.compile(r"\s[·|]\s")


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

# The word a job title ends with: 'Senior Data Engineer', 'Product Owner', 'Founder & CEO'
_ROLE_NOUNS = (
    "engineer|developer|programmer|architect|administrator|admin|analyst|scientist|researcher|designer|manager|"
    "director|lead|head|officer|executive|president|founder|owner|partner|consultant|specialist|strategist|"
    "coordinator|associate|assistant|technician|technologist|intern|trainee|apprentice|fellow|student|recruiter|"
    "sourcer|accountant|auditor|controller|bookkeeper|actuary|underwriter|economist|statistician|nurse|physician|"
    "therapist|pharmacist|counsel|counselor|counsellor|attorney|lawyer|paralegal|teacher|instructor|lecturer|"
    "professor|tutor|trainer|coach|educator|editor|writer|copywriter|journalist|producer|marketer|representative|rep|"
    "agent|advisor|adviser|supervisor|superintendent|planner|buyer|clerk|receptionist|secretary|operator|mechanic|"
    "electrician|inspector|investigator|contractor|freelancer|generalist|evangelist|advocate|liaison|facilitator|"
    "sre|sdet|dba|ceo|cto|cfo|coo|cio|ciso|cmo|cpo"
)
# ... then maybe a level ('II', '3', 'L5') and a note in brackets ('(Contract)', its ')' maybe cut off by the split)
TITLE_END_RE = re.compile(
    rf"\b(?:{_ROLE_NOUNS})(?:\s+(?:i{{1,3}}|iv|v|[1-5]|l\d))?(?:\s*\([^()]*\)?)?$", re.IGNORECASE
)
# A title that starts with its rank: 'Head of Data', 'Senior Director of Engineering', 'VP Sales', 'Chief of Staff'
TITLE_START_RE = re.compile(
    r"^(?:(?:senior|sr\.?|deputy|associate|assistant|interim|acting|executive)\s+)?"
    r"(?:(?:head|director|manager|vp|vice\s+president|chief|member|dean|professor)\s+of\b|"
    r"[sea]?vp\b|vice\s+president\b|chief\s+\w)",
    re.IGNORECASE,
)
# The one comma a title may hold, right after its rank: 'VP, Engineering', 'Senior Manager, Data Engineering'
_RANK_COMMA_RE = re.compile(r"^(?:senior\s+)?(?:[sea]?vp|vice\s+president|director|head|manager)\s*,\s", re.IGNORECASE)
# What only a company has: 'Inc.', 'LLC', 'Ltd', 'Corp', 'plc', 'GmbH', '& Co.', a web address ('lead.io'). Not
# '.net': 'ASP.NET Developer' is a title.
COMPANY_MARK_RE = re.compile(
    r"\b(?:inc|llc|ltd|limited|corp|corporation|plc|gmbh|llp|pvt|pty)\b|&\s*co\b|\bco\.(?=\s|$)|"
    r"\.(?:com|io|ai|org|co|dev|app)\b",
    re.IGNORECASE,
)
# A sub-heading inside a job, never its company (the whole line): 'Key Responsibilities', 'Achievements', 'Tech Stack'
_JOB_SUB_HEADING = re.compile(
    r"(?:key\s+|main\s+|major\s+|selected\s+|notable\s+)?(?:responsibilities|achievements|accomplishments|highlights|"
    r"projects|duties|results|impact|contributions|overview|summary|technologies|tools|tech(?:nology)?\s+stack|"
    r"environment|skills|clients)",
    re.IGNORECASE,
)
# Small words a name keeps in small letters: 'Bank of America', 'Procter & Gamble'
_NAME_LINKS = frozenset({"of", "and", "the", "for", "at", "in", "on", "de", "la", "le", "du", "da", "di", "der", "von",
                         "van", "y", "&", "+"})
# One date at the end of a job line, when only the start or the end is known: '... · Remote   Present'
_LONE_DATE = re.compile(rf"\s({_POINT}|Present|Current)\s*$", re.IGNORECASE)


def _normalize_mode(text: str) -> str:
    low = text.lower().replace("-", "")
    return {"remote": "Remote", "onsite": "Onsite", "hybrid": "Hybrid"}.get(low, "")


def _reads_as_title(part: str) -> bool:
    """True when a part of a job line reads as a job title: it ends with a role, maybe with a level ('Senior Data
    Engineer', 'Data Engineer II', 'Software Engineer (Contract)'), or starts with a rank ('Head of Data', 'VP Sales').
    Never for a company ('Acme Consulting LLC', 'lead.io', 'The Lakeside Economist': no title starts with 'The') or a
    part holding more than a title ('Graduate Analyst, Initech, Leeds', as a line split only by commas has it)."""
    text = part.strip(" ,;:")
    if not text or COMPANY_MARK_RE.search(text) or text.lower().startswith("the "):
        return False
    if "," in text:  # only a rank's own comma ('VP, Engineering')
        return text.count(",") == 1 and bool(_RANK_COMMA_RE.match(text))
    return bool(TITLE_START_RE.match(text) or TITLE_END_RE.search(text))


def _capitalized(text: str) -> bool:
    """True when each word has a capital letter or a digit, apart from small linking words: a name or a title, not a
    sentence ('Built the data platform')."""
    return all(word.lower() in _NAME_LINKS or any(ch.isupper() or ch.isdigit() for ch in word) for word in text.split())


def _reads_as_name(text: str) -> bool:
    """True when a line reads as a name ('Bluefin Commerce', 'Bank of America', 'eBay'), not a sentence: a few
    capitalized words."""
    return 1 <= len(text.split()) <= 8 and _capitalized(text)


def _is_header(line: str, title_check: bool = True) -> str | None:
    bare = line.strip(" :").strip()
    if len(bare) > 40:
        return None
    # A job title is no heading, alone or leading its job line ('Experience Designer', 'History Teacher · Lakeside
    # High   2019 – 2021'): the app's own job lines start with the title, and a dated ' · ' line is one whatever its
    # title says ('Education Aide · Contoso   2019 – 2021'). A word in capitals before other text is a side label run
    # into the first line beside it ('EXPERIENCE Senior Data Engineer'), so it still starts a section; one in title case
    # ('Experience Senior Data Engineer') does only without `title_check` (see parse_resume_text).
    if _DOT.search(bare) and (DATE_RANGE_RE.search(bare) or _LONE_DATE.search(bare)):
        return None
    side_label = bool(bare) and bare.split()[0].isupper() and not bare.isupper()
    # without `title_check`, only an Experience heading on a line with no dates may lead a title
    relaxed = not title_check and EXPERIENCE_HEAD.match(bare) and not DATE_RANGE_RE.search(bare)
    if not relaxed and not side_label and _reads_as_title(
            SEPARATOR_RE.split(DATE_RANGE_RE.sub(" ", bare), maxsplit=1)[0]):
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


def _split_sections(lines: list[str], title_check: bool = True) -> tuple[list[str], dict[str, list[str]]]:
    head: list[str] = []
    sections: dict[str, list[str]] = {"experience": [], "education": [], "certifications": [], "other": []}
    current: str | None = None
    for line in lines:
        inline = CERT_INLINE.match(_LIST_MARK.sub("", line.strip())) if current != "experience" else None
        if inline:  # a list on the heading's own line belongs to it; the section around it goes on
            sections["certifications"] += [part for part in re.split(r"\s*[,;]\s*", inline.group(1)) if part]
            continue
        kind = _is_header(line, title_check)
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


def _titled_job_line(text: str) -> bool:
    """True when text reads as a job line without its dates: parts split by ' · ' or ' | ', the first or the second of
    them a job title ('Data Engineer · Northwind Labs · Remote', 'Northwind Labs | Data Engineer')."""
    if not _ROW_SEPARATOR.search(text):
        return False
    parts = [part.strip(" ,()") for part in SEPARATOR_RE.split(text) if part.strip(" ,()")]
    return len(parts) >= 2 and any(_reads_as_title(part) for part in parts[:2])


def _job_dates(line: str) -> tuple[str, str, str] | None:
    """(the line without its dates, start, end) for a job's dated line: a range with any dash ('Mar 2022 – Present',
    '2019 - 2021'), or one date ending a job line whose start or end is not known ('Data Engineer · Northwind Labs ·
    Remote   Present': the end). None for any other line, such as a bullet whose words around the range read as a
    sentence ('• Led the 2019 – 2021 move to AWS.', '• Scaled the API from 2000 to 9000 requests per second.'), not as
    a job ('• Northwind Labs — Data Engineer (2019 – 2021)')."""
    match = DATE_RANGE_RE.search(line)
    if match:
        # without its list mark ('* Northwind Labs', a PDF's Word bullet U+F0B7), which is no part of the employer
        rest = _LIST_MARK.sub("", line[: match.start()] + " " + line[match.end() :]).strip(" ,;|-–—•·()")
        text = re.sub(r"\([^()]*\)?", " ", SEPARATOR_RE.sub(" ", rest))  # judged without notes in brackets and signs
        if _LIST_MARK.match(line) and not _capitalized(" ".join(w for w in text.split() if any(c.isalpha() for c in w))):
            return None
        start, end = match.group(1).strip(), match.group(2).strip()
    else:
        lone = _LONE_DATE.search(line)
        prefix = line[: lone.start()] if lone else ""
        kept, place, mode = _job_parts(prefix) if _DOT.search(prefix) else ([], "", "")
        app_line = len(kept) >= 2 and bool(place or mode)  # the app's 'Title · Company · Place · Mode   Feb 2022'
        if not lone or _LIST_MARK.match(line) or not (_titled_job_line(prefix) or app_line):
            return None
        rest, start, end = line[: lone.start()].strip(" ,;|-–—•·()"), "", lone.group(1).strip()
    return rest, start, "Present" if end.lower() in _PRESENT else end


def _job_parts(text: str) -> tuple[list[str], str, str]:
    """A job line's parts without its dates: (the parts kept, in order; location; work mode). A work mode in any part
    ('Remote', '(Hybrid)') is the mode; a place ('Dallas, TX') after the first part is the location. The app's own
    line is split only at its ' · ', so a title keeps its dash ('Engineer – Data Platform'), and its place is the last
    part that reads as one: a company before it may look like a place too ('Associate · Rivera, Quinn · New York,
    NY')."""
    dotted = bool(_DOT.search(text))
    mode = ""
    named: list[str] = []
    for part in (p.strip(" ,()") for p in (_DOT if dotted else SEPARATOR_RE).split(text)):
        found = MODE_RE.search(part)
        if found and dotted and not named and _MODE_WORD.sub("", part).strip(" ,-()") and not re.search(
                r"(?:\(|,)\s*(?:Remote|On-?site|Hybrid)\b", part, re.IGNORECASE):
            found = None  # a title holding a mode word ('Hybrid Cloud Engineer'), not the job's mode
        if found:
            mode = _normalize_mode(found.group(1))
            part = _MODE_WORD.sub("", part).strip(" ,-()")
        if part:
            named.append(part)
    places = [index for index, part in enumerate(named) if index and LOCATION_RE.fullmatch(part)]
    if dotted:
        places = places[-1:]
    location = named[places[-1]] if places else ""
    return [part for index, part in enumerate(named) if index not in places], location, mode


def _rest_of_job_line(line: str | None, need_company: bool) -> tuple[str, str, str] | None:
    """What a title-first job line too long for one line left for the line under it, as (company, location, mode):
    'Bluefin Commerce · Dallas, TX · Onsite' under a title alone, 'Dallas, TX · Onsite' under a title and its company.
    None when that line is anything else: a bullet, a dated line, a sub-heading ('Achievements:', 'Key Projects'), a
    sentence, another title (an old layout's title under its company), or (under a company) no place or work mode. A
    sentence that names a work mode is a bullet, as a Word file gives it without its mark ('Built a hybrid cloud
    platform.')."""
    if (not line or line.endswith(":") or _LIST_MARK.match(line) or _JOB_SUB_HEADING.fullmatch(line)
            or DATE_RANGE_RE.search(line) or POINT_RE.search(line)):
        return None
    dotted = bool(_DOT.search(line))
    mode = ""
    parts: list[str] = []
    for part in (p.strip(" ,()") for p in (_DOT if dotted else SEPARATOR_RE).split(line)):
        found = MODE_RE.search(part)
        if found and not mode:  # 'Onsite', or in a part: 'Contoso Systems, Remote', 'Hybrid (3 days onsite)'
            mode = _normalize_mode(found.group(1))
            # what the part says besides the mode and a note in brackets: a sentence is a bullet ('Led a remote team.')
            words = _MODE_WORD.sub("", re.sub(r"\([^()]*\)?", "", part)).strip(" ,-()")
            if words and not _reads_as_name(words):
                return None
            part = _MODE_WORD.sub("", part).strip(" ,-()")
        if part:
            parts.append(part)
    # the place is the first part that reads as one; in the app's own line its last ('Rivera, Quinn · New York, NY')
    places = [index for index, part in enumerate(parts) if LOCATION_RE.fullmatch(part)]
    place = (places[-1] if dotted else places[0]) if places else -1
    company = location = ""
    others = 0
    for index, part in enumerate(parts):
        if index == place:
            location = part
        elif need_company and not company and not others:  # the company comes first
            if ":" in part or not _reads_as_name(part) or _reads_as_title(part):
                return None
            company = part
        else:
            others += 1
    return (company, location, mode) if (company or location or mode) and others <= 1 else None


def _take_rest(company: Company, rest: tuple[str, str, str]) -> bool:
    """Fill what the employer's dated line left out from the line under it; True when that line gave anything."""
    name, location, mode = rest
    taken = False
    if name:  # under a title alone: the company's own name, in place of the title
        company.name, taken = name, True
    if location and not company.location:
        company.location, taken = location, True
    if mode and not company.mode:
        company.mode, taken = mode, True
    return taken


def _plain_order(first: str, second: str) -> int:
    """How a job line's first two parts plainly read: 1 the title first ('Data Engineer · Northwind Labs', 'Barista ·
    Fabrikam Inc.'), -1 the company first ('Northwind Labs · Data Engineer', 'Fabrikam Inc. · Barista'), 0 when no
    part has a company mark and both or neither read as a title ('Scrum Master · Northwind Labs')."""
    if COMPANY_MARK_RE.search(first):
        return -1
    if COMPANY_MARK_RE.search(second):
        return 1
    return int(_reads_as_title(first)) - int(_reads_as_title(second))


def _stacked_order(part: str, under: str) -> int:
    """How a dated line holding one part and the line under it plainly read: 1 a title over its company ('Senior Data
    Engineer   2019 – 2021' over 'Northwind Labs'), -1 a company over its title ('Northwind Labs   2019 – 2021' over
    'Senior Data Engineer'), 0 when neither reads as a title."""
    if COMPANY_MARK_RE.search(part):
        return -1
    if _reads_as_title(part):
        rest = _rest_of_job_line(under, need_company=True)
        return 1 if rest and rest[0] else 0
    if _job_dates(under) is None and not _LIST_MARK.match(under):
        first = SEPARATOR_RE.split(under, maxsplit=1)[0]
        if COMPANY_MARK_RE.search(first):  # a company under its title
            return 1
        app_break = len(part) + len(first) >= 45 and _DOT.search(under)  # the app's line broken before its company
        return -1 if _reads_as_title(first) and not app_break else 0
    return 0


class _Layout(NamedTuple):
    """How the Experience section lays its jobs out, by votes of the lines that show it plainly (_section_layout)."""
    dots: int  # its ' · ' job lines: > 0 the title first, < 0 the company first, 0 when no line decides or tied
    dotted: bool  # True when it has a dated ' · ' line
    stacks: int  # its dated lines holding one part: > 0 a title over its company, < 0 a company over its title


def _broken_app_line(kept: list[str], dotted: bool, under: str) -> bool:
    """True when a dated line holding only `kept` (a title, or a title and company joined by ' · ': `dotted`) is the
    app's line too long for the page: what did not fit beside the dates is under it, two of the company, its place and
    its work mode joined by ' · ' ('Lead Barista and Coffee Program Mentor for the Flagship Store   2016 – 2019' over
    'Fourth Coffee · Seattle, WA · Onsite', or the title and company over 'Seattle, WA · Onsite'), and the parts make
    45 characters or more (shorter ones would have fitted on one line)."""
    if not (len(kept) == 1 or (len(kept) == 2 and dotted)) or not _DOT.search(under):
        return False
    rest = _rest_of_job_line(under, need_company=len(kept) == 1)
    if not rest or (len(kept) == 1 and not rest[0]) or sum(map(bool, rest)) < 2:
        return False
    return len("".join(kept)) + len(rest[0] or rest[1]) >= 45


def _section_layout(lines: list[str]) -> _Layout:
    """How the Experience section lays its jobs out. Its dated ' · ' lines (see _plain_order) hold the title first, as
    the app writes them, or the company first, as a person may ('Northwind Labs · Data Engineer · Austin, TX   2021 –
    Present'). The app's lines too long for the page vote for the title first too (see _broken_app_line), so one line
    whose employer reads as a title ('Principal · Bluefin Coach') does not turn the app's other lines around. Its dated
    lines holding a title or a company alone (see _stacked_order) have the other under them."""
    dots, dotted, stacks = 0, False, 0
    for index, (line, under) in enumerate(zip(lines, [*lines[1:], ""])):
        dates = _job_dates(line)
        if dates and not dates[0] and index and _DOT.search(lines[index - 1]):  # dates alone under a ' · ' line
            above = _job_parts(lines[index - 1].strip(" ,|-–—•·"))[0]
            dots += _plain_order(above[0], above[1]) if len(above) > 1 else 0
            continue
        kept, location, mode = _job_parts(dates[0]) if dates and dates[0] else ([], "", "")
        dotted_line = bool(dates and _DOT.search(dates[0]))
        if dotted_line:
            dotted = True
            dots += _plain_order(kept[0], kept[1]) if len(kept) > 1 else 0
        if len(kept) == 1 and under:
            stacks += _stacked_order(kept[0], under)
        if kept and not (location or mode) and _broken_app_line(kept, dotted_line, under):
            dots += 1
    return _Layout(dots, dotted, stacks)


def _parse_company_line(line: str, previous: str | None, below: list[str], company_above: bool,
                        layout: _Layout) -> tuple[Company | None, int]:
    """(the employer on a dated experience line or None, how many of the lines under it were read as the rest of it).

    The company comes first in older layouts ('Northwind Labs | Data Engineer | Remote   Mar 2022 - Present') and
    second after a title in an upload's 'Title | Company'. The app's own line joins its parts with ' · ', the title
    first ('Scrum Master · Northwind Labs · Austin, TX · Remote   Mar 2022 – Present'); a person's may join them so
    too, the company first. A section's ' · ' lines are all read in the order they vote for (`layout.dots`): by
    position (the title first) when no line decides or the vote ties, whatever the titles say, unless a part has a
    company mark ('Fabrikam Inc.'). A line of dates alone takes the line above it (`previous`) as its text: never the
    app's line, it is read by what its parts say (see _plain_order). A title alone takes its company from the lines
    around it (`below`: the next two lines; `previous` too when `company_above`, in a layout with each company over its
    titles), see _company_of_title."""
    dates = _job_dates(line)
    if dates is None:
        return None, 0
    remainder, start, end = dates
    own_line = not remainder and bool(previous)
    if own_line:  # the date sits on its own line under the company name
        remainder, previous = previous.strip(" ,|-–—•·"), None
    kept, location, mode = _job_parts(remainder)
    if not kept:
        return None, 0
    following = below[0] if below else None
    dotted = bool(_DOT.search(remainder))
    if dotted and len(kept) == 1 and location:
        under = _rest_of_job_line(following, need_company=False)
        # a job has one place: with it under the line, or in a section of the app's lines, the part read as one is the
        # company ('Barista · Rivera, Quinn' over 'New York, NY · Onsite', or with only a work mode: 'Barista · Rivera,
        # Quinn · Onsite'), unless it is a town and its state
        if (under and under[1]) or (layout.dots > 0 and not own_line and not _TOWN_AND_STATE.search(location)):
            kept, location = [kept[0], location], ""
    if dotted and len(kept) > 1:
        # the section's order, unless a company mark tells the parts apart; a line of dates alone is never the app's
        if own_line or COMPANY_MARK_RE.search(kept[0]) or COMPANY_MARK_RE.search(kept[1]):
            order = _plain_order(kept[0], kept[1])
            title_first = order > 0 if order else layout.dots > 0
        else:
            title_first = layout.dots >= 0
        name = kept[1] if title_first else kept[0]
    else:
        title_first = _reads_as_title(kept[0])
        name = kept[1] if title_first and len(kept) > 1 and not _reads_as_title(kept[1]) else kept[0]
    company = Company(name=name, start=start, end=end, mode=mode, location=location)
    if len(kept) > 1:
        rest = _rest_of_job_line(following, need_company=False) if title_first else None
        return company, int(rest is not None and _take_rest(company, rest))
    taken, rest = 1, _rest_of_job_line(following, need_company=True)
    # A title too long for a line of its own wraps by words where the app draws it, before the rest of its job line (a
    # company and its place or work mode): 'Senior Software Engineer, Payments Infrastructure and Fraud Detection   Jan
    # 2020 – 11/2021' over 'Systems' over 'Fourth Coffee · Seattle, WA · Onsite'. The two lines are read as its rest.
    tail = following if len(below) > 1 and not (location or mode or company_above) else None
    if tail and len(kept[0]) >= 45 and _reads_as_name(tail) and not _job_dates(tail) and _DOT.search(below[1]):
        wrapped = _rest_of_job_line(below[1], need_company=True)
        if wrapped and wrapped[0] and (wrapped[1] or wrapped[2]):
            taken, rest = 2, wrapped
    # A part the rules do not know as a title ('Scrum Master') alone on the line is a title over its company in a
    # section whose lines like it hold a title over its company (`layout.stacks`). With none, only where the app breaks
    # a line too long for the page before the company, whatever the title says ('Principal Agile Delivery and Release
    # Train Scrum Master   Mar 2022 – Present' over 'Northwind Labs · Austin, TX · Remote'): in a section with the app's
    # ' · ' lines (a dated one, or the one holding the company), for a title and company of 45 characters or more
    # together (shorter ones would have fitted beside the dates). Otherwise it is a person's company over a title
    # ('Lakeside Unified School District   Aug 2016 – Present' over 'Special Education Paraprofessional'), as always in
    # a section whose ' · ' lines put the company first.
    if layout.stacks:
        title_over = layout.stacks > 0
    else:
        app_lines = layout.dotted or layout.dots > 0 or bool(_DOT.search(below[taken - 1] if below else ""))
        title_over = app_lines and len(kept[0]) + len(rest[0] if rest else "") >= 45
    if not title_first and not (
            title_over and layout.dots >= 0 and rest and rest[0] and not (location or mode) and _capitalized(kept[0])
            and not COMPANY_MARK_RE.search(kept[0])):
        return company, 0
    return company, taken if _company_of_title(
        company, rest, dotted, previous if company_above else None, below) else 0


def _company_of_title(company: Company, rest: tuple[str, str, str] | None, dotted: bool, previous: str | None,
                      below: list[str]) -> bool:
    """Find the company of a dated line holding a job title alone (in `company.name`, maybe with a place and work
    mode); True when the line under it (`rest`, as read by _rest_of_job_line) was read as the rest of it.

    The company is on the line under it, where the app puts what did not fit beside the dates ('Bluefin Commerce ·
    Dallas, TX · Onsite'). It is on the line above it (`previous`, given only in a layout with each company over its
    titles: 'Northwind Labs' over 'Data Engineer   2019 – 2021') when the line under it names no company or is the next
    job's company, over its own dated title. On the app's own line it is the part read as its place ('Associate ·
    Rivera, Quinn') when no line around names one, unless that part is a town and its state ('Austin, TX')."""
    above = _rest_of_job_line(previous, need_company=True)
    next_job = len(below) > 1 and _job_dates(below[1]) is not None
    if above and above[0] and (not rest or not rest[0] or next_job):
        _take_rest(company, above)
        return bool(rest and not rest[0]) and _take_rest(company, rest)
    if (not rest or not rest[0]) and dotted and company.location and not _TOWN_AND_STATE.search(company.location):
        company.name, company.location = company.location, ""
    return rest is not None and _take_rest(company, rest)


def _read_companies(lines: list[str]) -> list[Company]:
    """The employers in the Experience section, each from its dated line, in the resume's order. A layout with each
    company on the line over its titles shows at the first job: a company's name over the first dated line. The app's
    own resumes start with a job line. Which part of a line is the title, where the line does not say, the section's
    other lines tell (see _section_layout)."""
    first = next((index for index, line in enumerate(lines) if _job_dates(line)), 0)
    above = _rest_of_job_line(lines[first - 1], need_company=True) if first else None
    company_above = bool(above and above[0])
    layout = _section_layout(lines)
    companies: list[Company] = []
    previous: str | None = None  # the line above, unless it belongs to a job line read already
    index = 0
    while index < len(lines):
        line = lines[index]
        company, taken = _parse_company_line(line, previous, lines[index + 1 : index + 3], company_above, layout)
        index += 1 + taken
        if company:
            companies.append(company)
            previous = None
        elif not MODE_RE.fullmatch(line.strip()):
            previous = line
    return companies


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


def _degree_and_field(text: str) -> tuple[str, str]:
    """'B.S., Computer Science' or 'B.S. in Computer Science' -> ('B.S.', 'Computer Science'), cut at whichever of
    ', ' and ' in ' comes first. A degree without a field stays whole ('MBA')."""
    text = re.sub(r"\s{2,}", " ", text).strip(" ,-–—")
    cuts = [(text.find(mark), len(mark)) for mark in (", ", " in ") if text.find(mark) > 0]
    if not cuts:
        return text, ""
    index, width = min(cuts)
    return text[:index].strip(" ,"), text[index + width :].strip(" ,")


def _school_and_degree(text: str) -> tuple[str, str]:
    """An education line's (school, degree part), in either order: 'B.S., Computer Science · State University' (the
    app's own row) or 'State University | B.S. in Computer Science'. The school is the part naming a school, one that
    names no degree first (not 'Master of Science, School of Engineering'); without separators it is the whole line.
    A state's code is no degree: 'State University – Springfield, MA' names none."""
    parts = [part.strip(" ,-–—") for part in SEPARATOR_RE.split(text)]
    parts = [part for part in parts if part]
    schools = [part for part in parts if SCHOOL_RE.search(part)]
    school = next((part for part in schools if not DEGREE_RE.search(part)), schools[0] if schools else text)
    return school, next((part for part in parts
                         if part != school and DEGREE_RE.search(re.sub(r",\s*[A-Z]{2}$", "", part))), "")


def _parse_education(lines: list[str]) -> list[Education]:
    entries: list[Education] = []
    for line in lines:
        text, start, end = _take_dates(_LIST_MARK.sub("", line.strip()))
        if SCHOOL_RE.search(text):
            school, degree_part = _school_and_degree(text)
            entry = Education(school=school or text, start=start, end=end)
            if degree_part:
                entry.degree, entry.field = _degree_and_field(degree_part)
            entries.append(entry)
        elif DEGREE_RE.search(text):
            if not entries:
                entries.append(Education(school=""))
            entry = entries[-1]
            if not entry.degree:
                entry.degree, entry.field = _degree_and_field(text)
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


def _row_dates(line: str, found: re.Match[str]) -> bool:
    """True when the dates ending a certification row (`found`, from _TRAILING_DATES) are its own dates: after a ' · '
    or ' | ' separator ('Name · Issuer   2020') or a gap of three spaces or more (a tab's: the app's Word copy has
    'Name<TAB>2020'), or dates no name ends with: a month and year or a range ('Name Mar 2024', a row without an
    issuer). A bare year right after the name is part of it ('MCSA: Windows Server 2016')."""
    before = line[: found.start() + 1]  # with the space before the date, which closes a separator
    if not before.strip():
        return False
    if _ROW_SEPARATOR.search(before) or before.endswith("   "):
        return True
    return found.group(2) is not None or not _BARE_YEAR.fullmatch(found.group(1))


def _parse_certifications(lines: list[str]) -> list[Certification]:
    """One certification per line: 'Name, Issuer, 2023', 'Name | Issuer | Mar 2023 - Mar 2026', '• Name (2021,
    expires 2024)', the app's own 'Name · Issuer   Mar 2023 – Mar 2026'. A name is never split at a dash ('AWS Certified
    Solutions Architect – Professional'). LinkedIn's own lines ('Issued Jan 2023 · Expires Jan 2026', 'Expires Jan
    2026', 'Credential ID ...') are not certifications: their dates are the one above's."""
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
        trailing = _TRAILING_DATES.search(line)
        if trailing and _row_dates(line, trailing):  # 'Name · Issuer   Mar 2023 – Mar 2026'
            year, expires = year or trailing.group(1), expires or _expiry(trailing.group(2))
            # the separator or the word before the dates goes with them: 'CompTIA Security+ - Feb 2021', 'Name –
            # Issued Mar 2022', 'Six Sigma Green Belt: March 2019'
            line = _DATE_WORD.sub("", line[: trailing.start()].rstrip(" ·|,;:-–—")).rstrip(" ·|,;:-–—")
        elif dated := CERT_DATE_RE.search(" " + line):
            year, expires = year or dated.group(1), expires or _expiry(dated.group(2))
            spaced = " " + line
            line = (spaced[: dated.start()] + spaced[dated.end():]).strip()
        # ' | ' (the box's and older resumes') and ' · ' (the app's rows) separate the fields, so a name may keep a
        # comma ('Oracle Certified Professional, Java SE 11')
        pieces = re.split(r"\s+[|·]\s+", line) if _ROW_SEPARATOR.search(line) else re.split(r"\s*[|·]\s*|,\s+", line)
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


def _header_items(line: str) -> list[str]:
    """A header line's items: the parts between its separators and gaps. Three spaces or more (a tab, by now) part two
    items ('Senior Data Engineer<TAB>Austin, TX': a headline, then a place), and so do two spaces, as in
    'Austin, TX  78701', but not inside one place ('New  York, NY'), unless a job title comes before them ('Senior Data
    Engineer  Austin, TX')."""
    items: list[str] = []
    for part in SEPARATOR_RE.split(line):
        for piece in _WIDE_GAP.split(part):
            words = _GAP.split(piece)
            one_place = (LOCATION_RE.fullmatch(piece.strip(" ,")) and len(words[0].split()) == 1
                         and not _reads_as_title(words[0]))
            items += [piece] if one_place else words
    return items


def parse_resume_text(text: str) -> Profile:
    # A tab is a gap like the spaces before right-aligned dates: Word files (the app's own too) put dates on a tab stop.
    # A run of tabs or spaces is one gap.
    lines = [_GAPS.sub("   ", ln).strip() for ln in text.replace("\r", "").splitlines() if ln.strip()]
    head, sections = _split_sections(lines)
    if not sections["experience"]:
        # the only Experience heading may be a side label in title case run into the job title beside it, as a
        # two-column template gives it ('Experience Senior Data Engineer'): the job-title check takes it for a title
        without_check = _split_sections(lines, title_check=False)
        if without_check[1]["experience"]:
            head, sections = without_check
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
        for chunk in _header_items(cleaned):
            chunk = chunk.strip(" ,")
            if chunk != name and LOCATION_RE.fullmatch(chunk):
                location = chunk
                break
        if location:
            break

    return Profile(
        name=name,
        contact=Contact(email=email, phone=phone, linkedin=linkedin, location=location),
        companies=_read_companies(sections["experience"]),
        education=_parse_education(sections["education"]),
        certifications=_parse_certifications(sections["certifications"]),
    )
