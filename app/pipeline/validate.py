"""Checks that run on every result, whichever model produced it.

  build_resume            merge model output with the locked profile (the AI never touches fixed facts)
  assert_fixed_fields_intact   safety net that the fixed facts are byte-identical to the saved profile
  keyword_coverage        how many JD keywords appear in the resume
  ats_lint                format / length warnings
  claim_flags             numbers and skills the user never supplied, for the review step
  unshown_must_haves      required JD terms missing, or only listed, instead of shown in the summary or a bullet
  posting_match_flags     sentences added to match the JD that the user's hints do not back up
  resolve_certifications  the model's certification picks: only the person's own are printed, the rest are ideas
  certification_mentions  certification names in the text (they belong only in their own section)
"""
from __future__ import annotations

import re
import unicodedata
from itertools import pairwise

from ..schemas import (
    MAX_CERTIFICATIONS,
    Certification,
    CertificationChoice,
    CertificationPick,
    Coverage,
    CoverLetter,
    ExperienceOut,
    Flag,
    JDAnalysis,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
    TailoredSections,
    TailorOutput,
    TailorRevision,
)
from . import spelling
from .facts import is_expired, months_phrase, parse_point, stay_months

_PUNCT_MAP = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": " - ", "\u2015": " - ",
    "\u2026": "...", "\u00a0": " ", "\t": " ", "\u2192": " to ", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\n": " ", "\r": " ", "\x0b": " ", "\x0c": " ", "\x85": " ", "\u2028": " ", "\u2029": " ",
})
# A bullet's own mark; a dash touching a figure is its minus sign and stays: '-12% churn', '-$1.2M', '–12%'
_BULLET_LEAD = re.compile(r"^(?:[\s*•·▪◦‣—>]|[-–](?![\d$€£]|\.\d))+")
_NUMBER = re.compile(r"(?:[$€£]|(?<![\w.]))\d[\d,.]*\s?(?:%|[kKmMbB]\b|\+|x\b)?")
_FIRST_PERSON = re.compile(r"\b(I|me|my|mine|myself|I'm|I've)\b")


def clean_text(text: str) -> str:
    """Normalise punctuation, drop symbols/emoji/control characters, collapse whitespace."""
    text = text.translate(_PUNCT_MAP)
    text = "".join(ch for ch in text if (unicodedata.category(ch)[0] not in ("S", "C") or ch in "+#$%&<>=|€£±")
                   and not ("\ufe00" <= ch <= "\ufe0f" or unicodedata.category(ch) == "Me"))  # emoji: '✔️', '1️⃣'
    return re.sub(r"\s+", " ", text).strip()


def clean_bullet(text: str) -> str:
    return clean_text(_BULLET_LEAD.sub("", text))


# A dash used as punctuation ("Built X — cutting Y", "X -- Y", "X - Y") is a well-known sign of AI-written text
# (a recruiter's advice): it becomes a comma, and the prompt asks for a full stop or a comma in the first place.
# Kept: hyphens inside words (event-driven, AI-/ML-based), "Python- and SQL-based", "-12%", command-line flags
# (--dry-run, also after a bracket, a quote or another dash), and a range between two like values going up, which
# loses its spaces (2019 - 2021 -> 2019-2021; $500K - $2M; 10 - 20%; Q1 - Q3; Jan - Mar; 2021 - present) or reads
# "to" when its first end is two words (Jan 2021 - Mar 2024 -> Jan 2021 to Mar 2024; Tier 1 - Tier 3 -> Tier 1 to
# Tier 3). A spaced em dash joins only two dates: models write it as punctuation ("from 101 to 83 — 18 fewer"). A dash
# that starts a list item ("things:\n- Airflow DAGs") is punctuation, never one end of a range.
_SPACE_RUN = re.compile(r"\s+")  # newlines and every Unicode space too; one space keeps the scan linear
_HYPHENS = str.maketrans({"‐": "-", "‑": "-", "−": "-"})  # Unicode hyphens and the minus sign
_LINE_BREAKS = "\n\r\x0b\x0c\x85\u2028\u2029"
# A dash that starts a line ('\n- Cut costs', never '\n-12%'): the spaces read after a line break are never line
# breaks, so a run of them is no slow scan
_LINE_DASH = re.compile(f"[{_LINE_BREAKS}][^\\S{_LINE_BREAKS}]*[‒–—―-]+(?=[^\\S{_LINE_BREAKS}])")
_THREE_WORDS = re.compile(f"(?:[^\\S{_LINE_BREAKS}]+\\S+){{3}}")  # ... on the same line
_LIST_MARK = "\ufdd0"  # a noncharacter: stands in for a list marker's dash until the dash fix replaces it
_DASH_RUN = re.compile(f" ?[‒–—―{_LIST_MARK}-]+ ?")
_DASH_WINDOW = 120  # characters read on each side of a dash: enough for its words, and the scan stays linear
_MONTH_NUMBER = {name: number for number, names in enumerate((
    ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
    ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
    ("dec", "december")), 1) for name in names}
_MONTH_SHORT = frozenset({"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"})
_SEASON_NUMBER = {"winter": 0, "spring": 1, "summer": 2, "fall": 3, "autumn": 3}
_OPEN_END = frozenset({"present", "date"})  # always the end of a range: 2021-present
# ... only where the clause ends with them: (2021-now), never '2023 — now used' or '2019 — today, I lead'
_OPEN_NOW = frozenset({"now", "today", "current"})
_TOKEN_EDGES = "()[]{}\"'`,;:.!?"
_CLAUSE_END = ",;:.!?)]}"
_RANGE_NUMBER = re.compile(
    r"([$€£]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|bn|[kmb]|x|\+|ms|s|sec|secs|min|mins|h|hr|hrs|d|kb|mb|gb|tb|pb)?",
    re.IGNORECASE)
_RANGE_LABEL = re.compile(r"([A-Za-z]{1,4})(\d+)")  # v16, Sev1, p50, P0
_RANGE_MAGNITUDE = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9}
_YEAR = re.compile(r"(?:19|20)\d\d")
_DIGITS = re.compile(r"\d+")
_CLOCK = re.compile(r"(1[0-2]|0?[1-9])(?::([0-5]\d))?([ap])\.?m", re.IGNORECASE)  # 7am, 7:30pm, 7p.m.
# Words that put the year after them in time, so it is never a count: 'in 2000 — 5000 users signed up'
_YEAR_WORDS = frozenset({"in", "since", "by", "during", "until", "till", "through", "throughout", "before", "after",
                         "early", "mid", "late", "circa"})
# A full stop that ends an abbreviation, not a sentence: 'Contoso Ltd.', 'etc.', 'the U.S.', 'e.g.'
_ABBREVIATION_STOP = re.compile(r"(?:[A-Za-z]\.){2,}|(?:inc|ltd|corp|co|llc|plc|bros|etc|vs|approx|jr|sr|st)\.",
                                re.IGNORECASE)


def _bare(token: str) -> str:
    return token.strip(_TOKEN_EDGES)


def _runs_on(word: str) -> bool:
    """False when a clause ends at `word` ('in 2021.', 'Northwind,'); the full stop of 'Jan.' only ends the
    abbreviation."""
    return word[-1:] not in _CLAUSE_END or (word.endswith(".") and _bare(word).lower() in _MONTH_SHORT)


def _year_of(token: str) -> int | None:
    """2021 for '2021' and 'FY21' (also 'FY2021'), else None."""
    bare = _bare(token)
    if _YEAR.fullmatch(bare):
        return int(bare)
    fiscal = re.fullmatch(r"FY(\d\d|\d{4})", bare)
    return int(fiscal.group(1)) + (2000 if len(fiscal.group(1)) == 2 else 0) if fiscal else None


def _part_of(token: str) -> tuple[str, int] | None:
    """('month', 3) for 'Mar', ('quarter', 2) for 'Q2', ('half', 1) for 'H1', ('season', 1) for 'Spring'; 'may',
    'march' and 'spring' are words, not dates."""
    bare = _bare(token)
    if bare[:1].isupper() and bare.lower() in _MONTH_NUMBER:
        return "month", _MONTH_NUMBER[bare.lower()]
    if bare[:1].isupper() and bare.lower() in _SEASON_NUMBER:
        return "season", _SEASON_NUMBER[bare.lower()]
    if re.fullmatch(r"Q[1-4]|H[12]", bare):
        return ("quarter" if bare[0] == "Q" else "half"), int(bare[1])
    return None


def _left_date(words: list[str]) -> tuple[tuple[int | None, tuple[str, int] | None], int] | None:
    """((year, part of the year), how many words) for the date that ends `words`: 'Jan 2021', 'Jan. 2021', 'H1 2023',
    '2021', 'Jan'."""
    year, part = _year_of(words[-1]), _part_of(words[-1])
    if year is not None:
        month = _part_of(words[-2]) if len(words) > 1 and _runs_on(words[-2]) else None
        return (year, month), (2 if month else 1)
    return ((None, part), 1) if part else None


def _right_date(words: list[str]) -> tuple[int | None, tuple[str, int] | None] | None:
    """(year, part of the year) for the date that starts `words`: 'Mar 2024', 'Mar. 2024', 'Fall 2024', '2024',
    'Mar'."""
    year, part = _year_of(words[0]), _part_of(words[0])
    if year is not None:
        return year, None
    if part is None:
        return None
    return (_year_of(words[1]) if len(words) > 1 and _runs_on(words[0]) else None), part


def _dates_go_up(left: tuple[int | None, tuple[str, int] | None],
                 right: tuple[int | None, tuple[str, int] | None]) -> bool:
    (left_year, left_part), (right_year, right_part) = left, right
    if left_year is not None and right_year is not None and left_year != right_year:
        return left_year < right_year  # also between two kinds of part: 'H1 2023 - Mar 2024'
    if left_part and right_part and left_part[0] != right_part[0]:
        return False  # 'Jan - Q3'
    if left_year is not None and right_year is not None:
        return bool(left_part and right_part and left_part[1] < right_part[1])
    if left_year is not None or not left_part or not right_part:
        return False  # '2023 - Mar', 'Jan - 2024'
    # 'Jan - Mar 2023', 'Q1 - Q3'; months and seasons may wrap round the year end ('Nov - Feb', 'Fall - Spring'),
    # quarters and halves go up
    return left_part[1] != right_part[1] if left_part[0] in ("month", "season") else left_part[1] < right_part[1]


def _number_of(token: str) -> tuple[str, float, str, tuple[int, int] | None] | None:
    """(currency, value, unit, parts) for one end of a number range: '$500K' -> ('$', 500000.0, 'k', None). A plain
    decimal also keeps its parts, so 'Python 3.8 - 3.12' can go up as a version does."""
    match = _RANGE_NUMBER.fullmatch(_bare(token))
    if not match:
        return None
    currency, whole, decimal, unit = match.group(1), match.group(2).replace(",", ""), match.group(3), match.group(4)
    unit = (unit or "").lower()
    if len(whole) > 15:
        return None
    parts = (int(whole), int(decimal[1:])) if decimal and not currency and not unit else None
    return currency, float(whole + (decimal or "")) * _RANGE_MAGNITUDE.get(unit, 1), unit, parts


def _numbers_go_up(left: tuple[str, float, str, tuple[int, int] | None],
                   right: tuple[str, float, str, tuple[int, int] | None]) -> bool:
    (left_currency, left_value, left_unit, left_parts), (right_currency, right_value, right_unit, right_parts) = (
        left, right)
    if left_currency != right_currency and right_currency:
        return False
    magnitudes = left_unit in _RANGE_MAGNITUDE and right_unit in _RANGE_MAGNITUDE  # '$500K - $2M'
    if left_unit != right_unit and left_unit and not magnitudes:
        return False  # '5% - 10x'; a unit on the right end only is the unit of both ('10 - 20%', '2.5 - 3.5%')
    if not left_unit and right_unit in _RANGE_MAGNITUDE:
        left_value *= _RANGE_MAGNITUDE[right_unit]  # '10 - 50K', '1.5 - 2M'
    # by value ('2 - 2.5', '0.85 - 0.9'); two plain decimals that go down by value may go up part by part, as
    # versions do ('3.8 - 3.12')
    return left_value < right_value or bool(left_parts and right_parts and left_parts < right_parts)


def _shape_of(token: str) -> tuple[str, tuple[int, ...]] | None:
    """('v9.9', (1, 24)) for 'v1.24': the token with each run of digits as 9 (a year as Y, so it never looks like a
    count), and its numbers, years first ('11/2019' -> (2019, 11)); None without a digit."""
    bare = _bare(token)
    runs = _DIGITS.findall(bare)
    if not runs or len(bare) > 24:
        return None
    shape = _DIGITS.sub(lambda run: "Y" if _YEAR.fullmatch(run.group()) else "9", bare)
    return shape, tuple(int(run) for run in sorted(runs, key=lambda run: not _YEAR.fullmatch(run)))


def _clock_of(token: str) -> int | None:
    """Minutes after midnight for '7am' or '7:30pm', else None."""
    match = _CLOCK.fullmatch(_bare(token))
    if not match:
        return None
    hour, minute, half = match.groups()
    return (int(hour) % 12 + (12 if half.lower() == "p" else 0)) * 60 + int(minute or 0)


def _range_join(before: str, after: str, dates_only: bool) -> str | None:
    """'-' when the dash joins two like values going up, ' to ' when the first end is two words ('Jan 2021 to Mar
    2024', 'Tier 1 to Tier 3'); None when the dash is punctuation ('in 2022 — 14 services', 'from 101 to 83 — 18
    fewer', 'in 2019 — in 2022 I took over')."""
    left_words, right_words = before.split()[-6:], after.split()[:3]
    if not left_words or not right_words:
        return None
    left, right = left_words[-1], right_words[0]
    if not _runs_on(left):
        return None  # 'in 2021. — 2022 brought', 'to 45, — 75 fewer'
    lowered = [_bare(word).lower() for word in left_words]
    left_number, right_number = _number_of(left), _number_of(right)
    if len(lowered) > 2 and lowered[-2] == "to" and "from" in lowered[:-2]:
        # 'from A to B — C': B ends a pair, unless it is a bare number whose unit comes after the dash ('from 4 hours
        # to 30 - 45 minutes', 'from weekly to 2 - 3 times a day')
        if (left_number is None or left_number[0] or left_number[2] or _year_of(left) is not None
                or _year_of(right) is not None):
            return None  # 'from $18,600 to $12,400 — $74,400 saved a year', 'from 120 to 45 — 2022 brought'
        start = _number_of(left_words[-3]) if len(lowered) > 3 and lowered[-4] == "from" else None
        if start and not start[0] and not start[2] and right_number and (
                right_number[0] or right_number[2] or right_number[1] == abs(start[1] - left_number[1])):
            return None  # C is the change from A to B: 'from 120 to 45 — 75 fewer', 'from 10 to 13 — 30% more'
    left_date = _left_date(left_words)
    if left_date:
        (left_year, _part), size = left_date
        open_end = _bare(right).lower()
        after_word = right.lstrip(_TOKEN_EDGES)[len(open_end):][:1]  # what directly follows now/today/current
        now_ends = after_word in tuple(")]}.!?") or (not after_word and len(right_words) == 1)
        if left_year is not None and (open_end in _OPEN_END or (open_end in _OPEN_NOW and now_ends)):
            return " to " if size == 2 else "-"
        right_date = _right_date(right_words)
        if right_date and _dates_go_up(left_date[0], right_date):
            return " to " if size == 2 else "-"
    if len(left_words) > 1 and len(right_words) > 1 and _runs_on(left_words[-2]) and _runs_on(right):
        first, last = (_year_of(left_words[-2]), _part_of(left)), (_year_of(right), _part_of(right_words[1]))
        if None not in first + last and _dates_go_up(first, last):
            return " to "  # the year first: 'FY23 Q1 - FY23 Q3', '2023 H1 - 2024 H2'
    if dates_only:
        return None
    if left_number and right_number:
        # a year-like first end is a count when it is round and no word puts it in time: '2000 - 5000 requests';
        # 'Launched 2021 - 3000 users' and 'in 2000 - 5000 users signed up' stay punctuation
        count = (left_date is not None and left_date[1] == 1 and not right_number[0] and not right_number[2]
                 and _year_of(right) is None and left_number[1] % 100 == 0
                 and (len(lowered) == 1 or lowered[-2] not in _YEAR_WORDS))
        if (not left_date or count) and _numbers_go_up(left_number, right_number):
            return "-"
    left_label, right_label = _RANGE_LABEL.fullmatch(_bare(left)), _RANGE_LABEL.fullmatch(_bare(right))
    if (left_label and right_label and left_label.group(1).lower() == right_label.group(1).lower()
            and len(left_label.group(2)) < 8 and len(right_label.group(2)) < 8
            and int(left_label.group(2)) < int(right_label.group(2))):
        return "-"  # 'v16 - v18', 'Sev1 - Sev3', 'p50 - p99'
    if len(left_words) > 1 and len(right_words) > 1 and _runs_on(left_words[-2]) and _runs_on(right):
        label, first, last = _bare(left_words[-2]), _number_of(left), _number_of(right_words[1])
        if (label.isalpha() and len(label) > 1 and label.lower() == _bare(right).lower()
                and label.lower() not in _SMALL_WORDS and first and last and not first[0] and not last[0]
                and (first[2] == last[2] or (first[2] in _RANGE_MAGNITUDE and last[2] in _RANGE_MAGNITUDE))
                and _numbers_go_up(first, last)):
            return " to "  # 'Tier 1 - Tier 3', '(Level 2 - Level 4)', 'EUR 2M - EUR 5M'
    if (left_number and right_number) or (left_label and right_label):
        return None  # the value rules read both ends, and they do not go up
    # ends the value rules cannot read go up when they have the same shape and their numbers go up ('v1.24 -
    # v1.28', '$80/hr - $120/hr', '9:00 - 17:00', '#1 - #3', '03/2019 - 06/2021'), or are clock times ('7am - 7pm')
    left_shape, right_shape = _shape_of(left), _shape_of(right)
    if left_shape and right_shape and left_shape[0] == right_shape[0] and left_shape[1] < right_shape[1]:
        return "-"
    left_clock, right_clock = _clock_of(left), _clock_of(right)
    return "-" if left_clock is not None and right_clock is not None and left_clock < right_clock else None


def _dash_edits(text: str) -> list[tuple[int, int, str, bool]]:
    """(start, end, replacement, used as punctuation) for each dash that is not part of a word, a flag or a negative
    number; dashes next to each other count as one. Runs of spaces must already be one space (_SPACE_RUN)."""
    stripped = text.strip()
    first_text = len(text) - len(text.lstrip())  # where the text starts and ends: a dash outside is dropped
    last_text = first_text + len(stripped)
    found: list[list] = []  # [start, end, replacement or None for punctuation]
    for match in _DASH_RUN.finditer(text):
        raw = match.group(0)
        start, end = match.start(), match.end()
        dash, spaced_after = raw.strip(), raw.endswith(" ")
        # a space the dash before took still counts, so 'Northbeam — - Airflow' is one dash
        spaced_before = raw.startswith(" ") or text[start - 1: start] == " "
        at_edge = start <= first_text or end >= last_text  # nothing but spaces before it, or after it
        before_char, after_char = text[start - 1: start], text[end: end + 1]
        if dash == "-" and not (spaced_before and spaced_after):
            if at_edge and (spaced_before or spaced_after):
                found.append([start, end, None])  # a list marker: '- Led the team'
            continue  # inside a word or a number, 'Python- and', '-12%', '-v'
        if dash in ("‒", "–") and not spaced_before and not spaced_after:
            continue  # 2019–2021, New York–London: clean_text makes it a plain hyphen
        if (dash.startswith("--") and not spaced_after and after_char.islower()
                and (spaced_before or not before_char or before_char in " ([{\"'`/")):
            continue  # a command-line flag: --dry-run, (--dry-run), `--force`, '— --dry-run'
        if at_edge or _LIST_MARK in dash:
            found.append([start, end, None])  # at the very start or end, or a list marker after a line break
            continue
        dates_only = (spaced_before or spaced_after) and ("—" in dash or "―" in dash or "--" in dash)
        joined = _range_join(text[max(0, start - _DASH_WINDOW): start], text[end: end + _DASH_WINDOW], dates_only)
        found.append([start, end, joined])
    merged: list[list] = []
    for item in found:  # 'the migration — — of 14 services': one dash
        if merged and item[2] is None and merged[-1][2] is None and merged[-1][1] == item[0]:
            merged[-1][1] = item[1]
        else:
            merged.append(item)
    edits: list[tuple[int, int, str, bool]] = []
    for start, end, joined in merged:
        if joined is not None:
            edits.append((start, end, joined, False))
            continue
        if start <= first_text or end >= last_text:
            edits.append((start, end, "", True))  # at the very start or end: dropped
            continue
        before = text[max(0, start - _DASH_WINDOW): start].rstrip()
        after = text[end: end + _DASH_WINDOW].lstrip()
        last, first = before[-1:], after[:1]
        # after an abbreviation's full stop, or before a word in lower case, the dash still stands for a comma
        # ('Contoso Ltd., a 40-person fintech,'); after a sentence's full stop it goes ('on time. Then', '2021. 2022')
        aside = last == "." and first not in ",.;:!?)]}" and (
            _ABBREVIATION_STOP.fullmatch(before.rsplit(" ", 1)[-1].lstrip("([{\"'")) is not None
            or (first.islower() and after.split(" ", 1)[0].islower()))  # 'iOS', 'eBay': a name may start a sentence
        replacement = ("" if last in "([{" else ", " if aside else " " if last in ",.;:!?\u2026"
                       else "" if first in ",.;:!?)]}" else ", ")
        edits.append((start, end, replacement, True))
    return edits


def _marked_list_items(text: str) -> str:
    """The text with each dash that starts a list item replaced by _LIST_MARK: in a list of two or more, or before
    three words or more. Alone before one or two words, a dash that starts a line is a range or a dash wrapped onto
    it ('Jan 2020' and then '– Mar 2023' on the next line)."""
    starts = list(_LINE_DASH.finditer(text))
    parts: list[str] = []
    last = 0
    for match in starts:
        if len(starts) > 1 or _THREE_WORDS.match(text, match.end()):
            parts += [text[last:match.start()], " " + _LIST_MARK]
            last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def fix_dashes(text: str) -> str:
    """Dashes used as punctuation become commas, and a range loses its spaces. Run on the model's own text before
    clean_text, which turns any dash into a plain hyphen."""
    text = _SPACE_RUN.sub(" ", _marked_list_items(text.translate(_HYPHENS)))  # before the lines are joined
    parts: list[str] = []
    last = 0
    for start, end, replacement, _punctuation in _dash_edits(text):
        parts += [text[last:start], replacement]
        last = end
    parts.append(text[last:])
    return "".join(parts)


def clean_prose(text: str) -> str:
    """The model's summary, bullet or letter paragraph: no dash punctuation, then clean_text."""
    return clean_text(fix_dashes(text))


def prose_places(resume: Resume, letter: CoverLetter | None = None) -> list[tuple[str, str]]:
    """(where, text) for every sentence-like part a recruiter reads: the summary, each bullet (by employer) and the
    cover letter's paragraphs."""
    places = [("summary", resume.summary)] + [
        (resume.profile.companies[exp.company_index].name, bullet) for exp in resume.experience for bullet in exp.bullets
    ]
    if letter is not None:
        places += [(f"cover letter paragraph {number}", paragraph)
                   for number, paragraph in enumerate(letter.paragraphs, 1)]
    return places


def dash_problems(resume: Resume, letter: CoverLetter | None = None) -> list[str]:
    """Dashes used as punctuation that are still in the text: the model's text loses them (fix_dashes), so these
    come from the user's own edits. Ranges, hyphenated words and command-line flags are not reported."""
    problems: list[str] = []
    for where, text in prose_places(resume, letter):
        text = _SPACE_RUN.sub(" ", text.translate(_HYPHENS))
        for start, end, _replacement, punctuation in _dash_edits(text):
            if punctuation:
                before = " ".join(text[max(0, start - _DASH_WINDOW): start].split()[-3:])
                after = " ".join(text[end: end + _DASH_WINDOW].split()[:3])
                problems.append(f"'{' '.join(part for part in (before, text[start:end].strip(), after) if part)}'"
                                f" ({where})")
    return problems


def word_count(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------ keywords, not job-description phrases
_SMALL_WORDS = frozenset({
    "a", "an", "the", "and", "or", "nor", "but", "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "into",
    "onto", "over", "under", "via", "per", "across", "like", "than", "then", "our", "your", "their", "its", "his", "her",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "we", "you", "they",
    "using", "through", "within", "while", "when", "where", "which", "who", "how", "all", "each", "both", "not", "so",
    "if", "also", "up", "out", "about", "after", "before", "between", "during", "without", "can", "will", "has", "have",
    "had", "years", "experience",  # '7+ years of experience' opens every summary
})
MAX_TERM_WORDS = 4  # meaningful words: a longer phrase is a piece of a sentence, not what an ATS or a recruiter matches
MAX_NAME_WORDS = 8  # ... but a proper name may be longer: 'Payment Card Industry Data Security Standard (PCI DSS)'

# "AI tools like Copilot, Cursor, or Claude Code": the words after the lead are the tools themselves.
_EXAMPLES_LEAD = re.compile(r"(?<![\w-])(?:like|such as|including|e\.g\.?)(?![\w-])\s*[:,]?\s*", re.IGNORECASE)
_BRACKET_LEAD = re.compile(r"\(\s*(?:like|such as|including|e\.g\.?)(?![\w-])\s*[:,]?\s*", re.IGNORECASE)
_COMMA_SPLIT = re.compile(r"\s*[,;]\s*(?:(?:or|and|&)\s+)?", re.IGNORECASE)  # 'A, B, and C'
_OR_WORD = re.compile(r"\s+or\s+", re.IGNORECASE)  # 'OR-Tools' stays whole
_AND_WORD = re.compile(r"\s+(?:and|&)\s+", re.IGNORECASE)
_LOOSE_JOINER = re.compile(r"^(?:or|and|&)\s+|\s+(?:or|and|&)$", re.IGNORECASE)
# The end of a list, not an item: "Python, Go, etc.", "Copilot, Cursor or similar", "AWS or other cloud platforms"
_LIST_END = re.compile(r"(?:etc|more|others?|similar|the like|so on)\.?$|(?:other|similar) ", re.IGNORECASE)
_BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_SPELLED_OUT = re.compile(r"(.+?)\s*\(([A-Za-z&]{2,})\)")  # 'Extract, Transform, Load (ETL)'
# In the analysis, a bracket of examples or a list is detail, not the keyword: 'AWS (e.g., EC2, S3)', 'AWS (ECS, SQS)'
_DETAIL_BRACKET = re.compile(
    r"\s*\((?:\s*(?:like|such as|including|e\.g\.?)(?![\w-])[^()]*|[^()]*,[^()]*)\)", re.IGNORECASE)

# Prose around a keyword in a posting: "experience with Kafka", "advanced SQL statements", "AWS-hosted systems"
_PROSE_LEAD = re.compile(
    r"^.*?\b(?:experience|knowledge|understanding|familiarity|proficiency|expertise|background)\s+"
    r"(?:of|with|in|using)\s+", re.IGNORECASE)
_ABILITY = re.compile(r"\bability to\b", re.IGNORECASE)
_HOSTED = re.compile(r"^(.+?)-hosted\b.*$", re.IGNORECASE)
_QUALIFIER = re.compile(  # in lower case or capitalized, so that a name in capitals stays: 'SOLID principles'
    r"\b(?:[Aa]dvanced|[Bb]asic|[Ss]trong|[Ss]olid|[Pp]roven|[Ee]xcellent|[Dd]eep|[Gg]ood|[Hh]igh-scale|[Ll]arge-scale"
    r"|[Hh]ands-on|[Ee]xtensive)\s+")
_TOOL_SKILLS = re.compile(r"(\b[A-Z][\w.+#/-]*)\s+skills?$")  # 'Python skills' -> 'Python'; 'people skills' stays
# ... but not where such a word is part of a name ("Deep Learning", "Visual Basic", "self-hosted runners")
_NAMED_TERMS = re.compile(
    r"\b(?:deep (?:learning|neural|reinforcement|links?|linking|packet inspection)|visual basic|basic auth\w*"
    r"|strong (?:consistency|customer authentication|typing)"
    r"|advanced (?:analytics|encryption standard|persistent threats?)"
    r"|good (?:manufacturing|clinical|laboratory|distribution|documentation) practices?"
    r"|(?:self|github|gitlab|microsoft)-hosted"
    r"|solid\.?js|large-scale scrum|(?:basic|advanced(?: cardiac| cardiovascular)?|pediatric advanced) life support)\b",
    re.IGNORECASE,
)
_ABBREVIATION = re.compile(r"\(\s*[A-Z][A-Z0-9&/+.-]+(?:\s+[A-Z][A-Z0-9&/+.-]+)*\s*\)")  # '(BLS)', '(PCI DSS)'


def _meaningful(text: str) -> list[str]:
    """Words that carry meaning: no small words and no lone marks ('-', '&'); a word in capitals counts ('CAN')."""
    words = [word.strip(",.;:!?\"'") for word in text.split()]
    return [w for w in words if re.search(r"[A-Za-z0-9]", w) and (w.isupper() or w.lower() not in _SMALL_WORDS)]


def _is_name(term: str) -> bool:
    """A proper name rather than prose: it has an abbreviation in brackets ('Basic Life Support (BLS)') or is written
    in title case ('Solid Edge', 'Good Documentation Practices', 'SOLID', 'iOS')."""
    if _ABBREVIATION.search(term):
        return True
    words = _meaningful(re.sub(r"\([^)]*\)", "", term))
    return bool(words) and all(not w[0].isalpha() or any(ch.isupper() for ch in w) for w in words)


def _masked(text: str) -> tuple[str, list[str]]:
    """The text with each bracket (nested ones too) replaced by a placeholder, so lists inside brackets stay whole."""
    kept: list[str] = []

    def mask(match: re.Match) -> str:
        kept.append(match.group(0))
        return f"\0{len(kept) - 1}\0"

    text = text.replace("\0", "")
    while (masked := _BRACKETS.sub(mask, text)) != text:
        text = masked
    return text, kept


def _unmasked(text: str, kept: list[str]) -> str:
    while (restored := re.sub(r"\0(\d+)\0", lambda m: kept[int(m.group(1))], text)) != text:
        text = restored
    return text


def _spelled_out(item: str) -> bool:
    """A name with commas in it: 'Extract, Transform, Load (ETL)', 'Water, Sanitation and Hygiene (WASH)'. The
    abbreviation starts with the first word's initial and holds every word's initial, in order."""
    match = _SPELLED_OUT.fullmatch(item)
    if not match:
        return False
    initials, letters = [word[0].lower() for word in _meaningful(match.group(1))], match.group(2).lower()
    rest = iter(letters)
    return bool(initials) and letters[0] == initials[0] and all(initial in rest for initial in initials)


def _list_items(text: str, lead: bool) -> list[str]:
    """Split a list at commas and 'or'; after a lead ('such as AWS and GCP') also at the 'and' before its last item,
    so names that hold an 'and' stay whole: 'Profit and Loss, Balance Sheet and Cash Flow', 'Research and
    Development'."""
    parts = [piece for part in _COMMA_SPLIT.split(text) for piece in _OR_WORD.split(part)]
    if lead and not _OR_WORD.search(text):
        parts[-1:] = _AND_WORD.split(parts[-1])
    return parts


def _tidy(part: str) -> str:
    part = _LOOSE_JOINER.sub("", _BRACKET_LEAD.sub("(", part).strip(" ;:"))  # 'AWS (e.g., EC2, S3)' -> 'AWS (EC2, S3)'
    if not re.search(r"(?:^|[^A-Za-z])[A-Za-z]\.$", part):  # 'Ph.D.' keeps its dot, 'Node.js.' loses it
        part = part.rstrip(". ")  # a leading dot stays: '.NET'
    while part.endswith(")") and part.count(")") > part.count("("):  # "(like Copilot, Cursor)" leaves "Cursor)"
        part = part[:-1].rstrip()
    while part.startswith("(") and part.count("(") > part.count(")"):
        part = part[1:].lstrip()
    return part


def _examples_as_items(item: str) -> list[str]:
    """'AI tools like Copilot, Cursor, or Claude Code' -> ['Copilot', 'Cursor', 'Claude Code']; 'TypeScript or
    JavaScript' and 'Python, Go, and Rust' -> each one. A list in brackets stays with its keyword ('AWS (ECS, SQS)'),
    and so do names with commas in them: 'Extract, Transform, Load (ETL)', 'Microsoft Office: Word, Excel',
    'Java 8, 11, 17'."""
    masked, kept = _masked(item)
    lead = _EXAMPLES_LEAD.search(masked)
    if lead:
        head = masked[:lead.start()].rstrip()
        parts = _list_items(masked[lead.end():], lead=True)
        if head.endswith(","):  # 'SQL, including window functions': SQL is an item too; 'AI tools like ...' is not
            parts.insert(0, head[:-1])
    elif _spelled_out(item) or ":" in masked:
        parts = [masked]
    else:
        parts = _list_items(masked, lead=False)
    parts = [_tidy(_unmasked(part, kept)) for part in parts]
    if not lead and not all(re.search(r"[A-Za-z]", part) for part in parts if part):  # '11' is no item
        parts = [_tidy(item)]
    return [part for part in parts if _meaningful(part) and not _LIST_END.match(part)
            and not _EXAMPLES_LEAD.fullmatch(part)]


def _keyword_of(term: str) -> str:
    """The keyword inside a piece of the posting's prose: 'experience with Kafka' -> 'Kafka', 'advanced SQL' -> 'SQL',
    'AWS-hosted systems' -> 'AWS', 'Strong Python skills' -> 'Python'. A name stays as it is ('Basic Life Support
    (BLS)', 'Solid Edge', 'Deep Learning'); an 'ability to ...' has no keyword ('')."""
    if _ABILITY.search(term):
        return ""
    term = _PROSE_LEAD.sub("", term)
    if not _NAMED_TERMS.search(term):
        term = _HOSTED.sub(r"\1", term)
        if not _is_name(term):
            term = _TOOL_SKILLS.sub(r"\1", _QUALIFIER.sub("", term))
    return term.strip()


def is_keyword(term: str) -> bool:
    """A term as people list it on a resume ('SQL', 'AWS (ECS, SQS)', 'Basic Life Support (BLS)'), not a piece of the
    posting's prose: nothing around the keyword ('advanced', 'AWS-hosted', 'experience with') and at most four
    meaningful words, or eight for a proper name. An abbreviation in brackets does not count."""
    words = _meaningful(re.sub(r"\([^)]*\)", "", term))
    limit = MAX_NAME_WORDS if _is_name(term) else MAX_TERM_WORDS
    return 0 < len(words) <= limit and _keyword_of(term) == term.strip()


def clean_terms(items: list[str]) -> list[str]:
    """Job-description terms as keywords: a list of examples becomes the examples themselves, a piece of the posting's
    prose becomes the keyword in it, or is dropped when it has none, since the coverage rule would otherwise push it
    into a bullet word for word. Duplicates go."""
    terms: list[str] = []
    for item in items:
        for part in _examples_as_items(_DETAIL_BRACKET.sub("", clean_text(item))):
            term = _keyword_of(part)
            if is_keyword(term) and term.lower() not in {t.lower() for t in terms}:
                terms.append(term)
    return terms


def clean_analysis(analysis: JDAnalysis, names: list[str] | tuple = ()) -> JDAnalysis:
    """The analysis with keyword lists only (changed in place, and returned). A certification (one of `names`, the
    model's picks and the person's own, or plainly a credential) is never a must-have or nice-to-have skill (the
    prompt says so, and this makes sure): the fix round would otherwise push its name into a bullet, claiming it.
    The keywords keep it, so the keyword score stays honest and a printed one counts."""
    names = list(names)
    analysis.must_have_skills = [t for t in clean_terms(analysis.must_have_skills) if not names_certification(t, names)]
    analysis.nice_to_have_skills = [t for t in clean_terms(analysis.nice_to_have_skills)
                                    if not names_certification(t, names)]
    analysis.keywords = clean_terms(analysis.keywords)
    return analysis


# ------------------------------------------------------------------ certifications
# The model picks the 4 certifications that best support the application (see CERTIFICATIONS in the prompt). Only
# the person's own are printed, in their own section; the others are shown on the Tailor page to tick ("I have it")
# or to earn. A certification is never named in the headline, summary, skills or bullets.
CERT_NAME_LIMIT = 120
_BRACKETED = re.compile(r"\(([A-Za-z0-9][A-Za-z0-9+&./ -]{0,19})\)")
# An abbreviation or exam code ('PMP', 'CKAD', 'SAA-C03', 'AZ-900'): capitals and digits, with 2+ capitals
_CERT_ABBREVIATION = re.compile(r"(?=(?:[^A-Z]*[A-Z]){2})[A-Z0-9][A-Z0-9+&./-]{1,14}")
# Short names that are tools, platforms or issuers, never one certification ('Solutions Architect (AWS)')
_COMMON_ACRONYMS = frozenset({
    "AWS", "GCP", "IBM", "SAP", "SQL", "API", "AI", "ML", "NLP", "ETL", "UI", "UX", "IT", "QA", "BI", "ERP", "CRM",
    "HTML", "CSS", "JSON", "XML", "REST", "HTTP", "DNS", "VPN", "LLM", "RAG", "SDK", "OOP", "TDD", "BDD", "SRE", "IOT",
    "IAM", "EC2", "EKS", "ECS", "AKS", "GKE", "CI", "CD", "OCI", "IEEE", "ISC", "ISC2", "PMI", "CNCF", "EDB", "NVIDIA",
})
# Words that make a name read as a credential. A certification named like a tool ('GitHub Copilot', 'GitHub
# Actions') or a skill ('Python') is printed when picked, but never makes a skill or a sentence count as one.
_CREDENTIAL_NAME = re.compile(
    r"\b(?:certified|certifications?|certificate|licen[cs](?:e|ed|es)|credential|associate|professional|expert|"
    r"practitioner|specialist|fundamentals|foundations?|comptia|isc2|isaca|giac|pmi)\b", re.IGNORECASE)
# Before 'certification' or 'certified', these name work on a product or a standard, not a person's credential
_STANDARD_WORDS = frozenset({
    "faa", "fcc", "fda", "iso", "soc", "hipaa", "pci", "fedramp", "gdpr", "sox", "ce", "ul", "nist", "cmmc", "itar",
    "cmmi", "easa", "iec", "type", "product", "safety", "compliance", "regulatory", "airworthiness", "process",
    "device", "environmental", "energy", "organic", "vendor", "partner", "quality",
})
_CERT_TAIL = re.compile(r"(?:^|[\s-])([A-Za-z][\w+.]*)[\s-]+(?:certified|certifications?|licen[cs]es?)$", re.IGNORECASE)
_CERT_GROUP = re.compile(r"certif|licen[cs]|credential", re.IGNORECASE)  # a skills group of certifications
# A claim in a sentence: 'AWS Certified Solutions Architect', 'AWS-certified', 'PMP certification', 'Certified ...'
_CERT_CLAIM = re.compile(r"\b(?:[A-Z][\w+.]*[- ])+(?:[Cc]ertified|[Cc]ertifications?)\b(?:\s+[A-Z][\w+.-]*)*"
                         r"|\bCertified(?:\s+[A-Z][\w+.-]*)+")


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _without_brackets(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()


def _in_order(letters: str, capitals: str) -> bool:
    """Every letter of the abbreviation is one of the name's capitals, in order ('PCA' in 'Google Cloud Professional
    Cloud Architect')."""
    rest = iter(capitals)
    return bool(letters) and all(any(c == letter for c in rest) for letter in letters)


def _abbreviation(name: str) -> str:
    """The abbreviation or exam code in a name's brackets that stands for the name, or ''. Its letters are the name's
    capitals in order ('PMP' in 'Project Management Professional (PMP)', 'CSM' in 'Certified ScrumMaster (CSM)'), or
    it is an exam code after a name of two or more words ('SAA-C03'). Not '(Coursera)', '(Python)', '(Associate)',
    '(AWS)' after 'Solutions Architect Associate', or '(CKA)' after the one word 'Kubernetes'."""
    bare = _without_brackets(name)
    capitals = "".join(ch for ch in bare if ch.isupper())
    for inner in (part.strip() for part in _BRACKETED.findall(name)):
        if not _CERT_ABBREVIATION.fullmatch(inner) or inner.upper() in _COMMON_ACRONYMS:
            continue
        if any(ch.isdigit() for ch in inner) and len(bare.split()) >= 2:
            return inner
        if _in_order("".join(ch for ch in inner.upper() if ch.isalpha()), capitals):
            return inner
    return ""


def _spellings(name: str) -> set[str]:
    """The name's letters and digits in lower case ('AWS Certified Solutions Architect – Associate' ~ 'AWS Certified
    Solutions Architect Associate'), plus the name without its abbreviation ('Certified ScrumMaster (CSM)' ~
    'Certified ScrumMaster')."""
    spellings = {_letters(name)}
    if _abbreviation(name) and len(_without_brackets(name).split()) >= 2:
        spellings.add(_letters(_without_brackets(name)))
    spellings.discard("")
    return spellings


def same_certification(a: str, b: str) -> bool:
    """One certification in two spellings. Two different abbreviations or exam codes are two certifications ('...
    (SAA-C03)' and '... (SAP-C02)'). An abbreviation matches only a name that is that abbreviation alone ('PMP' ~
    'Project Management Professional (PMP)'), so two names that share a bracket never match."""
    a_short, b_short = _abbreviation(a), _abbreviation(b)
    if a_short and b_short and _letters(a_short) != _letters(b_short):
        return False
    if _spellings(a) & _spellings(b):
        return True
    a_alone = a.strip() if _CERT_ABBREVIATION.fullmatch(a.strip()) else ""
    b_alone = b.strip() if _CERT_ABBREVIATION.fullmatch(b.strip()) else ""
    return bool(a_alone and _letters(a_alone) == _letters(b_short) or b_alone and _letters(b_alone) == _letters(a_short))


def certification_terms(names: list[str]) -> list[str]:
    """What marks a certification in a job-description term, a skill or a sentence, for each name that reads as a
    credential: the name without its bracket, its abbreviation ('CISSP') and its capitals ('CKAD' for 'Certified
    Kubernetes Application Developer'); or a name that is an abbreviation alone ('PMP'). A name that is also a tool
    or a skill ('GitHub Copilot', 'GitHub Actions', 'Python', 'AWS') marks nothing."""
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if len(term) >= 2 and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)

    for name in (n.strip() for n in names):
        if _CERT_ABBREVIATION.fullmatch(name):
            if name.upper() not in _COMMON_ACRONYMS:
                add(name)
            continue
        if not _CREDENTIAL_NAME.search(name):
            continue
        bare = _without_brackets(name)
        if len(bare.split()) >= 2:
            add(bare)
        add(_abbreviation(name))
        capitals = "".join(ch for ch in bare if ch.isupper())
        if 3 <= len(capitals) <= 6 and capitals not in _COMMON_ACRONYMS:
            add(capitals)
    return terms


def is_credential_phrase(term: str) -> bool:
    """A term, skill or phrase that names a person's credential: 'Certified ScrumMaster', 'AWS Certified Solutions
    Architect', 'PMP certification', 'AWS-certified', 'RN license'. Work on a product or to a standard is not one:
    'SOC 2 certification', 'FAA type certification', 'HIPAA-certified', 'certificate management', 'software
    licensing'."""
    text = term.strip()
    if re.match(r"(?i)certified\b", text):
        return True
    words = re.findall(r"[A-Za-z0-9+]+", text)
    for index, word in enumerate(words[1:], 1):
        if word == "Certified":
            before = words[index - 1]
            return before.lower() not in _STANDARD_WORDS and not before.isdigit()
    tail = _CERT_TAIL.search(text)
    return bool(tail and tail.group(1)[:1].isupper() and tail.group(1).lower() not in _STANDARD_WORDS)


def names_certification(term: str, names: list[str] | tuple = ()) -> bool:
    """True for a job-description term or a skill that is a certification or license: one of these names in any
    spelling ('CISSP' for 'Certified Information Systems Security Professional (CISSP)'), or plainly a credential
    ('PMP certification', 'Certified ScrumMaster', 'RN license'). 'AWS', 'GitHub Actions', 'certificate management',
    'software licensing' and 'SOC 2 certification' are skills or work."""
    key = _letters(term)
    return any(key == _letters(t) for t in certification_terms(list(names))) or is_credential_phrase(term)


def resolve_certifications(
    profile: Profile, picks: list[CertificationPick], today=None
) -> tuple[list[Certification], list[CertificationChoice]]:
    """(printed, rows) for the model's picks. A pick is one of the person's own when its number points at one of
    their certifications, or when its name is one of theirs in another spelling: that certification is then printed
    exactly as the person wrote it, with its dates. One of theirs that has expired is never printed (nor shown).
    Any other pick is a row the person can tick on the Tailor page, never printed on its own. Duplicates go; at
    most MAX_CERTIFICATIONS rows, in the model's order (best first)."""
    own = profile.certifications
    rows: list[CertificationChoice] = []
    for pick in picks:
        name = clean_text(pick.name)[:CERT_NAME_LIMIT].strip()
        index = pick.candidate_index if 0 <= pick.candidate_index < len(own) else -1
        if index < 0 and name:
            index = next((i for i, cert in enumerate(own) if same_certification(cert.name, name)), -1)
        if index >= 0:
            cert = own[index]
            if is_expired(cert.expires, today):
                continue
            row = CertificationChoice(name=cert.name, issuer=cert.issuer, year=cert.year, expires=cert.expires,
                                      have=True)
        elif name:
            row = CertificationChoice(name=name, issuer=clean_text(pick.issuer)[:CERT_NAME_LIMIT].strip())
        else:
            continue
        if any(same_certification(row.name, other.name) for other in rows):
            continue
        rows.append(row)
        if len(rows) == MAX_CERTIFICATIONS:
            break
    return printed_certifications(rows), rows


def printed_certifications(rows: list[CertificationChoice]) -> list[Certification]:
    """What the resume prints for these rows: the ones the person has, best first."""
    return [Certification(name=r.name, issuer=r.issuer, year=r.year, expires=r.expires)
            for r in rows if r.have][:MAX_CERTIFICATIONS]


def change_certifications(profile: Profile, added: list[Certification], dropped: list[str]) -> None:
    """Remember ticks from the Tailor page on a profile (changed in place): `added` are certifications the person
    has (one that is already there in another spelling is not added twice); `dropped` are names of ones they
    unticked (every spelling of each goes)."""
    kept = [cert for cert in profile.certifications if not any(same_certification(cert.name, d) for d in dropped)]
    for cert in added:
        if not any(same_certification(cert.name, other.name) for other in kept):
            kept.append(cert.model_copy())
    profile.certifications = kept


def assert_certifications_held(resume: Resume) -> None:
    """The one rule that never bends: every printed certification is one the person has."""
    for cert in resume.certifications:
        if cert not in resume.profile.certifications:
            raise ValueError(f"'{cert.name}' is not one of the person's certifications, so it cannot be printed.")


def remove_certification_skills(resume: Resume, names: list[str]) -> list[str]:
    """Take certifications out of the skills list (changed in place): they are printed only in their own section,
    and only when the person has them. A group of certifications goes whole; a group left empty goes too. Returns
    what was removed."""
    terms = {_letters(term) for term in certification_terms(names)}
    removed: list[str] = []
    for group in resume.skills:
        if _CERT_GROUP.search(group.category):
            removed += group.skills
            group.skills = []
            continue
        kept = []
        for item in group.skills:
            (removed if _letters(item) in terms or is_credential_phrase(item) else kept).append(item)
        group.skills = kept
    if removed and any(group.skills for group in resume.skills):
        resume.skills = [group for group in resume.skills if group.skills]
    return removed


def certification_mentions(resume: Resume, names: list[str]) -> list[tuple[str, str]]:
    """(where, name) for each certification named in the headline, summary or a bullet ('summary',
    'experience:<company_index>'): one of these names, or any credential claim ('AWS Certified Solutions Architect',
    'AWS-certified', 'PMP certification'). Certifications belong only in their own section: a name in the text would
    claim one the person may not have, and unticking it on the Tailor page could not take it off the resume."""
    terms = certification_terms(names)
    places = [("headline", resume.headline), ("summary", resume.summary)]
    places += [(f"experience:{exp.company_index}", bullet) for exp in resume.experience for bullet in exp.bullets]
    found: list[tuple[str, str]] = []
    for where, text in places:
        named = [term for term in terms if mentions(text, term)]
        named += [m.group(0).strip() for m in _CERT_CLAIM.finditer(text) if is_credential_phrase(m.group(0))]
        for term in named:
            if (where, term) not in found:
                found.append((where, term))
    return found


# ------------------------------------------------------------------ merge
def build_resume(profile: Profile, out: TailorOutput, from_model: bool = True) -> Resume:
    """Raise ValueError (retried by the pipeline with this message) if the output cannot be used. The model's summary
    and bullets lose their dash punctuation; text the user typed in the editor (from_model=False) is kept as typed
    (run_checks notes its dashes)."""
    prose = clean_prose if from_model else clean_text
    sections = out.sections
    n = len(profile.companies)
    indices = [e.company_index for e in sections.experience]
    if sorted(indices) != list(range(n)):
        raise ValueError(
            f"`sections.experience` must have exactly one entry for each company_index 0..{n - 1} "
            f"(got {sorted(indices)})."
        )
    headline = clean_text(sections.headline)
    summary = prose(sections.summary)
    if not headline or not summary:
        raise ValueError("`headline` and `summary` must not be empty.")

    skills: list[SkillGroup] = []
    seen: set[str] = set()
    for group in sections.skills:
        cleaned = []
        for raw in group.skills:
            for item in _examples_as_items(clean_text(raw)):  # "AI tools like Copilot, Cursor" -> the tools
                if item and item.lower() not in seen:
                    seen.add(item.lower())
                    cleaned.append(item)
        if cleaned and clean_text(group.category):
            skills.append(SkillGroup(category=clean_text(group.category), skills=cleaned))
    if not skills:
        raise ValueError("`skills` must contain at least one group with skills.")

    experience: list[ResumeExperience] = []
    for entry in sorted(sections.experience, key=lambda e: e.company_index):
        title = clean_text(entry.job_title)
        bullets = [b for b in (prose(_BULLET_LEAD.sub("", x)) for x in entry.bullets) if b]
        if not title or len(title) > 80:
            raise ValueError(f"Job title for company_index {entry.company_index} is empty or too long.")
        if not bullets:
            raise ValueError(f"company_index {entry.company_index} has no bullets.")
        experience.append(ResumeExperience(company_index=entry.company_index, job_title=title, bullets=bullets))

    resume = Resume(
        profile=profile.model_copy(deep=True), headline=headline, summary=summary, skills=skills, experience=experience
    )
    assert_fixed_fields_intact(profile, resume)
    return resume


_INDEX_LABEL = re.compile(r"^\s*\[\d+\]\s*")
_FIELD_LABEL = re.compile(r"^\s*(?:headline|summary)\s*:\s*", re.IGNORECASE)
# An edit whose text is clearly another bullet of the same employer rewritten (it shares at least half its words,
# and clearly more than with the bullet it claims to replace) points at the wrong bullet_index.
SHIFT_MIN_OVERLAP, SHIFT_MARGIN = 0.5, 0.2


def _overlap(a: str, b: str) -> float:
    words_a = {w.lower() for w in _WORD.findall(a)}
    words_b = {w.lower() for w in _WORD.findall(b)}
    return len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0.0


def apply_revision(out: TailorOutput, draft: Resume, rev: TailorRevision) -> TailorOutput:
    """The first answer with a fix applied: `draft` is the resume made from `out` (cleaned, titles corrected) and
    exactly what the model was shown; `rev` lists only what changes, so everything else comes from `draft`. The
    analysis, the career plan and the number of bullets stay as they are. Raises ValueError for an edit that
    cannot be applied as meant, so the caller keeps the first draft. Neither input is changed."""
    bullets = {exp.company_index: list(exp.bullets) for exp in draft.experience}
    originals = {index: list(items) for index, items in bullets.items()}
    titles = {exp.company_index: exp.job_title for exp in draft.experience}
    for change in rev.titles:
        if change.company_index not in titles:
            raise ValueError(f"titles: there is no company_index {change.company_index}.")
        if change.job_title.strip():
            titles[change.company_index] = change.job_title
    seen: set[tuple[int, int]] = set()
    for change in rev.bullets:
        where = f"company_index {change.company_index}, bullet_index {change.bullet_index}"
        current = bullets.get(change.company_index)
        if current is None or not 0 <= change.bullet_index < len(current):
            raise ValueError(f"bullets: there is no bullet at {where}.")
        if (change.company_index, change.bullet_index) in seen:
            raise ValueError(f"bullets: {where} is changed twice.")
        seen.add((change.company_index, change.bullet_index))
        text = _INDEX_LABEL.sub("", change.text).strip()
        if not text:
            continue
        own = originals[change.company_index]
        target = _overlap(text, own[change.bullet_index])
        for other_index, other in enumerate(own):
            other_overlap = _overlap(text, other)
            if other_index != change.bullet_index and other_overlap >= SHIFT_MIN_OVERLAP \
                    and other_overlap >= target + SHIFT_MARGIN:
                raise ValueError(f"bullets: the new text for {where} rewrites bullet_index {other_index} instead.")
        current[change.bullet_index] = text
    skills = [SkillGroup(category=group.category, skills=list(group.skills)) for group in draft.skills]
    for change in rev.skill_changes:
        old, new = clean_text(change.old).lower(), clean_text(change.new)
        for group in skills:
            group.skills = [new if item.lower() == old else item for item in group.skills if item.lower() != old or new]
    sections = TailoredSections(
        headline=_FIELD_LABEL.sub("", rev.headline).strip() or draft.headline,
        summary=_FIELD_LABEL.sub("", rev.summary).strip() or draft.summary,
        skills=skills,
        experience=[ExperienceOut(company_index=i, job_title=titles[i], bullets=bullets[i]) for i in sorted(bullets)],
    )
    return out.model_copy(update={"sections": sections})


def assert_fixed_fields_intact(original: Profile, resume: Resume) -> None:
    if resume.profile.model_dump() != original.model_dump():
        raise ValueError("Fixed profile fields (name, contact, companies, dates, education, certifications) were "
                         "altered.")
    if [e.company_index for e in resume.experience] != list(range(len(original.companies))):
        raise ValueError("Experience entries no longer map one-to-one onto the profile's companies.")


# ------------------------------------------------------------------ text helpers
def resume_text(resume: Resume) -> str:
    parts = [resume.headline, resume.summary]
    for group in resume.skills:
        parts.append(group.category)
        parts.extend(group.skills)
    for exp in resume.experience:
        parts.append(exp.job_title)
        parts.extend(exp.bullets)
    parts.extend(cert.name for cert in resume.certifications)  # a printed certification the posting names is found
    return "\n".join(parts)


def contains_term(text: str, term: str) -> bool:
    term = term.strip()
    if not term:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9+#])"
    flags = 0 if len(term) <= 3 else re.IGNORECASE
    return re.search(pattern, text, flags) is not None


def either_spelling(word: str) -> str:
    """A regex for the word in both spelling styles; punctuation inside it stays literal ('modeling/warehousing',
    "Organization's", '(search')."""
    parts = []
    for piece in re.split(r"([A-Za-z]+)", word):
        forms = spelling.twins(piece) if piece.isalpha() else frozenset()
        if len(forms) > 1:
            parts.append("(?:" + "|".join(sorted(map(re.escape, forms))) + ")")
        elif piece:
            parts.append(re.escape(piece))
    return "".join(parts)


def _word_pattern(word: str, last: bool) -> str:
    """One word of a term, in either spelling style ('optimisation' ~ 'optimization'); the last one also matches its
    plural or singular ('API' ~ 'APIs', 'pipelines')."""
    if not word.isalpha():  # 'Node.js', 'CI/CD' stay literal; each part of 'modeling/warehousing' in either spelling
        return either_spelling(word)
    forms = spelling.twins(word) if len(word) > 3 else ()  # 'DATA MODELING' too: the match ignores case
    if len(forms) > 1:
        return "(?:" + "|".join(_one_word_pattern(form, last) for form in sorted(forms)) + ")"
    return _one_word_pattern(word, last)


def _one_word_pattern(word: str, last: bool) -> str:
    if not last:
        return re.escape(word)
    if word.isupper() or len(word) <= 3:  # acronyms: 'API' ~ 'APIs'
        return re.escape(word) + "s?"
    if word.lower().endswith("ss"):
        return re.escape(word) + "(?:es)?"
    if word.lower().endswith("s"):
        return re.escape(word[:-1]) + "(?:s|es)?"
    return re.escape(word) + "(?:s|es)?"


def _term_regex(term: str, case_sensitive: bool | None = None) -> re.Pattern | None:
    words = [w for w in re.split(r"[\s-]+", term.strip()) if w]
    if not words:
        return None
    body = r"[\s-]+".join(_word_pattern(w, i == len(words) - 1) for i, w in enumerate(words))
    if case_sensitive is None:
        case_sensitive = len(term.strip()) <= 3
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9+#])", 0 if case_sensitive else re.IGNORECASE)


def mentions(text: str, term: str, case_sensitive: bool | None = None) -> bool:
    """contains_term, tolerant of the forms an ATS also matches: hyphen or space between words
    ('event-driven' ~ 'event driven') and a plural or singular last word ('REST APIs' ~ 'REST API')."""
    pattern = _term_regex(term, case_sensitive)
    return pattern is not None and pattern.search(text) is not None


def mention_count(text: str, term: str) -> int:
    """How many times `mentions` would find the term."""
    pattern = _term_regex(term)
    return len(pattern.findall(text)) if pattern else 0


def _unique(terms: list[str]) -> list[str]:
    out: list[str] = []
    for term in terms:
        term = term.strip()
        if term and term.lower() not in {t.lower() for t in out}:
            out.append(term)
    return out


def evidence_text(resume: Resume) -> str:
    """Where a skill counts as shown, not just listed: the summary and the bullets."""
    return "\n".join([resume.summary, *(b for exp in resume.experience for b in exp.bullets)])


def unshown_must_haves(resume: Resume, analysis: JDAnalysis) -> tuple[list[str], list[str]]:
    """(absent, listed_only): must-have terms found nowhere in the resume, and ones that appear only in the
    skills list, headline or titles but in no summary sentence or bullet."""
    evidence, everything = evidence_text(resume), resume_text(resume)
    absent, listed_only = [], []
    for term in _unique(analysis.must_have_skills):
        if mentions(evidence, term):
            continue
        (listed_only if mentions(everything, term) else absent).append(term)
    return absent, listed_only


# ------------------------------------------------------------------ checks
def keyword_coverage(resume: Resume, analysis: JDAnalysis) -> Coverage:
    terms = _unique([*analysis.must_have_skills, *analysis.keywords])
    text = resume_text(resume)
    found = [t for t in terms if mentions(text, t)]
    missing = [t for t in terms if t not in found]
    _absent, listed_only = unshown_must_haves(resume, analysis)
    return Coverage(
        score=round(len(found) / len(terms), 3) if terms else 0.0, found=found, missing=missing,
        skills_only=listed_only,
    )


MIN_SUMMARY_WORDS = 55
MIN_BULLET_WORDS = 15
WEAK_OPENERS = (
    "responsible for", "worked on", "helped", "assisted", "participated in", "involved in", "utilized", "utilised",
    "build and maintain", "built and maintained",
)


def thin_text_problems(resume: Resume) -> list[str]:
    """Signs of a weak draft: a short summary, two or more thin bullets, bullets that open like a duty list."""
    problems: list[str] = []
    words = word_count(resume.summary)
    if words < MIN_SUMMARY_WORDS:
        problems.append(f"the summary has only {words} words")
    bullets = [b for exp in resume.experience for b in exp.bullets]
    thin = [b for b in bullets if word_count(b) < MIN_BULLET_WORDS]
    if len(thin) >= 2:
        problems.append(
            f"{len(thin)} bullets are under {MIN_BULLET_WORDS} words (" + "; ".join(f"'{b}'" for b in thin[:3]) + ")"
        )
    weak = [b for b in bullets if b.lower().startswith(WEAK_OPENERS)]
    if weak:
        problems.append("these bullets open like a duty list: " + "; ".join(f"'{b}'" for b in weak[:3]))
    return problems


# A capitalized pronoun counts only where a sentence starts, and a lowercase one never inside a hyphenated word,
# so standard names stay fine: 'Know Your Customer (KYC)', 'Bring Your Own Device', 'pay-as-you-go'.
_POSTING_VOICE = re.compile(
    r"(?:^|(?<=[.!?]\s))(?:You|Your|Yours|Yourself|We|Our|Ours)\b(?!-)"
    r"|(?<![\w-])(?:you|your|yours|yourself|we|our|ours)(?![\w-])"
)


MIN_COPIED_WORDS = 4  # a run of four words shared with the posting is a copied sentence piece, not a keyword
MIN_COPIED_CONTENT = 3  # ... when at least three of them carry meaning
MIN_COPIED_PROSE = 2  # ... and at least two of those are not keywords: the JD's keywords are meant to be matched
MIN_COPIED_FOR_RETRY = 2  # one copied run is only noted; two or more are worth the single fix round


_WORD_PART = re.compile(r"[A-Za-z][\w+#]*")  # a hyphen splits: 'end-to-end' is matched as 'end to end'
_VOWEL = re.compile(r"[aeiouy]")


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    """(lowercase word, start, end) for each word, and each part of a hyphenated word."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_PART.finditer(text)]


def _in_either_spelling(item: str, text: str) -> bool:
    """The item appears in the text, in either spelling style (a JD skill the run spelled the other way)."""
    return item.lower() in text.lower() or spelling.us_text(item) in spelling.us_text(text)


def _stem(word: str) -> str:
    """One form for 'contribute', 'contributes', 'contributed' and 'contributing' (and 'identify', 'identified';
    'focus', 'focused'; 'need', 'needed'), so a copy is found in any tense, and in either spelling style
    ('optimised' ~ 'optimized')."""
    word = spelling.us_word(word)
    if len(word) > 4 and word.endswith(("ies", "ied")):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "e", "s"):
        stem = word[:-len(suffix)]
        if word.endswith(suffix) and len(stem) >= 2 and _VOWEL.search(stem):
            if not (suffix == "s" and stem.endswith("u")) and not (suffix == "ed" and stem.endswith("e")):
                word = stem  # but 'focus' and 'need' are not 'focu' + s and 'ne' + ed
            break
    return word[:-1] if len(word) > 3 and word[-1] == word[-2] else word  # 'debugged' -> 'debugg' -> 'debug'


def copied_posting_spans(resume: Resume, jd: str, analysis: JDAnalysis) -> list[str]:
    """Pieces of the summary and the bullets copied word for word (in any tense) from the job description: runs of
    4+ words found in it, holding at least 3 meaningful words of which at least 2 are not keywords. Keywords (the
    JD's, the target role and the resume's own skills) may be used as they are, so a run made of them ('TypeScript,
    React and PostgreSQL', 'Senior Backend Engineer with') is fine. Overlapping runs are reported once."""
    posting = [_stem(word) for word, _start, _end in _word_spans(clean_text(jd))]
    n = MIN_COPIED_WORDS
    shared = {tuple(posting[i:i + n]) for i in range(len(posting) - n + 1)}
    terms = [*analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords,
             *(item for group in resume.skills for item in group.skills)]
    known = {_stem(word) for term in [*clean_terms(terms), analysis.target_role, analysis.general_role]
             for word, _start, _end in _word_spans(term)}
    places = [("summary", resume.summary)] + [
        (resume.profile.companies[exp.company_index].name, bullet) for exp in resume.experience for bullet in exp.bullets
    ]
    found: list[str] = []
    for where, text in places:
        words = _word_spans(text)
        stems = [_stem(word) for word, _start, _end in words]
        runs: list[list[int]] = []
        for i in range(len(words) - n + 1):
            if tuple(stems[i:i + n]) in shared:
                if runs and i <= runs[-1][1] + 1:  # windows that overlap or touch become one run
                    runs[-1][1] = i + n - 1
                else:
                    runs.append([i, i + n - 1])
        for first, last in runs:  # judged as a whole: 'Improve the patient experience across all touchpoints'
            content = {stems[j] for j in range(first, last + 1) if words[j][0] not in _SMALL_WORDS}
            if len(content) >= MIN_COPIED_CONTENT and len(content - known) >= MIN_COPIED_PROSE:
                found.append(f"'{text[words[first][1]:words[last][2]]}' ({where})")
    return found


def skill_phrase_problems(resume: Resume) -> list[str]:
    """Skills-list items written as a piece of the posting's prose ('advanced SQL statements', 'AWS-hosted
    systems') rather than as the keyword itself ('SQL', 'AWS'). Lists are already split into their items."""
    return [item for group in resume.skills for item in group.skills if not is_keyword(item)]


def posting_voice_problems(resume: Resume) -> list[str]:
    """Summary sentences and bullets that talk like the posting ("Refine your craft by..."): a resume never
    says you, your, we or our, so these were copied from the JD."""
    texts = [resume.summary, *(b for exp in resume.experience for b in exp.bullets)]
    return [text for text in texts if _POSTING_VOICE.search(text)]


MAX_PHRASE_REPEATS = 2
MIN_PHRASE_WORDS = 3  # 'modern software development', 'test-driven development'; not 'Spring Boot' or 'REST APIs'
_NAME_JOINERS = {"on", "of", "and", "for", "the", "&", "to", "in", "with"}


def _looks_like_name(term: str) -> bool:
    """'Azure Data Factory', 'Ruby on Rails', 'Microsoft Power BI': every main word capitalized, so a product or
    tool, which recurs wherever it was really used. The JD's spelling is kept, and descriptive phrases are lowercase."""
    words = [w for w in re.split(r"[\s-]+", term.strip()) if w and w.lower() not in _NAME_JOINERS]
    return bool(words) and all(w[:1].isupper() or w[:1].isdigit() for w in words)


def repeated_phrase_problems(resume: Resume, analysis: JDAnalysis) -> list[str]:
    """Descriptive JD phrases of three or more words used more than twice in the summary and bullets: keyword
    stuffing. Hyphen-joined parts count as words, as in matching ('test-driven development' is three)."""
    evidence = evidence_text(resume)
    terms = [t for t in _unique([*analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords])
             if len(re.split(r"[\s-]+", t.strip())) >= MIN_PHRASE_WORDS and not _looks_like_name(t)]
    problems = []
    for term in terms:
        count = mention_count(evidence, term)
        if count > MAX_PHRASE_REPEATS:
            problems.append(f"'{term}' ({count} times)")
    return problems


_QUALITY_WORDS = re.compile(
    r"\b(scalable(?!\s+vector\s+graphics)|maintainable|low[\s-]+maintenance|high[\s-]+quality"
    r"|robust(?!\s+(?:control|optimi[sz]ation|statistics|regression)\b)|reliable)\b",
    re.IGNORECASE,
)


def quality_skill_problems(resume: Resume) -> list[str]:
    """Skills-list items that are qualities or outcomes ('High-Quality Code', 'Scalable and Maintainable
    Applications') rather than something a person uses or does."""
    return [skill for group in resume.skills for skill in group.skills if _QUALITY_WORDS.search(skill)]


# Words that make a resume read as machine-written or as marketing copy (the prompt's PLAIN WORDS rule). Words
# with an everyday technical meaning stay allowed: orchestration (Kubernetes), dynamic (pricing), enhance.
_BUZZ = re.compile(
    r"\b(?:leverag(?:e|ed|es|ing)|utili[sz](?:e|ed|es|ing|ation)|spearhead(?:ed|s|ing)?|seamless(?:ly)?|robust(?:ly|ness)?"
    r"|cutting[\s-]edge|state[\s-]of[\s-]the[\s-]art|synerg(?:y|ies|istic)|holistic(?:ally)?|innovative|transformative"
    r"|pivotal|paradigm|best[\s-]in[\s-]class|world[\s-]class|game[\s-]chang(?:er|ing)|delv(?:e|ed|es|ing)"
    r"|foster(?:ed|s|ing)?|empower(?:ed|s|ing|ment)?|meticulous(?:ly)?|plethora|myriad|in order to|a wide range of)\b",
    re.IGNORECASE,
)
# A vague result says nothing a recruiter can check ("improved operational efficiency", "increased decision making"):
# a recruiter's advice is to say what changed instead ("cut weekly reporting from 8 hours to 2"). Specific nouns stay
# allowed: query performance, developer productivity, energy efficiency, visibility into failed payments.
_VAGUE_VERB = (r"(?:improv(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|enhanc(?:e|es|ed|ing)|boost(?:s|ed|ing)?"
               r"|dr(?:ive|ives|ove|iving|iven)|strengthen(?:s|ed|ing)?|optimi[sz](?:e|es|ed|ing)"
               r"|maximi[sz](?:e|es|ed|ing)|elevat(?:e|es|ed|ing))")
_VAGUE_QUAL = (r"(?:(?:overall|operational|organi[sz]ational|business|team|process|system|cross[\s-]functional"
               r"|cross[\s-]team|company[\s-]wide|workflow|general|data[\s-]driven)\s+)?")
# A technical term, not a result: 'performance tests', 'value objects', 'value-based pricing', 'value based care',
# 'performance-critical'; 'alignment based on shared OKRs' is still vague
_COMPOUND = (r"(?!-\w)(?!\s+(?:objects?|types?|at\s+risk|streams?|tests?|testing|reviews?|schemas?|insights|accuracy"
             r"|scores?|pages?|cache|caching|monitoring|metrics|profiling|tuning|engineering|budgets?|dashboards?"
             r"|regressions?|bottlenecks?|benchmarks?|tooling|alert(?:s|ing)|observability|management|reporting|reports"
             r"|logging|tracking|analytics|analysis|attribution|tools|features|baselines|slos"
             r"|based(?!\s+(?:on|upon|around|off)\b))\b)")
# 'the performance of the checkout API' names what got better; 'the efficiency of our operations' does not. A word
# for people is the owner after any two words ('the data engineering team'); operations, functions, groups and units
# only after a determiner or an organisation's word ('our operations', 'business operations'), never after a
# technical one ('CRUD operations', 'AWS Lambda functions'). The owner ends its noun phrase: 'the business logic
# layer', "the company's order API" and 'business-critical reports' are specific.
_ORG_WORD = (r"(?:business|company|corporate|commercial|day[\s-]to[\s-]day|daily|core|internal|global|regional"
             r"|cross[\s-]functional|back[\s-]office|front[\s-]?line|finance|financial|sales|marketing|hr|it|people"
             r"|support|customer|engineering|operating)\s+")
_OWNER_ENDS = (r"(?![\w'’-])(?!\s+(?!(?:by|with|for|in|of|on|at|to|from|across|between|into|via|over|under|through"
               r"|throughout|within|without|during|using|after|before|around|since|than|as|and|or|but|so|while|when"
               r"|where|which|that|who|whose|because|thanks|each|every|i|we|they|you)\b)[a-z])")
_GENERIC_OWNER = (r"(?:(?:the|our|its|their|your|my|his|her|all)\s+)?(?:[\w&.-]+['’]s\s+)?"  # "Northwind's"
                  r"(?:(?:entire|whole|overall|wider|broader)\s+)?"  # an intensifier is not one of the two words
                  r"(?:(?:(?!(?:by|with|for|in|of|on|at|to|from|across|between|into|via)\b)[\w&-]+\s+){0,2}?"
                  r"(?:teams?|staff|employees|people|company|companies|business(?:es)?|organi[sz]ations?|orgs?"
                  r"|departments?|stakeholders|firm|divisions?)"
                  rf"|(?:{_ORG_WORD}){{0,2}}(?<!\bof\s)(?:operations|functions?|groups?|units?)){_OWNER_ENDS}")
_VAGUE_NOUN = (r"(?:efficienc(?:y|ies)|productivity|decision[\s-]making|collaboration|visibility(?!\s+into\b)|alignment"
               r"|effectiveness|outcomes|agility|performance|results|engagement|value|excellence)"
               rf"(?!\s+of\b(?!\s+{_GENERIC_OWNER})){_COMPOUND}")
# Can be made specific by a figure in the same clause: 'improved team productivity by 18% (from 11 to 13 tickets)'
_VAGUE_RESULT = re.compile(
    rf"(?<![\w-]){_VAGUE_VERB}\s+(?:(?:the|its|their)\s+)?{_VAGUE_QUAL}{_VAGUE_NOUN}\b"  # not 'event-driven ...'
    r"|\bstreamlin(?:e|es|ed|ing)\s+(?:(?:the|its|their|\w+)\s+)?(?:process(?:es)?|operations|workflows?|procedures)\b"
    rf"{_COMPOUND}", re.IGNORECASE)
# Cliches whatever follows them: 'Results-driven Data Engineer with 7+ years ...' is no more specific for the figure
_VAGUE_CLICHE = re.compile(
    r"\bdata[\s-]driven\s+decision[\s-]making\b|\boperational\s+excellence\b|\bresults[\s-]driven\b"
    rf"|\bdeliver(?:s|ed|ing)?\s+(?:(?:strong|measurable|tangible|significant|real)\s+)?results\b{_COMPOUND}"
    rf"|\badd(?:s|ed|ing)?\s+(?:(?:significant|real|tangible)\s+)?value\b{_COMPOUND}", re.IGNORECASE)
# A figure that measures something: not a year (2024, the 2010s) unless a count follows ('2047 requests'), no digit
# inside a name (Q3, S3, EC2, B2B, GPT-4, Web3), and no number that names a version, a standard or a product (Java 17,
# SOC 2, Fortune 500, 5G, 2FA, 24/7, 1:1s) unless a measure follows. 'sub-50ms', 'top-10' and '(-38%)' are figures.
_MEASURE = re.compile(r"(?:(?<![\w-])|(?<=[\s(\[]-)|(?<=\bsub-)|(?<=\btop-)|(?<=\bunder-)|(?<=\bover-)|(?<=\bnear-))"
                      r"([$€£]?)(\d+(?:[.,]\d+)*(?:[/:]\d+)?)([A-Za-z]*)", re.IGNORECASE)
_NAME_NUMBER = re.compile(r"[2-6]G|[23]D|2FA|[1-3]P|24/7|1:1s?")  # not '16G heaps' or '40ms'
# 'to Java 17', 'with SOC 2'; but a metric's value ('NPS 52', 'CSAT 4.7') and a rank ('Top 10') are figures
_NAME_BEFORE_NUMBER = re.compile(r"(?:[^\s.!?:;]\s+|\()(?!(?:NPS|CSAT|DAU|MAU|ARR|MRR|GMV|AOV|LTV|CAC|CTR|MTTR|SLA"
                                 r"|Top)\s)[A-Z](?:[\w#+.]*[\w#+])?\s+$")
_YEAR_NUMBER = re.compile(r"(?:19|20)\d\d(?:/(?:19|20)?\d\d)?|(?:19|20)\d0s")  # 2024, 2024/25, the 2010s
_FAST_UNIT = r"ms|µs|μs|milliseconds?|microseconds?|seconds?|secs?|minutes?|mins?"
_TIME_SPAN_UNIT = rf"{_FAST_UNIT}|hours?|hrs?|days?|weeks?|months?"
_MEASURE_WORD = re.compile(rf"(?:\s+[\w-]+)?\s+(?:{_TIME_SPAN_UNIT}|quarters?|years?|points?|pts|percent|users|people|ftes?)\b",
                           re.IGNORECASE)  # 'saving Finance 120 hours'
_COUNT_WORD = re.compile(  # '1900-person', '2047 requests per second', '1938 concurrent users'
    rf"-[a-z]|(?:\s+[\w-]+)?\s+(?:{_TIME_SPAN_UNIT}|per|percent|[kmgt]b|rps|qps|tps|requests|users|tickets|customers|clients"
    r"|servers|nodes|records|rows|events|orders|transactions|queries|jobs|accounts|people|employees|staff|agents"
    r"|engineers|students|patients|members|merchants|stores|sites|locations|devices|endpoints|pages|documents|files"
    r"|calls|messages|emails|items|products|skus|hosts|instances|pods|builds|deployments|sessions|visits|visitors"
    r"|subscribers|leads|deals|shipments|invoices|claims|cases|seats|licen[cs]es|vendors|suppliers|connections"
    r"|threads|partitions|shards|containers|vms|cores|tables|pipelines|repos|repositories|services|microservices"
    r"|apis|lines|alerts|incidents|tasks|units|packages|parcels|responses|applicants|candidates)\b", re.IGNORECASE)
# 'the catalog search delivers results in 90 ms': query results and their speed, not the 'delivered results' cliche
_SPEED = re.compile(rf"(?<![\w.,])(?!(?:19|20)\d0s\b)\d[\d.,]*(?:\s?(?:{_FAST_UNIT})|s)\b|\breal[\s-]time\b",
                    re.IGNORECASE)
_CLAUSE_BREAK = re.compile(r"[.;](?:\s|$)|,\s+(?:and|but|while)\b")
_CLAUSE_WINDOW = 300  # characters read for a figure: a whole bullet, and linear time on any text
# ', procurement' right before ', and payroll': that ', and' closes a list ('across finance, procurement, and payroll')
_LIST_ITEM = r"(?!(?:and|or|then)\b)(?:[^\s,;]+\s+){0,3}[^\s,;]+"
_LAST_ITEM = re.compile(rf",\s+{_LIST_ITEM}$")
_ONLY_ITEMS = re.compile(rf"(?:,\s+(?!\w+ing\b){_LIST_ITEM})+")  # ', release cadence', not ', cutting reviews'
# 'U.S.', 'e.g.', 'Inc.': the full stop of an abbreviation ends no sentence when a lowercase word or a figure follows
# ('across U.S. warehouses by 18%'); 'in the U.S. Cut pick time ...' starts a new one
_CLAUSE_ABBREVIATION = re.compile(r"(?<![\w.])(?:[a-z]\.){2,}$|\b(?:inc|ltd|co|corp|vs|approx|etc|incl|est)\.$", re.IGNORECASE)
_SAME_SENTENCE = re.compile(r"\s+[a-z\d$€£]")
_ING_CLAUSE = re.compile(r",\s+(?:and\s+)?(?!during\b)[a-z]{4,}ing\b")  # ", ensuring ...", ", and enabling ..."
MAX_ING_CLAUSES = 2  # a third in one bullet reads as a template
MIN_PLAIN_PROBLEMS_FOR_RETRY = 2  # one stray word is only noted; it is not worth a model call


def _abbreviation_stop(text: str, stop: int) -> bool:
    """The full stop at `stop` ends an abbreviation, not the sentence: a lowercase word or a figure follows it."""
    return (text.startswith(".", stop) and _SAME_SENTENCE.match(text, stop + 1) is not None
            and _CLAUSE_ABBREVIATION.search(text, max(0, stop - 12), stop + 1) is not None)


def _clause(text: str, end: int) -> str:
    """The rest of the clause after `end`: up to '. ', '; ' or ', and/but/while' that starts a new clause. Neither an
    abbreviation ('U.S. warehouses') nor the ', and' that closes a list after the phrase ('across finance, procurement,
    and payroll') ends it. A list that the phrase itself starts does ('improved team productivity, release cadence,
    and on-call health'): a figure after it belongs to the last item."""
    rest = text[end:end + _CLAUSE_WINDOW]
    if text[end + _CLAUSE_WINDOW:end + _CLAUSE_WINDOW + 1].strip():  # no word cut in half: '20' of '2024' is no figure
        rest = rest[:re.search(r"\S*$", rest).start()]
    for found in _CLAUSE_BREAK.finditer(rest):
        stop = found.start()
        if _abbreviation_stop(rest, stop) or (
                found.group(0).endswith("and") and _LAST_ITEM.search(rest, max(0, stop - 80), stop)
                and not _ONLY_ITEMS.fullmatch(rest, 0, stop)):
            continue
        return rest[:stop]
    return rest


def _is_figure(text: str, number: re.Match, stop: int) -> bool:
    """`number`, a _MEASURE match in `text` inside a clause that ends at `stop`, measures something."""
    currency, digits, letters = number.groups()
    if currency or text.startswith(("%", "+"), number.end()):
        return True  # $41K, 40%, 2000+
    if _NAME_NUMBER.fullmatch(digits + letters) or (
            not letters and _NAME_BEFORE_NUMBER.search(text, max(0, number.start() - 40), number.start())):
        return _MEASURE_WORD.match(text, number.end(), stop) is not None  # SOC 2; but 'saving Finance 120 hours'
    if _YEAR_NUMBER.fullmatch(digits + letters):
        return _COUNT_WORD.match(text, number.end(), stop) is not None  # 'in 2024'; but '2047 requests per second'
    return True


def _measured(text: str, end: int) -> bool:
    """A real figure follows in the same clause, so the result is specific: 'improved team productivity by 18.3% (from
    11 to 13 tickets a sprint)'. Never a year, a name like 'Q3', 'GPT-4' or 'Java 17', or a figure after ', and'."""
    stop = end + len(_clause(text, end))
    return any(_is_figure(text, number, stop) for number in _MEASURE.finditer(text, end, stop))


def _literal_delivery(text: str, match: re.Match) -> bool:
    """Bare 'delivers results' with a speed figure or 'real time' in its clause: a system that returns results."""
    return (re.fullmatch(r"deliver(?:s|ed|ing)?\s+results", match.group(0), re.IGNORECASE) is not None
            and _SPEED.search(_clause(text, match.end())) is not None)


def _posting_phrase(phrase: str, terms: list[str]) -> bool:
    """The phrase is itself one of the posting's terms (never 'results-driven', a cliche in any posting)."""
    return not re.match(r"results[\s-]driven", phrase, re.IGNORECASE) and any(mentions(term, phrase) for term in terms)


def _plain_problems(places: list[tuple[str, str]], exempt: list[str] | None = None) -> list[str]:
    """Buzzwords and vague results in each (where, text), and texts that chain three or more '-ing' clauses. A vague
    phrase that is itself one of the posting's terms (`exempt`: it asks for "operational excellence") is the keyword
    the posting wants; with a verb in front ("improved operational excellence") it is still vague. One phrase is one
    problem: the cliche inside a vague result already reported ('Drove operational excellence') is not counted again."""
    problems: list[str] = []
    for where, text in places:
        problems += [f"'{match.group(0)}' ({where})" for match in _BUZZ.finditer(text)]
        vague = [match for match in _VAGUE_RESULT.finditer(text)
                 if not _measured(text, match.end()) and not _posting_phrase(match.group(0), exempt or [])]
        reported = {i for match in vague for i in range(*match.span())}
        problems += [f"'{match.group(0)}' ({where})" for match in vague]
        problems += [f"'{match.group(0)}' ({where})" for match in _VAGUE_CLICHE.finditer(text)
                     if not reported.intersection(range(*match.span()))
                     and not _posting_phrase(match.group(0), exempt or []) and not _literal_delivery(text, match)]
        if len(_ING_CLAUSE.findall(text)) > MAX_ING_CLAUSES:
            problems.append(f"a chain of '-ing' clauses ({where}: '{' '.join(text.split()[:8])} ...')")
    return problems


def plain_word_problems(resume: Resume, analysis: JDAnalysis | None = None) -> list[str]:
    """Machine-sounding wording in the summary and the bullets: buzzwords, and bullets that chain three or more
    '-ing' clauses. Skills are a list of tool names and are not read."""
    return _plain_problems([("summary", resume.summary)] + [
        (resume.profile.companies[exp.company_index].name, bullet) for exp in resume.experience for bullet in exp.bullets
    ], _posting_terms(analysis))


def _posting_terms(analysis: JDAnalysis | None) -> list[str]:
    if analysis is None:
        return []
    return [*analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords]


def _sentences(paragraph: str) -> list[str]:
    """The paragraph's sentences. An abbreviation that a lowercase word or a figure follows ('U.S. warehouses') ends
    none."""
    sentences, start = [], 0
    for gap in re.finditer(r"(?<=[.!?])\s+", paragraph):
        if not _abbreviation_stop(paragraph, gap.start() - 1):
            sentences.append(paragraph[start:gap.start()])
            start = gap.end()
    return [sentence for sentence in [*sentences, paragraph[start:]] if sentence]


def cover_plain_problems(letter: CoverLetter, analysis: JDAnalysis | None = None) -> list[str]:
    """The same check on the cover letter, sentence by sentence, so a long paragraph is not read as one '-ing'
    chain."""
    return _plain_problems([(f"paragraph {number}", sentence) for number, paragraph in enumerate(letter.paragraphs, 1)
                            for sentence in _sentences(paragraph)], _posting_terms(analysis))


_WORD = re.compile(r"[A-Za-z][\w+#]*(?:-[\w+#]+)*")
_REAL_DOUBLES = {"had", "that"}  # 'what it had had', 'said that that': correct English
# 'a plug-in in Java', 'an add-on on Shopify', 'a follow-up up front': a hyphenated word ending in one of these
# may be followed by the same small word on purpose
_PARTICLES = {"in", "on", "up", "out", "off", "to", "of", "by", "at", "for", "with", "as", "over", "down", "back"}


def _repeats(text: str) -> list[tuple[int, int, str]]:
    """Each word that repeats the word just before it, with only a space between: 'the the', and a hyphenated
    word followed by its own last part, 'multi-tenant tenant'. (start, end) of the repeat, and the two words."""
    found = []
    tokens = list(re.finditer(r"\S+", text))
    for before, after in pairwise(tokens):
        first, second = before.group(0), after.group(0).rstrip(".,;:!?)")
        if not (_WORD.fullmatch(first) and _WORD.fullmatch(second)) or text[before.end():after.start()] != " ":
            continue  # 'Java, Java' or a line break: not the same slip
        word = second.lower()
        same = word == first.lower() and word not in _REAL_DOUBLES
        tail = "-" in first and word == first.split("-")[-1].lower() and word not in _PARTICLES
        if same or tail:
            found.append((before.end(), after.start() + len(second), f"{first} {second}"))
    return found


def drop_repeated_words(text: str) -> str:
    """'multi-tenant tenant data schemas' -> 'multi-tenant data schemas'."""
    for start, end, _pair in reversed(_repeats(text)):
        text = text[:start] + text[end:]
    return text


def _resume_texts(resume: Resume) -> list[str]:
    return [resume.headline, resume.summary, *(b for exp in resume.experience for b in exp.bullets)]


def repeated_word_problems(resume: Resume) -> list[str]:
    """Words the model repeated by mistake in the headline, summary and bullets ('multi-tenant tenant')."""
    return [pair for text in _resume_texts(resume) for _s, _e, pair in _repeats(text)]


def fix_repeated_words(resume: Resume) -> list[str]:
    """Remove the repeats that are still there after the rewrite pass; returns what was removed."""
    fixed = repeated_word_problems(resume)
    resume.headline, resume.summary = drop_repeated_words(resume.headline), drop_repeated_words(resume.summary)
    for exp in resume.experience:
        exp.bullets = [drop_repeated_words(b) for b in exp.bullets]
    return fixed


_LEVEL_WORDS = {
    "intern": -1, "trainee": -1, "junior": 0, "jr": 0, "associate": 0, "graduate": 0,
    "senior": 3, "sr": 3, "staff": 4, "lead": 4, "principal": 5,
}
_LEVEL_NUMBERS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}


def title_level(title: str) -> tuple[str, int]:
    """('software engineer', 2) for 'Software Engineer II': the role without its level words, and the level
    (intern -1, junior/associate 0, plain 1, II 2, III or senior 3, staff or lead 4, principal 5)."""
    words = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", title.lower().replace(".", ""))
    level, base = None, []
    for index, word in enumerate(words):
        if word in _LEVEL_WORDS and level is None:
            level = _LEVEL_WORDS[word]
        elif index == len(words) - 1 and index > 0 and word in _LEVEL_NUMBERS:
            level = max(level or 0, _LEVEL_NUMBERS[word])
        else:
            base.append(word)
    return spelling.us_text(" ".join(base)), (1 if level is None else level)  # either spelling: one role


_ROLE_LEVEL = re.compile(r"^(?P<role>.*?\S)\s+(?P<level>I{1,3}|IV|V|[1-5]|L[1-9]|Level\s+[1-9])$", re.IGNORECASE)


def role_level_fixes(resume: Resume, analysis: JDAnalysis) -> list[str]:
    """The headline and summary name the posting's role without the hiring company's level number (II, L5) when
    the candidate's latest title does not have that level: "Senior Software Engineer II with 7+ years" claims a
    level the candidate never held. Changes the resume in place and returns a note per changed field."""
    match = _ROLE_LEVEL.match(analysis.target_role.strip())
    if not match or not resume.experience:
        return []
    role, level = match.group("role"), match.group("level")
    level_pattern = r"\s+".join(map(re.escape, level.split()))
    if re.search(rf"(?<![\w-]){level_pattern}$", resume.experience[0].job_title.strip(), re.IGNORECASE):
        return []  # the candidate's latest title really has this level
    role_pattern = r"\s+".join(map(either_spelling, role.split()))  # the draft may spell it the run's way
    phrase = re.compile(rf"(?<![\w-])({role_pattern})\s+{level_pattern}(?![\w-])", re.IGNORECASE)
    notes = []
    for field in ("headline", "summary"):
        text = getattr(resume, field)
        fixed = phrase.sub(lambda m: m.group(1), text)
        if fixed != text:
            setattr(resume, field, fixed)
            notes.append(f"{field.title()}: wrote '{role}' instead of the posting's level '{analysis.target_role}', "
                         "which is not your title.")
    return notes


def _steps_down(newer: str, older: str) -> bool:
    """A newer title below the older one: same role at a lower level, or a different role 2+ levels lower."""
    (new_base, new_level), (old_base, old_level) = title_level(newer), title_level(older)
    return new_level < old_level and (new_base == old_base or old_level - new_level >= 2)


def fix_title_regressions(resume: Resume) -> list[str]:
    """Make sure no job title steps down from one employer to the next (for example 'Software Engineer II'
    then 'Software Engineer'). Changes the resume in place and returns a note per change.

    Same role at a lower level: the newer title is raised to the older one, unless the newer job is the most
    recent (its title matches the target role) or raising it would step above the job after it; then the older
    title is lowered instead. Different roles are only reported by title_growth_problems."""
    exps, notes = resume.experience, []  # most recent first
    for i in range(1, len(exps)):
        newer, older = exps[i - 1], exps[i]
        (new_base, new_level), (old_base, old_level) = title_level(newer.job_title), title_level(older.job_title)
        if new_base != old_base or new_level >= old_level:
            continue
        above = title_level(exps[i - 2].job_title) if i >= 2 else None
        can_raise = i - 1 > 0 and above is not None and (above[0] != old_base or above[1] >= old_level)
        changed, before = (newer, newer.job_title) if can_raise else (older, older.job_title)
        changed.job_title = older.job_title if can_raise else newer.job_title
        company = resume.profile.companies[changed.company_index].name
        notes.append(f"{company}: title changed from '{before}' to '{changed.job_title}' so the career path never "
                     "steps down.")
    return notes


# When each tool became COMMON at work (not its first release): a job that ended before that year should not
# mention it. Lenient on purpose: compared with the job's END year. Only tools a recent career can get wrong.
COMMON_YEAR: dict[str, int] = {
    "Kubernetes": 2017, "Helm": 2018, "Amazon EKS": 2019, "EKS": 2019, "Istio": 2019, "Next.js": 2019,
    "GitHub Actions": 2020, "dbt": 2020, "Argo CD": 2021, "Tailwind CSS": 2021, "FastAPI": 2021, "Delta Lake": 2021,
    "Azure Synapse": 2021, "OpenTelemetry": 2022, "Vite": 2022, "Astro": 2023, "Polars": 2023, "OpenTofu": 2024,
    "Bun": 2024, "Microsoft Fabric": 2024,
}
# AI tools, with the year they became popular at work. They also belong only to the most recent AI_RECENT_JOBS jobs.
AI_TOOLS: dict[str, int] = {
    "ChatGPT": 2023, "LLM": 2023, "large language model": 2023, "GPT-4": 2023, "LangChain": 2023,
    "prompt engineering": 2023, "vector database": 2023, "retrieval-augmented generation": 2023,
    "AI coding assistant": 2023, "Copilot": 2022, "agentic": 2025, "coding agent": 2025,
}
AI_RECENT_JOBS = 2
_EVERYDAY_WORDS = {"Helm", "Astro", "Vite", "Copilot", "Bun", "Polars"}  # matched only as written: "at the helm" is fine
_CURRENT = ("", "present", "current", "now", "today")


def _mentions_tool(text: str, term: str) -> bool:
    return mentions(text, term, case_sensitive=term in _EVERYDAY_WORDS or None)


def era_problems(resume: Resume) -> list[str]:
    """Tools a job could not really have used: an AI tool outside the most recent one or two jobs, or any tool in a
    job that ended before it became common (e.g. OpenTofu at a job that ended in 2021). The current job, and jobs
    whose end date cannot be read, are never flagged for dates."""
    problems: list[str] = []
    for index, exp in enumerate(resume.experience):  # most recent first
        company = resume.profile.companies[exp.company_index]
        text = "\n".join([exp.job_title, *exp.bullets])
        when = f"{company.name} ({company.start or '?'} - {company.end or 'Present'})"
        if index >= AI_RECENT_JOBS:
            for term in AI_TOOLS:
                if _mentions_tool(text, term):
                    problems.append(f"{when} mentions {term}: AI tools belong only to the most recent one or two jobs.")
        end = parse_point(company.end, is_end=True)
        if end is None or company.end.strip().lower() in _CURRENT:
            continue
        for term, year in {**COMMON_YEAR, **(AI_TOOLS if index < AI_RECENT_JOBS else {})}.items():
            if year > end[0] and _mentions_tool(text, term):
                problems.append(f"{when} mentions {term}, which only became common in {year}.")
    return problems


def title_growth_problems(resume: Resume) -> list[str]:
    """At most one warning about the career path: a title that steps down from one job to the next, the same
    title everywhere, the same title at 3+ employers in a row, or a stay of 2+ years with the same title as the
    job before it."""
    exps = resume.experience  # most recent first
    titles = [e.job_title.strip().lower() for e in exps]
    for i in range(len(exps) - 1):
        if _steps_down(exps[i].job_title, exps[i + 1].job_title):
            company = resume.profile.companies[exps[i].company_index].name
            return [(f"{company}: '{exps[i].job_title}' comes after '{exps[i + 1].job_title}', which reads as a step "
                     "down. Use the same level or higher than the job before it.")]
    if len(titles) > 1 and len(set(titles)) == 1:
        return ["The same job title is used at every employer, so there is no visible career progression."]
    for i in range(2, len(titles)):
        if titles[i] == titles[i - 1] == titles[i - 2]:
            return [(f"The title '{exps[i].job_title}' is used at 3 employers in a row, so there is no visible "
                     "career progression.")]
    problems = []
    for i in range(len(exps) - 1):  # the oldest job has nothing before it to grow from
        if titles[i] != titles[i + 1]:
            continue
        company = resume.profile.companies[exps[i].company_index]
        months = stay_months(company)
        if months is not None and months >= 24:
            problems.append(
                f"{company.name}: {months_phrase(months)} with the same title as the job before it. Show growth, "
                "for example a level step ('Software Engineer II') or the specialty title."
            )
    return problems[:1]


def ats_lint(resume: Resume) -> list[str]:
    warnings: list[str] = []
    if word_count(resume.headline) > 14:
        warnings.append(f"Headline is long ({word_count(resume.headline)} words).")
    words = word_count(resume.summary)
    if words > 95:
        warnings.append(f"Summary is long ({words} words; aim for 60-85).")
    elif words < MIN_SUMMARY_WORDS:
        warnings.append(f"Summary is short ({words} words; aim for 60-85).")
    if _FIRST_PERSON.search(resume.summary):
        warnings.append("Summary uses first-person pronouns.")
    total_skills = sum(len(g.skills) for g in resume.skills)
    if not 12 <= total_skills <= 40:
        warnings.append(f"{total_skills} skills listed (aim for roughly 24-32).")
    warnings += title_growth_problems(resume)
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index].name
        if not 2 <= len(exp.bullets) <= 8:
            warnings.append(f"{company}: {len(exp.bullets)} bullets (aim for 3-8, the most in recent roles).")
        openers = [b.split()[0].lower() for b in exp.bullets if b.split()]
        if len(openers) != len(set(openers)):
            warnings.append(f"{company}: two bullets start with the same word.")
        for bullet in exp.bullets:
            if word_count(bullet) > 36:
                warnings.append(f"{company}: a bullet is {word_count(bullet)} words long.")
                break
            if word_count(bullet) < MIN_BULLET_WORDS:
                warnings.append(
                    f"{company}: a bullet has only {word_count(bullet)} words; say how it was done and what changed."
                )
                break
            if _FIRST_PERSON.search(bullet):
                warnings.append(f"{company}: a bullet uses first-person pronouns.")
                break
    return warnings


# A product or standard name before a bare number or version: Microsoft 365, Fortune 500, Postgres 15, Node.js 20, iOS 17
_NAME_BEFORE = re.compile(r"[^\s.!?:;]\s+(?:[A-Z][\w#+.-]*|[a-z]+[A-Z][\w#+.-]*|[a-z]+\.(?:js|net|io))\s$")
_NAMED_BEFORE = re.compile(r"(?:\b[A-Z][A-Za-z]*-|\+\+)$")  # GPT-4, COVID-19, SEV-2, C++17: part of a name
# A count after a metric or a team is a figure: 'NPS 12 points', 'Finance 20 hours', 'Postgres 40 TB'
_METRIC_NAMES = frozenset({
    "NPS", "eNPS", "CSAT", "CES", "MTTR", "MTTD", "MTTA", "MTBF", "AHT", "FCR", "DSO", "DPO", "DAU", "MAU", "WAU", "ARR",
    "MRR", "ARPU", "LTV", "CAC", "ROAS", "ROI", "CTR", "CVR", "AOV", "GMV", "TPS", "QPS", "RPS", "SLA", "SLO", "KPI",
})
_MEASURE_AFTER = re.compile(
    r"\s*(?:points?|pts|bps|basis\s+points|percentage\s+points|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?"
    r"|months?|years?|ms|[kmgtp]b|people|engineers|analysts|developers|users|customers|clients|accounts|tickets"
    r"|dashboards|reports|services|teams|stores|sites|countries|markets|apps|models|pipelines|jobs|requests"
    r"|transactions|orders|calls|incidents|alerts)\b", re.IGNORECASE)


def _numbers_in(text: str) -> list[str]:
    """The figures in a text; never the digits of a name or a version (p95, EC2, GPT-4, C++17, Microsoft 365, Node.js
    20, Python 3.12), but a count after a metric or a team is one ('NPS 12 points', 'Finance 20 hours')."""
    found = []
    for match in _NUMBER.finditer(text):
        token, before = match.group(0).strip(" ,."), text[max(0, match.start() - 60): match.start()]
        if _NAMED_BEFORE.search(before):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*", token):
            name = _NAME_BEFORE.search(before)
            if name and name.group(0).split()[-1] not in _METRIC_NAMES and not _MEASURE_AFTER.match(text, match.end()):
                continue
        found.append(token)
    return found


_FIGURE_CORE = re.compile(r"[$€£]?(\d[\d,]*)(\.\d+)?")
_APPROX = re.compile(r"~\s?[$€£]?\d[\d,.]*\s?%?|\b(?:approximately|approx\.)\s+[$€£]?\d[\d,.]*\s?%?", re.IGNORECASE)


def is_round_figure(token: str) -> bool:
    """25, 30%, 40+, $120K: whole numbers of 10 or more ending in 0 or 5. Years (1900-2100) are not figures."""
    match = _FIGURE_CORE.match(token)
    if not match or match.group(2):
        return False
    digits = match.group(1).replace(",", "") or "0"
    if len(digits) > 18:
        return False
    value = int(digits)
    if 1900 <= value <= 2100 and token.isdigit():
        return False
    thousands = 0
    if token.lstrip()[:1] in "$€£":  # money: $41,000 reads like $41K and $41,300 like 413: only those digits count
        while value >= 1000 and value % 1000 == 0:
            value //= 1000
            thousands += 1
        if value >= 1000 and value % 100 == 0:
            value //= 100
            thousands += 1
        if token.rstrip()[-1:] in "kKmMbB":
            thousands += 1
    return (value >= 10 and value % 5 == 0) or (thousands > 0 and value < 10)


def figure_style_problems(resume: Resume, allowed_text: str) -> list[str]:
    """Estimated figures that look like round guesses or approximations ("35%", "40+", "~30%").

    Figures that appear in `allowed_text` (the user's hints and the computed facts) are never reported.
    """
    allowed = allowed_text.lower()
    problems: list[str] = []
    texts = [resume.summary] + [bullet for exp in resume.experience for bullet in exp.bullets]
    for text in texts:
        for match in _APPROX.finditer(text):
            problems.append(match.group(0).strip())
        for token in _numbers_in(text):
            if token and token.lower() not in allowed and is_round_figure(token):
                problems.append(token)
    return list(dict.fromkeys(problems))


MAX_FIGURE_BULLETS_PER_JOB = 2
MAX_FIGURE_SHARE = 0.5


def has_figure(text: str) -> bool:
    """True if the text carries a number other than a plain year."""
    for token in _numbers_in(text):
        if token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100:
            continue
        if token:
            return True
    return False


def metric_density_problems(resume: Resume) -> list[str]:
    """Employers with more than 2 bullets carrying figures, and an overall share above half."""
    problems: list[str] = []
    total = with_figure = 0
    for exp in resume.experience:
        count = sum(has_figure(b) for b in exp.bullets)
        total += len(exp.bullets)
        with_figure += count
        if count > MAX_FIGURE_BULLETS_PER_JOB:
            company = resume.profile.companies[exp.company_index].name
            problems.append(f"{company}: {count} of {len(exp.bullets)} bullets have figures")
    if total and with_figure / total > MAX_FIGURE_SHARE:
        problems.append(f"{with_figure} of {total} bullets overall have figures")
    return problems


MIN_FIRST_BULLETS_FOR_RETRY = 2  # two jobs that open without a figure (or the most recent one) are worth the fix round


# A percentage means little without its base ("by 20%" of 5,000 or of 500,000?), a recruiter's advice: every
# percentage that describes a change comes with its base in absolute terms, right beside it: "by 17.8% ($41,000 a
# year)", "by 22% (from 41 to 32 seconds)", "by 22%, from 41 to 32 seconds", "from 41 to 32 seconds, a 22% drop". A
# level or a share ("99.9% uptime", "used by 38% of users", "from 61% to 87%") needs none.
_PCT = r"(?<![\d,.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?%"  # '1,200%' is one figure
_NOT_SHARE = r"(?!\s+of\b)"
_CHANGE_NOUN = (r"(?:increase|decrease|reduction|improvement|growth|drop|decline|rise|gain|cut|saving|uplift|lift"
                r"|boost|jump|fall)")
# Words between a percentage and its change noun name what changed ('a 17.8% cost reduction'), never a level
# ('99.9% uptime during growth') or a new phrase ('lift-and-shift', 'a 6% decline rate')
_FILLER = (r"(?:(?!(?:of|to|in|from|and|or|the|a|an|during|through|despite|amid|while|for|over|with|on|at|by|across"
           r"|after|before|since|uptime|availability|accuracy|coverage|slas?|slos?|share|rates?|precision|recall"
           r"|utili[sz]ation|occupancy)\b)[a-z][\w-]*\s+)")
# After a verb, the words before the percentage name what changed ('Cut AWS spend 17.8%'), never a level kept
# alongside ('Cut costs while maintaining 99.9% uptime', 'for top 5% accounts')
_VERB_FILLER = (r"(?:(?!(?:by|to|from|at|of|with|in|on|and|or|while|whilst|yet|but|without|despite|within|under|below"
                r"|above|for|top|bottom|kept|keep|keeps|keeping|maintain|maintains|maintained|maintaining|hold|holds"
                r"|held|holding|sustain|sustains|sustained|sustaining|hit|hits|hitting|reach|reached|reaching"
                r"|achieve|achieved|achieving)\b)[\w-]+\s+)")
_LEVEL_AFTER = r"(?!\s+(?:uptime|availability|accuracy|coverage|sla|slo|of)\b)"
_CHANGE_PCT = re.compile(
    rf"\bby\s+(?:(?:about|almost|nearly|over|more\s+than|up\s+to)\s+)?{_PCT}{_NOT_SHARE}"
    rf"|(?P<adj>{_PCT}\s+(?:faster|slower|fewer|less|lower|higher|more|smaller|larger|shorter|cheaper|quicker)\b)"
    rf"|(?P<noun>{_PCT}\s+(?:(?:year[\s-](?:over|on)[\s-]year|yoy|annual|monthly|quarterly)\s+)?"
    rf"(?P<filler>{_FILLER}{{0,2}}?){_CHANGE_NOUN}s?\b(?!\s+rates?\b)(?!-(?!off\b)))"
    rf"|{_PCT}\s+(?:year[\s-](?:over|on)[\s-]year|yoy|qoq|mom|quarter[\s-](?:over|on)[\s-]quarter"
    rf"|month[\s-](?:over|on)[\s-]month)\b"
    rf"|\b{_CHANGE_NOUN}s?\s+of\s+{_PCT}"
    rf"|\b(?:cut|cutting|reduced|reducing|lowered|lowering|increased|increasing|raised|raising|improved|improving"
    rf"|boosted|boosting|lifted|lifting|grew|growing|decreased|decreasing|saved|saving)\s+"
    rf"{_VERB_FILLER}{{0,3}}?{_PCT}{_NOT_SHARE}{_LEVEL_AFTER}"
    rf"|\b(?:up|down|grew|rose|fell|dropped|declined|jumped|climbed)\s+{_PCT}{_NOT_SHARE}", re.IGNORECASE)
_VALUE = r"[$€£]?\d[\d,]*(?:\.\d+)?"
_UNIT = (r"(?:ms|msec|milliseconds?|millis|million|billion|mn|bn|secs?|seconds?|s|mins?|minutes?|m|hrs?|hours?|h"
         r"|days?|d|weeks?|months?|years?"
         r"|[kKmMgGtTpP][bB]|[kKmMbB])")
_PAIR_BODY = (rf"(?P<was>{_VALUE})(?!\.?\d)(?:\s?(?P<was_unit>{_UNIT})\b)?(?!\s?%)"
              rf"(?P<gap>(?:(?!\bfrom\b)[^.;()]){{0,30}}?)\bto\s+"
              rf"(?P<now>{_VALUE})(?!\.?\d)(?:\s?(?P<now_unit>{_UNIT})\b)?(?!\s?%)")
_PAIR_FROM = re.compile(rf"\bfrom\s+{_PAIR_BODY}", re.IGNORECASE)  # '(p95 from 820 to 648 ms)', '(down from ...)'
_PAIR_IN_BRACKET = re.compile(rf"^\s*(?:[\w-]+:\s*)?(?:from\s+)?{_PAIR_BODY}", re.IGNORECASE)  # '(p95: 820 to 648 ms)'
_PAIR_AFTER = re.compile(rf"^,?\s*(?:(?:down|up)\s+)?from\s+{_PAIR_BODY}", re.IGNORECASE)  # ', down from 41 to 32 s'
_SAME_CLAUSE = r"(?!\b(?:and|but|while|then|whereas)\b)"  # 'from 41 to 32 minutes and drove a 30% drop': two claims
_PAIR_BEFORE = re.compile(rf"\bfrom\s+{_PAIR_BODY}(?:{_SAME_CLAUSE}[^.;)%]){{0,25}}$", re.IGNORECASE)  # '..., a 22% drop'
_BRACKET_AFTER = re.compile(r"^(?:\.\d|[^.;()%]){0,60}?\(([^()]*)\)")
_LEVELS = re.compile(rf"(?:\bfrom\s+)?{_PCT}[^()]{{0,30}}?\bto\s+{_PCT}", re.IGNORECASE)  # '(from 3.1% to 3.2%)'
_LEVELS_AFTER = re.compile(rf"^,?\s*from\s+{_PCT}[^()]{{0,30}}?\bto\s+{_PCT}", re.IGNORECASE)
_LEVELS_BEFORE = re.compile(rf"\bfrom\s+{_PCT}[^()]{{0,30}}?\bto\s+{_PCT}(?:{_SAME_CLAUSE}[^.;)%]){{0,25}}$",
                            re.IGNORECASE)
# An amount with its period in brackets is a base too: '(12 hours a week)', '(1,200 fewer alerts a month)'
_AMOUNT_IN_BRACKET = re.compile(r"\d[\d,.]*\s*(?:[a-z%]+\s+){0,4}?(?:(?:a|per|each|every)\s+(?:day|week|month|quarter"
                                r"|year|sprint|release|night|run)|fewer|more|less|saved)\b", re.IGNORECASE)
# Money after a percentage is its base only when it is the saving itself: never 'by 21%, protecting $2.3M in revenue'
_MONEY_AFTER = re.compile(r"^[,:]?\s*(?:(?:saving|saved|worth|or|about|roughly|which\s+is|that\s+is|for|(?:which|that)"
                          r"\s+saved|a\s+saving\s+of)\s+(?:(?:the\s+)?(?:company|team|business|clients?|customers?|us)"
                          r"\s+)?)?[$€£]\s?\d", re.IGNORECASE)
_SAVING_AFTER = re.compile(r"^,?\s*(?:saving|saved|freeing(?:\s+up)?|(?:which|that)\s+saved)\s+(?:(?:about|roughly"
                           r"|over|nearly)\s+)?[$€£]?\d", re.IGNORECASE)  # ', saving 12 hours a week'
_TO_AFTER = re.compile(rf"^,?\s*(?:(?:down|up)\s+)?to\s+{_VALUE}", re.IGNORECASE)  # 'by 23.4% to 11 minutes'
# A period after the percentage does not hide the base after it: 'Grew ARR 24.3% year-over-year to $12.4M'
_PERIOD_AFTER = re.compile(r"^\s+(?:(?:year|quarter|month)[\s-](?:over|on)[\s-](?:year|quarter|month)|yoy|qoq|mom"
                           r"|annually|yearly|quarterly|monthly|(?:a|per|each|every)\s+(?:year|quarter|month|week))\b",
                           re.IGNORECASE)
_MONEY_BEFORE = re.compile(rf"[$€£]\d(?:\.\d|\([^()]*\)|{_SAME_CLAUSE}[^.;%()]){{0,60}}$")  # 'saving $41,300 a year, a 17.8% cut'
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
MIN_UNBASED_FOR_RETRY = 2  # one is only noted: like a single rounded figure, not worth a model call
MAX_BASE_GAP = 1.0  # percentage points between a percentage and the change its base shows
_TIME = {"ms": 0.001, "msec": 0.001, "millisecond": 0.001, "milliseconds": 0.001, "millis": 0.001, "s": 1, "sec": 1,
         "secs": 1, "second": 1, "seconds": 1, "min": 60, "mins": 60, "minute": 60, "minutes": 60, "h": 3600,
         "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "d": 86400, "day": 86400, "days": 86400,
         "week": 604800, "weeks": 604800, "month": 2629800, "months": 2629800, "year": 31557600, "years": 31557600}
_DATA = {"kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "pb": 1e15}
_COUNT = {"k": 1e3, "b": 1e9, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
# 'Kept checkout up 99.95% through Black Friday': a level held, not a change; never 'Kept churn flat while pushing
# activation up 12.4%'
_KEPT = re.compile(r"\b(?:kept|keep|keeps|keeping|stayed|stays|remained|remains|held|holding)\s+"
                   r"(?:(?!(?:while|whilst|and|but|then|whereas|yet)\b)[\w-]+\s+){0,3}$", re.IGNORECASE)
# The job post's own growth figures: a cover letter may quote them ('your 40% annual growth')
_GROWTH_WORDS = re.compile(r"\b(?:grow|grows|grew|grown|growing|growth|expand(?:s|ed|ing)?|expansion|increase[sd]?"
                           r"|rise|rises|rose|risen|rising|scal(?:e|es|ed|ing)|doubled|tripled|annually"
                           r"|year[\s-](?:over|on)[\s-]year|yoy)\b", re.IGNORECASE)
_SPEED_UP = re.compile(r"\b(?:sped|speed(?:s|ing)?)\s+up\b|\baccelerat\w*", re.IGNORECASE)


def _change_percentages(sentence: str):
    """(match, its base) for each percentage that describes a change; text before a percentage starts after the
    previous percentage's own base, so one base never counts for another."""
    matches = [m for m in _CHANGE_PCT.finditer(sentence)
               if not (m.group(0)[:2].lower() == "up" and _KEPT.search(sentence[: m.start()]))]
    start = 0
    for index, match in enumerate(matches):
        before = sentence[start: match.start()]
        after = sentence[match.end(): matches[index + 1].start() if index + 1 < len(matches) else len(sentence)]
        based, pair, used = _attached_base(match, before, after)
        start = match.end() + used
        yield match, (based, pair)


def _years(pair: re.Match | None) -> bool:
    """'from 2021 to 2023': a span of years, not a base; never '1,950 to 2,040' (a count)."""
    return pair is not None and not (pair.group("was_unit") or pair.group("now_unit")) and all(
        re.fullmatch(r"(?:19|20)\d\d", pair.group(group)) for group in ("was", "now"))


def _attached_base(match: re.Match, before: str, after: str) -> tuple[bool, re.Match | None, int]:
    """(has its base beside it, the 'from A to B' to check against it, where its base after it ends)."""
    bracket = _BRACKET_AFTER.match(after)
    if bracket:
        content = bracket.group(1)
        pair = _PAIR_FROM.search(content) or _PAIR_IN_BRACKET.search(content)
        pair = None if _years(pair) else pair
        if (pair or re.search(r"[$€£]\s?\d", content) or _LEVELS.search(content)
                or _AMOUNT_IN_BRACKET.search(content)):
            return True, pair, bracket.end()
    period = _PERIOD_AFTER.match(after)
    skipped = period.end() if period else 0
    rest = after[skipped:]
    pair = _PAIR_AFTER.match(rest)
    if pair and not _years(pair):
        return True, pair, skipped + pair.end()
    found = _TO_AFTER.match(rest) or _MONEY_AFTER.match(rest) or _SAVING_AFTER.match(rest) or _LEVELS_AFTER.match(rest)
    if found:
        return True, None, skipped + found.end()
    if match.group("noun") or match.group("adj"):
        pair = _PAIR_BEFORE.search(before)
        # 'Consolidated Kafka clusters from 14 to 6, a 17.8% cost reduction': the pair counts clusters, not costs
        if pair and not _years(pair) and not (match.group("noun") and match.group("filler")):
            return True, pair, 0
        if _MONEY_BEFORE.search(before) or _LEVELS_BEFORE.search(before):
            return True, None, 0
    return False, None, 0


def _scale(unit: str, other: str, money: bool) -> tuple[str, float] | None:
    """(dimension, factor) of a unit; 'm' is minutes beside hours, millions beside money."""
    if unit in _DATA:
        return "data", _DATA[unit]
    if unit == "m":
        if other in _TIME and other != "m":
            return "time", 60
        return ("count", 1e6) if money or other in _COUNT else ("m", 1)
    if unit in _TIME:
        return "time", _TIME[unit]
    if unit in _COUNT:
        return "count", _COUNT[unit]
    return None


def _number(value: str) -> float:
    return float(value.lstrip("$€£").replace(",", ""))


def pair_values(pair: re.Match) -> tuple[float, float] | None:
    """(A, B) of 'from A to B' on one scale ('from 1.2s to 820ms' is (1.2, 0.82)), or None when the two ends cannot
    be compared (different kinds of unit, years, a second number before 'to')."""
    if re.search(r"\d", pair.group("gap") or "") or _years(pair):
        return None  # 'from 1h 20m to 41m': a second number before 'to'; 'from 2019 to 2022': years, not a base
    raw_was, raw_now = (pair.group("was_unit") or "").lower(), (pair.group("now_unit") or "").lower()
    money_was, money_now = pair.group("was")[:1] in "$€£", pair.group("now")[:1] in "$€£"
    money = money_was or money_now
    plain_unit = raw_now if money_was else raw_was
    if money_was != money_now and plain_unit != "m" and (plain_unit in _TIME or plain_unit in _DATA):
        return None  # 'from $6.20 to 3 minutes per invoice': money and time
    was, now = _number(pair.group("was")), _number(pair.group("now"))
    if bool(raw_was) != bool(raw_now):
        unit = raw_was or raw_now
        bare = (pair.group("now") if raw_was else pair.group("was")).lstrip("$€£").split(".")[0]
        full = "," in bare or len(bare) >= 4
        scale = ("count", 1e6) if unit == "m" and full else _scale(unit, unit, money)  # beside 980,000, M is millions
        if scale and scale[0] == "count" and full:
            # '980,000 to 1.2M': the bare end is written in full, only the other one in millions
            was, now = (was * scale[1], now) if raw_was else (was, now * scale[1])
            return (was, now) if was > 0 else None
    was_unit, now_unit = raw_was or raw_now, raw_now or raw_was  # 'from 41 to 32 seconds': one unit for both
    if was_unit:
        scale_was, scale_now = _scale(was_unit, now_unit, money), _scale(now_unit, was_unit, money)
        if scale_was is None or scale_now is None or scale_was[0] != scale_now[0]:
            return None
        was, now = was * scale_was[1], now * scale_now[1]
    return (was, now) if was > 0 else None


def pair_change(pair: re.Match) -> float | None:
    """The change, in percent, from A to B ('from 1.2s to 820ms' is 31.7), or None when the two ends cannot be
    compared."""
    values = pair_values(pair)
    return abs(values[0] - values[1]) / values[0] * 100 if values else None


def unbased_percentages(text: str, ignore: frozenset[str] = frozenset()) -> list[str]:
    """The percentages in a text that describe a change but have no base beside them ('cut costs by 23.4%').
    `ignore`: figures that are not the candidate's (the company's own growth, quoted from the job post)."""
    found: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        for match, (based, _pair) in _change_percentages(sentence):
            figure = re.search(_PCT, match.group(0)).group(0).replace(" ", "")  # type: ignore[union-attr]
            if figure not in ignore and not based:
                found.append(match.group(0))
    return found


def base_mismatches(text: str) -> list[str]:
    """Bases that do not agree with their own percentage ('by 22% (from 41 to 38 seconds)')."""
    found: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        for match, (_based, pair) in _change_percentages(sentence):
            values = pair_values(pair) if pair else None
            if values is None:
                continue
            was, now = values
            percent = float(re.search(_PCT, match.group(0)).group(0).replace(" ", "").replace(",", "")  # type: ignore[union-attr]
                            .rstrip("%"))
            changes = [abs(was - now) / was * 100]
            speed = (re.search(r"\b(?:faster|quicker)\b", match.group(0), re.IGNORECASE)
                     or _SPEED_UP.search(sentence[: match.start()]))
            if now > 0 and speed:
                changes.append((was / now - 1) * 100)  # '140% faster (from 12 to 5 minutes)': 2.4 times the speed
            if all(abs(change - percent) > MAX_BASE_GAP for change in changes):
                found.append(pair.group(0).strip())  # type: ignore[union-attr]
    return found


def posting_figures(jd: str) -> frozenset[str]:
    """The job post's own change figures ('We grew 40% last year', 'grown 40% annually'), which a cover letter may
    quote; never its benefits ('up to 20% travel', 'a 4% 401(k) match')."""
    figures = {re.search(_PCT, m.group(0)).group(0).replace(" ", "") for m in _CHANGE_PCT.finditer(jd)}  # type: ignore[union-attr]
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", jd):
        if _GROWTH_WORDS.search(sentence):
            figures.update(match.replace(" ", "") for match in re.findall(_PCT, sentence))
    return frozenset(figures)


def unbased_percentage_problems(resume: Resume, letter: CoverLetter | None = None, jd: str = "") -> list[str]:
    """"'by 23.4%' (Northwind Labs)" for every percentage without its base, in the summary, the bullets and the
    cover letter. A percentage the job post itself gives (the company's growth) is the company's, not a claim."""
    posting = posting_figures(jd)
    return [f"'{found}' ({where})" for where, text in prose_places(resume, letter)
            for found in unbased_percentages(text, posting if where.startswith("cover letter") else frozenset())]


def base_mismatch_problems(resume: Resume, letter: CoverLetter | None = None) -> list[str]:
    return [f"'{found}' ({where})" for where, text in prose_places(resume, letter) for found in base_mismatches(text)]


def first_bullet_problems(resume: Resume) -> list[str]:
    """Jobs whose first bullet carries no figure, most recent first. A recruiter's six-second scan reads only that
    bullet, so it should show the job's biggest win with its size and its result."""
    return [resume.profile.companies[exp.company_index].name for exp in resume.experience
            if exp.bullets and not has_figure(exp.bullets[0])]


# What a draft adds after the role: a domain, areas of work or skills ("AI Engineer | AI Platforms", "Full-Stack
# Engineer - Payments")
_AFTER_ROLE = re.compile(r"\s*\|.*$|\s+-\s+.*$")


# Level words before a role ("Senior Data Engineer"); "Tech Lead" and a bare "Lead" stay
_LEADING_LEVEL = re.compile(
    r"^(?:(?:senior|sr|staff|lead|principal|junior|jr|associate|graduate|mid-level|entry-level)\.?\s+)+(?=\S)",
    re.IGNORECASE,
)


def _without_level(role: str) -> str:
    """'Senior Data Engineer' -> 'Data Engineer', 'AI Engineer II' -> 'AI Engineer'."""
    role = _LEADING_LEVEL.sub("", role.strip())
    match = _ROLE_LEVEL.match(role)
    return match.group("role") if match else role


def _role_core(title: str) -> str:
    """A title's short, common name: what follows a '|' or a spaced dash goes, and so do brackets, the part after a
    comma and the level ('Staff Data Engineer, Analytics Infrastructure' -> 'Data Engineer'). A bare role noun
    takes its function from after the comma ('Manager, Customer Success' -> 'Customer Success Manager')."""
    text = re.sub(r"\s*\([^)]*\)", "", _AFTER_ROLE.sub("", title or ""))
    role, _comma, rest = text.partition(",")
    role, rest = _without_level(role), _without_level(rest)
    if len(role.split()) == 1 and rest:
        role = f"{rest} {role}"
    return role.strip()


def short_role_headline(resume: Resume) -> None:
    """The headline is the role's short, common name, never the posting's full title (see _role_core). Changes the
    resume in place."""
    role = _role_core(resume.headline)
    if role:
        resume.headline = role


# A broad field is part of a common title, never a specialty: 'AI Platform Engineer', 'Data Platform Engineer' and
# 'Cloud Security Engineer' are used as they are; 'Voice' in 'Voice AI Engineer' is a specialty
_BROAD_FIELDS = frozenset({
    "ai", "ml", "machine", "learning", "data", "cloud", "platform", "platforms", "infrastructure", "infra", "backend",
    "back-end", "frontend", "front-end", "full-stack", "fullstack", "full", "stack", "software", "mobile", "ios",
    "android", "web", "security", "devops", "mlops", "dataops", "devsecops", "site", "reliability", "sre", "systems",
    "embedded", "network",
})


def _extra_words(special: str, common: str) -> list[str]:
    """The words of the special title that the common title lacks, in either spelling: ['Voice'] for 'Voice AI
    Engineer' -> 'AI Engineer'."""
    kept = {spelling.us_word(word) for word in common.split()}
    return [word for word in special.split() if spelling.us_word(word) not in kept]


def common_title(analysis: JDAnalysis | None) -> tuple[str, str]:
    """(the posting's special title, the most similar common title): ('Voice AI Engineer', 'AI Engineer'); ('', '')
    when the posting's title is a common one already ('Senior Data Engineer' -> 'Data Engineer'), also when only broad
    fields set it apart ('AI Platform Engineer' stays, whatever the model offers instead)."""
    if analysis is None or not analysis.general_role.strip():
        return "", ""
    special, common = _role_core(analysis.target_role), _role_core(analysis.general_role)
    if (not special or not common or len(common.split()) > 4 or spelling.us_text(special) == spelling.us_text(common)
            or all(spelling.us_word(word) in _BROAD_FIELDS for word in _extra_words(special, common))):
        return "", ""
    return special, common


def specialty(analysis: JDAnalysis | None) -> str:
    """What makes the posting's title a special one, when the common title is the special one without it: 'Voice'
    for 'Voice AI Engineer' -> 'AI Engineer'. '' otherwise."""
    special, common = common_title(analysis)
    words, kept = special.split(), common.split()
    if len(words) > len(kept) and spelling.us_text(" ".join(words[-len(kept):])) == spelling.us_text(common):
        return " ".join(words[: -len(kept)])
    return ""


def _article_for(word: str) -> str:
    """'an' before 'AI Engineer', 'ML Engineer' or 'Engineering Manager'; 'a' before 'UX Designer' or 'User
    Researcher'."""
    if len(word) > 1 and word.isupper():  # an acronym is read letter by letter
        return "an" if word[0] in "AEFHILMNORSX" else "a"
    lower = word.lower()
    if lower.startswith(("uni", "use", "usu", "uti", "eu", "one", "ure")):
        return "a"
    return "an" if lower[:1] in "aeiou" else "a"


def general_role_fixes(resume: Resume, analysis: JDAnalysis | None) -> None:
    """A special posting title is never copied: the headline is the most similar common title, and the summary and
    the job titles say it instead ('Voice AI Engineer' -> 'AI Engineer', 'a Senior Voice AI Engineer' -> 'a Senior AI
    Engineer'). The specialty belongs in the bullets. Changes the resume in place."""
    special, common = common_title(analysis)
    if not special:
        _keep_posting_headline(resume, analysis)
        return
    resume.headline = common
    words = r"[\s-]+".join(either_spelling(word) for word in special.split())
    pattern = re.compile(rf"(?<![\w-])(?:(?P<article>an?)\s+)?{words}(?![\w-])", re.IGNORECASE)

    def swap(match: re.Match) -> str:
        article = match.group("article")
        if not article:
            return common
        new = _article_for(common.split()[0])
        return (new[0].upper() + new[1:] if article[0].isupper() else new) + " " + common

    resume.summary = pattern.sub(swap, resume.summary)
    for exp in resume.experience:
        exp.job_title = pattern.sub(swap, exp.job_title)


def _keep_posting_headline(resume: Resume, analysis: JDAnalysis | None) -> None:
    """A common posting title stays whole in the headline when a draft drops only its broad fields: 'AI Engineer' ->
    'AI Platform Engineer' for an 'AI Platform Engineer' post, never 'Full-Stack Engineer' -> 'Software Engineer'."""
    if analysis is None:
        return
    posting = _role_core(analysis.target_role)
    words, headline = posting.split(), resume.headline.split()
    if not headline or len(headline) >= len(words) or len(words) > 4:
        return
    position, missing = 0, []  # the headline is the posting's title with some words left out, in order
    for word in words:
        if position < len(headline) and spelling.us_word(word) == spelling.us_word(headline[position]):
            position += 1
        else:
            missing.append(word)
    if position == len(headline) and all(spelling.us_word(word) in _BROAD_FIELDS for word in missing):
        resume.headline = posting


def specialty_problems(resume: Resume, analysis: JDAnalysis | None) -> list[str]:
    """Where the headline, the summary or a job title names what makes the posting's title special ('voice' for a
    'Voice AI Engineer' post): it belongs only in the bullets."""
    words = specialty(analysis)
    if not words:
        return []
    pattern = re.compile(r"(?<![\w-])" + r"[\s-]+".join(either_spelling(word) for word in words.split())
                         + r"(?![\w-])", re.IGNORECASE)
    places = [("headline", resume.headline), ("summary", resume.summary)] + [
        (f"job title at {resume.profile.companies[exp.company_index].name}", exp.job_title) for exp in resume.experience]
    return [where for where, text in places if pattern.search(text)]


def _quoted(items: list[str]) -> str:
    """'a'; 'a' and 'b'; 'a', 'b' and 'c'."""
    quoted = [f"'{item}'" for item in items]
    return " and ".join(quoted) if len(quoted) <= 2 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def claim_flags(resume: Resume, jd: str, hints: str) -> list[Flag]:
    """Things the user did not provide and should confirm before sending the resume."""
    flags: list[Flag] = []
    hints_l = hints.lower()

    def check_numbers(section: str, text: str) -> None:
        """One flag naming every estimate in the text (a job's first bullet gives its size and its result)."""
        tokens = list(dict.fromkeys(t for t in _numbers_in(text) if t and t.lower() not in hints_l))
        if not tokens:
            return
        if len(tokens) == 1:
            reason = f"'{tokens[0]}' is an estimate, not a number you gave. Replace it with your real figure or remove it."
        else:
            reason = (f"{_quoted(tokens)} are estimates, not numbers you gave. Replace them with your real figures or "
                      "remove them.")
        flags.append(Flag(section=section, text=text, level="verify", reason=reason))

    check_numbers("headline", resume.headline)
    check_numbers("summary", resume.summary)
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index].name
        flags.append(
            Flag(section=f"experience:{exp.company_index}", text=exp.job_title, level="info",
                 reason=f"Suggested job title at {company}. Use a title you can honestly claim.")
        )
        for bullet in exp.bullets:
            check_numbers(f"experience:{exp.company_index}", bullet)
    for group in resume.skills:
        for item in group.skills:
            if not _in_either_spelling(item, jd) and not _in_either_spelling(item, hints):
                flags.append(
                    Flag(section="skills", text=item, level="info",
                         reason="Related skill added (not in the job description). Keep it only if you have it.")
                )
    return flags


MAX_POSTING_FLAGS = 10


def posting_match_flags(resume: Resume, analysis: JDAnalysis, hints: str) -> list[Flag]:
    """One 'info' flag per bullet or summary sentence that carries must-have terms the user's hints do not back
    up: the app writes those in to match the posting, so the user should confirm them. The first place each term
    appears is flagged, most recent employer first, then the summary."""
    places: list[tuple[str, str]] = [
        (f"experience:{exp.company_index}", bullet) for exp in resume.experience for bullet in exp.bullets
    ]
    places += [("summary", s) for s in re.split(r"(?<=[.!?])\s+", resume.summary) if s.strip()]
    grouped: dict[tuple[str, str], list[str]] = {}
    for term in _unique(analysis.must_have_skills):
        if mentions(hints, term):
            continue
        for place in places:
            if mentions(place[1], term):
                grouped.setdefault(place, []).append(term)
                break
    flags = [
        Flag(section=section, text=text, level="info",
             reason="Added to match the posting (" + ", ".join(terms) + "). Confirm you can speak to it in an "
                    "interview, or remove it.")
        for (section, text), terms in grouped.items()
    ]
    return flags[:MAX_POSTING_FLAGS]


def cover_flags(letter: CoverLetter, hints: str) -> list[Flag]:
    flags: list[Flag] = []
    hints_l = hints.lower()
    for paragraph in letter.paragraphs:
        for token in _numbers_in(paragraph):
            if token.lower() not in hints_l:
                flags.append(
                    Flag(section="cover_letter", text=paragraph, level="verify",
                         reason=f"Contains '{token}', which is not in your hints. Only keep figures you can back up.")
                )
                break
    return flags


# ------------------------------------------------------------------ US style / EU style spelling
_LEGAL_SUFFIX = re.compile(r"[,\s]+(?:inc|llc|ltd|limited|plc|gmbh|corp|corporation|co|company|s\.?a|ag|b\.?v|n\.?v|pty"
                           r"|llp|lp)\.?$", re.IGNORECASE)


def _name_forms(name: str) -> list[str]:
    """The name as written and its short form: 'Color Health, Inc.' -> 'Color Health'; 'Center for Internet
    Security (CIS)' -> 'Center for Internet Security'; 'The Behavior Lab' -> 'Behavior Lab'."""
    core = _LEGAL_SUFFIX.sub("", re.sub(r"\s*\([^)]*\)", "", name).strip()).strip(" ,")
    core = re.sub(r"^the\s+", "", core, flags=re.IGNORECASE)
    return [form for form in dict.fromkeys([name.strip(), core]) if form]


_PERSON_TITLES = frozenset({"mr", "mrs", "ms", "mx", "dr", "prof", "sir", "dame"})


def known_names(resume: Resume, analysis: JDAnalysis | None = None, company: str = "", manager: str = "") -> list[str]:
    """What keeps its own spelling: the employers, the schools, the hiring company (each also in its short form) and
    the hiring manager, also by each part of their name ('Ms. Gray')."""
    names = [c.name for c in resume.profile.companies] + [e.school for e in resume.profile.education]
    names += [cert.name for cert in resume.certifications]
    if analysis is not None:
        names.append(analysis.company)
    kept = [form for name in [*names, company, manager] if name and name.strip() for form in _name_forms(name)]
    for hiring in (company, analysis.company if analysis is not None else ""):  # 'Color' for 'Color Health, Inc.'
        words = _name_forms(hiring)[-1].split() if hiring and hiring.strip() else []
        if len(words) > 1 and words[0][:1].isupper() and len(words[0]) > 2:
            kept.append(words[0])
    kept += [part for part in re.findall(r"[A-Z][A-Za-z'\u2019-]+", manager or "") if part.lower() not in _PERSON_TITLES]
    return kept


def posting_terms(analysis: JDAnalysis | None) -> list[str]:
    """The job post's own keywords: written as the post spells them, so an ATS that searches for them letter for
    letter finds them."""
    return _posting_terms(analysis)


def skill_terms(resume: Resume, analysis: JDAnalysis | None = None) -> list[str]:
    """Skills, changed to the style even when capitalised inside a sentence ('led Data Modeling')."""
    return [skill for group in resume.skills for skill in group.skills if skill and skill.strip()]


def role_terms(resume: Resume, analysis: JDAnalysis | None = None) -> list[str]:
    """The role the resume aims at, its headline and its job titles: changed to the style everywhere, even where a
    word looks like part of a product's name ('Senior Data Modeler' -> 'Senior Data Modeller')."""
    terms = [resume.headline, *(exp.job_title for exp in resume.experience)]
    if analysis is not None:
        terms += [analysis.target_role, _without_level(analysis.target_role), analysis.general_role]
    return [term for term in terms if term and term.strip()]


def spell_resume(resume: Resume, style: str, analysis: JDAnalysis | None = None,
                 names: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Every word the model wrote in the run's style: "us" (American) or "eu" (British), the job post's keywords as
    the post spells them. The profile's facts (name, employers, schools) are never touched. Returns the changes."""
    keep, exact = known_names(resume, analysis, *names), posting_terms(analysis)
    terms, roles = skill_terms(resume, analysis), role_terms(resume, analysis)
    changes: list[tuple[str, str]] = []

    def convert(text: str, title: bool, role_spans: list[str], brand: bool = False) -> str:
        new, done = spelling.convert_and_list(text, style, keep, terms, title, exact=exact, roles=role_spans,
                                              brand=brand)
        changes.extend(done)
        return new

    resume.headline = convert(resume.headline, True, roles)
    seen: set[str] = set()
    for group in resume.skills:
        brand = spelling.names_brand(group.category)  # 'AWS': 'Organizations', 'Compute Optimizer'
        group.category = convert(group.category, True, [])
        items = []
        for item in (convert(skill, True, [], brand) for skill in group.skills):
            if item.lower() not in seen:  # 'Data Visualisation' twice once both spellings are one
                seen.add(item.lower())
                items.append(item)
        group.skills = items
    resume.skills = [group for group in resume.skills if group.skills]
    resume.summary = convert(resume.summary, False, roles)
    for exp in resume.experience:
        exp.job_title = convert(exp.job_title, True, roles)
        exp.bullets = [convert(bullet, False, roles) for bullet in exp.bullets]
    return changes


def spell_letter(letter: CoverLetter, style: str, keep: list[str], terms: list[str], exact: list[str] | None = None,
                 roles: list[str] | None = None) -> None:
    """The cover letter's paragraphs in the run's style (the job post's keywords as the post spells them), 'my
    resume' called a CV in EU style, and the sign-off of that style ('Kind regards,' / 'Sincerely,'). The greeting
    holds names and stays."""
    letter.paragraphs = [spelling.cv_wording(spelling.convert(paragraph, style, keep, terms, exact=exact or [],
                                                              roles=roles or []), style)
                         for paragraph in letter.paragraphs]
    letter.closing = spelling.sign_off(letter.closing, style)


def spelling_problems(resume: Resume, letter: CoverLetter | None, style: str, analysis: JDAnalysis | None = None,
                      names: tuple[str, ...] = ()) -> list[str]:
    """Words in the other spelling style: the model's text is converted, so these come from the user's own edits.
    The job post's keywords keep the post's spelling on purpose and are not reported, in whatever case the post or
    the text writes them ('Data Modeling' in the post, 'data modeling' in a sentence)."""
    keep, exact = known_names(resume, analysis, *names), posting_terms(analysis)
    terms, roles = skill_terms(resume, analysis), role_terms(resume, analysis)
    places = [(resume.headline, True, roles, False),
              *((skill, True, [], spelling.names_brand(group.category)) for group in resume.skills
                for skill in group.skills),
              *((exp.job_title, True, roles, False) for exp in resume.experience), (resume.summary, False, roles, False),
              *((bullet, False, roles, False) for exp in resume.experience for bullet in exp.bullets)]
    if letter is not None:
        places += [(paragraph, False, roles, False) for paragraph in letter.paragraphs]
    found = [word for text, title, role_spans, brand in places
             for word in spelling.other_style_words(text, style, keep, terms, title, roles=role_spans, brand=brand,
                                                    exact=exact)]
    return list(dict.fromkeys(found))


def clean_cover_letter(
    greeting: str, paragraphs: list[str], closing: str, from_model: bool = True
) -> tuple[str, list[str], str]:
    """The model's paragraphs lose their dash punctuation; text the user typed in the editor (from_model=False) is
    kept as typed."""
    prose = clean_prose if from_model else clean_text
    paras = [p for p in (drop_repeated_words(prose(x)) for x in paragraphs) if p]
    if not 2 <= len(paras) <= 6:
        raise ValueError("The cover letter needs between 2 and 6 non-empty paragraphs.")
    greeting, closing = clean_text(greeting), clean_text(closing)
    if not greeting or not closing:
        raise ValueError("`greeting` and `closing` must not be empty.")
    return greeting, paras, closing
