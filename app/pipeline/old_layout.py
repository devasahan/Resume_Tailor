"""The resume layout from before the restyle, measured without drawing.

Until the restyle the resume was set in Carlito (metrically compatible with Calibri) on five size steps from 10.5 to
9.5 pt (fit.OLD_LADDER), with each job on one line: the company in bold, the title in italics, then its place and the
dates. The bullet plan (fit.bullet_budget) and the page count of a fitted resume (fit.try_styles) are still worked out
on that layout, so the restyle changes only the look: every plan is the one the app made before, and no resume takes
more pages than it did.

This is that layout (app/render/pdf.py at commit aa88a1a: Sheet and resume_sheet, the one-line job header) reduced to
arithmetic: the same measuring, wrapping, page breaks and keep rules, with the same numbers in the same order, so it
gives exactly the page count and last-page fill the old code did, without fpdf2 or the font files. Text is measured
with Carlito's advance widths as fpdf2 measured them (carlito/widths.json, made by carlito/make_widths.py). A layout
takes a few milliseconds, and less for text already measured.
"""
from __future__ import annotations

import json
import re
from functools import cache, lru_cache
from pathlib import Path

from ..schemas import Certification, Contact, FitStyle, Resume

WIDTHS = Path(__file__).parent / "carlito" / "widths.json"
REGULAR, BOLD, ITALIC = "regular", "bold", "italic"
PAPER = {"Letter": (612.0, 792.0), "A4": (595.28, 841.89)}  # fpdf2's page sizes in points

# The old layout's numbers (points, or a multiple of the font size for the line height)
RESUME_LINE = 1.27
TEXT_X = 13.0  # a bullet's hanging indent
ROW_GAP = 10.0  # the least room between a row's text and its dates
HEADLINE_EXTRA = 2.0  # the title line under the name is this much larger than the body text
HEADER_GAP = 0.75
# fpdf2 reads "{nb}" in any text as its page count alias and measures it as "000", between the text around it
PAGE_COUNT_ALIAS, ALIAS_WIDTH_TEXT = "{nb}", "000"

SPACES = re.compile(r"(\s+)")

Runs = tuple[tuple[str, str], ...]  # (text, style) pieces of one paragraph or row


# ---------------------------------------------------------------------- measuring, as fpdf2 did
@cache
def carlito() -> dict[str, tuple[dict[int, int], int]]:
    """For each style: {code point: width in thousandths of an em} and the width of a character Carlito does not
    have. Read once per process."""
    data = json.loads(WIDTHS.read_text(encoding="utf-8"))
    fonts = {}
    for style, entry in data.items():
        widths = {first + offset: width for first, run in entry["ranges"] for offset, width in enumerate(run)}
        fonts[style] = (widths, entry["missing"])
    return fonts


def _units_of(text: str, style: str) -> tuple[int, ...]:
    """The text's width in thousandths of an em, for each piece fpdf2 measured on its own: the whole text, unless it
    holds the page count alias."""
    widths, missing = carlito()[style]
    if PAGE_COUNT_ALIAS not in text:
        return (sum(widths.get(ord(char), missing) for char in text),) if text else ()
    pieces = []
    for number, part in enumerate(text.split(PAGE_COUNT_ALIAS)):
        if number:
            pieces.append(sum(widths.get(ord(char), missing) for char in ALIAS_WIDTH_TEXT))
        if part:
            pieces.append(sum(widths.get(ord(char), missing) for char in part))
    return tuple(pieces)


_units = lru_cache(maxsize=8192)(_units_of)  # words come back often: every layout of a resume measures them again


def text_width(text: str, style: str, size: float) -> float:
    """The width of `text` in points, the same number fpdf2's get_string_width gave (the same sums, in its order)."""
    return sum(units * size * 0.001 for units in _units(text, style))


def _split_width(text: str, style: str, size: float) -> float:
    """text_width without the cache, for the pieces of a word split by characters (each a new string)."""
    return sum(units * size * 0.001 for units in _units_of(text, style))


@lru_cache(maxsize=4096)
def line_count(runs: Runs, width: float, size: float) -> int:
    """How many lines the old typesetter's wrap gave `runs` in `width` points at `size`: words fill a line while they
    fit (with 1e-6 to spare), spaces never start a line, and a word wider than the line is split by characters."""
    tokens = []
    for text, style in runs:
        for part in SPACES.split(text):
            if part:
                tokens.append((" " if part.isspace() else part, style, part.isspace()))
    lines, filling, used = 0, False, 0.0  # `filling`: the current line holds a word (it never starts with a space)
    for text, style, is_space in tokens:
        w = text_width(text, style, size)
        if is_space:
            if filling:
                used += w
            continue
        if used + w <= width + 1e-6:
            filling, used = True, used + w
            continue
        lines, filling, used = lines + int(filling), False, 0.0  # the line so far is done
        if w <= width:
            filling, used = True, w
            continue
        chunk = ""  # one unbreakable token wider than the line (a long URL): split it by characters
        for char in text:
            if chunk and _split_width(chunk + char, style, size) > width:
                lines, chunk = lines + 1, char
            else:
                chunk += char
        if chunk:
            filling, used = True, _split_width(chunk, style, size)
    return lines + int(filling)


# ---------------------------------------------------------------------- the old page arithmetic
class OldSheet:
    """The old Sheet's page arithmetic: `y` is the top of the next line in points from the page top, `pages` the pages
    begun so far. Every method moves `y` exactly as the old one did."""

    def __init__(self, paper: str, margin: float, size: float):
        page_width, page_height = PAPER[paper]
        self.margin = margin
        self.size = size
        self.line = RESUME_LINE
        self.width = page_width - 2 * margin
        self.bottom = page_height - margin
        self.y = margin
        self.pages = 1
        self._wraps: dict[tuple, int] = {}

    def lines(self, runs: Runs, width: float, size: float) -> int:
        """line_count, kept per sheet under the old wrap cache's key (the width rounded to 3 decimals)."""
        key = (runs, round(width, 3), size)
        if key not in self._wraps:
            self._wraps[key] = line_count(runs, width, size)
        return self._wraps[key]

    def fit_size(self, text: str, size: float, width: float, floor: float = 11.0) -> float:
        """Shrink a single-line bold text (the name, the title line) until it fits the width."""
        while size > floor and text_width(text, BOLD, size) > width:
            size -= 0.5
        return size

    def para_height(self, runs: Runs) -> float:
        return self.lines(runs, self.width, self.size) * self.size * self.line

    def bullet_height(self, text: str) -> float:
        """A bullet's lines, without the gap after it."""
        return self.lines(((text, REGULAR),), self.width - TEXT_X, self.size) * self.size * self.line

    def _row_lines(self, left: Runs, right: str) -> int:
        right_w = text_width(right, REGULAR, self.size) if right else 0.0
        return self.lines(left, self.width - (right_w + ROW_GAP if right else 0.0), self.size)

    def row_height(self, left: Runs, right: str) -> float:
        return max(self._row_lines(left, right), 1) * self.size * self.line

    def ensure(self, height: float) -> None:
        if self.y + height > self.bottom + 0.01:
            self.pages += 1
            self.y = self.margin

    def keep(self, height: float) -> None:
        self.ensure(min(height, self.bottom - self.margin))

    def space(self, points: float) -> None:
        self.y += points

    def para(self, runs: Runs, size: float | None = None, after: float = 0.0) -> None:
        size = size or self.size
        height = size * self.line
        for _line in range(self.lines(runs, self.width, size)):
            self.ensure(height)
            self.y += height
        self.y += after

    def bullet(self, text: str, after: float = 1.0) -> None:
        height = self.size * self.line
        count = self.lines(((text, REGULAR),), self.width - TEXT_X, self.size)
        self.keep(count * height)
        for _line in range(count):
            self.ensure(height)
            self.y += height
        self.y += after

    def row(self, left: Runs, right: str) -> None:
        height = self.size * self.line
        for _line in range(max(self._row_lines(left, right), 1)):
            self.ensure(height)
            self.y += height

    def heading(self, keep: float = 0.0) -> None:
        """A section heading and its rule, kept on the page of its first entry (`keep` is that entry's height)."""
        size = self.size + 0.5
        height = size * self.line
        self.y += 8
        self.keep(height + 5 + keep)
        self.y += height + 1
        self.y += 0.75 + 3

    def last_page_fill(self) -> float:
        return max(0.0, min(1.0, (self.y - self.margin) / (self.bottom - self.margin)))


# ---------------------------------------------------------------------- the old resume
def display_url(url: str) -> str:
    """The link as the old contact line showed it: no https://, www. or trailing slash."""
    value = url.strip()
    for prefix in ("https://", "http://"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
    if value.lower().startswith("www."):
        value = value[4:]
    return value.rstrip("/")


def contact_line(contact: Contact) -> str:
    """The old contact line: City, State | Phone | Email | LinkedIn | Portfolio, leaving out what is missing."""
    parts = [contact.location, contact.phone, contact.email, display_url(contact.linkedin),
             display_url(contact.website)]
    return " | ".join(part.strip() for part in parts if part and part.strip())


def fmt_dates(start: str, end: str) -> str:
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


def certification_dates(cert: Certification) -> str:
    if cert.year and cert.expires:
        return fmt_dates(cert.year, cert.expires)
    return cert.year or (f"Expires {cert.expires}" if cert.expires else "")


def resume_layout(resume: Resume, style: FitStyle) -> tuple[int, float]:
    """(page count, how full the last page is) of the resume laid out before the restyle at `style` (a step of
    fit.OLD_LADDER, on its paper), as the old resume_layout gave them."""
    p = resume.profile
    sheet = OldSheet(style.paper, style.margin_in * 72, style.font_pt)
    size = sheet.size

    sheet.para(((p.name, BOLD),), size=sheet.fit_size(p.name, size + 9, sheet.width))
    sheet.space(HEADER_GAP)
    sheet.para(((resume.headline, BOLD),), size=sheet.fit_size(resume.headline, size + HEADLINE_EXTRA, sheet.width,
                                                               floor=size))
    line = contact_line(p.contact)
    if line:
        sheet.space(HEADER_GAP)
        sheet.para(((line, REGULAR),))

    sheet.heading()  # SUMMARY
    sheet.para(((resume.summary, REGULAR),))

    sheet.heading()  # SKILLS
    for group in resume.skills:
        sheet.para(((f"{group.category}:", BOLD), (" " + ", ".join(group.skills), REGULAR)), after=1)

    jobs = []  # (job line, dates, bullets, height that must stay together: the job line and the first bullet)
    for exp in resume.experience:
        company = p.companies[exp.company_index]
        place = "".join(f" | {x}" for x in (company.location, company.mode) if x)
        left = ((company.name, BOLD), (" | ", REGULAR), (exp.job_title, ITALIC), (place, REGULAR))
        dates = fmt_dates(company.start, company.end)
        first = sheet.bullet_height(exp.bullets[0]) if exp.bullets else 0.0
        jobs.append((left, dates, exp.bullets, sheet.row_height(left, dates) + 1.5 + first))
    sheet.heading(keep=jobs[0][3] if jobs else 0.0)  # EXPERIENCE
    for index, (left, dates, bullets, start) in enumerate(jobs):
        sheet.space(4 if index else 0)
        sheet.keep(start)
        sheet.row(left, dates)
        sheet.space(1.5)
        for bullet in bullets:
            sheet.bullet(bullet)

    if p.education:
        entries = []
        for edu in p.education:
            main = ", ".join(x for x in (edu.degree, edu.field) if x)
            left = ((main, BOLD), (f" | {edu.school}", REGULAR)) if main else ((edu.school, BOLD),)
            dates, extra = fmt_dates(edu.start, edu.end), ((edu.details, REGULAR),) if edu.details else ()
            height = sheet.row_height(left, dates) + (sheet.para_height(extra) if extra else 0.0)
            entries.append((left, dates, extra, height))
        sheet.heading(keep=entries[0][3])  # EDUCATION
        for left, dates, extra, height in entries:
            sheet.keep(height)
            sheet.row(left, dates)
            if extra:
                sheet.para(extra)

    if resume.certifications:
        rows = [(((cert.name, BOLD),) + (((f" | {cert.issuer}", REGULAR),) if cert.issuer else ()),
                 certification_dates(cert)) for cert in resume.certifications]
        heights = [sheet.row_height(left, dates) for left, dates in rows]
        sheet.heading(keep=heights[0])  # CERTIFICATIONS
        for (left, dates), height in zip(rows, heights):
            sheet.keep(height)
            sheet.row(left, dates)
    return sheet.pages, sheet.last_page_fill()
