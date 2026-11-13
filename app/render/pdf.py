"""Resume and cover letter PDFs, drawn directly with fpdf2.

Pure Python: no system libraries (no Pango/Cairo/GTK), so the file is the same on Windows, macOS and Linux. One bundled
font family (Inter, in four weights), single column, real text, no tables, images or icons: the layout ATS parsers read
best.

The look is the owner's design "standard" (a US Letter page drawn at 816 x 1056 CSS px): a centred header, small spaced
capitals in navy for the section headings with a thin gray rule beside them, the job title first on each job line,
gray bullet markers and a column for the skills labels. Every size and gap below is the design's px value as a share of
its 13.5 px body text, so each step of the size ladder (app/pipeline/fit.py) is the same design, only larger or smaller.
The ladder's last step also brings the gaps between the body's parts closer (FitStyle.spacing), so a long career of many
short jobs keeps the pages it took before the restyle.

The code is a small typesetter: measure text, wrap it into lines, draw the lines top to bottom. Drawing in reading order
also gives a clean text layer for ATS extraction. A page break never splits a bullet, and never leaves a section heading
or a job line alone at the bottom of a page.

The latest resume PDFs are kept in memory (remember_resume_pdf), so the preview and the downloads that follow a run
serve the file its job made instead of laying the resume out again.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from fontTools.ttLib import TTFont
from fpdf import FPDF
from fpdf.enums import XPos, YPos

try:
    from fpdf.fonts import SubsetMap
except ImportError:  # an fpdf2 without it has nothing to forget
    SubsetMap = None

from ..schemas import (
    Certification,
    Company,
    Contact,
    CoverLetter,
    Education,
    FitStyle,
    Resume,
    SkillGroup,
)
from ..timing import FONTS

FONT_DIR = Path(__file__).parent / "fonts"

# Inter's four weights as fpdf2 font families, with their files. fpdf2 keeps at most four styles in a family ("", "B",
# "I" and "BI"), so Medium and SemiBold have families of their own.
REGULAR, MEDIUM, SEMIBOLD, BOLD = "regular", "medium", "semibold", "bold"
WEIGHTS = {
    REGULAR: ("Inter", "", "Inter-Regular.ttf"),  # body text
    MEDIUM: ("InterMedium", "", "Inter-Medium.ttf"),  # the title line under the name
    SEMIBOLD: ("InterSemiBold", "", "Inter-SemiBold.ttf"),  # job titles, skills labels, degrees, certification names
    BOLD: ("Inter", "B", "Inter-Bold.ttf"),  # the name and the section headings
}
CAP_HEIGHT = 0.7275  # Inter's capital letters are this share of the font size tall (1490 of 2048 units)

# The design's colours
INK = "#0F1216"  # the name, job titles, skills labels, degrees and certification names
TEXT = "#1B1F24"  # body text
SOFT = "#3A4049"  # what follows a job title (company, place, work mode), the school, the issuer
MUTED = "#454B54"  # the contact line and the dates
MARKER = "#5F6672"  # bullet markers
RULE = "#CDD3DB"  # the thin rules beside the headings and under the letter's header
ACCENT = "#1F3A5F"  # navy: the title line and the headings

DESIGN_PX = 13.5  # the design's body text in CSS px: an em below


def em(px: float) -> float:
    """A length of the design, in CSS px, as a share of its body text size."""
    return px / DESIGN_PX


DESIGN_SIZE = 10.125  # the design's body text in points (13.5 px): the first step of the size ladder
DESIGN_MARGIN = 0.667  # its side margins in inches (64 px)
TOP_MARGIN = 56 / 64  # the top and bottom margins as a share of the side ones (56 and 64 px in the design)

# Font sizes, as shares of the body size
NAME_SIZE = em(30)
TITLE_SIZE = em(15)  # the title line under the name
CONTACT_SIZE = em(12.5)
HEADING_SIZE = em(11.5)
JOB_SIZE = em(14)  # a job line
DATE_SIZE = em(12.5)  # the dates on the right of a row

# Line heights, as multiples of the font size (the design's own)
LINE = 1.42  # body text and the rows
NAME_LINE, TITLE_LINE, HEADING_LINE = 1.1, 1.3, 1.2

# Letter spacing in em of the text's own size. ATS readers built like pdfminer read letters 0.10 em or more apart as
# separate words, so the headings' spacing stays at or below 0.09 em.
NAME_TRACKING = -0.02
HEADING_TRACKING = 0.08
MAX_TRACKING = 0.09
assert HEADING_TRACKING <= MAX_TRACKING

# Gaps, as shares of the body size. Those between the body's parts (sections, jobs, bullets, rows: not the header's) are
# measured in Sheet.gap_unit, the body size times FitStyle.spacing (1, or less at the ladder's last step).
SECTION_GAP = em(20)  # above a section heading (left out at the top of a page)
HEADER_GAP = em(6)  # between the name, the title line and the contact line
HEADING_GAP = em(8)  # from a heading to its first line
JOB_GAP = em(12)  # between jobs
BULLETS_GAP = em(4)  # from a job line to its first bullet
BULLET_GAP = em(2)  # between bullets
SKILLS_ROW_GAP = em(3)  # between skills rows
ENTRY_GAP = em(4)  # between education and certification rows
LABEL_COLUMN = em(112)  # the skills labels' column
LABEL_GAP = em(16)  # from the labels' column to the values
LABEL_COLUMN_LIMIT = 0.4  # a label column wider than this share of the text width: labels in line with the values
RULE_GAP = em(12)  # from a heading's text to its rule
DATE_GAP = 1.0  # the least room between a row's text and its right-aligned dates
MARKER_END = em(17)  # a bullet's marker ("• ") ends here, from the left margin (where CSS puts an outside marker)
TEXT_INDENT = em(20)  # where a bullet's text starts: its hanging indent
RULE_WIDTH = 0.75  # points: the design's 1 px rules

SPACES = re.compile(r"(\s+)")


def display_url(url: str) -> str:
    """linkedin.com/in/name instead of https://www.linkedin.com/in/name/.

    PDF viewers (Chrome, Edge, Acrobat) turn text that starts with http(s):// or www. into a clickable
    link by themselves; without the prefix it stays plain text, and it is shorter and still ATS-readable.
    """
    value = url.strip()
    for prefix in ("https://", "http://"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
    if value.lower().startswith("www."):
        value = value[4:]
    return value.rstrip("/")


def contact_parts(contact: Contact) -> list[str]:
    """The usual resume order, whatever order the base resume used: City, State, Phone, Email, LinkedIn,
    Portfolio/GitHub. Missing parts are left out."""
    parts = [contact.location, contact.phone, contact.email, display_url(contact.linkedin), display_url(contact.website)]
    return [part.strip() for part in parts if part and part.strip()]


def contact_line(contact: Contact) -> str:
    """The contact details on one line, joined by middle dots as in the design."""
    return " · ".join(contact_parts(contact))


def person_name(name: str) -> str:
    """Normal capitals for a name written all in capitals (or all small): 'JUAN DANIEL RAMIREZ' -> 'Juan Daniel Ramirez'.
    A name that already mixes them ('Ann McDonald') stays as it is."""
    return name.title() if name.isupper() or name.islower() else name


def fmt_dates(start: str, end: str) -> str:
    """'Feb 2025 – Present': the design's en dash between two dates (a job's, a school's, or a certification's earned
    and expiry dates); one date alone as it is."""
    if start and end:
        return f"{start} – {end}"
    return start or end or ""


# ---------------------------------------------------------------------- measuring
@cache
def advance_widths(weight: str) -> tuple[dict[int, int], int]:
    """Each character's advance width in a weight, in thousandths of an em rounded as fpdf2 rounds them, and the width
    fpdf2 gives a character the font does not have (its .notdef glyph). Read once per process: measuring by table is
    many times faster than asking fpdf2, which also writes a font change into the page for every measurement."""
    font = TTFont(FONT_DIR / WEIGHTS[weight][2], lazy=True)
    scale = 1000 / font["head"].unitsPerEm
    metrics = font["hmtx"].metrics
    widths = {code: round(scale * metrics[glyph][0] + 0.001) for code, glyph in font.getBestCmap().items()}
    return widths, round(scale * metrics[".notdef"][0])


@cache
def _covered() -> frozenset[int]:
    """The characters every weight has (Inter's four weights share one character map)."""
    return frozenset.intersection(*(frozenset(advance_widths(weight)[0]) for weight in WEIGHTS))


class _Drawable(dict):
    """A str.translate table for the text a PDF draws, filled in as characters are met. Inter lacks a few characters
    that the font before it had: the ligatures in text copied out of a PDF (ﬁ ﬂ ﬀ ﬃ ﬄ) and invisible marks such as a
    soft hyphen or a zero-width joiner. fpdf2 leaves a character the font lacks out of the page, while measuring counts
    it, so 'Certiﬁed' would print as 'Certied' with a gap after it. Such a character becomes its compatibility form
    when the fonts have all of that (ﬁ -> fi, a full-width Ａ -> A); an invisible one is left out; any other stays as it
    is, and fpdf2 leaves it out as before (as it did Chinese and emoji with the old font too)."""

    def __missing__(self, code: int) -> int | str | None:
        covered, char = _covered(), chr(code)
        form = unicodedata.normalize("NFKC", char)
        if code in covered:
            value: int | str | None = code
        elif all(ord(part) in covered for part in form):
            value = form
        elif unicodedata.category(char) == "Cf":
            value = None
        else:
            value = code
        self[code] = value
        return value


DRAWABLE = _Drawable()


def char_spacing(tracking: float, size: float) -> float:
    """Letter spacing in points, as the PDF holds it (fpdf2 writes it with two decimals)."""
    return round(tracking * size, 2)


def text_width(text: str, weight: str, size: float, tracking: float = 0.0) -> float:
    """The width of `text` in points as the PDF draws it (see _Drawable), the same as fpdf2's get_string_width of that.
    Letter spacing follows every character, the last one too, as in the PDF (and in CSS)."""
    widths, missing = advance_widths(weight)
    text = text.translate(DRAWABLE)
    return sum(widths.get(ord(ch), missing) for ch in text) * size * 0.001 + char_spacing(tracking, size) * len(text)


def fit_size(text: str, weight: str, size: float, width: float, floor: float = 11.0, tracking: float = 0.0) -> float:
    """Shrink a one-line text (the name, the title line) half a point at a time until it fits `width`, never below
    `floor`."""
    while size > floor and text_width(text, weight, size, tracking) > width:
        size = max(floor, size - 0.5)
    return size


def headline_size(headline: str, body: float, width: float) -> float:
    """The title line's size: the design's 15/13.5 of the body size, less only as far as needed to keep it on one line
    (never below the body size). Pure, so the Word file uses the same size without laying out a PDF."""
    return fit_size(headline, MEDIUM, TITLE_SIZE * body, width, floor=body)


@dataclass
class Run:
    """Text in one style. `bold` stays for callers that pass it (Run(text, True)); a `weight` wins over it, and `bold`
    then says whether the weight reads as bold (SemiBold and Bold do: the Word file uses it). `color` is a hex colour
    ("" for the body text's), `tracking` the letter spacing in em."""
    text: str
    bold: bool = False
    weight: str = ""
    color: str = ""
    tracking: float = 0.0

    def __post_init__(self) -> None:
        self.weight = self.weight or (BOLD if self.bold else REGULAR)
        self.bold = self.weight in (SEMIBOLD, BOLD)
        self.color = self.color or TEXT


@dataclass
class Piece:
    """A measured word or space of a wrapped line, in its run's style."""
    text: str
    weight: str
    width: float
    color: str = TEXT
    tracking: float = 0.0


Line = list[Piece]


def forget_glyph_caches() -> None:
    """fpdf2 caches two methods of its SubsetMap with functools.cache, keyed by the map itself, so every document's four
    fonts (fontTools' parsed glyphs, about 10 MB) stay in memory for the life of the process, and a tailoring run lays
    out many documents. The cache only saves a lookup: pick() gives the same number for a character again, so emptying
    it is safe at any time, also while another thread lays out a document."""
    for name in ("pick", "get_glyph"):
        clear = getattr(getattr(SubsetMap, name, None), "cache_clear", None)
        if clear is not None:
            clear()


class Sheet:
    """A PDF being laid out. `y` is the current top of the next line, in points from the page top. `margin` is the left
    and right margin (x = 0 is there), `top` the top and bottom one (the same as `margin` when not given). `spacing`
    scales the gaps between the body's parts (FitStyle.spacing): they are measured in `gap_unit`."""

    def __init__(self, paper: str, margin: float, size: float, line: float, title: str, author: str,
                 lang: str = "en-US", top: float | None = None, spacing: float = 1.0):
        forget_glyph_caches()  # the documents laid out before this one are let go
        top = margin if top is None else top
        pdf = FPDF(orientation="P", unit="pt", format=paper)
        pdf.set_auto_page_break(False)  # pagination is handled here so a line is never split in two
        pdf.set_margins(margin, top, margin)
        pdf.c_margin = 0  # no inner padding: widths measured here are the widths drawn
        began = time.perf_counter()  # fpdf2 reads and parses the four font files again for every sheet
        for family, style, file in WEIGHTS.values():
            pdf.add_font(family, style, str(FONT_DIR / file))
        FONTS.add(time.perf_counter() - began)  # counted for the job lines (app/timing.py)
        pdf.set_title(title)
        pdf.set_author(author)
        try:
            pdf.set_lang(lang)  # en-GB for an EU style CV and its letter
        except Exception:  # older fpdf2 without set_lang
            pass
        pdf.add_page()
        pdf.set_draw_color(RULE)  # the only lines drawn are the thin rules
        pdf.set_line_width(RULE_WIDTH)
        self.pdf = pdf
        self.margin = margin
        self.top = top
        self.size = size
        self.gap_unit = size * spacing  # exactly `size` at spacing 1: every gap as the design has it
        self.line = line
        self.width = pdf.w - 2 * margin
        self.bottom = pdf.h - top
        self.y = top
        self.text_x = TEXT_INDENT * size  # a bullet's text, in points from the left margin
        self.bullet_x = MARKER_END * size - text_width("• ", REGULAR, size)  # its marker: "• " ends at MARKER_END
        self._wraps: dict[tuple, list[Line]] = {}  # the keep rules measure a block before drawing it

    # ------------------------------------------------------------------ measuring and wrapping
    def measure(self, text: str, weight: str, size: float, tracking: float = 0.0) -> float:
        return text_width(text, weight, size, tracking)

    def headline_size(self, headline: str, body: float) -> float:
        """The title line's size on this sheet (see headline_size)."""
        return headline_size(headline, body, self.width)

    def wrap(self, runs: list[Run], width: float, size: float) -> list[Line]:
        key = (tuple((run.text, run.weight, run.color, run.tracking) for run in runs), round(width, 3), size)
        if key not in self._wraps:
            self._wraps[key] = self._wrap(runs, width, size)
        return self._wraps[key]

    def _wrap(self, runs: list[Run], width: float, size: float) -> list[Line]:
        tokens: list[tuple[str, Run, bool]] = []
        for run in runs:
            for part in SPACES.split(run.text):
                if part:
                    tokens.append((" " if part.isspace() else part, run, part.isspace()))
        lines: list[Line] = []
        current: Line = []
        used = 0.0

        def flush() -> None:
            nonlocal current, used
            while current and current[-1].text == " ":
                current.pop()
            if current:
                lines.append(current)
            current, used = [], 0.0

        for text, run, is_space in tokens:
            w = text_width(text, run.weight, size, run.tracking)
            piece = Piece(text, run.weight, w, run.color, run.tracking)
            if is_space:
                if current:  # no leading spaces on a line
                    current.append(piece)
                    used += w
                continue
            if used + w <= width + 1e-6:
                current.append(piece)
                used += w
                continue
            flush()
            if w <= width:
                current.append(piece)
                used = w
                continue
            # one unbreakable token wider than the line (a long URL): split it by characters
            chunk = ""
            for ch in text:
                if chunk and text_width(chunk + ch, run.weight, size, run.tracking) > width:
                    lines.append([Piece(chunk, run.weight, text_width(chunk, run.weight, size, run.tracking),
                                        run.color, run.tracking)])
                    chunk = ch
                else:
                    chunk += ch
            if chunk:
                used = text_width(chunk, run.weight, size, run.tracking)
                current = [Piece(chunk, run.weight, used, run.color, run.tracking)]
        flush()
        return lines

    def para_height(self, runs: list[Run], size: float | None = None, x: float = 0.0) -> float:
        size = size or self.size
        return len(self.wrap(runs, self.width - x, size)) * size * self.line

    def bullet_height(self, text: str, after: float = 0.0) -> float:
        return len(self.wrap([Run(text)], self.width - self.text_x, self.size)) * self.size * self.line + after

    def _row_lines(self, left: list[Run], right: str, rest: list[Run] | None,
                   size: float) -> tuple[list[Line], list[Line], float]:
        """A row's lines beside the dates, its lines under them (`rest`) and the dates' width."""
        right_w = text_width(right, REGULAR, DATE_SIZE * self.size) if right else 0.0
        beside = self.width - (right_w + DATE_GAP * self.size if right else 0.0)
        first = self.wrap(left, beside, size) or [[]]
        later = self.wrap(rest, self.width, size) if rest else []
        return first, later, right_w

    def row_height(self, left: list[Run], right: str, rest: list[Run] | None = None,
                   size: float | None = None) -> float:
        first, later, _width = self._row_lines(left, right, rest, size or self.size)
        return (len(first) + len(later)) * self.size * self.line

    def job_line(self, company: Company, title: str, dates: str) -> tuple[list[Run], list[Run]]:
        """A job line's two parts on this sheet: what goes beside the dates, and what goes on the line under it (see
        split_job_line)."""
        size = JOB_SIZE * self.size
        dates_w = text_width(dates, REGULAR, DATE_SIZE * self.size) + DATE_GAP * self.size if dates else 0.0
        return split_job_line(company, title, size, self.width - dates_w)

    # ------------------------------------------------------------------ drawing
    def _ensure(self, height: float) -> None:
        if self.y + height > self.bottom + 0.01:
            self.pdf.add_page()
            self.y = self.top

    def keep(self, height: float) -> None:
        """Start a new page unless the next `height` points fit on this one (never more than a page body, so a
        block taller than a page cannot cause an empty page)."""
        self._ensure(min(height, self.bottom - self.top))

    def _draw(self, line: Line, x: float, y: float, size: float, height: float) -> None:
        merged: list[list] = []  # neighbours in the same style become one text string
        for piece in line:
            style = (piece.weight, piece.color, piece.tracking)
            if merged and merged[-1][1] == style:
                merged[-1][0] += piece.text
                merged[-1][2] += piece.width
            else:
                merged.append([piece.text, style, piece.width])
        for text, (weight, color, tracking), width in merged:
            family, style, _file = WEIGHTS[weight]
            self.pdf.set_font(family, style, size)
            self.pdf.set_text_color(color)
            self.pdf.set_char_spacing(char_spacing(tracking, size))
            self.pdf.set_xy(x, y)
            self.pdf.cell(w=width, h=height, text=text.translate(DRAWABLE), new_x=XPos.RIGHT, new_y=YPos.TOP)
            x += width
        self.pdf.set_char_spacing(0)  # written into the page only when it was set

    def space(self, points: float) -> None:
        self.y += points

    def para(self, runs: list[Run], size: float | None = None, after: float = 0.0, x: float = 0.0, align: str = "L",
             line: float | None = None) -> None:
        """Wrapped text from `x`; `line` is its line height when not the sheet's (as a multiple of the size)."""
        size = size or self.size
        height = size * (line or self.line)
        width = self.width - x
        for wrapped in self.wrap(runs, width, size):
            self._ensure(height)
            total = sum(p.width for p in wrapped)
            left = self.margin + x + ((width - total) / 2 if align == "C" else 0.0)
            self._draw(wrapped, left, self.y, size, height)
            self.y += height
        self.y += after

    def bullet(self, text: str, after: float = 0.0) -> None:
        """A whole bullet on one page, its gray marker hanging left of the text (the gap after it is space, not content,
        so it is not reserved)."""
        height = self.size * self.line
        lines = self.wrap([Run(text)], self.width - self.text_x, self.size)
        self.keep(len(lines) * height)
        for index, line in enumerate(lines):
            self._ensure(height)
            if index == 0:
                marker = Piece("•", REGULAR, text_width("•", REGULAR, self.size), MARKER)
                self._draw([marker], self.margin + self.bullet_x, self.y, self.size, height)
            self._draw(line, self.margin + self.text_x, self.y, self.size, height)
            self.y += height
        self.y += after

    def row(self, left: list[Run], right: str, rest: list[Run] | None = None, size: float | None = None) -> None:
        """Text on the left at `size` (the body size when not given), its dates right-aligned on the first line in
        smaller gray type, on the same baseline; `rest` goes on the lines under it, from the left margin.

        The gap before the dates is filled with real spaces. They are invisible, but without them PDF readers that
        group text by position (pdfminer, and ATS built like it) see the dates as a separate block and put them
        inside the first bullet."""
        size = size or self.size
        height = self.size * self.line
        first, later, right_w = self._row_lines(left, right, rest, size)
        date_size = DATE_SIZE * self.size
        # fpdf2 puts a cell's baseline at y + h/2 + 0.3 x the font size: a taller cell for the smaller dates puts them
        # on the left text's baseline
        date_height = height + 0.6 * (size - date_size)
        for index, line in enumerate(first + later):
            self._ensure(height)
            self._draw(line, self.margin, self.y, size, height)
            if index == 0 and right:
                used = sum(piece.width for piece in line)
                space_w = text_width(" ", REGULAR, date_size)
                count = int((self.width - right_w - used) // space_w)
                if count > 0:
                    filler = Piece(" " * count, REGULAR, count * space_w, MUTED)
                    self._draw([filler], self.margin + used, self.y, date_size, date_height)
                self._draw([Piece(right, REGULAR, right_w, MUTED)], self.margin + self.width - right_w, self.y,
                           date_size, date_height)
            self.y += height

    def label_row(self, label: str, values: str, values_x: float) -> None:
        """A skills row: the label ('Languages:') from the left margin, the values in their own column from
        `values_x`, wrapping inside it. Real spaces fill the gap on every line of the row: a reader that groups text by
        position (pdfminer, and ATS built like it) then keeps the label and its values on one line, and keeps a wrapped
        line with its row (a line starting at the values' column was read after the next row). The row stays on one
        page."""
        height = self.size * self.line
        lines = self.wrap([Run(values)], self.width - values_x, self.size) or [[]]
        self.keep(len(lines) * height)
        label_w = text_width(label, SEMIBOLD, self.size)
        space_w = text_width(" ", REGULAR, self.size)
        count, indent = int((values_x - label_w) // space_w), int(values_x // space_w)
        for index, line in enumerate(lines):
            self._ensure(height)
            if index == 0:
                pieces = [Piece(label, SEMIBOLD, label_w, INK)]
                if count > 0:
                    pieces.append(Piece(" " * count, REGULAR, count * space_w))
                self._draw(pieces, self.margin, self.y, self.size, height)
            elif indent > 0:
                self._draw([Piece(" " * indent, REGULAR, indent * space_w)], self.margin, self.y, self.size, height)
            self._draw(line, self.margin + values_x, self.y, self.size, height)
            self.y += height

    def rule(self, before: float = 0.0, after: float = 0.0, width: float = RULE_WIDTH) -> None:
        """A full-width thin gray line, as under the cover letter's header."""
        self.y += before
        self.pdf.set_line_width(width)
        self.pdf.line(self.margin, self.y + width / 2, self.margin + self.width, self.y + width / 2)
        self.y += width + after

    def heading(self, text: str, keep: float = 0.0) -> None:
        """A section heading: small spaced capitals in the accent colour, then a thin rule from the text to the right
        margin, level with the middle of the capitals. Kept on the page of its first entry (`keep` is that entry's
        height); the gap above it is left out at the top of a page."""
        size = HEADING_SIZE * self.size
        height = HEADING_LINE * size
        if self.y > self.top + 0.01:
            self.y += SECTION_GAP * self.gap_unit
        self.keep(height + HEADING_GAP * self.gap_unit + keep)  # a new page drops the gap added above
        text = text.upper()
        width = text_width(text, BOLD, size, HEADING_TRACKING)
        self._draw([Piece(text, BOLD, width, ACCENT, HEADING_TRACKING)], self.margin, self.y, size, height)
        start, end = self.margin + width + RULE_GAP * self.size, self.margin + self.width
        if start < end:
            middle = self.y + height / 2 + 0.3 * size - CAP_HEIGHT * size / 2  # the baseline, less half the capitals
            self.pdf.set_line_width(RULE_WIDTH)
            self.pdf.line(start, middle, end, middle)
        self.y += height + HEADING_GAP * self.gap_unit

    def last_page_fill(self) -> float:
        """How full the last page is, 0..1: a second page holding a few lines is worth avoiding."""
        return max(0.0, min(1.0, (self.y - self.top) / (self.bottom - self.top)))

    def finish(self) -> tuple[bytes, int]:
        return bytes(self.pdf.output()), self.pdf.page


# ---------------------------------------------------------------------- the header (resume and cover letter)
def contact_lines(contact: Contact, size: float, width: float) -> list[str]:
    """The contact line, split between items (never inside an email or link) when it is too wide for one line."""
    lines: list[str] = []
    for item in contact_parts(contact):
        joined = f"{lines[-1]} · {item}" if lines else item
        if lines and text_width(joined, REGULAR, size) <= width:
            lines[-1] = joined
        else:
            lines.append(item)
    return lines


def draw_header(sheet: Sheet, name: str, headline: str, contact: Contact, body: float) -> None:
    """The name, the title line and the contact line, centred, in sizes taken from `body`. The resume draws it at its
    own size step; the cover letter at the resume's first step, so the two look like one set."""
    name_size = fit_size(name, BOLD, NAME_SIZE * body, sheet.width, tracking=NAME_TRACKING)
    sheet.para([Run(name, weight=BOLD, color=INK, tracking=NAME_TRACKING)], size=name_size, align="C",
               line=NAME_LINE)
    if headline:
        sheet.space(HEADER_GAP * body)
        sheet.para([Run(headline, weight=MEDIUM, color=ACCENT)], size=sheet.headline_size(headline, body),
                   align="C", line=TITLE_LINE)
    lines = contact_lines(contact, CONTACT_SIZE * body, sheet.width)
    if lines:
        sheet.space(HEADER_GAP * body)
    for line in lines:
        sheet.para([Run(line, color=MUTED)], size=CONTACT_SIZE * body, align="C", line=LINE)


# ---------------------------------------------------------------------- resume
def document_name(spelling: str | None) -> str:
    """What a run calls its resume, in the PDF's title and the file names: a CV in EU style (British usage), a
    resume otherwise (US style, and runs made before the style buttons)."""
    return "CV" if spelling == "eu" else "Resume"


def render_resume_pdf(resume: Resume, style: FitStyle, document: str = "Resume") -> tuple[bytes, int]:
    """Return (pdf bytes, page count). `document` names it in the file's title: "Resume", or "CV" in EU style.
    Always lays the resume out (the server's warm-up relies on it): recent_resume_pdf serves a kept file."""
    return resume_sheet(resume, style, document).finish()


def resume_layout(resume: Resume, style: FitStyle) -> tuple[int, float]:
    """(page count, how full the last page is), without building the PDF file."""
    sheet = resume_sheet(resume, style)
    return sheet.pdf.page, sheet.last_page_fill()


def job_parts(company: Company) -> list[str]:
    """What follows a job title on its line, in order: the company, its place (when known), the work mode (when
    known)."""
    return [part for part in (company.name, company.location, company.mode) if part]


def job_runs(company: Company, title: str) -> list[Run]:
    """A job line as one line: the title first (SemiBold, dark), then ' · Company · Place · Mode' in gray."""
    return [Run(title, weight=SEMIBOLD, color=INK)] + _after_title(job_parts(company))


def _after_title(parts: list[str]) -> list[Run]:
    """The parts that follow a job title on its line, each after ' · ', in gray ([] for none)."""
    return [Run("".join(f" · {part}" for part in parts), color=SOFT)] if parts else []


def split_job_line(company: Company, title: str, size: float, width: float) -> tuple[list[Run], list[Run]]:
    """A job line in two when it is too wide: (what goes beside the dates, what goes on the line under it).

    The first part is the title and as many whole ' · ' parts after it as fit in `width` (the room left of the dates,
    at font size `size`); the rest goes under it, from the left margin and without its leading '· ', so a long line
    breaks between two parts (before the company, the place or the work mode), never inside one. A title too long on
    its own wraps by words where it is drawn. The second part is [] when one line holds everything. The PDF and the
    Word file both use it."""
    parts = job_parts(company)
    used, kept = text_width(title, SEMIBOLD, size), 0
    for part in parts:
        extra = text_width(f" · {part}", REGULAR, size)
        if used + extra > width + 1e-6:
            break
        used, kept = used + extra, kept + 1
    first = [Run(title, weight=SEMIBOLD, color=INK)] + _after_title(parts[:kept])
    rest = [Run(" · ".join(parts[kept:]), color=SOFT)] if kept < len(parts) else []
    return first, rest


def education_runs(edu: Education) -> list[Run]:
    """An education line: 'Degree, Field' (SemiBold, dark) then ' · School' in gray, or the school alone."""
    main = ", ".join(x for x in (edu.degree, edu.field) if x)
    if main:
        return [Run(main, weight=SEMIBOLD, color=INK), Run(f" · {edu.school}", color=SOFT)]
    return [Run(edu.school, weight=SEMIBOLD, color=INK)]


def certification_runs(cert: Certification) -> list[Run]:
    """A certification's line, like an education line: the name (SemiBold, dark), then ' · Issuer' in gray."""
    issuer = [Run(f" · {cert.issuer}", color=SOFT)] if cert.issuer else []
    return [Run(cert.name, weight=SEMIBOLD, color=INK)] + issuer


def certification_dates(cert: Certification) -> str:
    """Its dates on the right, like a job's: 'Mar 2023 – Mar 2026' (earned – expires), 'Feb 2020' when it does not
    expire, 'Expires Mar 2026' when the date it was earned is not known."""
    if cert.year and cert.expires:
        return fmt_dates(cert.year, cert.expires)
    return cert.year or (f"Expires {cert.expires}" if cert.expires else "")


def skills_column(groups: list[SkillGroup], size: float, width: float) -> float | None:
    """Where the skills values start, in points from the left margin: after the labels' column (the design's 112 px,
    or the widest label) and its gap. None when that column would take more than LABEL_COLUMN_LIMIT of the text width
    (`width`): the labels then go in line with their values, so a label never wraps. The Word file uses it too."""
    widest = max((text_width(f"{group.category}:", SEMIBOLD, size) for group in groups), default=0.0)
    column = max(LABEL_COLUMN * size, widest)
    if column > LABEL_COLUMN_LIMIT * width:
        return None
    return column + LABEL_GAP * size


def draw_skills(sheet: Sheet, groups: list[SkillGroup]) -> None:
    """Each skills group on its own row: its label in a column, the values beside it (or in line, see skills_column)."""
    values_x = skills_column(groups, sheet.size, sheet.width)
    for index, group in enumerate(groups):
        if index:
            sheet.space(SKILLS_ROW_GAP * sheet.gap_unit)
        label, values = f"{group.category}:", ", ".join(group.skills)
        if values_x is not None:
            sheet.label_row(label, values, values_x)
            continue
        runs = [Run(label, weight=SEMIBOLD, color=INK), Run(" " + values)]
        sheet.keep(sheet.para_height(runs))
        sheet.para(runs)


def resume_sheet(resume: Resume, style: FitStyle, document: str = "Resume") -> Sheet:
    """The resume laid out on pages: `pdf.page` pages, `last_page_fill()`, and the file from `finish()`."""
    p = resume.profile
    side = style.margin_in * 72
    sheet = Sheet(style.paper, side, style.font_pt, LINE, f"{p.name} - {document}", p.name,
                  "en-GB" if document == "CV" else "en-US", top=TOP_MARGIN * side, spacing=style.spacing)
    size, unit = sheet.size, sheet.gap_unit  # unit: the gaps between the body's parts

    draw_header(sheet, p.name, resume.headline, p.contact, size)

    sheet.heading("SUMMARY")
    sheet.para([Run(resume.summary)])

    sheet.heading("SKILLS")
    draw_skills(sheet, resume.skills)

    jobs = []  # (line beside the dates, line under it, dates, bullets, height kept together: the job line + 1st bullet)
    for exp in resume.experience:
        company = p.companies[exp.company_index]
        dates = fmt_dates(company.start, company.end)
        left, rest = sheet.job_line(company, exp.job_title, dates)
        start = sheet.row_height(left, dates, rest, JOB_SIZE * size)
        if exp.bullets:
            start += BULLETS_GAP * unit + sheet.bullet_height(exp.bullets[0])
        jobs.append((left, rest, dates, exp.bullets, start))
    sheet.heading("EXPERIENCE", keep=jobs[0][4] if jobs else 0.0)
    for index, (left, rest, dates, bullets, start) in enumerate(jobs):
        sheet.space(JOB_GAP * unit if index else 0.0)
        sheet.keep(start)  # a job line never sits alone at the bottom of a page
        sheet.row(left, dates, rest, JOB_SIZE * size)
        if bullets:
            sheet.space(BULLETS_GAP * unit)
        for number, bullet in enumerate(bullets):
            sheet.bullet(bullet, after=BULLET_GAP * unit if number < len(bullets) - 1 else 0.0)

    if p.education:
        entries = []
        for edu in p.education:
            left, dates = education_runs(edu), fmt_dates(edu.start, edu.end)
            extra = [Run(edu.details)] if edu.details else []
            height = sheet.row_height(left, dates) + (sheet.para_height(extra) if extra else 0.0)
            entries.append((left, dates, extra, height))
        sheet.heading("EDUCATION", keep=entries[0][3])
        for index, (left, dates, extra, height) in enumerate(entries):
            sheet.space(ENTRY_GAP * unit if index else 0.0)
            sheet.keep(height)
            sheet.row(left, dates)
            if extra:
                sheet.para(extra)

    if resume.certifications:  # only ones the person has, best for the job first
        rows = [(certification_runs(cert), certification_dates(cert)) for cert in resume.certifications]
        heights = [sheet.row_height(left, dates) for left, dates in rows]
        sheet.heading("CERTIFICATIONS", keep=heights[0])
        for index, ((left, dates), height) in enumerate(zip(rows, heights)):
            sheet.space(ENTRY_GAP * unit if index else 0.0)
            sheet.keep(height)
            sheet.row(left, dates)
    return sheet


# ---------------------------------------------------------------------- recent resume PDFs
RECENT_PDFS = 8  # files kept in memory: a run's preview and downloads come soon after it
_recent_pdfs: OrderedDict[str, bytes] = OrderedDict()  # least recently used first
_recent_lock = threading.Lock()  # a job's worker thread and the PDF requests' threads use it at once


def _pdf_key(resume: Resume, style: FitStyle, document: str) -> str:
    """What the file is made of: the resume's text and facts, the layout style and the document word. An edit, a
    ticked certification or another style is another key, so a kept file never stands in for a changed resume."""
    parts = (resume.model_dump_json(), style.model_dump_json(), document)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def remember_resume_pdf(resume: Resume, style: FitStyle, document: str, pdf: bytes) -> None:
    """Keep a resume PDF just made, such as a job's fitted file. `pdf` must be the file of exactly this resume, style
    and document (what render_resume_pdf would make of them, or for a run saved before the restyle what
    fit.saved_resume_pdf makes of its saved style). The least recently used file goes when more than RECENT_PDFS are
    kept."""
    key, pdf = _pdf_key(resume, style, document), bytes(pdf)
    with _recent_lock:
        _recent_pdfs[key] = pdf
        _recent_pdfs.move_to_end(key)
        while len(_recent_pdfs) > RECENT_PDFS:
            _recent_pdfs.popitem(last=False)


def recent_resume_pdf(resume: Resume, style: FitStyle, document: str) -> bytes | None:
    """The kept PDF of this resume, style and document, or None when it is not kept (any more)."""
    key = _pdf_key(resume, style, document)
    with _recent_lock:
        pdf = _recent_pdfs.get(key)
        if pdf is not None:
            _recent_pdfs.move_to_end(key)  # read again: kept longer
        return pdf


def forget_resume_pdfs() -> None:
    """Drop every kept PDF (the admin deleted an account, or all data)."""
    with _recent_lock:
        _recent_pdfs.clear()


# ---------------------------------------------------------------------- cover letter
# (body size, line height, space after a paragraph in em): the first that keeps the letter on one page is used
COVER_STYLES = [(9.5, LINE, 0.8), (9.0, LINE, 0.8), (8.6, LINE, 0.8)]
COVER_MARGIN = 72.0  # 1 inch all round
SIGNATURE_ROOM = 18.0  # points under the closing for a handwritten signature on a printout


def render_cover_pdf(resume: Resume, letter: CoverLetter, paper: str = "Letter",
                     lang: str = "en-US") -> tuple[bytes, int]:
    """The letter under the same header as the resume (name, headline, contact line), so the two look like one set:
    a thin rule, then the letter from its greeting to the typed signature (no date, address or subject line). One
    column of real text in reading order, like the resume, so applicant systems read it too."""
    for size, line_height, gap in COVER_STYLES:
        data, pages = _cover(resume, letter, paper, size, line_height, gap, lang)
        if pages == 1:
            break
    return data, pages


def _cover(resume: Resume, letter: CoverLetter, paper: str, size: float, line_height: float,
           gap: float, lang: str = "en-US") -> tuple[bytes, int]:
    p = resume.profile
    sheet = Sheet(paper, COVER_MARGIN, size, line_height, f"{p.name} - Cover Letter", p.name, lang)
    draw_header(sheet, p.name, resume.headline, p.contact, DESIGN_SIZE)  # the resume's header at its first step
    sheet.rule(before=size, after=2 * size)

    after = gap * size
    sheet.para([Run(letter.greeting)], after=after)
    for paragraph in letter.paragraphs:
        sheet.para([Run(paragraph)], after=after)
    sheet.para([Run(letter.closing)], after=after + SIGNATURE_ROOM)
    sheet.para([Run(person_name(p.name))])
    return sheet.finish()
