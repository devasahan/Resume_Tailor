"""Resume and cover letter PDFs, drawn directly with fpdf2.

Pure Python: no system libraries (no Pango/Cairo/GTK), so the file is the same on Windows, macOS and
Linux. One bundled font (Carlito, metrically compatible with Calibri), single column, real text, no
tables, images or icons: the layout ATS parsers read best.

The code is a small typesetter: measure text, wrap it into lines, draw the lines top to bottom.
Drawing in reading order also gives a clean text layer for ATS extraction. A page break never splits a bullet,
and never leaves a section heading or a job title alone at the bottom of a page.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from ..schemas import Certification, Company, Contact, CoverLetter, FitStyle, Resume
from ..timing import FONTS

FONT_DIR = Path(__file__).parent / "fonts"
FAMILY = "Carlito"

RESUME_LINE = 1.27  # line height as a multiple of the font size
BULLET_X = 4.0  # points from the left margin to the bullet glyph
TEXT_X = 13.0  # points from the left margin to bullet text (hanging indent)
ROW_GAP = 10.0  # minimum space between the left text and the right-aligned dates
HEADLINE_EXTRA = 2.0  # the title line under the name is this many points larger than the body text
HEADER_GAP = 0.75  # points above and below the title line (its larger size brings its own room)


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


def contact_line(contact: Contact) -> str:
    """The usual resume order, whatever order the base resume used: City, State | Phone | Email | LinkedIn |
    Portfolio/GitHub. Missing parts are left out."""
    parts = [contact.location, contact.phone, contact.email, display_url(contact.linkedin), display_url(contact.website)]
    return " | ".join(part.strip() for part in parts if part and part.strip())


def person_name(name: str) -> str:
    """Normal capitals for a name written all in capitals (or all small): 'JUAN DANIEL RAMIREZ' -> 'Juan Daniel Ramirez'.
    A name that already mixes them ('Ann McDonald') stays as it is."""
    return name.title() if name.isupper() or name.islower() else name


def fmt_dates(start: str, end: str) -> str:
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


@dataclass
class Run:
    text: str
    bold: bool = False
    italic: bool = False


@dataclass
class Piece:
    text: str
    bold: bool
    width: float
    italic: bool = False


def font_style(bold: bool, italic: bool = False) -> str:
    return ("B" if bold else "") + ("I" if italic else "")


Line = list[Piece]


class Sheet:
    """A PDF being laid out. `y` is the current top of the next line, in points from the page top."""

    def __init__(self, paper: str, margin: float, size: float, line: float, title: str, author: str,
                 lang: str = "en-US"):
        pdf = FPDF(orientation="P", unit="pt", format=paper)
        pdf.set_auto_page_break(False)  # pagination is handled here so a line is never split in two
        pdf.set_margins(margin, margin, margin)
        pdf.c_margin = 0  # no inner padding: widths measured here are the widths drawn
        began = time.perf_counter()  # fpdf2 reads and parses the three font files again for every sheet
        pdf.add_font(FAMILY, "", str(FONT_DIR / "Carlito-Regular.ttf"))
        pdf.add_font(FAMILY, "B", str(FONT_DIR / "Carlito-Bold.ttf"))
        pdf.add_font(FAMILY, "I", str(FONT_DIR / "Carlito-Italic.ttf"))
        FONTS.add(time.perf_counter() - began)  # counted for the job lines (app/timing.py)
        pdf.set_title(title)
        pdf.set_author(author)
        try:
            pdf.set_lang(lang)  # en-GB for an EU style CV and its letter
        except Exception:  # older fpdf2 without set_lang
            pass
        pdf.add_page()
        pdf.set_text_color(0, 0, 0)
        pdf.set_draw_color(0, 0, 0)
        self.pdf = pdf
        self.margin = margin
        self.size = size
        self.line = line
        self.width = pdf.w - 2 * margin
        self.bottom = pdf.h - margin
        self.y = margin
        self._wraps: dict[tuple, list[Line]] = {}  # the keep rules measure a block before drawing it

    # ------------------------------------------------------------------ measuring and wrapping
    def measure(self, text: str, bold: bool, size: float, italic: bool = False) -> float:
        self.pdf.set_font(FAMILY, font_style(bold, italic), size)
        return self.pdf.get_string_width(text)

    def fit_size(self, text: str, bold: bool, size: float, width: float, floor: float = 11.0) -> float:
        """Shrink a single-line text (the name) until it fits the width."""
        while size > floor and self.measure(text, bold, size) > width:
            size -= 0.5
        return size

    def headline_size(self, headline: str, body: float) -> float:
        """The title line's size: HEADLINE_EXTRA above the body text, less only as far as needed to keep it on one
        line."""
        return self.fit_size(headline, True, body + HEADLINE_EXTRA, self.width, floor=body)

    def wrap(self, runs: list[Run], width: float, size: float) -> list[Line]:
        key = (tuple((run.text, run.bold, run.italic) for run in runs), round(width, 3), size)
        if key not in self._wraps:
            self._wraps[key] = self._wrap(runs, width, size)
        return self._wraps[key]

    def _wrap(self, runs: list[Run], width: float, size: float) -> list[Line]:
        tokens: list[tuple[str, bool, bool, bool]] = []
        for run in runs:
            for part in re.split(r"(\s+)", run.text):
                if part:
                    tokens.append((" " if part.isspace() else part, run.bold, run.italic, part.isspace()))
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

        for text, bold, italic, is_space in tokens:
            w = self.measure(text, bold, size, italic)
            if is_space:
                if current:  # no leading spaces on a line
                    current.append(Piece(text, bold, w, italic))
                    used += w
                continue
            if used + w <= width + 1e-6:
                current.append(Piece(text, bold, w, italic))
                used += w
                continue
            flush()
            if w <= width:
                current.append(Piece(text, bold, w, italic))
                used = w
                continue
            # one unbreakable token wider than the line (a long URL): split it by characters
            chunk = ""
            for ch in text:
                if chunk and self.measure(chunk + ch, bold, size, italic) > width:
                    lines.append([Piece(chunk, bold, self.measure(chunk, bold, size, italic), italic)])
                    chunk = ch
                else:
                    chunk += ch
            if chunk:
                used = self.measure(chunk, bold, size, italic)
                current = [Piece(chunk, bold, used, italic)]
        flush()
        return lines

    def para_height(self, runs: list[Run], size: float | None = None, x: float = 0.0) -> float:
        size = size or self.size
        return len(self.wrap(runs, self.width - x, size)) * size * self.line

    def bullet_height(self, text: str, after: float = 1.0) -> float:
        return len(self.wrap([Run(text)], self.width - TEXT_X, self.size)) * self.size * self.line + after

    def row_height(self, left: list[Run], right: str) -> float:
        right_w = self.measure(right, False, self.size) if right else 0.0
        lines = self.wrap(left, self.width - (right_w + ROW_GAP if right else 0.0), self.size)
        return max(len(lines), 1) * self.size * self.line

    # ------------------------------------------------------------------ drawing
    def _ensure(self, height: float) -> None:
        if self.y + height > self.bottom + 0.01:
            self.pdf.add_page()
            self.y = self.margin

    def keep(self, height: float) -> None:
        """Start a new page unless the next `height` points fit on this one (never more than a page body, so a
        block taller than a page cannot cause an empty page)."""
        self._ensure(min(height, self.bottom - self.margin))

    def _draw(self, line: Line, x: float, y: float, size: float, height: float) -> None:
        merged: list[list] = []  # neighbours in the same style become one text string
        for piece in line:
            style = font_style(piece.bold, piece.italic)
            if merged and merged[-1][1] == style:
                merged[-1][0] += piece.text
                merged[-1][2] += piece.width
            else:
                merged.append([piece.text, style, piece.width])
        for text, style, width in merged:
            self.pdf.set_font(FAMILY, style, size)
            self.pdf.set_xy(x, y)
            self.pdf.cell(w=width, h=height, text=text, new_x=XPos.RIGHT, new_y=YPos.TOP)
            x += width

    def space(self, points: float) -> None:
        self.y += points

    def para(self, runs: list[Run], size: float | None = None, after: float = 0.0, x: float = 0.0, align: str = "L") -> None:
        size = size or self.size
        height = size * self.line
        width = self.width - x
        for line in self.wrap(runs, width, size):
            self._ensure(height)
            total = sum(p.width for p in line)
            left = self.margin + x + ((width - total) / 2 if align == "C" else 0.0)
            self._draw(line, left, self.y, size, height)
            self.y += height
        self.y += after

    def bullet(self, text: str, after: float = 1.0) -> None:
        """A whole bullet on one page (the gap after it is space, not content, so it is not reserved)."""
        height = self.size * self.line
        lines = self.wrap([Run(text)], self.width - TEXT_X, self.size)
        self.keep(len(lines) * height)
        for index, line in enumerate(lines):
            self._ensure(height)
            if index == 0:
                self._draw([Piece("•", False, self.measure("•", False, self.size))], self.margin + BULLET_X, self.y, self.size, height)
            self._draw(line, self.margin + TEXT_X, self.y, self.size, height)
            self.y += height
        self.y += after

    def row(self, left: list[Run], right: str) -> None:
        """Bold-led text on the left, dates right-aligned on the first line.

        The gap before the dates is filled with real spaces. They are invisible, but without them PDF
        readers that group text by position (pdfminer, and ATS built like it) see the dates as a separate
        block and put them inside the first bullet."""
        height = self.size * self.line
        right_w = self.measure(right, False, self.size) if right else 0.0
        lines = self.wrap(left, self.width - (right_w + ROW_GAP if right else 0.0), self.size)
        for index, line in enumerate(lines or [[]]):
            self._ensure(height)
            self._draw(line, self.margin, self.y, self.size, height)
            if index == 0 and right:
                used = sum(piece.width for piece in line)
                space_w = self.measure(" ", False, self.size)
                count = int((self.width - right_w - used) // space_w)
                if count > 0:
                    self._draw([Piece(" " * count, False, count * space_w)], self.margin + used, self.y, self.size, height)
                self._draw([Piece(right, False, right_w)], self.margin + self.width - right_w, self.y, self.size, height)
            self.y += height

    def rule(self, before: float = 0.0, after: float = 0.0, width: float = 0.75) -> None:
        """A full-width line, as under the resume's section headings."""
        self.y += before
        self.pdf.set_line_width(width)
        self.pdf.line(self.margin, self.y + width / 2, self.margin + self.width, self.y + width / 2)
        self.y += width + after

    def heading(self, text: str, keep: float = 0.0) -> None:
        """A section heading, kept on the page of its first entry (`keep` is that entry's height)."""
        size = self.size + 0.5
        height = size * self.line
        self.y += 8
        self.keep(height + 5 + keep)
        self._draw([Piece(text, True, self.measure(text, True, size))], self.margin, self.y, size, height)
        self.y += height + 1
        self.pdf.set_line_width(0.75)
        self.pdf.line(self.margin, self.y + 0.375, self.margin + self.width, self.y + 0.375)
        self.y += 0.75 + 3

    def last_page_fill(self) -> float:
        """How full the last page is, 0..1: a second page holding a few lines is worth avoiding."""
        return max(0.0, min(1.0, (self.y - self.margin) / (self.bottom - self.margin)))

    def finish(self) -> tuple[bytes, int]:
        return bytes(self.pdf.output()), self.pdf.page


# ---------------------------------------------------------------------- resume
def render_resume_pdf(resume: Resume, style: FitStyle, document: str = "Resume") -> tuple[bytes, int]:
    """Return (pdf bytes, page count). `document` names it in the file's title: "Resume", or "CV" in EU style."""
    return resume_sheet(resume, style, document).finish()


def resume_layout(resume: Resume, style: FitStyle) -> tuple[int, float]:
    """(page count, how full the last page is), without building the PDF file."""
    sheet = resume_sheet(resume, style)
    return sheet.pdf.page, sheet.last_page_fill()


def headline_size(headline: str, body: float, paper: str, margin: float) -> float:
    """The PDF's title-line size, for the Word file to use the same one."""
    return Sheet(paper, margin, body, RESUME_LINE, "", "").headline_size(headline, body)


def job_header(company: Company, title: str, stacked: bool) -> list[tuple[list[Run], str]]:
    """The rows that open a job, as (left text, right-aligned dates). The company comes first, where a recruiter's
    eye runs down the page. Stacked: the company and the dates, then the title on its own line; otherwise all of it
    on one line, which takes no more room than a single row."""
    place = "".join(f" | {x}" for x in (company.location, company.mode) if x)
    dates = fmt_dates(company.start, company.end)
    if stacked:
        return [([Run(company.name, True), Run(place)], dates), ([Run(title, italic=True)], "")]
    return [([Run(company.name, True), Run(" | "), Run(title, italic=True), Run(place)], dates)]


def certification_runs(cert: Certification) -> list[Run]:
    """A certification's line, like an education line: the name in bold, then the issuer."""
    return [Run(cert.name, True)] + ([Run(f" | {cert.issuer}")] if cert.issuer else [])


def certification_dates(cert: Certification) -> str:
    """Its dates on the right, like a job's: 'Mar 2023 - Mar 2026' (earned - expires), 'Feb 2020' when it does not
    expire, 'Expires Mar 2026' when the date it was earned is not known."""
    if cert.year and cert.expires:
        return fmt_dates(cert.year, cert.expires)
    return cert.year or (f"Expires {cert.expires}" if cert.expires else "")


def resume_sheet(resume: Resume, style: FitStyle, document: str = "Resume") -> Sheet:
    """The resume laid out on pages: `pdf.page` pages, `last_page_fill()`, and the file from `finish()`."""
    p = resume.profile
    sheet = Sheet(style.paper, style.margin_in * 72, style.font_pt, RESUME_LINE, f"{p.name} - {document}", p.name,
                  "en-GB" if document == "CV" else "en-US")
    size = sheet.size

    # Header: name, then the headline, then the contact line.
    sheet.para([Run(p.name, True)], size=sheet.fit_size(p.name, True, size + 9, sheet.width), align="C")
    sheet.space(HEADER_GAP)
    sheet.para([Run(resume.headline, True)], size=sheet.headline_size(resume.headline, size), align="C")
    line = contact_line(p.contact)
    if line:
        sheet.space(HEADER_GAP)
        sheet.para([Run(line)], align="C")

    sheet.heading("SUMMARY")
    sheet.para([Run(resume.summary)])

    sheet.heading("SKILLS")
    for group in resume.skills:
        sheet.para([Run(f"{group.category}:", True), Run(" " + ", ".join(group.skills))], after=1)

    jobs = []  # (header rows, bullets, height that must stay together: the header and the first bullet)
    for exp in resume.experience:
        rows = job_header(p.companies[exp.company_index], exp.job_title, style.stacked)
        first = sheet.bullet_height(exp.bullets[0], after=0.0) if exp.bullets else 0.0
        jobs.append((rows, exp.bullets, sum(sheet.row_height(left, right) for left, right in rows) + 1.5 + first))
    sheet.heading("EXPERIENCE", keep=jobs[0][2] if jobs else 0.0)
    for index, (rows, bullets, start) in enumerate(jobs):
        sheet.space(4 if index else 0)
        sheet.keep(start)  # a job title never sits alone at the bottom of a page
        for left, right in rows:
            sheet.row(left, right)
        sheet.space(1.5)
        for bullet in bullets:
            sheet.bullet(bullet)

    if p.education:
        entries = []
        for edu in p.education:
            main = ", ".join(x for x in (edu.degree, edu.field) if x)
            left = [Run(main, True), Run(f" | {edu.school}")] if main else [Run(edu.school, True)]
            dates, extra = fmt_dates(edu.start, edu.end), [Run(edu.details)] if edu.details else []
            entries.append((left, dates, extra, sheet.row_height(left, dates) + (sheet.para_height(extra) if extra else 0.0)))
        sheet.heading("EDUCATION", keep=entries[0][3])
        for left, dates, extra, height in entries:
            sheet.keep(height)
            sheet.row(left, dates)
            if extra:
                sheet.para(extra)

    if resume.certifications:  # only ones the person has, best for the job first
        rows = [(certification_runs(cert), certification_dates(cert)) for cert in resume.certifications]
        heights = [sheet.row_height(left, year) for left, year in rows]
        sheet.heading("CERTIFICATIONS", keep=heights[0])
        for (left, year), height in zip(rows, heights):
            sheet.keep(height)
            sheet.row(left, year)
    return sheet


# ---------------------------------------------------------------------- cover letter
# (body size, line height, space after a paragraph): the first that keeps the letter on one page is used
COVER_STYLES = [(11.0, 1.35, 9.0), (10.5, 1.3, 8.0), (10.0, 1.25, 7.0)]


def contact_lines(sheet: Sheet, contact: Contact, size: float) -> list[str]:
    """The contact line, split between items (never inside an email or link) when it is too wide for one line."""
    lines: list[str] = []
    for item in [part for part in contact_line(contact).split(" | ") if part]:
        joined = f"{lines[-1]} | {item}" if lines else item
        if lines and sheet.measure(joined, False, size) <= sheet.width:
            lines[-1] = joined
        else:
            lines.append(item)
    return lines


def render_cover_pdf(resume: Resume, letter: CoverLetter, paper: str = "Letter",
                     lang: str = "en-US") -> tuple[bytes, int]:
    """The letter under the same header as the resume (name, headline, contact line), so the two look like one set:
    a rule, then the letter from its greeting to the typed signature (no date, address or subject line). One column
    of real text in reading order, like the resume, so applicant systems read it too."""
    for size, line_height, gap in COVER_STYLES:
        data, pages = _cover(resume, letter, paper, size, line_height, gap, lang)
        if pages == 1:
            break
    return data, pages


def _cover(resume: Resume, letter: CoverLetter, paper: str, size: float, line_height: float,
           gap: float, lang: str = "en-US") -> tuple[bytes, int]:
    p = resume.profile
    sheet = Sheet(paper, 72, size, line_height, f"{p.name} - Cover Letter", p.name, lang)
    # Letterhead, as on the resume
    sheet.para([Run(p.name, True)], size=sheet.fit_size(p.name, True, 20, sheet.width), align="C")
    if resume.headline:
        sheet.space(HEADER_GAP)
        sheet.para([Run(resume.headline, True)], size=sheet.headline_size(resume.headline, 10.5), align="C")
    lines = contact_lines(sheet, p.contact, 10)
    if lines:
        sheet.space(HEADER_GAP if resume.headline else 2)
    for line in lines:
        sheet.para([Run(line)], size=10, align="C")
    sheet.rule(before=8, after=22)

    sheet.para([Run(letter.greeting)], after=gap)
    for paragraph in letter.paragraphs:
        sheet.para([Run(paragraph)], after=gap)
    sheet.para([Run(letter.closing)], after=gap + 18)  # room for a handwritten signature on a printout
    sheet.para([Run(person_name(p.name))])
    return sheet.finish()
