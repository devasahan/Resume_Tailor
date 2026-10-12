"""Resume and cover letter PDFs, drawn directly with fpdf2.

Pure Python: no system libraries (no Pango/Cairo/GTK), so the file is the same on Windows, macOS and
Linux. One bundled font (Carlito, metrically compatible with Calibri), single column, real text, no
tables, images or icons: the layout ATS parsers read best.

The code is a small typesetter: measure text, wrap it into lines, draw the lines top to bottom.
Drawing in reading order also gives a clean text layer for ATS extraction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from ..schemas import Contact, CoverLetter, FitStyle, Resume

FONT_DIR = Path(__file__).parent / "fonts"
FAMILY = "Carlito"

RESUME_LINE = 1.27  # line height as a multiple of the font size
BULLET_X = 4.0  # points from the left margin to the bullet glyph
TEXT_X = 13.0  # points from the left margin to bullet text (hanging indent)
ROW_GAP = 10.0  # minimum space between the left text and the right-aligned dates


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
    parts = [contact.email, contact.phone, display_url(contact.linkedin), contact.location, display_url(contact.website)]
    return " | ".join(part for part in parts if part)


def fmt_dates(start: str, end: str) -> str:
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


@dataclass
class Run:
    text: str
    bold: bool = False


@dataclass
class Piece:
    text: str
    bold: bool
    width: float


Line = list[Piece]


class Sheet:
    """A PDF being laid out. `y` is the current top of the next line, in points from the page top."""

    def __init__(self, paper: str, margin: float, size: float, line: float, title: str, author: str):
        pdf = FPDF(orientation="P", unit="pt", format=paper)
        pdf.set_auto_page_break(False)  # pagination is handled here so a line is never split in two
        pdf.set_margins(margin, margin, margin)
        pdf.c_margin = 0  # no inner padding: widths measured here are the widths drawn
        pdf.add_font(FAMILY, "", str(FONT_DIR / "Carlito-Regular.ttf"))
        pdf.add_font(FAMILY, "B", str(FONT_DIR / "Carlito-Bold.ttf"))
        pdf.set_title(title)
        pdf.set_author(author)
        pdf.set_creator("Resume Tailor")
        try:
            pdf.set_lang("en-US")
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

    # ------------------------------------------------------------------ measuring and wrapping
    def measure(self, text: str, bold: bool, size: float) -> float:
        self.pdf.set_font(FAMILY, "B" if bold else "", size)
        return self.pdf.get_string_width(text)

    def fit_size(self, text: str, bold: bool, size: float, width: float, floor: float = 11.0) -> float:
        """Shrink a single-line text (the name) until it fits the width."""
        while size > floor and self.measure(text, bold, size) > width:
            size -= 0.5
        return size

    def wrap(self, runs: list[Run], width: float, size: float) -> list[Line]:
        tokens: list[tuple[str, bool, bool]] = []
        for run in runs:
            for part in re.split(r"(\s+)", run.text):
                if part:
                    tokens.append((" " if part.isspace() else part, run.bold, part.isspace()))
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

        for text, bold, is_space in tokens:
            w = self.measure(text, bold, size)
            if is_space:
                if current:  # no leading spaces on a line
                    current.append(Piece(text, bold, w))
                    used += w
                continue
            if used + w <= width + 1e-6:
                current.append(Piece(text, bold, w))
                used += w
                continue
            flush()
            if w <= width:
                current.append(Piece(text, bold, w))
                used = w
                continue
            # one unbreakable token wider than the line (a long URL): split it by characters
            chunk = ""
            for ch in text:
                if chunk and self.measure(chunk + ch, bold, size) > width:
                    lines.append([Piece(chunk, bold, self.measure(chunk, bold, size))])
                    chunk = ch
                else:
                    chunk += ch
            if chunk:
                current, used = [Piece(chunk, bold, self.measure(chunk, bold, size))], self.measure(chunk, bold, size)
        flush()
        return lines

    # ------------------------------------------------------------------ drawing
    def _ensure(self, height: float) -> None:
        if self.y + height > self.bottom + 0.01:
            self.pdf.add_page()
            self.y = self.margin

    def _draw(self, line: Line, x: float, y: float, size: float, height: float) -> None:
        merged: list[list] = []  # neighbours in the same style become one text string
        for piece in line:
            if merged and merged[-1][1] == piece.bold:
                merged[-1][0] += piece.text
                merged[-1][2] += piece.width
            else:
                merged.append([piece.text, piece.bold, piece.width])
        for text, bold, width in merged:
            self.pdf.set_font(FAMILY, "B" if bold else "", size)
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
        height = self.size * self.line
        lines = self.wrap([Run(text)], self.width - TEXT_X, self.size)
        for index, line in enumerate(lines):
            self._ensure(height)
            if index == 0:
                self._draw([Piece("•", False, self.measure("•", False, self.size))], self.margin + BULLET_X, self.y, self.size, height)
            self._draw(line, self.margin + TEXT_X, self.y, self.size, height)
            self.y += height
        self.y += after

    def row(self, left: list[Run], right: str) -> None:
        """Bold-led text on the left, dates right-aligned on the first line."""
        height = self.size * self.line
        right_w = self.measure(right, False, self.size) if right else 0.0
        lines = self.wrap(left, self.width - (right_w + ROW_GAP if right else 0.0), self.size)
        for index, line in enumerate(lines or [[]]):
            self._ensure(height)
            self._draw(line, self.margin, self.y, self.size, height)
            if index == 0 and right:
                self._draw([Piece(right, False, right_w)], self.margin + self.width - right_w, self.y, self.size, height)
            self.y += height

    def heading(self, text: str) -> None:
        size = self.size + 0.5
        height = size * self.line
        self.y += 8
        self._ensure(height + 5)
        self._draw([Piece(text, True, self.measure(text, True, size))], self.margin, self.y, size, height)
        self.y += height + 1
        self.pdf.set_line_width(0.75)
        self.pdf.line(self.margin, self.y + 0.375, self.margin + self.width, self.y + 0.375)
        self.y += 0.75 + 3

    def finish(self) -> tuple[bytes, int]:
        return bytes(self.pdf.output()), self.pdf.page


# ---------------------------------------------------------------------- resume
def render_resume_pdf(resume: Resume, style: FitStyle) -> tuple[bytes, int]:
    """Return (pdf bytes, page count)."""
    p = resume.profile
    sheet = Sheet(style.paper, style.margin_in * 72, style.font_pt, RESUME_LINE, f"{p.name} - Resume", p.name)
    size = sheet.size

    # Header: name, then the headline, then the contact line.
    sheet.para([Run(p.name, True)], size=sheet.fit_size(p.name, True, size + 9, sheet.width), align="C")
    sheet.space(2)
    sheet.para([Run(resume.headline, True)], align="C")
    line = contact_line(p.contact)
    if line:
        sheet.space(2)
        sheet.para([Run(line)], align="C")

    sheet.heading("SUMMARY")
    sheet.para([Run(resume.summary)])

    sheet.heading("SKILLS")
    for group in resume.skills:
        sheet.para([Run(f"{group.category}:", True), Run(" " + ", ".join(group.skills))], after=1)

    sheet.heading("EXPERIENCE")
    for index, exp in enumerate(resume.experience):
        company = p.companies[exp.company_index]
        details = " | ".join(x for x in (company.name, company.location, company.mode) if x)
        sheet.space(4 if index else 0)
        sheet.row([Run(exp.job_title, True), Run(f" | {details}")], fmt_dates(company.start, company.end))
        sheet.space(1.5)
        for bullet in exp.bullets:
            sheet.bullet(bullet)

    if p.education:
        sheet.heading("EDUCATION")
        for edu in p.education:
            main = ", ".join(x for x in (edu.degree, edu.field) if x)
            left = [Run(main, True), Run(f" | {edu.school}")] if main else [Run(edu.school, True)]
            sheet.row(left, fmt_dates(edu.start, edu.end))
            if edu.details:
                sheet.para([Run(edu.details)])
    return sheet.finish()


# ---------------------------------------------------------------------- cover letter
def render_cover_pdf(resume: Resume, letter: CoverLetter, paper: str = "Letter") -> tuple[bytes, int]:
    p = resume.profile
    sheet = Sheet(paper, 72, 11, 1.35, f"{p.name} - Cover Letter", p.name)
    sheet.para([Run(p.name, True)], size=sheet.fit_size(p.name, True, 16, sheet.width))
    line = contact_line(p.contact)
    if line:
        sheet.space(2)
        sheet.para([Run(line)], after=14)
    else:
        sheet.space(14)
    today = date.today()
    sheet.para([Run(f"{today:%B} {today.day}, {today.year}")], after=9)
    recipient = [x for x in (letter.hiring_manager, letter.company_name) if x]
    for value in recipient:
        sheet.para([Run(value)])
    if recipient:
        sheet.space(14)
    sheet.para([Run(letter.greeting)], after=9)
    for paragraph in letter.paragraphs:
        sheet.para([Run(paragraph)], after=9)
    sheet.para([Run(letter.closing)], after=9)
    sheet.para([Run(p.name)])
    return sheet.finish()
