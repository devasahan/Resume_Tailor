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
TECH_LABEL = "Tech:"  # the line under a job's bullets: "Tech: Python, FastAPI, PostgreSQL"
TECH_GAP = 1.5  # points between a job's last bullet and its tech line


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


def letter_date(today: date | None = None) -> str:
    today = today or date.today()
    return f"{today:%B} {today.day}, {today.year}"


def recipient_lines(letter: CoverLetter) -> list[str]:
    """Who the letter is for: the hiring manager's name, or 'Hiring Manager' when only the company is known."""
    manager, company = letter.hiring_manager.strip(), letter.company_name.strip()
    if not manager and not company:
        return []
    return [manager or "Hiring Manager", *([company] if company else [])]


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
        self._wraps: dict[tuple, list[Line]] = {}  # the keep rules measure a block before drawing it

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
        key = (tuple((run.text, run.bold) for run in runs), round(width, 3), size)
        if key not in self._wraps:
            self._wraps[key] = self._wrap(runs, width, size)
        return self._wraps[key]

    def _wrap(self, runs: list[Run], width: float, size: float) -> list[Line]:
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

    def bullet(self, text: str, after: float = 1.0, keep: float = 0.0) -> None:
        """A whole bullet on one page; `keep` reserves room for what must follow it (a job's tech line)."""
        height = self.size * self.line
        lines = self.wrap([Run(text)], self.width - TEXT_X, self.size)
        self.keep(len(lines) * height + after + keep)
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
def tech_runs(tech: list[str]) -> list[Run]:
    """'Tech: Python, FastAPI, AWS (ECS, SQS)': the label bold, the tools in plain text, commas only (no pipes or
    dots, which an ATS could read as a new bullet). Nothing for a job without a tech line (older runs)."""
    return [Run(TECH_LABEL, True), Run(" " + ", ".join(tech))] if tech else []


def render_resume_pdf(resume: Resume, style: FitStyle) -> tuple[bytes, int]:
    """Return (pdf bytes, page count)."""
    return _resume_sheet(resume, style).finish()


def resume_layout(resume: Resume, style: FitStyle) -> tuple[int, float]:
    """(page count, how full the last page is), without building the PDF file."""
    sheet = _resume_sheet(resume, style)
    return sheet.pdf.page, sheet.last_page_fill()


def _resume_sheet(resume: Resume, style: FitStyle) -> Sheet:
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

    jobs = []  # (row text, dates, bullets, tech line, tech height, height that must stay with the row)
    for exp in resume.experience:
        company = p.companies[exp.company_index]
        details = " | ".join(x for x in (company.name, company.location, company.mode) if x)
        left, dates, tech = [Run(exp.job_title, True), Run(f" | {details}")], fmt_dates(company.start, company.end), tech_runs(exp.tech)
        tech_h = TECH_GAP + sheet.para_height(tech, x=TEXT_X) if tech else 0.0
        first = sheet.bullet_height(exp.bullets[0]) if exp.bullets else 0.0
        start = sheet.row_height(left, dates) + 1.5 + first + (tech_h if len(exp.bullets) <= 1 else 0.0)
        jobs.append((left, dates, exp.bullets, tech, tech_h, start))
    sheet.heading("EXPERIENCE", keep=jobs[0][5] if jobs else 0.0)
    for index, (left, dates, bullets, tech, tech_h, start) in enumerate(jobs):
        sheet.space(4 if index else 0)
        sheet.keep(start)  # a job title never sits alone at the bottom of a page
        sheet.row(left, dates)
        sheet.space(1.5)
        for number, bullet in enumerate(bullets):
            sheet.bullet(bullet, keep=tech_h if number == len(bullets) - 1 else 0.0)
        if tech:
            sheet.space(TECH_GAP)
            sheet.para(tech, x=TEXT_X)

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


def render_cover_pdf(resume: Resume, letter: CoverLetter, paper: str = "Letter", role: str = "") -> tuple[bytes, int]:
    """A business letter under the same header as the resume (name, headline, contact line), so the two look like
    one set: a rule, the date, the recipient, a subject line with the role, the letter, and the typed signature.
    One column of real text in reading order, like the resume, so applicant systems read it too."""
    for size, line_height, gap in COVER_STYLES:
        data, pages = _cover(resume, letter, paper, role, size, line_height, gap)
        if pages == 1:
            break
    return data, pages


def _cover(resume: Resume, letter: CoverLetter, paper: str, role: str, size: float, line_height: float,
           gap: float) -> tuple[bytes, int]:
    p = resume.profile
    sheet = Sheet(paper, 72, size, line_height, f"{p.name} - Cover Letter", p.name)
    # Letterhead, as on the resume
    sheet.para([Run(p.name, True)], size=sheet.fit_size(p.name, True, 20, sheet.width), align="C")
    if resume.headline:
        sheet.space(2)
        sheet.para([Run(resume.headline, True)], size=10.5, align="C")
    lines = contact_lines(sheet, p.contact, 10)
    if lines:
        sheet.space(2)
    for line in lines:
        sheet.para([Run(line)], size=10, align="C")
    sheet.rule(before=8, after=22)

    sheet.para([Run(letter_date())], after=gap + 5)
    recipient = recipient_lines(letter)
    for value in recipient:
        sheet.para([Run(value)])
    if recipient:
        sheet.space(gap + 5)
    if role.strip():
        sheet.para([Run("Re: ", True), Run(role.strip(), True)], after=gap + 5)
    sheet.para([Run(letter.greeting)], after=gap)
    for paragraph in letter.paragraphs:
        sheet.para([Run(paragraph)], after=gap)
    sheet.para([Run(letter.closing)], after=gap + 18)  # room for a handwritten signature on a printout
    sheet.para([Run(person_name(p.name))])
    return sheet.finish()
