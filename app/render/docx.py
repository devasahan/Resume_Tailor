"""DOCX export (editable copy). Plain paragraphs, tab stops and the built-in bullet style: no tables or text boxes."""
from __future__ import annotations

from datetime import date
from io import BytesIO

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from ..schemas import CoverLetter, FitStyle, Resume
from .pdf import contact_line, fmt_dates

PAPER = {"Letter": (Inches(8.5), Inches(11)), "A4": (Inches(8.27), Inches(11.69))}


def _setup(style: FitStyle, margin_in: float, font_pt: float) -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = PAPER[style.paper]
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(margin_in))
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.font.size = Pt(font_pt)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    return doc


def _bottom_border(paragraph) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for key, value in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "000000")):
        bottom.set(qn(key), value)
    border.append(bottom)
    pPr.append(border)


def _heading(doc: Document, text: str, font_pt: float) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(font_pt + 0.5)
    _bottom_border(p)


def _row(doc: Document, left_bold: str, left_rest: str, right: str, text_width) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.tab_stops.add_tab_stop(text_width, WD_TAB_ALIGNMENT.RIGHT)
    p.add_run(left_bold).bold = True
    if left_rest:
        p.add_run(left_rest)
    if right:
        p.add_run("\t" + right)


def resume_docx(resume: Resume, style: FitStyle) -> bytes:
    p = resume.profile
    doc = _setup(style, style.margin_in, style.font_pt)
    text_width = PAPER[style.paper][0] - Inches(style.margin_in * 2)

    name = doc.add_paragraph()
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = name.add_run(p.name)
    run.bold = True
    run.font.size = Pt(style.font_pt + 9)
    line = contact_line(p.contact)
    if line:
        c = doc.add_paragraph(line)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_before = Pt(4)
    h.add_run(resume.headline).bold = True

    _heading(doc, "SUMMARY", style.font_pt)
    doc.add_paragraph(resume.summary)

    _heading(doc, "SKILLS", style.font_pt)
    for group in resume.skills:
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(1)
        para.add_run(f"{group.category}: ").bold = True
        para.add_run(", ".join(group.skills))

    _heading(doc, "EXPERIENCE", style.font_pt)
    for exp in resume.experience:
        company = p.companies[exp.company_index]
        rest = f" | {company.name}"
        if company.location:
            rest += f" | {company.location}"
        if company.mode:
            rest += f" | {company.mode}"
        _row(doc, exp.job_title, rest, fmt_dates(company.start, company.end), text_width)
        for bullet in exp.bullets:
            para = doc.add_paragraph(bullet, style="List Bullet")
            para.paragraph_format.space_after = Pt(1)

    if p.education:
        _heading(doc, "EDUCATION", style.font_pt)
        for edu in p.education:
            main = ", ".join(x for x in (edu.degree, edu.field) if x)
            if main:
                _row(doc, main, f" | {edu.school}", fmt_dates(edu.start, edu.end), text_width)
            else:
                _row(doc, edu.school, "", fmt_dates(edu.start, edu.end), text_width)
            if edu.details:
                doc.add_paragraph(edu.details)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def cover_docx(resume: Resume, letter: CoverLetter, paper: str = "Letter") -> bytes:
    p = resume.profile
    style = FitStyle(font_pt=11, margin_in=1.0, paper=paper)  # type: ignore[arg-type]
    doc = _setup(style, 1.0, 11)
    name = doc.add_paragraph()
    run = name.add_run(p.name)
    run.bold = True
    run.font.size = Pt(16)
    line = contact_line(p.contact)
    if line:
        doc.add_paragraph(line).paragraph_format.space_after = Pt(14)
    today = date.today()
    doc.add_paragraph(f"{today:%B} {today.day}, {today.year}").paragraph_format.space_after = Pt(9)
    for value in (letter.hiring_manager, letter.company_name):
        if value:
            doc.add_paragraph(value)
    if letter.hiring_manager or letter.company_name:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
    for text in [letter.greeting, *letter.paragraphs, letter.closing, p.name]:
        para = doc.add_paragraph(text)
        para.paragraph_format.space_after = Pt(9)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
