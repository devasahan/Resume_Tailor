"""DOCX export (editable copy). Plain paragraphs, tab stops and the built-in bullet style: no tables or text boxes.
Word paginates the resume itself; keep-with-next and keep-together ask it to break pages the way the PDF does."""
from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from ..schemas import CoverLetter, FitStyle, Resume
from .pdf import (
    HEADER_GAP,
    Run,
    certification_dates,
    certification_runs,
    contact_line,
    fmt_dates,
    headline_size,
    job_header,
    person_name,
)

PAPER = {"Letter": (Inches(8.5), Inches(11)), "A4": (Inches(8.27), Inches(11.69))}


def _setup(style: FitStyle, margin_in: float, font_pt: float, *, author: str, title: str,
           lang: str = "en-US") -> Document:
    doc = Document()
    # python-docx's template says "python-docx" and a 2013 date; use the candidate's own details instead
    props, now = doc.core_properties, datetime.now(UTC).replace(microsecond=0)
    props.author = props.last_modified_by = author
    props.title, props.comments, props.language = title, "", lang
    props.created = props.modified = now
    # The text's own language, which Word proofs it in (en-GB: 'optimise' is right); the template says en-US
    defaults = doc.styles.element.find(f"{qn('w:docDefaults')}/{qn('w:rPrDefault')}/{qn('w:rPr')}")
    if defaults is not None:
        language = defaults.find(qn("w:lang"))
        if language is None:
            language = OxmlElement("w:lang")
            defaults.append(language)
        language.set(qn("w:val"), lang)
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
    p.paragraph_format.keep_with_next = True  # never alone at the bottom of a page
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(font_pt + 0.5)
    _bottom_border(p)


def _row(doc: Document, left: list[Run], right: str, text_width, space_before: float = 0):
    """The left text (in the runs' styles) with the right-aligned dates on a tab stop."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.keep_together = True
    p.paragraph_format.tab_stops.add_tab_stop(text_width, WD_TAB_ALIGNMENT.RIGHT)
    for part in left:
        if part.text:
            run = p.add_run(part.text)
            run.bold, run.italic = part.bold or None, part.italic or None
    if right:
        p.add_run("\t" + right)
    return p


def resume_docx(resume: Resume, style: FitStyle, document: str = "Resume") -> bytes:
    p = resume.profile
    doc = _setup(style, style.margin_in, style.font_pt, author=p.name, title=f"{p.name} - {document}",
                 lang="en-GB" if document == "CV" else "en-US")
    text_width = PAPER[style.paper][0] - Inches(style.margin_in * 2)
    normal = doc.styles["Normal"].paragraph_format
    normal.line_spacing = 1.04  # the PDF's line height (1.27 x the font size), so Word keeps the same page count
    normal.widow_control = True

    name = doc.add_paragraph()
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = name.add_run(p.name)
    run.bold = True
    run.font.size = Pt(style.font_pt + 9)
    # Header: name, then the headline, then the contact line.
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_before = Pt(HEADER_GAP)
    run = h.add_run(resume.headline)
    run.bold = True
    run.font.size = Pt(headline_size(resume.headline, style.font_pt, style.paper, style.margin_in * 72))
    line = contact_line(p.contact)
    if line:
        c = doc.add_paragraph(line)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_before = Pt(HEADER_GAP)

    _heading(doc, "SUMMARY", style.font_pt)
    doc.add_paragraph(resume.summary)

    _heading(doc, "SKILLS", style.font_pt)
    for group in resume.skills:
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(1)
        para.add_run(f"{group.category}: ").bold = True
        para.add_run(", ".join(group.skills))

    _heading(doc, "EXPERIENCE", style.font_pt)
    for index, exp in enumerate(resume.experience):
        for number, (left, right) in enumerate(job_header(p.companies[exp.company_index], exp.job_title, style.stacked)):
            row = _row(doc, left, right, text_width, space_before=4 if index and not number else 0)
            row.paragraph_format.keep_with_next = True  # a job's header stays with its first bullet
        for bullet in exp.bullets:
            para = doc.add_paragraph(bullet, style="List Bullet")
            para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.keep_together = True

    if p.education:
        _heading(doc, "EDUCATION", style.font_pt)
        for edu in p.education:
            main = ", ".join(x for x in (edu.degree, edu.field) if x)
            left = [Run(main, True), Run(f" | {edu.school}")] if main else [Run(edu.school, True)]
            row = _row(doc, left, fmt_dates(edu.start, edu.end), text_width)
            row.paragraph_format.keep_with_next = bool(edu.details)
            if edu.details:
                doc.add_paragraph(edu.details)

    if resume.certifications:  # only ones the person has, best for the job first
        _heading(doc, "CERTIFICATIONS", style.font_pt)
        for cert in resume.certifications:
            _row(doc, certification_runs(cert), certification_dates(cert), text_width)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def cover_docx(resume: Resume, letter: CoverLetter, paper: str = "Letter", lang: str = "en-US") -> bytes:
    """The same letter as the PDF: the resume's header, a rule, then the letter from its greeting to the signature."""
    p = resume.profile
    style = FitStyle(font_pt=11, margin_in=1.0, paper=paper)  # type: ignore[arg-type]
    doc = _setup(style, 1.0, 11, author=p.name, title=f"{p.name} - Cover Letter", lang=lang)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.15

    def centered(text: str, size: float, bold: bool, before: float = 0.0):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(before)
        run = para.add_run(text)
        run.bold, run.font.size = bold, Pt(size)
        return para

    last = centered(p.name, 20, True)
    if resume.headline:
        last = centered(resume.headline, headline_size(resume.headline, 10.5, paper, 72), True, before=HEADER_GAP)
    line = contact_line(p.contact)
    if line:
        last = centered(line, 10, False, before=HEADER_GAP if resume.headline else 2)
    last.paragraph_format.space_after = Pt(4)
    _bottom_border(last)

    def para(text: str, after: float = 0.0, before: float = 0.0, bold: bool = False):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(before)
        paragraph.paragraph_format.space_after = Pt(after)
        paragraph.add_run(text).bold = bold
        return paragraph

    para(letter.greeting, after=9, before=22)
    for text in letter.paragraphs:
        para(text, after=9)
    para(letter.closing, after=27)  # room for a handwritten signature on a printout
    para(person_name(p.name))
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
