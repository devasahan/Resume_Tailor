"""DOCX export (editable copy), in the PDF's look: Inter (Word shows Arial where Inter is not installed), the same
sizes, colours, gaps and order, from the same helpers. Plain paragraphs, tab stops and the built-in bullet style: no
tables or text boxes. Word paginates the resume itself; keep-with-next and keep-together ask it to break pages the way
the PDF does. SemiBold and Bold text is bold here, Medium text regular: Word knows bold or not."""
from __future__ import annotations

import math
from datetime import UTC, datetime
from io import BytesIO

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

from ..schemas import Contact, CoverLetter, FitStyle, Resume
from .pdf import (
    ACCENT,
    BOLD,
    BULLET_GAP,
    BULLETS_GAP,
    CONTACT_SIZE,
    COVER_STYLES,
    DATE_GAP,
    DATE_SIZE,
    DESIGN_SIZE,
    ENTRY_GAP,
    HEADER_GAP,
    HEADING_GAP,
    HEADING_SIZE,
    HEADING_TRACKING,
    INK,
    JOB_GAP,
    JOB_SIZE,
    LINE,
    MARKER,
    MARKER_END,
    MEDIUM,
    MUTED,
    NAME_LINE,
    NAME_SIZE,
    NAME_TRACKING,
    REGULAR,
    RULE,
    RULE_GAP,
    SECTION_GAP,
    SEMIBOLD,
    SKILLS_ROW_GAP,
    TEXT,
    TEXT_INDENT,
    TITLE_LINE,
    TOP_MARGIN,
    Run,
    certification_dates,
    certification_runs,
    contact_lines,
    education_runs,
    fit_size,
    fmt_dates,
    headline_size,
    person_name,
    skills_column,
    split_job_line,
    text_width,
)

PAPER = {"Letter": (Inches(8.5), Inches(11)), "A4": (Inches(8.27), Inches(11.69))}
FONT = "Inter"
FALLBACK_FONT = "Arial"  # named in the font table: what Word uses where Inter is not installed
WORD_LINE = (1984 + 494) / 2048  # Word's single line for Inter (its ascender and descender), as a share of the size
# the run properties that follow w:spacing, in the order the schema keeps them
_AFTER_SPACING = ("w:w", "w:kern", "w:position", "w:sz", "w:szCs", "w:highlight", "w:u", "w:effect", "w:bdr", "w:shd",
                  "w:fitText", "w:vertAlign", "w:rtl", "w:cs", "w:em", "w:lang", "w:eastAsianLayout", "w:specVanish",
                  "w:oMath")


def _size(points: float) -> Pt:
    """A font size as Word keeps it, in whole half points, rounded down (as python-docx stored sizes before): Word's
    text is then never wider than the text measured at `points`, so a contact line or a job line broken to fit on the
    PDF stays on its lines in Word too."""
    return Pt(math.floor(points * 2 + 1e-9) / 2)


def _rgb(color: str) -> RGBColor:
    return RGBColor.from_string(color.lstrip("#").upper())


def _add_run(paragraph, run: Run, size: float):
    """`run` in its weight, colour and letter spacing, at `size` points."""
    added = paragraph.add_run(run.text)
    added.bold = run.bold or None  # SemiBold and Bold read as bold; Medium and Regular as regular
    added.font.size = _size(size)
    added.font.color.rgb = _rgb(run.color)
    if run.tracking:  # w:spacing, in twentieths of a point
        spacing = OxmlElement("w:spacing")
        spacing.set(qn("w:val"), str(round(run.tracking * size * 20)))
        added._r.get_or_add_rPr().insert_element_before(spacing, *_AFTER_SPACING)
    return added


def _fallback_font(doc) -> None:
    """Name Arial as Inter's alternative in the font table, so Word shows a similar sans serif where Inter is not
    installed. Only a hint: a template without a font table is left as it is."""
    try:
        part = doc.part.part_related_by(RELATIONSHIP_TYPE.FONT_TABLE)
    except KeyError:
        return
    fonts = parse_xml(part.blob)
    font = OxmlElement("w:font")
    font.set(qn("w:name"), FONT)
    for tag, value in (("w:altName", FALLBACK_FONT), ("w:family", "swiss"), ("w:pitch", "variable")):
        child = OxmlElement(tag)
        child.set(qn("w:val"), value)
        font.append(child)
    fonts.append(font)
    part._blob = etree.tostring(fonts, xml_declaration=True, encoding="UTF-8", standalone=True)  # kept as bytes


def _setup(style: FitStyle, margin_in: float, font_pt: float, *, author: str, title: str,
           lang: str = "en-US", top_in: float | None = None) -> Document:
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
    top = TOP_MARGIN * margin_in if top_in is None else top_in
    section.left_margin = section.right_margin = Inches(margin_in)
    section.top_margin = section.bottom_margin = Inches(top)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    fonts = normal.element.rPr.rFonts
    for key in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(key), FONT)
    normal.font.size = _size(font_pt)
    normal.font.color.rgb = _rgb(TEXT)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.line_spacing = LINE / WORD_LINE  # the PDF's line height (1.42 x the size)
    _fallback_font(doc)
    return doc


def _text_width(style: FitStyle, margin_in: float) -> float:
    """The width between the side margins, in points."""
    return PAPER[style.paper][0].pt - 2 * 72 * margin_in


def _bottom_border(paragraph, space: float) -> None:
    """A thin gray rule under the paragraph, `space` points below its text."""
    pPr = paragraph._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for key, value in (("w:val", "single"), ("w:sz", "6"), ("w:space", str(min(31, round(space)))),
                       ("w:color", RULE.lstrip("#"))):
        bottom.set(qn(key), value)
    border.append(bottom)
    pPr.insert_element_before(border, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
                              "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi",
                              "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
                              "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
                              "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
                              "w:pPrChange")


def _header(doc, name: str, headline: str, contact: Contact, body: float, width: float):
    """The name, the title line and the contact line, centred, as on the PDF (draw_header); returns the last of them."""
    def centred(runs: list[Run], size: float, line: float, before: float):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(before)
        paragraph.paragraph_format.line_spacing = line / WORD_LINE
        for index, run in enumerate(runs):
            if index:
                paragraph.add_run().add_break()
            _add_run(paragraph, run, size)
        return paragraph

    size = fit_size(name, BOLD, NAME_SIZE * body, width, tracking=NAME_TRACKING)
    last = centred([Run(name, weight=BOLD, color=INK, tracking=NAME_TRACKING)], size, NAME_LINE, 0.0)
    if headline:
        last = centred([Run(headline, weight=MEDIUM, color=ACCENT)], headline_size(headline, body, width), TITLE_LINE,
                       HEADER_GAP * body)
    lines = contact_lines(contact, CONTACT_SIZE * body, width)  # broken between items, as on the PDF
    if lines:
        last = centred([Run(line, color=MUTED) for line in lines], CONTACT_SIZE * body, LINE, HEADER_GAP * body)
    return last


def _heading(doc, text: str, body: float, width: float) -> None:
    """A section heading as on the PDF: small spaced capitals in the accent colour, then a thin gray rule to the right
    margin, drawn as the leader of a right tab. Kept with the paragraph after it."""
    paragraph = doc.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.space_before, fmt.space_after = Pt(SECTION_GAP * body), Pt(HEADING_GAP * body)
    fmt.keep_with_next = True  # never alone at the bottom of a page
    fmt.tab_stops.add_tab_stop(Pt(width), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.LINES)
    size = HEADING_SIZE * body
    _add_run(paragraph, Run(text.upper(), weight=BOLD, color=ACCENT, tracking=HEADING_TRACKING), size)
    gap = RULE_GAP * body - text_width(" ", REGULAR, size)  # a space widened to the design's gap before the rule
    _add_run(paragraph, Run(" ", color=RULE, tracking=gap / size), size)
    _add_run(paragraph, Run("\t", color=RULE), size)  # the leader takes the tab's colour


def _row(doc, left: list[Run], right: str, width: float, size: float, body: float, rest: list[Run] | None = None):
    """The left text (in the runs' styles) at `size`, the dates right-aligned on a tab stop in smaller gray type, and
    `rest` (a long job line's second part) on a line of its own under them."""
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.tab_stops.add_tab_stop(Pt(width), WD_TAB_ALIGNMENT.RIGHT)
    for run in left:
        if run.text:
            _add_run(paragraph, run, size)
    if right:
        _add_run(paragraph, Run("\t" + right, color=MUTED), DATE_SIZE * body)
    if rest:
        paragraph.add_run().add_break()
        for run in rest:
            _add_run(paragraph, run, size)
    return paragraph


def _skills(doc, resume: Resume, body: float, width: float) -> None:
    """'Label:' then a tab to the values' column, with a hanging indent so the values wrap inside it; labels too wide
    for the column go in line with their values, as on the PDF (skills_column)."""
    values_x = skills_column(resume.skills, body, width)
    for index, group in enumerate(resume.skills):
        paragraph = doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.space_before = Pt(SKILLS_ROW_GAP * body if index else 0.0)
        fmt.keep_together = True
        _add_run(paragraph, Run(f"{group.category}:", weight=SEMIBOLD, color=INK), body)
        if values_x is None:
            _add_run(paragraph, Run(" " + ", ".join(group.skills)), body)
            continue
        fmt.tab_stops.add_tab_stop(Pt(values_x))
        fmt.left_indent, fmt.first_line_indent = Pt(values_x), Pt(-values_x)
        _add_run(paragraph, Run("\t" + ", ".join(group.skills)), body)


def _bullet(doc, text: str, body: float, last: bool) -> None:
    """A bullet in the built-in List Bullet style, its marker gray and where the PDF draws it."""
    paragraph = doc.add_paragraph(style="List Bullet")
    _add_run(paragraph, Run(text), body)
    fmt = paragraph.paragraph_format
    text_x = TEXT_INDENT * body
    marker_x = MARKER_END * body - text_width("• ", REGULAR, body)
    fmt.left_indent, fmt.first_line_indent = Pt(text_x), Pt(marker_x - text_x)
    fmt.space_after = Pt(0.0 if last else BULLET_GAP * body)
    fmt.keep_together = True
    mark = OxmlElement("w:rPr")  # the paragraph mark's colour, which Word gives the bullet
    color = OxmlElement("w:color")
    color.set(qn("w:val"), MARKER.lstrip("#"))
    mark.append(color)
    paragraph._p.get_or_add_pPr().insert_element_before(mark, "w:sectPr", "w:pPrChange")


def resume_docx(resume: Resume, style: FitStyle, document: str = "Resume") -> bytes:
    p = resume.profile
    body = style.font_pt
    doc = _setup(style, style.margin_in, body, author=p.name, title=f"{p.name} - {document}",
                 lang="en-GB" if document == "CV" else "en-US")
    width = _text_width(style, style.margin_in)
    doc.styles["Normal"].paragraph_format.widow_control = True
    bullets = doc.styles["List Bullet"].element.pPr  # its contextual spacing would drop the gap between two bullets
    for flag in [] if bullets is None else bullets.findall(qn("w:contextualSpacing")):
        bullets.remove(flag)

    _header(doc, p.name, resume.headline, p.contact, body, width)

    _heading(doc, "SUMMARY", body, width)
    doc.add_paragraph().add_run(resume.summary)

    _heading(doc, "SKILLS", body, width)
    _skills(doc, resume, body, width)

    _heading(doc, "EXPERIENCE", body, width)
    for index, exp in enumerate(resume.experience):
        company = p.companies[exp.company_index]
        dates = fmt_dates(company.start, company.end)
        room = width - (text_width(dates, REGULAR, DATE_SIZE * body) + DATE_GAP * body if dates else 0.0)
        left, rest = split_job_line(company, exp.job_title, JOB_SIZE * body, room)
        row = _row(doc, left, dates, width, JOB_SIZE * body, body, rest)
        row.paragraph_format.space_before = Pt(JOB_GAP * body if index else 0.0)
        row.paragraph_format.space_after = Pt(BULLETS_GAP * body if exp.bullets else 0.0)
        row.paragraph_format.keep_with_next = True  # a job's line stays with its first bullet
        for number, bullet in enumerate(exp.bullets):
            _bullet(doc, bullet, body, last=number == len(exp.bullets) - 1)

    if p.education:
        _heading(doc, "EDUCATION", body, width)
        for index, edu in enumerate(p.education):
            row = _row(doc, education_runs(edu), fmt_dates(edu.start, edu.end), width, body, body)
            row.paragraph_format.space_before = Pt(ENTRY_GAP * body if index else 0.0)
            row.paragraph_format.keep_with_next = bool(edu.details)
            if edu.details:
                doc.add_paragraph().add_run(edu.details)

    if resume.certifications:  # only ones the person has, best for the job first
        _heading(doc, "CERTIFICATIONS", body, width)
        for index, cert in enumerate(resume.certifications):
            row = _row(doc, certification_runs(cert), certification_dates(cert), width, body, body)
            row.paragraph_format.space_before = Pt(ENTRY_GAP * body if index else 0.0)

    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def cover_docx(resume: Resume, letter: CoverLetter, paper: str = "Letter", lang: str = "en-US") -> bytes:
    """The same letter as the PDF: the resume's header (at its first size step), a thin gray rule, then the letter
    from its greeting to the signature."""
    p = resume.profile
    size, _line, gap = COVER_STYLES[0]
    style = FitStyle(font_pt=size, margin_in=1.0, paper=paper)  # type: ignore[arg-type]
    doc = _setup(style, 1.0, size, author=p.name, title=f"{p.name} - Cover Letter", lang=lang, top_in=1.0)
    last = _header(doc, p.name, resume.headline, p.contact, DESIGN_SIZE, _text_width(style, 1.0))
    _bottom_border(last, size)  # the rule under the header, 1 em below it

    def para(text: str, after: float = 0.0, before: float = 0.0):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(before)
        paragraph.paragraph_format.space_after = Pt(after)
        paragraph.add_run(text)
        return paragraph

    para(letter.greeting, after=gap * size, before=2 * size)
    for text in letter.paragraphs:
        para(text, after=gap * size)
    para(letter.closing, after=gap * size + 18)  # room for a handwritten signature on a printout
    para(person_name(p.name))
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
