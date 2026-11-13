import io
import logging
import random
import re
import unicodedata

import pytest
from docx import Document
from docx.oxml.ns import qn
from pypdf import PdfReader

from app.pipeline import validate
from app.pipeline.fit import LADDER, MAX_PAGES, styles, trim_to_fit, try_styles
from app.render import pdf as pdf_module
from app.render.docx import cover_docx, resume_docx
from app.render.pdf import (
    CONTACT_SIZE, DRAWABLE, DRAWABLE_LIMIT, REGULAR, SEMIBOLD, Run, Sheet, advance_widths, contact_line, fmt_dates,
    render_cover_pdf, render_resume_pdf, text_width,
)
from app.schemas import (
    Certification, Contact, CoverLetter, ExperienceOut, FitStyle, SkillGroup, TailoredSections, TailorOutput,
)

from .test_validate import make_output


def text_of(pdf: bytes) -> str:
    return "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)


@pytest.fixture
def resume(profile):
    return validate.build_resume(profile, make_output())


def test_one_page_with_text_in_reading_order(resume):
    pdf, pages = render_resume_pdf(resume, FitStyle())
    assert pages == 1
    text = text_of(pdf)
    order = ["Jordan Rivera", "SUMMARY", "SKILLS", "EXPERIENCE", "Northwind Labs", "Contoso Systems", "Fabrikam", "EDUCATION", "University of Texas at Austin"]
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)
    assert "Mar 2022 – Present" in text and "Remote" in text and "B.S., Computer Science" in text


def test_dates_stay_on_the_job_line_for_layout_readers(resume):
    """pdfminer groups text by position, like many ATS: right-aligned dates must not land inside a bullet."""
    pdfminer = pytest.importorskip("pdfminer.high_level")
    sentence = "Designed and operated data pipelines with Python and SQL for several product teams, documenting each decision."
    for exp in resume.experience:
        exp.bullets = [sentence, sentence]
    pdf, _ = render_resume_pdf(resume, FitStyle())
    lines = [" ".join(line.split()) for line in pdfminer.extract_text(io.BytesIO(pdf)).splitlines()]
    for name, dates in (("Northwind Labs", "Mar 2022 – Present"), ("Contoso Systems", "Jun 2019 – Feb 2022"),
                        ("Fabrikam", "Jul 2016 – May 2019"), ("University of Texas at Austin", "2016")):
        line = next(line for line in lines if name in line)
        assert line.endswith(dates), line
    assert text_of(pdf).count("Mar 2022 – Present") == 1  # the plain reader sees the dates once too


def test_metadata_and_embedded_font(resume):
    pdf, _ = render_resume_pdf(resume, FitStyle())
    reader = PdfReader(io.BytesIO(pdf))
    assert reader.metadata.title == "Jordan Rivera - Resume" and reader.metadata.author == "Jordan Rivera"
    assert reader.metadata.creator is None  # the file does not name the tool that made it
    fonts = reader.pages[0]["/Resources"]["/Font"]
    embedded = []
    for ref in fonts.values():
        font = ref.get_object()
        descendant = font["/DescendantFonts"][0].get_object()
        descriptor = descendant["/FontDescriptor"].get_object()
        file = descriptor.get("/FontFile2")
        truetype = file is not None and file.get_object().get_data()[:4] == b"\x00\x01\x00\x00"  # glyf outlines
        embedded.append(("Inter" in str(font["/BaseFont"]), truetype))
    assert embedded and all(is_inter and truetype for is_inter, truetype in embedded)


def test_docx_metadata_names_the_candidate_not_the_library(resume):
    props = Document(io.BytesIO(resume_docx(resume, FitStyle()))).core_properties
    assert props.author == props.last_modified_by == "Jordan Rivera" and props.title == "Jordan Rivera - Resume"
    assert "python-docx" not in props.comments and props.created.year >= 2026


@pytest.mark.parametrize("paper,size", [("Letter", (612, 792)), ("A4", (595, 842))])
def test_paper_sizes(resume, paper, size):
    pdf, pages = render_resume_pdf(resume, FitStyle(paper=paper))
    box = PdfReader(io.BytesIO(pdf)).pages[0].mediabox
    assert (round(float(box.width)), round(float(box.height))) == size and pages == 1


def long_resume(profile, bullets=12) -> "object":
    sentence = "Designed and operated reliable data pipelines with Python and SQL for several product teams, documenting every decision clearly."
    out = TailorOutput(
        jd_analysis=make_output().jd_analysis,
        sections=TailoredSections(
            headline="Senior Data Engineer | Python | SQL", summary="Senior Data Engineer. " * 5,
            skills=[SkillGroup(category="Core", skills=["Python", "SQL", "Airflow"])],
            experience=[
                ExperienceOut(company_index=i, job_title="Data Engineer", bullets=[f"{sentence} Item {j}." for j in range(bullets)])
                for i in range(len(profile.companies))
            ],
        ),
    )
    return validate.build_resume(profile, out)


def test_trimming_reports_a_few_removed_bullets_and_how_many_more(profile):
    _trimmed, _style, pages, _pdf, notes = trim_to_fit(long_resume(profile, bullets=32))
    note = next(n for n in notes if n.startswith("Removed"))
    removed = int(note.split()[1])
    assert pages <= MAX_PAGES and removed > 5 and note.count("...") == 5 and note.endswith(f"and {removed - 5} more")


def test_a_bullet_that_fits_at_the_bottom_of_a_page_stays_there():
    """The 1 pt gap after a bullet is space, not content: it never pushes a bullet that fits to the next page."""
    sheet = Sheet("Letter", 36, 10.5, 1.27, "t", "a")
    height = sheet.size * sheet.line
    sheet.y = sheet.bottom - height  # room for exactly one line
    sheet.bullet("A one-line bullet.")
    assert sheet.pdf.page == 1


def test_overflow_makes_more_pages_and_the_fit_steps_repair_it(profile):
    resume = long_resume(profile, bullets=30)
    _, pages = render_resume_pdf(resume, FitStyle())
    assert pages == 4
    assert try_styles(resume) is None  # shrinking alone cannot save this much text (the last step holds 28 a job)
    trimmed, style, pages, _pdf, notes = trim_to_fit(resume)
    assert pages <= MAX_PAGES and render_resume_pdf(trimmed, style)[1] == pages
    assert all(len(e.bullets) >= 2 for e in trimmed.experience)
    assert any("Removed" in n for n in notes)
    assert sum(len(e.bullets) for e in trimmed.experience) < sum(len(e.bullets) for e in resume.experience)


def test_moderate_overflow_is_fixed_by_shrinking_only(profile):
    resume = long_resume(profile, bullets=14)
    _, pages = render_resume_pdf(resume, FitStyle())
    found = try_styles(resume)
    assert pages == 3 and found is not None and found[1] == 2 and (found[0].font_pt, found[0].margin_in) == LADDER[2]
    assert any("Font reduced" in n and "two pages" in n for n in found[3])


def test_a_full_second_page_keeps_the_largest_font(profile):
    found = try_styles(long_resume(profile, bullets=10))  # page 2 well filled at the design's own sizes
    assert found[1] == 2 and (found[0].font_pt, found[0].margin_in) == LADDER[0] and found[3] == []


def test_a_nearly_empty_second_page_becomes_one_page_at_a_smaller_font(profile):
    resume = long_resume(profile, bullets=6)  # page 2 would hold a few lines at 10.5 pt
    assert render_resume_pdf(resume, FitStyle())[1] == 2
    found = try_styles(resume)
    assert found[1] == 1 and found[0].font_pt < 10.5 and any("one page" in n for n in found[3])
    nothing_fits_on_one = long_resume(profile, bullets=7)  # no step reaches one page: the largest font, two pages
    found = try_styles(nothing_fits_on_one)
    assert found[1] == 2 and (found[0].font_pt, found[0].margin_in) == LADDER[0]


def _lines_by_page(resume, style):
    drawn, indents = [], {}
    original = pdf_module.Sheet._draw

    def spy(self, line, x, y, size, height):
        drawn.append((self.pdf.page, y, round(x - self.margin, 2), "".join(p.text for p in line)))
        indents.update(text=round(self.text_x, 2), marker=round(self.bullet_x, 2))  # a bullet's text and its marker
        return original(self, line, x, y, size, height)

    pdf_module.Sheet._draw = spy
    try:
        pages = render_resume_pdf(resume, style)[1]
    finally:
        pdf_module.Sheet._draw = original
    return pages, drawn, indents


@pytest.mark.parametrize("seed", range(12))
def test_page_breaks_never_split_a_bullet_or_leave_a_job_title_alone(profile, seed):
    rng = random.Random(seed)
    resume = long_resume(profile, bullets=rng.randint(8, 12))
    for exp in resume.experience:  # bullets of different lengths move the break around
        exp.bullets = [b + " Extra detail." * rng.randint(0, 4) for b in exp.bullets]
    font_pt, margin_in = rng.choice(LADDER)
    assert_page_breaks_keep_blocks_whole(resume, FitStyle(font_pt=font_pt, margin_in=margin_in))


@pytest.mark.parametrize("seed", range(4))
def test_the_last_step_keeps_the_page_break_rules(profile, seed):
    """The ladder's last two steps (closer gaps, then the skills in line too) keep the same rules, on resumes long
    enough to break there too."""
    rng = random.Random(seed)
    resume = long_resume(profile, bullets=rng.randint(18, 24))
    for exp in resume.experience:
        exp.bullets = [b + " Extra detail." * rng.randint(0, 4) for b in exp.bullets]
    assert_page_breaks_keep_blocks_whole(resume, styles(rng.choice(["Letter", "A4"]))[len(LADDER) + seed % 2])


def test_the_last_step_draws_the_same_lines_closer_together(profile):
    """The last step is the floor's sizes with the gaps between the body's parts brought closer: the same lines at the
    same places across the page and the same header, every line at most as low as at the plain floor."""
    resume = long_resume(profile, bullets=6)
    plain, last = styles("Letter")[len(LADDER) - 1], styles("Letter")[len(LADDER)]
    _pages, at_floor, _indents = _lines_by_page(resume, plain)
    _pages, closer, _indents = _lines_by_page(resume, last)
    assert [(x, text) for _page, _y, x, text in closer] == [(x, text) for _page, _y, x, text in at_floor]
    assert closer[:3] == at_floor[:3]  # the name, the title line and the contact line
    assert all(a[:2] <= b[:2] for a, b in zip(closer, at_floor)) and closer[-1][:2] < at_floor[-1][:2]


def test_the_very_last_step_puts_the_skills_in_line(profile):
    """The very last step is the compact one with each skills row in line: its label, then the values across the full
    width (as a label too wide for the column already gets). Every other line is drawn where the compact step draws it,
    and the Word copy puts the values right after their labels too, with no tab to a column."""
    resume = long_resume(profile, bullets=6)
    resume.skills = [SkillGroup(category="Core", skills=["Python", "SQL", "Airflow"]),
                     SkillGroup(category="Cloud and Infrastructure", skills=["AWS", "Docker", "Terraform"])]
    texts = ["Core: Python, SQL, Airflow", "Cloud and Infrastructure: AWS, Docker, Terraform"]
    compact, last = styles("Letter")[len(LADDER):]
    _pages, column, _indents = _lines_by_page(resume, compact)
    _pages, in_line, _indents = _lines_by_page(resume, last)
    rows = [line for line in in_line if line[3] in texts]
    assert [(x, text) for _page, _y, x, text in rows] == [(0.0, text) for text in texts]
    at_rows = {line[:2] for line in rows}  # (page, y) of the skills rows: the column draws a label and its values there
    assert len(column) == len(in_line) + len(rows)
    assert [line for line in in_line if line[:2] not in at_rows] == [line for line in column if line[:2] not in at_rows]
    for style, shown in ((compact, ["Core:\tPython, SQL, Airflow", "Cloud and Infrastructure:\tAWS, Docker, Terraform"]),
                         (last, texts)):
        paragraphs = Document(io.BytesIO(resume_docx(resume, style))).paragraphs
        assert [p.text for p in paragraphs if p.text.startswith(("Core:", "Cloud and"))] == shown


def assert_page_breaks_keep_blocks_whole(resume, style):
    """A page never ends with a job title or a section heading, and never starts in the middle of a bullet."""
    pages, drawn, indents = _lines_by_page(resume, style)
    assert pages >= 2
    for page in range(1, pages):
        bottom_y = max(y for p, y, _x, _t in drawn if p == page)
        bottom = [t for p, y, _x, t in drawn if p == page and y == bottom_y]
        assert not any(t.startswith("Data Engineer") or t in ("SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "CERTIFICATIONS")
                       for t in bottom), bottom
        top_y = min(y for p, y, _x, _t in drawn if p == page + 1)
        top = [(x, t) for p, y, x, t in drawn if p == page + 1 and y == top_y]
        xs = {x for x, _t in top}
        assert not (indents["text"] in xs and indents["marker"] not in xs), top  # never starts mid-bullet


def test_docx_keeps_job_titles_with_their_bullets(profile):
    doc = Document(io.BytesIO(resume_docx(long_resume(profile, bullets=2), FitStyle())))
    paragraphs = doc.paragraphs
    rows = [p for p in paragraphs if p.text.startswith("Data Engineer · ")]  # the title first, then the company
    assert len(rows) == 3
    assert all(p.paragraph_format.keep_with_next for p in rows)  # a job's header stays with its first bullet
    assert all(p.paragraph_format.keep_together for p in paragraphs if p.style.name == "List Bullet")
    assert not any(p.text.startswith("Tech") for p in paragraphs) and not doc.tables


def test_a_block_taller_than_a_page_does_not_leave_a_blank_page():
    sheet = Sheet("Letter", 36, 10.5, 1.27, "t", "a")
    sheet.keep(10_000)
    assert sheet.pdf.page == 1  # nothing would fit better on a new page


def test_accented_and_non_latin1_names_render_and_extract(profile):
    fancy = profile.model_copy(deep=True)
    fancy.name = "José Álvarez-Łukasz"
    fancy.companies[1].name = "Zażółć Sp. z o.o."
    fancy.contact.location = "Zürich, Switzerland"
    pdf, pages = render_resume_pdf(validate.build_resume(fancy, make_output()), FitStyle())
    text = text_of(pdf)
    assert pages == 1 and "José Álvarez-Łukasz" in text and "Zażółć Sp. z o.o." in text and "Zürich" in text


def test_glyph_missing_from_the_font_does_not_crash(profile):
    odd = profile.model_copy(deep=True)
    odd.name = "王小明 Wang Xiaoming"
    pdf, pages = render_resume_pdf(validate.build_resume(odd, make_output()), FitStyle())
    assert pages == 1 and "Wang Xiaoming" in text_of(pdf)


def test_long_name_shrinks_to_fit_one_line(profile):
    long_name = profile.model_copy(deep=True)
    long_name.name = "Maria-Alexandra Konstantinopoulou-Papadopoulos de la Fuente Alvarez"
    pdf, _ = render_resume_pdf(validate.build_resume(long_name, make_output()), FitStyle())
    first_line = text_of(pdf).splitlines()[0]
    assert first_line == long_name.name


def make_sheet():
    return Sheet("Letter", 36, 10.5, 1.27, "t", "a")


def test_wrapping_respects_width_and_never_leaves_stray_spaces():
    sheet, rng = make_sheet(), random.Random(7)
    words = ["alpha", "be", "constellation", "x", "internationalization", "ok", "Python,", "CI/CD", "a" * 70]
    for _ in range(40):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 60)))
        width = rng.choice([120.0, 250.0, 400.0])
        lines = sheet.wrap([Run(text), Run(" tail", True)], width, 10.5)
        for line in lines:
            assert sum(p.width for p in line) <= width + 0.01
            assert line[0].text != " " and line[-1].text != " "
        joined = "".join(p.text for line in lines for p in line).replace(" ", "")
        assert joined == (text + " tail").replace(" ", "")  # nothing lost or duplicated


def test_wrapping_splits_a_token_longer_than_the_line():
    lines = make_sheet().wrap([Run("https://example.com/" + "segment/" * 30)], 150.0, 10.5)
    assert len(lines) > 2 and all(sum(p.width for p in line) <= 150.01 for line in lines)


def test_cover_letter_pdf_and_docx(resume):
    letter = CoverLetter(greeting="Dear Sam,", paragraphs=["First paragraph here.", "Second paragraph here.", "Third."], closing="Sincerely,", company_name="Acme Corp", hiring_manager="Sam Lee")
    pdf, pages = render_cover_pdf(resume, letter)
    text = text_of(pdf)
    assert pages == 1 and all(item in text for item in ("Jordan Rivera", "Dear Sam,", "Sincerely,"))
    assert PdfReader(io.BytesIO(pdf)).metadata.title == "Jordan Rivera - Cover Letter"
    paragraphs = [p.text for p in Document(io.BytesIO(cover_docx(resume, letter))).paragraphs]
    assert "Dear Sam," in paragraphs and "Sincerely," in paragraphs


def test_cover_letter_starts_at_the_greeting_under_the_resume_header(resume):
    """Reading order (what an applicant system extracts): the resume's header, then the letter from its greeting,
    the closing, and the signature in normal capitals. No date, recipient or subject line, even when the company
    and the hiring manager are known."""
    resume.profile.name = "JORDAN RIVERA"
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["One.", "Two."], closing="Sincerely,",
                         company_name="Acme Corp", hiring_manager="Sam Lee")
    lines = [line.strip() for line in text_of(render_cover_pdf(resume, letter)[0]).splitlines() if line.strip()]
    order = ["JORDAN RIVERA", resume.headline, "Dear Hiring Manager,", "One.", "Two.", "Sincerely,", "Jordan Rivera"]
    assert [line for line in lines if line in order] == order
    assert lines.index("Dear Hiring Manager,") == 3  # name, headline, contact line, then straight to the greeting
    docx = [p.text for p in Document(io.BytesIO(cover_docx(resume, letter))).paragraphs]
    assert [t for t in docx if t in order] == order and docx.index("Dear Hiring Manager,") == 3
    for text in ("\n".join(lines), "\n".join(docx)):
        assert "Acme Corp" not in text and "Sam Lee" not in text and "Re:" not in text
        assert not re.search(r"(?:January|February|March|April|May|June|July|August|September|October|November|"
                             r"December) \d{1,2}, \d{4}", text)  # no date


def test_a_long_cover_letter_stays_on_one_page(resume):
    long = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", company_name="Acme Corp",
                       paragraphs=[" ".join(["Delivered reliable data pipelines for the analytics team."] * 16)] * 4)
    from app.render.pdf import COVER_STYLES, _cover

    assert _cover(resume, long, "Letter", *COVER_STYLES[0])[1] == 2  # too long at full size
    pdf, pages = render_cover_pdf(resume, long)
    assert pages == 1 and "Sincerely," in text_of(pdf)


def test_the_cover_contact_line_breaks_between_items(resume):
    from app.render.pdf import CONTACT_SIZE, COVER_STYLES, DESIGN_SIZE, contact_lines

    resume.profile.contact = Contact(location="San Francisco Bay Area, CA", phone="+1 (555) 010-2030",
                                     email="jordan.alexander.rivera@example-mail.com",
                                     linkedin="https://www.linkedin.com/in/jordan-alexander-rivera-123456",
                                     website="https://github.com/jordan-alexander-rivera")
    sheet = Sheet("Letter", 72, COVER_STYLES[0][0], 1.42, "t", "a")
    size = CONTACT_SIZE * DESIGN_SIZE  # the letter's header is the resume's at its first step
    lines = contact_lines(resume.profile.contact, size, sheet.width)
    assert len(lines) == 2 and " · ".join(lines) == contact_line(resume.profile.contact)
    assert all(text_width(line, REGULAR, size) <= sheet.width for line in lines)
    text = text_of(render_cover_pdf(resume, CoverLetter(greeting="Dear Sam,", paragraphs=["One."], closing="Best,"))[0])
    assert text.splitlines()[2:4] == lines  # as drawn: the lines break between items


def test_docx_resume_content_and_layout_primitives(resume):
    doc = Document(io.BytesIO(resume_docx(resume, FitStyle())))
    texts = [p.text.strip() for p in doc.paragraphs]  # a heading ends in the tab that draws its rule
    assert resume.headline in texts and "SUMMARY" in texts and "EXPERIENCE" in texts
    assert "Engineer 0 · Northwind Labs · Remote\tMar 2022 – Present" in texts  # the title first, on one line
    assert not doc.tables and not doc.inline_shapes  # ATS-friendly: no tables or images


def test_small_helpers():
    assert contact_line(Contact(email="a@b.c", phone="1", location="X")) == "X · 1 · a@b.c"
    assert contact_line(Contact()) == ""
    assert fmt_dates("Jan 2020", "Present") == "Jan 2020 – Present"
    assert fmt_dates("", "2016") == "2016" and fmt_dates("", "") == ""


def test_contact_line_uses_the_usual_resume_order_whatever_the_input_order():
    contact = Contact(website="https://github.com/jrivera/", linkedin="https://www.linkedin.com/in/jrivera/",
                      email="jordan@example.com", phone="(555) 010-2030", location="Austin, TX")
    assert contact_line(contact) == (
        "Austin, TX · (555) 010-2030 · jordan@example.com · linkedin.com/in/jrivera · github.com/jrivera")
    assert contact_line(Contact(email="jordan@example.com", location="Austin, TX", phone="  ")) == (
        "Austin, TX · jordan@example.com")  # missing parts are left out


# ------------------------------------------------------------------ characters the font lacks
def test_ligatures_and_invisible_marks_print_as_plain_letters(profile, caplog):
    """Text copied out of a PDF often holds ligature characters (ﬁ ﬂ ﬀ ﬃ), and pasted text invisible marks (a soft
    hyphen, a zero-width joiner). Inter has none of them: they print as their plain letters, or not at all, with no gap
    where they were measured, and fpdf2 has no glyph to leave out."""
    pdfminer = pytest.importorskip("pdfminer.high_level")
    odd = profile.model_copy(deep=True)
    odd.name = "Jor‍dan Rivera"
    odd.companies[0].name = "Bluefin Oﬃce Supplies"
    resume = validate.build_resume(odd, make_output())
    resume.summary = "Data engineer who builds pro­gramming tools for the ﬁnance team."
    resume.experience[0].bullets[0] = "Built the workﬂow that cut eﬀort and hand-oﬀs for the ﬁnance team."
    resume.certifications = [Certification(name="AWS Certiﬁed Solutions Architect", issuer="Amazon Web Services",
                                           year="Mar 2023")]
    with caplog.at_level(logging.WARNING):
        pdf = render_resume_pdf(resume, FitStyle())[0]
    assert not [record for record in caplog.records if "missing" in record.getMessage()]
    plain = text_of(pdf)
    assert "AWS Certified Solutions Architect · Amazon Web Services" in plain  # no gap after the name
    for text in (plain, pdfminer.extract_text(io.BytesIO(pdf))):
        flat = " ".join(text.split())
        assert "Jordan Rivera" in flat and "Bluefin Office Supplies" in flat
        assert "builds programming tools for the finance team." in flat
        assert "Built the workflow that cut effort and hand-offs for the finance team." in flat
    assert text_width("Certiﬁed", REGULAR, 10) == text_width("Certified", REGULAR, 10)  # measured as drawn


def test_accents_typed_apart_print_on_their_letters(profile):
    """Some systems type an accented letter as the letter and then the accent (Unicode's decomposed form: e + U+0301).
    fpdf2 places no accents, so each would print beside its letter ('Jose´'); the PDF composes them first and draws
    exactly what the composed text draws, and readers get the composed letters back."""
    pdfplumber = pytest.importorskip("pdfplumber")
    composed = ["José Müller · Nguyễn Thị Hạnh", "Café Société", "Mérida, Yucatán",
                "Created a résumé reader for the café's Zürich team."]

    def render(texts):
        person = profile.model_copy(deep=True)
        person.name, person.companies[0].name, person.contact.location = texts[:3]
        resume = validate.build_resume(person, make_output())
        resume.experience[0].bullets[0] = texts[3]
        pdf = render_resume_pdf(resume, styles("A4")[2])[0]
        return pdf, [page.get_contents().get_data() for page in PdfReader(io.BytesIO(pdf)).pages]

    decomposed = [unicodedata.normalize("NFD", text) for text in composed]
    assert decomposed != composed
    pdf, pages = render(decomposed)
    assert pages == render(composed)[1]  # the same drawing, letter for letter
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        text = unicodedata.normalize("NFC", document.pages[0].extract_text())
    assert all(part in text for part in composed)
    assert text_width(decomposed[0], REGULAR, 10) == text_width(composed[0], REGULAR, 10)


def test_a_mark_the_font_lacks_prints_nothing_and_leaves_no_gap(profile, caplog):
    """An emoji keyboard types ™ and ® with a variation selector after them (U+FE0F), and a keycap mark (U+20E3) after a
    digit: Inter has neither, so they are left out like the invisible marks, with no gap where they were measured and
    no missing glyph for fpdf2."""
    resume = validate.build_resume(profile, make_output())
    resume.certifications = [Certification(name="Fabrikam Sheets™️ Specialist", issuer="Fabrikam", year="2021"),
                             Certification(name="Contoso Cloud®️ Level 2⃣", issuer="Contoso", year="2022")]
    with caplog.at_level(logging.WARNING):
        pdf = render_resume_pdf(resume, FitStyle())[0]
    assert not [record for record in caplog.records if "missing" in record.getMessage()]
    flat = " ".join(text_of(pdf).split())
    assert "Fabrikam Sheets™ Specialist · Fabrikam" in flat and "Contoso Cloud® Level 2 · Contoso" in flat
    assert text_width("Sheets™️", SEMIBOLD, 10) == text_width("Sheets™", SEMIBOLD, 10)


def test_the_character_table_stays_small_whatever_text_it_meets():
    """The table of how each character is drawn keeps what the fonts have and the characters it rewrites, but stops
    keeping characters it leaves as they are (Chinese, emoji) once it is large: crafted text of many thousands of
    different characters does not fill a worker's memory, and is still measured the same."""
    text = "".join(map(chr, [*range(0x3400, 0x4DC0), *range(0x4E00, 0xA000)]))  # 27,584 Chinese characters
    assert len(text) > DRAWABLE_LIMIT
    missing = advance_widths(REGULAR)[1]
    assert text_width(text, REGULAR, 10) == pytest.approx(len(text) * missing * 0.01)
    assert len(DRAWABLE) <= DRAWABLE_LIMIT
    assert text_width("Certiﬁed", REGULAR, 10) == text_width("Certified", REGULAR, 10)  # what it rewrites still works


# ------------------------------------------------------------------ skills rows that wrap
def test_a_wrapped_skills_row_is_read_before_the_next_row(profile):
    """A label wider than the design's column widens it, and a long row wraps inside the values' column. Filler spaces
    from the margin start each of its lines where the row starts, so readers that group text by position keep a
    wrapped line with its row (pdfminer read it after the next row)."""
    pdfminer = pytest.importorskip("pdfminer.high_level")
    pdfplumber = pytest.importorskip("pdfplumber")
    resume = validate.build_resume(profile, make_output())
    data = ["Airflow", "Spark", "Snowflake", "dbt", "Kafka", "Flink", "Postgres", "BigQuery", "Looker",
            "Great Expectations", "Delta Lake", "Iceberg", "Trino", "Redis", "MongoDB", "Elasticsearch", "Redshift"]
    resume.skills = [SkillGroup(category="Machine Learning", skills=["PyTorch", "scikit-learn", "XGBoost", "MLflow"]),
                     SkillGroup(category="Data", skills=data),
                     SkillGroup(category="Frameworks", skills=["FastAPI", "Django", "Flask", "React"])]
    for step in (0, 1, len(LADDER) - 1, len(LADDER), len(LADDER) + 1):  # the design, the first calibrated step, the
        pdf = render_resume_pdf(resume, styles("Letter")[step])[0]  # floor, closer gaps, and the skills in line too
        with pdfplumber.open(io.BytesIO(pdf)) as document:
            plumbed = document.pages[0].extract_text()
        for text in (pdfminer.extract_text(io.BytesIO(pdf)), text_of(pdf), plumbed):
            flat = " ".join(text.split())
            assert "Data: " + ", ".join(data) + " Frameworks: FastAPI, Django" in flat, step


# ------------------------------------------------------------------ the Word copy
def test_word_text_is_never_larger_than_it_was_measured(resume):
    """Word keeps sizes in half points. They are rounded down, so text broken into lines at its exact size on the PDF
    (the contact line here, a job line) still fits its line in Word."""
    resume.profile.contact = Contact(location="New York, NY", phone="+1 (555) 010-2030", email="lschmidt@example-mail.com",
                                     linkedin="linkedin.com/in/l-schmidt", website="github.com/lschmidt")
    for style in styles("Letter"):
        doc = Document(io.BytesIO(resume_docx(resume, style)))
        width = doc.sections[0].page_width.pt - doc.sections[0].left_margin.pt - doc.sections[0].right_margin.pt
        contact = next(p for p in doc.paragraphs if "lschmidt@example-mail.com" in p.text)
        for run in (run for run in contact.runs if run.text):
            assert run.font.size.pt <= CONTACT_SIZE * style.font_pt
            assert text_width(run.text, REGULAR, run.font.size.pt) <= width
        assert doc.styles["Normal"].font.size.pt <= style.font_pt


def test_word_bullets_keep_the_gap_between_them(resume):
    """The built-in List Bullet style's contextual spacing would drop the space after a bullet when the next one
    follows; the Word copy turns it off, so its bullets are as far apart as the PDF's."""
    doc = Document(io.BytesIO(resume_docx(resume, FitStyle())))
    assert doc.styles["List Bullet"].element.find(f".//{qn('w:contextualSpacing')}") is None
    bullets = [p for p in doc.paragraphs if p.style.name == "List Bullet"]
    assert any(p.paragraph_format.space_after.pt > 0 for p in bullets)
