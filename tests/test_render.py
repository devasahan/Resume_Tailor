import io
import random

import pytest
from docx import Document
from pypdf import PdfReader

from app.pipeline import validate
from app.pipeline.fit import trim_to_fit, try_styles
from app.render.docx import cover_docx, resume_docx
from app.render.pdf import Run, Sheet, contact_line, fmt_dates, render_cover_pdf, render_resume_pdf
from app.schemas import (
    Contact, CoverLetter, ExperienceOut, FitStyle, SkillGroup, TailoredSections, TailorOutput,
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
    assert "Mar 2022 - Present" in text and "Remote" in text and "B.S., Computer Science" in text


def test_dates_stay_on_the_job_line_for_layout_readers(resume):
    """pdfminer groups text by position, like many ATS: right-aligned dates must not land inside a bullet."""
    pdfminer = pytest.importorskip("pdfminer.high_level")
    sentence = "Designed and operated data pipelines with Python and SQL for several product teams, documenting each decision."
    for exp in resume.experience:
        exp.bullets = [sentence, sentence]
    pdf, _ = render_resume_pdf(resume, FitStyle())
    lines = [" ".join(line.split()) for line in pdfminer.extract_text(io.BytesIO(pdf)).splitlines()]
    for name, dates in (("Northwind Labs", "Mar 2022 - Present"), ("Contoso Systems", "Jun 2019 - Feb 2022"),
                        ("Fabrikam", "Jul 2016 - May 2019"), ("University of Texas at Austin", "2016")):
        line = next(line for line in lines if name in line)
        assert line.endswith(dates), line
    assert text_of(pdf).count("Mar 2022 - Present") == 1  # the plain reader sees the dates once too


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
        embedded.append(("Carlito" in str(font["/BaseFont"]), "/FontFile2" in descriptor))
    assert embedded and all(is_carlito and has_file for is_carlito, has_file in embedded)


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
                ExperienceOut(company_index=i, job_title="Data Engineer", bullets=[f"{sentence} Item {'abcdefghijkl'[j]}." for j in range(bullets)])
                for i in range(len(profile.companies))
            ],
        ),
    )
    return validate.build_resume(profile, out)


def test_overflow_makes_more_pages_and_the_fit_steps_repair_it(profile):
    resume = long_resume(profile)
    _, pages = render_resume_pdf(resume, FitStyle())
    assert pages > 1
    assert try_styles(resume) is None  # shrinking alone cannot save this much text
    trimmed, style, pages, _pdf, notes = trim_to_fit(resume)
    assert pages == 1 and style.font_pt == 9.5
    assert all(len(e.bullets) >= 2 for e in trimmed.experience)
    assert any("Removed" in n for n in notes)
    assert sum(len(e.bullets) for e in trimmed.experience) < sum(len(e.bullets) for e in resume.experience)


def test_moderate_overflow_is_fixed_by_shrinking_only(profile):
    resume = long_resume(profile, bullets=6)
    _, pages = render_resume_pdf(resume, FitStyle())
    found = try_styles(resume)
    assert pages == 2 and found is not None and found[1] == 1 and found[0].font_pt < 10.5
    assert any("Font reduced" in n for n in found[3])


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
    pdf, pages = render_cover_pdf(resume, letter, role="Senior Data Engineer")
    text = text_of(pdf)
    assert pages == 1 and all(item in text for item in ("Jordan Rivera", "Sam Lee", "Acme Corp", "Dear Sam,", "Sincerely,"))
    assert PdfReader(io.BytesIO(pdf)).metadata.title == "Jordan Rivera - Cover Letter"
    paragraphs = [p.text for p in Document(io.BytesIO(cover_docx(resume, letter, role="Senior Data Engineer"))).paragraphs]
    assert "Dear Sam," in paragraphs and "Acme Corp" in paragraphs and "Re: Senior Data Engineer" in paragraphs


def test_cover_letter_is_a_business_letter_under_the_resume_header(resume):
    """Reading order (what an applicant system extracts): the resume's header, date, recipient, subject, letter,
    closing, and the signature in normal capitals."""
    resume.profile.name = "JORDAN RIVERA"
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["One.", "Two."], closing="Sincerely,",
                         company_name="Acme Corp")
    pdf, _ = render_cover_pdf(resume, letter, role="Senior Data Engineer")
    lines = [line.strip() for line in text_of(pdf).splitlines() if line.strip()]
    order = ["JORDAN RIVERA", resume.headline, "Hiring Manager", "Acme Corp", "Re: Senior Data Engineer",
             "Dear Hiring Manager,", "One.", "Two.", "Sincerely,", "Jordan Rivera"]
    assert [line for line in lines if line in order] == order
    assert any(line.startswith("Austin, TX") or "@" in line for line in lines[2:4])  # the contact line, under the headline
    docx = [p.text for p in Document(io.BytesIO(cover_docx(resume, letter, role="Senior Data Engineer"))).paragraphs]
    assert [t for t in docx if t in order] == order


def test_cover_letter_recipient_and_subject_only_when_known(resume):
    bare = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["One."], closing="Sincerely,")
    text = text_of(render_cover_pdf(resume, bare)[0])
    assert "Hiring Manager\n" not in text.replace("Dear Hiring Manager,", "") and "Re:" not in text
    named = bare.model_copy(update={"hiring_manager": "Sam Lee"})
    assert "Sam Lee" in text_of(render_cover_pdf(resume, named)[0])


def test_a_long_cover_letter_stays_on_one_page(resume):
    long = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", company_name="Acme Corp",
                       paragraphs=[" ".join(["Delivered reliable data pipelines for the analytics team."] * 11)] * 4)
    from app.render.pdf import COVER_STYLES, _cover

    assert _cover(resume, long, "Letter", "Senior Data Engineer", *COVER_STYLES[0])[1] == 2  # too long at full size
    pdf, pages = render_cover_pdf(resume, long, role="Senior Data Engineer")
    assert pages == 1 and "Sincerely," in text_of(pdf)


def test_the_cover_contact_line_breaks_between_items(resume):
    from app.render.pdf import COVER_STYLES, contact_lines

    resume.profile.contact = Contact(location="San Francisco Bay Area, CA", phone="+1 (555) 010-2030",
                                     email="jordan.alexander.rivera@example-mail.com",
                                     linkedin="https://www.linkedin.com/in/jordan-alexander-rivera-123456",
                                     website="https://github.com/jordan-alexander-rivera")
    sheet = Sheet("Letter", 72, COVER_STYLES[0][0], 1.35, "t", "a")
    lines = contact_lines(sheet, resume.profile.contact, 10)
    assert len(lines) == 2 and " | ".join(lines) == contact_line(resume.profile.contact)
    assert all(sheet.measure(line, False, 10) <= sheet.width for line in lines)


def test_docx_resume_content_and_layout_primitives(resume):
    doc = Document(io.BytesIO(resume_docx(resume, FitStyle())))
    texts = [p.text for p in doc.paragraphs]
    assert resume.headline in texts and "SUMMARY" in texts and "EXPERIENCE" in texts
    assert any(t.startswith("Engineer 0") and "Northwind Labs" in t and "Mar 2022 - Present" in t for t in texts)
    assert not doc.tables and not doc.inline_shapes  # ATS-friendly: no tables or images


def test_small_helpers():
    assert contact_line(Contact(email="a@b.c", phone="1", location="X")) == "X | 1 | a@b.c"
    assert contact_line(Contact()) == ""
    assert fmt_dates("Jan 2020", "Present") == "Jan 2020 - Present" and fmt_dates("", "2016") == "2016" and fmt_dates("", "") == ""


def test_contact_line_uses_the_usual_resume_order_whatever_the_input_order():
    contact = Contact(website="https://github.com/jrivera/", linkedin="https://www.linkedin.com/in/jrivera/",
                      email="jordan@example.com", phone="(555) 010-2030", location="Austin, TX")
    assert contact_line(contact) == (
        "Austin, TX | (555) 010-2030 | jordan@example.com | linkedin.com/in/jrivera | github.com/jrivera")
    assert contact_line(Contact(email="jordan@example.com", location="Austin, TX", phone="  ")) == (
        "Austin, TX | jordan@example.com")  # missing parts are left out
