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


def test_metadata_and_embedded_font(resume):
    pdf, _ = render_resume_pdf(resume, FitStyle())
    reader = PdfReader(io.BytesIO(pdf))
    assert reader.metadata.title == "Jordan Rivera - Resume" and reader.metadata.author == "Jordan Rivera"
    fonts = reader.pages[0]["/Resources"]["/Font"]
    embedded = []
    for ref in fonts.values():
        font = ref.get_object()
        descendant = font["/DescendantFonts"][0].get_object()
        descriptor = descendant["/FontDescriptor"].get_object()
        embedded.append(("Carlito" in str(font["/BaseFont"]), "/FontFile2" in descriptor))
    assert embedded and all(is_carlito and has_file for is_carlito, has_file in embedded)


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
    pdf, pages = render_cover_pdf(resume, letter)
    text = text_of(pdf)
    assert pages == 1 and all(item in text for item in ("Jordan Rivera", "Sam Lee", "Acme Corp", "Dear Sam,", "Sincerely,"))
    assert PdfReader(io.BytesIO(pdf)).metadata.title == "Jordan Rivera - Cover Letter"
    paragraphs = [p.text for p in Document(io.BytesIO(cover_docx(resume, letter))).paragraphs]
    assert "Dear Sam," in paragraphs and "Acme Corp" in paragraphs


def test_docx_resume_content_and_layout_primitives(resume):
    doc = Document(io.BytesIO(resume_docx(resume, FitStyle())))
    texts = [p.text for p in doc.paragraphs]
    assert resume.headline in texts and "SUMMARY" in texts and "EXPERIENCE" in texts
    assert any(t.startswith("Engineer 0") and "Northwind Labs" in t and "Mar 2022 - Present" in t for t in texts)
    assert not doc.tables and not doc.inline_shapes  # ATS-friendly: no tables or images


def test_small_helpers():
    assert contact_line(Contact(email="a@b.c", phone="1", location="X")) == "a@b.c | 1 | X"
    assert contact_line(Contact()) == ""
    assert fmt_dates("Jan 2020", "Present") == "Jan 2020 - Present" and fmt_dates("", "2016") == "2016" and fmt_dates("", "") == ""
