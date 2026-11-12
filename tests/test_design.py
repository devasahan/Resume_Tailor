"""The owner's resume design ("standard"): Inter in four weights, embedded as TrueType; the navy title line and headings
with thin gray rules; the job title first on each job line; the skills labels in a column; the cover letter under the
same header; a Word copy in the same look; and runs saved before the restyle shown in the new look, within the pages
they had and with every bullet."""
import gc
import io
import json
import re
import sqlite3
from pathlib import Path

import pytest
from docx import Document
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import RGBColor
from fontTools.ttLib import TTFont
from fpdf import FPDF
from fpdf.fonts import TTFFont as TTFFont_fpdf
from pypdf import PdfReader

from app.pipeline import fit, validate
from app.pipeline.fit import (
    CALIBRATED,
    LADDER,
    OLD_LADDER,
    current_style,
    saved_resume_pdf,
    styles,
)
from app.render.docx import cover_docx, resume_docx
from app.render.pdf import (
    ACCENT,
    FONT_DIR,
    HEADING_SIZE,
    HEADING_TRACKING,
    MAX_TRACKING,
    NAME_TRACKING,
    RULE,
    WEIGHTS,
    contact_line,
    render_cover_pdf,
    render_resume_pdf,
    resume_layout,
    skills_column,
    split_job_line,
    text_width,
)
from app.schemas import (
    Certification,
    Company,
    Contact,
    CoverLetter,
    Education,
    FitStyle,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
)
from app.timing import FONTS

from .conftest import SAMPLE_JD
from .test_api import save_profile
from .test_render import long_resume
from .test_validate import make_output

HEADINGS = ("SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "CERTIFICATIONS")
TEXT_WIDTH = 612 - 2 * 72 * FitStyle().margin_in  # points between the margins at the default step (Letter)


def fonts_of(pdf: bytes) -> dict[str, bytes]:
    """The fonts embedded for the first page: their name without the subset tag -> the first bytes of the font file."""
    found = {}
    for ref in PdfReader(io.BytesIO(pdf)).pages[0]["/Resources"]["/Font"].values():
        font = ref.get_object()
        descriptor = font["/DescendantFonts"][0].get_object()["/FontDescriptor"].get_object()
        assert "/FontFile3" not in descriptor  # never CFF outlines
        found[str(font["/BaseFont"]).split("+")[-1]] = descriptor["/FontFile2"].get_object().get_data()[:4]
    return found


def lines_of(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines()]


def certified(profile):
    """The shared resume with a certification printed, so every section heading is there."""
    resume = validate.build_resume(profile, make_output())
    resume.profile.certifications = [Certification(name="Cloud Data Engineer Associate", issuer="Contoso Academy",
                                                   year="Mar 2023", expires="Mar 2026")]
    resume.certifications = list(resume.profile.certifications)
    return resume


# ------------------------------------------------------------------ the font
def test_the_four_weights_are_embedded_as_truetype_inter(profile):
    found = fonts_of(render_resume_pdf(validate.build_resume(profile, make_output()), FitStyle())[0])
    assert set(found) == {"Inter", "InterMedium", "InterSemiBold", "InterBold"}  # body, title line, job titles, name
    assert set(found.values()) == {b"\x00\x01\x00\x00"}  # TrueType outlines, as /FontFile2 requires


@pytest.mark.parametrize("weight", sorted(WEIGHTS))
def test_the_bundled_fonts_are_truetype_and_measure_as_fpdf2_does(weight):
    family, style, file = WEIGHTS[weight]
    font = TTFont(FONT_DIR / file)
    assert font.sfntVersion == "\x00\x01\x00\x00" and "glyf" in font and "CFF " not in font
    source = Path("/usr/share/fonts/opentype/inter") / file.replace(".ttf", ".otf")
    if source.exists():  # where the source is installed: the conversion kept its character map and every width
        original = TTFont(source)
        assert original.getBestCmap() == font.getBestCmap()
        assert original["hmtx"].metrics.keys() == font["hmtx"].metrics.keys()
        assert all(original["hmtx"][name][0] == font["hmtx"][name][0] for name in font["hmtx"].metrics)
    pdf = FPDF(unit="pt")
    pdf.add_font(family, style, str(FONT_DIR / file))
    samples = ("Senior Data Engineer · Northwind Labs · Remote", "Feb 2025 – Present", "• Zürich, Łódź",
               "王 x")  # the last one the font does not have
    for text in samples:
        for size, tracking in ((10.125, 0.0), (8.45, 0.0), (HEADING_SIZE * 10.125, HEADING_TRACKING),
                               (22.5, NAME_TRACKING)):
            pdf.set_font(family, style, size)
            pdf.set_char_spacing(round(tracking * size, 2))  # as the sheet sets it before drawing
            assert text_width(text, weight, size, tracking) == pytest.approx(pdf.get_string_width(text), abs=1e-9)


def test_heading_letter_spacing_stays_readable_for_applicant_systems():
    """pdfminer, and applicant systems built like it, read letters 0.10 em or more apart as separate words."""
    assert 0 < HEADING_TRACKING <= MAX_TRACKING <= 0.09


# ------------------------------------------------------------------ colours and rules
def colours(pdf: bytes, operator: str) -> list[tuple[float, ...]]:
    """The RGB colours set with `operator` in the first page (rg: text, RG: lines)."""
    data = PdfReader(io.BytesIO(pdf)).pages[0].get_contents().get_data().decode("latin-1")
    return [tuple(float(x) for x in found) for found in re.findall(rf"([\d.]+) ([\d.]+) ([\d.]+) {operator}\b", data)]


def same_colour(found: tuple[float, ...], color: str) -> bool:
    return all(abs(value - int(color[i:i + 2], 16) / 255) < 0.002 for value, i in zip(found, (1, 3, 5)))


def test_the_title_line_and_headings_are_navy_and_the_rules_thin_and_gray(profile):
    pdf = render_resume_pdf(certified(profile), FitStyle())[0]
    assert any(same_colour(found, ACCENT) for found in colours(pdf, "rg"))  # the title line and the headings
    strokes = colours(pdf, "RG")
    assert strokes and all(same_colour(found, RULE) for found in strokes)  # every line drawn is a gray rule
    data = PdfReader(io.BytesIO(pdf)).pages[0].get_contents().get_data().decode("latin-1")
    assert data.count(" l S") == len(HEADINGS) and "0.75 w" in data  # one 0.75 pt rule beside each heading


# ------------------------------------------------------------------ what applicant systems read
def test_headings_are_read_as_whole_words(profile):
    pdfminer = pytest.importorskip("pdfminer.high_level")
    pdfplumber = pytest.importorskip("pdfplumber")
    pdf = render_resume_pdf(certified(profile), FitStyle())[0]
    texts = {"pypdf": PdfReader(io.BytesIO(pdf)).pages[0].extract_text(),
             "pdfminer": pdfminer.extract_text(io.BytesIO(pdf))}
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        texts["pdfplumber"] = document.pages[0].extract_text()
        words = {word["text"] for word in document.pages[0].extract_words()}
    for reader, text in texts.items():
        assert set(HEADINGS) <= set(lines_of(text)), reader  # each heading on its own line, its letters together
    assert set(HEADINGS) <= words


def test_each_skills_label_is_read_on_the_line_of_its_values(profile):
    pdfminer = pytest.importorskip("pdfminer.high_level")
    pdfplumber = pytest.importorskip("pdfplumber")
    resume = validate.build_resume(profile, make_output())
    resume.skills = [SkillGroup(category="Languages", skills=["Python", "SQL", "Scala"]),
                     SkillGroup(category="Data Platforms", skills=["Airflow", "Spark", "Snowflake", "dbt"])]
    assert skills_column(resume.skills, FitStyle().font_pt, TEXT_WIDTH) is not None  # the labels have a column
    pdf = render_resume_pdf(resume, FitStyle())[0]
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        plumbed = document.pages[0].extract_text()
    for text in (PdfReader(io.BytesIO(pdf)).pages[0].extract_text(), pdfminer.extract_text(io.BytesIO(pdf)), plumbed):
        lines = lines_of(text)
        assert "Languages: Python, SQL, Scala" in lines and "Data Platforms: Airflow, Spark, Snowflake, dbt" in lines
    paragraphs = [p.text for p in Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs]
    assert "Languages:\tPython, SQL, Scala" in paragraphs  # Word: a tab to the values' column


def test_labels_too_wide_for_a_column_go_in_line_with_their_values(profile):
    resume = validate.build_resume(profile, make_output())
    label = "Cloud Infrastructure, Platform Engineering and Site Reliability"
    resume.skills = [SkillGroup(category=label, skills=["AWS", "Terraform"]),
                     SkillGroup(category="Languages", skills=["Python"])]
    assert skills_column(resume.skills, FitStyle().font_pt, TEXT_WIDTH) is None  # it would take over 40% of the width
    lines = lines_of(PdfReader(io.BytesIO(render_resume_pdf(resume, FitStyle())[0])).pages[0].extract_text())
    assert f"{label}: AWS, Terraform" in lines and "Languages: Python" in lines  # every row in line: no label wraps
    paragraphs = [p.text for p in Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs]
    assert f"{label}: AWS, Terraform" in paragraphs and "Languages: Python" in paragraphs


# ------------------------------------------------------------------ the job line
LONG_TITLE = "Senior Staff Software Engineer, Payments Data Platform"
LONG_COMPANY = Company(name="Bluefin Commerce International Holdings", location="San Francisco Bay Area, CA",
                       mode="Hybrid", start="Feb 2021", end="Present")
AFTER_TITLE = "Bluefin Commerce International Holdings · San Francisco Bay Area, CA · Hybrid"


def test_split_job_line_keeps_whole_parts_beside_the_dates():
    size = 10.5
    title_only = text_width(LONG_TITLE, "semibold", size) + 1
    first, rest = split_job_line(LONG_COMPANY, LONG_TITLE, size, title_only)
    assert [run.text for run in first] == [LONG_TITLE] and [run.text for run in rest] == [AFTER_TITLE]  # no '· '
    with_company = title_only + text_width(" · Bluefin Commerce International Holdings", "regular", size)
    first, rest = split_job_line(LONG_COMPANY, LONG_TITLE, size, with_company)
    assert "".join(run.text for run in first) == f"{LONG_TITLE} · Bluefin Commerce International Holdings"
    assert [run.text for run in rest] == ["San Francisco Bay Area, CA · Hybrid"]  # whole parts only
    first, rest = split_job_line(LONG_COMPANY, LONG_TITLE, size, 2000.0)
    assert "".join(run.text for run in first) == f"{LONG_TITLE} · {AFTER_TITLE}" and rest == []
    assert [run.weight for run in first] == ["semibold", "regular"]  # the title stands out, the rest is gray
    plain = Company(name="Northwind Labs", start="Mar 2022", end="Present")
    first, _rest = split_job_line(plain, "Engineer", size, 2000.0)
    assert "".join(run.text for run in first) == "Engineer · Northwind Labs"  # no place or mode: just the company


def test_a_long_job_line_breaks_before_the_company_with_the_dates_on_the_title_line(profile):
    long = profile.model_copy(deep=True)
    long.companies[0] = LONG_COMPANY
    resume = validate.build_resume(long, make_output())
    resume.experience[0].job_title = LONG_TITLE
    pdf = render_resume_pdf(resume, FitStyle())[0]
    lines = lines_of(PdfReader(io.BytesIO(pdf)).pages[0].extract_text())
    row = lines.index(f"{LONG_TITLE} Feb 2021 – Present")
    assert lines[row + 1] == AFTER_TITLE
    paragraphs = [p.text for p in Document(io.BytesIO(resume_docx(resume, FitStyle()))).paragraphs]
    assert f"{LONG_TITLE}\tFeb 2021 – Present\n{AFTER_TITLE}" in paragraphs  # Word breaks the line in the same place


# ------------------------------------------------------------------ the cover letter
def test_the_cover_letter_has_the_resume_header_and_stays_on_one_page(profile):
    resume = validate.build_resume(profile, make_output())
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["One.", "Two."], closing="Sincerely,")
    cover, pages = render_cover_pdf(resume, letter)
    designed = render_resume_pdf(resume, styles("Letter")[0])[0]  # the resume at the design's own sizes
    first = lines_of(PdfReader(io.BytesIO(cover)).pages[0].extract_text())[:3]
    assert pages == 1 and first == lines_of(PdfReader(io.BytesIO(designed)).pages[0].extract_text())[:3]
    assert first == ["Jordan Rivera", resume.headline, contact_line(profile.contact)]

    def header_sizes(pdf: bytes) -> list[str]:
        data = PdfReader(io.BytesIO(pdf)).pages[0].get_contents().get_data().decode("latin-1")
        return re.findall(r"/F\d+ ([\d.]+) Tf", data)[:3]

    assert header_sizes(cover) == header_sizes(designed)  # the name, title and contact line in the same sizes
    assert colours(cover, "RG") and all(same_colour(found, RULE) for found in colours(cover, "RG"))


# ------------------------------------------------------------------ the Word copy
def test_word_headings_are_navy_capitals_with_a_gray_rule(profile):
    doc = Document(io.BytesIO(resume_docx(validate.build_resume(profile, make_output()), FitStyle())))
    heading = next(p for p in doc.paragraphs if p.text.strip() == "EXPERIENCE")
    assert heading.paragraph_format.keep_with_next and heading.runs[0].bold
    assert heading.runs[0].font.color.rgb == RGBColor.from_string(ACCENT[1:])
    assert heading.runs[-1].text == "\t" and heading.runs[-1].font.color.rgb == RGBColor.from_string(RULE[1:])
    stop = heading.paragraph_format.tab_stops[0]
    assert stop.alignment == WD_TAB_ALIGNMENT.RIGHT and stop.leader == WD_TAB_LEADER.LINES  # the rule: a leader
    assert doc.styles["Normal"].font.name == "Inter"
    fonts = doc.part.part_related_by(RELATIONSHIP_TYPE.FONT_TABLE).blob
    assert b'w:name="Inter"' in fonts and b'w:altName w:val="Arial"' in fonts  # where Inter is not installed


def test_a_word_file_loads_no_pdf_fonts(profile):
    resume = validate.build_resume(profile, make_output())
    loads = FONTS.count
    resume_docx(resume, FitStyle())
    cover_docx(resume, CoverLetter(greeting="Dear Sam,", paragraphs=["One."], closing="Best,"))
    assert FONTS.count == loads  # sizes come from the font tables, never from a PDF laid out on the side


# ------------------------------------------------------------------ the size steps
def test_a_short_resume_has_the_design_sizes(profile):
    style, pages, _pdf, notes = fit.try_styles(long_resume(profile, bullets=3))
    assert ((style.font_pt, style.margin_in), pages, notes) == (LADDER[0], 1, [])


def test_the_design_sizes_never_add_a_page(profile):
    """Two pages at the design's sizes, the second one well filled, while a calibrated step fits everything on one
    page: the resume keeps the one page it had before the restyle."""
    resume = fit._plan_resume(profile, [6, 4, 3])
    pages, fill = resume_layout(resume, styles("Letter")[0])
    assert pages == 2 and fill >= fit.MIN_LAST_PAGE_FILL  # the fill rule alone would keep two pages at the design size
    style, pages, _pdf, notes = fit.try_styles(resume)
    assert pages == 1 and (style.font_pt, style.margin_in) != LADDER[0] and any("one page" in n for n in notes)


def many_short_jobs() -> Resume:
    """Ten jobs with two one-line bullets each. Laid out before the restyle at its smallest step (9.5 pt Carlito,
    0.5 in) it filled 0.99 of one page: the design gives every job more room, and the new smallest step still fits it
    on one."""
    bullet = ("Moved 120 nightly reports to Snowflake and dbt, cutting the build from 40 to 12 minutes for the "
              "finance team.")
    skills = ["Python", "SQL", "Airflow", "Snowflake", "AWS", "Docker", "Kafka", "dbt"]
    summary = " ".join(("Senior engineer with many years of experience building data platforms and services for "
                        "product teams, specializing in pipelines, integrations and cloud infrastructure. " * 4
                        ).split()[:80])
    companies = [Company(name=f"Company Number {i}", start=f"Jan {2024 - 2 * i}",
                         end="Present" if i == 0 else f"Dec {2025 - 2 * i}", mode="Remote" if i % 3 else "Hybrid")
                 for i in range(10)]
    profile = Profile(name="Taylor Quinn", contact=Contact(email="taylor@example.com", phone="(555) 010-4477",
                      linkedin="linkedin.com/in/taylorquinn", location="Austin, TX"), companies=companies,
                      education=[Education(school="State University", degree="B.S.", field="Computer Science",
                                           end="2014")])
    return Resume(profile=profile, headline="Senior Data Engineer | Python | AWS", summary=summary,
                  skills=[SkillGroup(category=c, skills=skills) for c in ("Languages", "Data", "Cloud", "Practices")],
                  experience=[ResumeExperience(company_index=i, job_title="Senior Data Engineer",
                                               bullets=[bullet] * 2) for i in range(10)])


def test_the_smallest_step_holds_what_the_old_one_did_for_a_career_of_many_short_jobs():
    assert resume_layout(many_short_jobs(), styles("Letter")[-1])[0] == 1


def test_laid_out_documents_are_let_go(profile):
    """fpdf2 caches glyph lookups for each document with functools.cache, which kept every document's four fonts
    (about 10 MB) in memory for good: each new sheet empties that cache, so fonts no longer pile up as runs go by."""
    def live_fonts() -> int:
        gc.collect()
        return sum(isinstance(found, TTFFont_fpdf) for found in gc.get_objects())

    resume = long_resume(profile, bullets=2)
    resume_layout(resume, FitStyle())
    before = live_fonts()
    for _ in range(5):
        resume_layout(resume, FitStyle())
    assert live_fonts() <= before


# ------------------------------------------------------------------ runs saved before the restyle
def test_old_ladder_steps_map_to_the_calibrated_steps():
    assert not set(LADDER) & set(OLD_LADDER)  # an old step is never mistaken for a new one
    for index, (font_pt, margin_in) in enumerate(OLD_LADDER):
        new_pt, new_in = LADDER[CALIBRATED + index]
        assert current_style(FitStyle(font_pt=font_pt, margin_in=margin_in, paper="A4")) == FitStyle(
            font_pt=new_pt, margin_in=new_in, paper="A4")
    for font_pt, margin_in in LADDER:
        style = FitStyle(font_pt=font_pt, margin_in=margin_in)
        assert current_style(style) is style  # already the new look
    error_run = FitStyle.model_validate({"font_pt": 10.5, "margin_in": 0.65, "paper": "Letter", "stacked": False})
    assert current_style(error_run) == FitStyle()  # the default of then is the default of now


def test_a_saved_run_tries_smaller_steps_only_when_it_needs_more_pages(profile, monkeypatch):
    resume = long_resume(profile, bullets=6)
    old = FitStyle(font_pt=10.5, margin_in=0.65)
    laid_out = []
    real = fit.render_resume_pdf
    monkeypatch.setattr(fit, "render_resume_pdf", lambda r, s, d="Resume": laid_out.append(s) or real(r, s, d))
    two_pages = saved_resume_pdf(resume, old, 2)
    assert laid_out == [current_style(old)] and len(PdfReader(io.BytesIO(two_pages)).pages) == 2  # once: it fits
    laid_out.clear()
    one_page = saved_resume_pdf(resume, old, 1)  # it was one page then: smaller steps until it is one page again
    assert laid_out[0] == current_style(old) and len(laid_out) > 1 and len(PdfReader(io.BytesIO(one_page)).pages) == 1
    text = " ".join(" ".join(page.extract_text() for page in PdfReader(io.BytesIO(one_page)).pages).split())
    assert all(bullet in text for exp in resume.experience for bullet in exp.bullets)  # never trimmed


def test_a_run_saved_before_the_restyle_shows_in_the_new_look_with_every_bullet(client):
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()
    db = client.app.state.db
    with sqlite3.connect(db.path) as raw:  # as stored before the restyle: an old step and the stacked header
        stored = json.loads(raw.execute("SELECT result FROM runs WHERE id = ?", (run["run_id"],)).fetchone()[0])
        stored["style"] = {"font_pt": 10.0, "margin_in": 0.55, "paper": "Letter", "stacked": True}
        stored["pages"] = 2
        raw.execute("UPDATE runs SET result = ? WHERE id = ?", (json.dumps(stored), run["run_id"]))
    url = f"/api/runs/{run['run_id']}/resume.pdf"
    pdf = client.get(url).content
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) <= 2 and set(fonts_of(pdf)) <= {"Inter", "InterMedium", "InterSemiBold", "InterBold"}
    text = " ".join(" ".join(page.extract_text() for page in reader.pages).split())
    assert all(" ".join(bullet.split()) in text for exp in run["resume"]["experience"] for bullet in exp["bullets"])
    loads = FONTS.count
    assert client.get(url + "?download=1").content == pdf and FONTS.count == loads  # kept for the download
    saved = db.get_run(run["run_id"])
    assert saved.style == FitStyle(font_pt=10.0, margin_in=0.55) and saved.pages == 2  # nothing written back
    docx = Document(io.BytesIO(client.get(f"/api/runs/{run['run_id']}/resume.docx").content))
    assert docx.styles["Normal"].font.size.pt == round(current_style(saved.style).font_pt * 2) / 2  # the new look
