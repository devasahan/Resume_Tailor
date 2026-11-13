"""Re-uploads: a resume the app made, read back by the rule-based reader (Settings > reader "rules").

The lines are what pdfplumber extracts from the app's PDFs (extract_text in app/parsing/resume_parser.py, as uploads are
read). Since the restyle a job line starts with the title and joins its parts with ' · ', with en-dash dates on the
right; a line too long for the page breaks before the company or a place. Resumes made before put the company first,
with ' | ' and hyphens, and some stacked the title on its own line. Every fact the profile holds comes back: the name,
the contact details, each employer (name, dates, work mode, place), each school and each certification. The profile
keeps no job titles (each run writes its own), so a title must never come back as an employer or a place.
"""
import io
import time

import pytest
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from app.parsing.heuristic import parse_resume_text
from app.parsing.resume_parser import extract_text
from app.pipeline.fit import LADDER, styles
from app.render.docx import resume_docx
from app.render.pdf import render_resume_pdf
from app.schemas import (
    Certification,
    Company,
    Contact,
    Education,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
)

NAME = "Taylor Quinn"
CONTACT = Contact(email="taylor.quinn@example.com", phone="(555) 010-4477", linkedin="linkedin.com/in/taylorquinn",
                  location="Austin, TX")
COMPANIES = [
    Company(name="Northwind Labs", start="Mar 2022", end="Present", mode="Remote", location="Austin, TX"),
    Company(name="Bluefin Commerce", start="Jun 2019", end="Feb 2022", mode="Onsite", location="Dallas, TX"),
    Company(name="Fabrikam Inc.", start="2016", end="2019", mode="Hybrid"),
    Company(name="Contoso Systems", start="Jan 2014", end="May 2016"),
]
TITLES = ["Senior Data Engineer", "Principal Data Platform Engineer", "Software Developer", "Junior Analyst"]
EDUCATION = [
    Education(school="State University", degree="B.S.", field="Computer Science", start="2010", end="2014"),
    Education(school="Lakeside Community College"),
]
CERTIFICATIONS = [
    Certification(name="AWS Certified Solutions Architect – Associate", issuer="Amazon Web Services", year="Mar 2023",
                  expires="Mar 2026"),
    Certification(name="Certified ScrumMaster (CSM)", issuer="Scrum Alliance", year="2020"),
    Certification(name="Certified Kubernetes Administrator", expires="Mar 2027"),
    Certification(name="HashiCorp Certified: Terraform Associate", year="Mar 2024"),
]

TOP = ["Taylor Quinn", "Data Engineer"]
MIDDLE = ["SUMMARY", "Data engineer with ten years of pipelines for product and finance teams.",
          "SKILLS", "Languages:{gap}Python, SQL, Scala", "Cloud:{gap}AWS, Snowflake, Airflow", "EXPERIENCE"]
BULLETS = ["• Built streaming pipelines in Python and Kafka for the finance team.",
           "• Moved 120 reports to Snowflake and retired the old warehouse.",
           "• Wrote the order services in Java.",
           "• Cleaned the sales data in Excel every week."]


def new_design(gap: str = "   ", wrapped_contact: bool = False, long_break: str = "company") -> str:
    """The text of a resume made since the restyle. `gap` is what extraction leaves of the filler before the dates (one
    space or several); the second job's line is too long and breaks before the company or before the place."""
    contact = ["Austin, TX · (555) 010-4477 · taylor.quinn@example.com",
               "linkedin.com/in/taylorquinn · taylorquinn.dev"]
    second = {
        "company": [f"Principal Data Platform Engineer{gap}Jun 2019 – Feb 2022",
                    "Bluefin Commerce · Dallas, TX · Onsite"],
        "place": [f"Principal Data Platform Engineer · Bluefin Commerce{gap}Jun 2019 – Feb 2022",
                  "Dallas, TX · Onsite"],
    }[long_break]
    lines = [
        *TOP, *(contact if wrapped_contact else [" · ".join(contact)]), *(line.format(gap=gap) for line in MIDDLE),
        f"Senior Data Engineer · Northwind Labs · Austin, TX · Remote{gap}Mar 2022 – Present", BULLETS[0],
        *second, BULLETS[1],
        f"Software Developer · Fabrikam Inc. · Hybrid{gap}2016 – 2019", BULLETS[2],
        f"Junior Analyst · Contoso Systems{gap}Jan 2014 – May 2016", BULLETS[3],
        "EDUCATION",
        f"B.S., Computer Science · State University{gap}2010 – 2014",
        "Lakeside Community College", "Dean's list",
        "CERTIFICATIONS",
        f"AWS Certified Solutions Architect – Associate · Amazon Web Services{gap}Mar 2023 – Mar 2026",
        f"Certified ScrumMaster (CSM) · Scrum Alliance{gap}2020",
        f"Certified Kubernetes Administrator{gap}Expires Mar 2027",
        f"HashiCorp Certified: Terraform Associate{gap}Mar 2024",
    ]
    return "\n".join(lines)


def old_design(stacked: bool = False) -> str:
    """The text of a resume made before the restyle, as pdfplumber extracts it (one space before the dates)."""
    jobs = [("Northwind Labs", "Austin, TX | Remote", "Mar 2022 - Present"),
            ("Bluefin Commerce", "Dallas, TX | Onsite", "Jun 2019 - Feb 2022"),
            ("Fabrikam Inc.", "Hybrid", "2016 - 2019"),
            ("Contoso Systems", "", "Jan 2014 - May 2016")]
    experience = []
    for (company, place, dates), title, bullet in zip(jobs, TITLES, BULLETS):
        place = f" | {place}" if place else ""
        rows = [f"{company}{place} {dates}", title] if stacked else [f"{company} | {title}{place} {dates}"]
        experience += [*rows, bullet]
    lines = [
        *TOP, "Austin, TX | (555) 010-4477 | taylor.quinn@example.com | linkedin.com/in/taylorquinn | taylorquinn.dev",
        *(line.format(gap=" ") for line in MIDDLE), *experience,
        "EDUCATION", "B.S., Computer Science | State University 2010 - 2014", "Lakeside Community College",
        "Dean's list",
        "CERTIFICATIONS",
        "AWS Certified Solutions Architect – Associate | Amazon Web Services Mar 2023 - Mar 2026",
        "Certified ScrumMaster (CSM) | Scrum Alliance 2020",
        "Certified Kubernetes Administrator Expires Mar 2027",
        "HashiCorp Certified: Terraform Associate Mar 2024",
    ]
    return "\n".join(lines)


def assert_reads_back(text: str) -> None:
    profile = parse_resume_text(text)
    assert profile.name == NAME
    assert (profile.contact.email, profile.contact.phone, profile.contact.linkedin, profile.contact.location) == (
        CONTACT.email, CONTACT.phone, CONTACT.linkedin, CONTACT.location)
    assert [c.model_dump() for c in profile.companies] == [c.model_dump() for c in COMPANIES]
    places = {value for company in profile.companies for value in (company.name, company.location)}
    assert not places & set(TITLES)  # a job title is never read as an employer or a place
    assert [(e.school, e.degree, e.field, e.start, e.end) for e in profile.education] == [
        (e.school, e.degree, e.field, e.start, e.end) for e in EDUCATION]
    assert [c.model_dump() for c in profile.certifications] == [c.model_dump() for c in CERTIFICATIONS]


@pytest.mark.parametrize("gap", ["   ", " "])
@pytest.mark.parametrize("long_break", ["company", "place"])
def test_a_resume_in_the_new_design_reads_back(gap, long_break):
    assert_reads_back(new_design(gap=gap, long_break=long_break))


def test_a_wrapped_contact_line_reads_back():
    """A long contact line breaks between its items, onto a second line."""
    assert_reads_back(new_design(wrapped_contact=True))


@pytest.mark.parametrize("stacked", [False, True])
def test_a_resume_made_before_the_restyle_reads_back(stacked):
    assert_reads_back(old_design(stacked=stacked))


def test_both_designs_read_alike():
    assert parse_resume_text(new_design()) == parse_resume_text(old_design())


# ---------------------------------------------------------------- the real files, as an upload reads them
def app_made_resume() -> Resume:
    """The same person as a resume the app makes: the second job's title is long, so its line breaks before the
    company; the contact line is long too, so it breaks between its items."""
    titles = list(TITLES)
    titles[1] = "Principal Data Platform and Analytics Infrastructure Engineer"
    contact = CONTACT.model_copy(update={"linkedin": "linkedin.com/in/taylor-quinn-data-engineer-austin",
                                         "website": "taylorquinn-data-engineering-portfolio.dev"})
    education = [EDUCATION[0], EDUCATION[1].model_copy(update={"details": "Dean's list"})]
    profile = Profile(name=NAME, contact=contact, companies=COMPANIES, education=education,
                      certifications=CERTIFICATIONS)
    return Resume(
        profile=profile, headline="Data Engineer",
        summary="Data engineer with ten years of pipelines for product and finance teams.",
        skills=[SkillGroup(category="Languages", skills=["Python", "SQL", "Scala"]),
                SkillGroup(category="Cloud", skills=["AWS", "Snowflake", "Airflow"])],
        experience=[ResumeExperience(company_index=i, job_title=title, bullets=[bullet.removeprefix("• ")])
                    for i, (title, bullet) in enumerate(zip(titles, BULLETS))],
        certifications=CERTIFICATIONS,
    )


def assert_file_reads_back(text: str, resume: Resume) -> None:
    profile = parse_resume_text(text)
    contact = resume.profile.contact
    assert profile.name == NAME
    assert (profile.contact.email, profile.contact.phone, profile.contact.linkedin, profile.contact.location) == (
        contact.email, contact.phone, contact.linkedin, contact.location)
    assert [c.model_dump() for c in profile.companies] == [c.model_dump() for c in COMPANIES]
    titles = {e.job_title for e in resume.experience}
    assert not {value for c in profile.companies for value in (c.name, c.location)} & titles
    assert [(e.school, e.degree, e.field, e.start, e.end) for e in profile.education] == [
        (e.school, e.degree, e.field, e.start, e.end) for e in EDUCATION]
    assert [c.model_dump() for c in profile.certifications] == [c.model_dump() for c in CERTIFICATIONS]


@pytest.mark.parametrize("paper", ["Letter", "A4"])
@pytest.mark.parametrize("step", [0, len(LADDER) - 1, len(LADDER), len(LADDER) + 1])
def test_a_pdf_the_app_made_reads_back_as_an_upload(paper, step):
    """render -> extract_text (pdfplumber, as an upload is read) -> the rules: every fact comes back, at the design's
    sizes, at the floor and at the last two steps (the floor with closer gaps, then with the skills in line too), where
    the lines break in other places."""
    resume = app_made_resume()
    pdf, _pages = render_resume_pdf(resume, styles(paper)[step])
    text = extract_text(pdf, "resume.pdf")
    if step == 0:  # at the design's sizes the long job line breaks, before the company or before the place
        assert "\nBluefin Commerce · Dallas" in text or "\nDallas, TX · Onsite" in text
    assert_file_reads_back(text, resume)


def test_a_word_file_the_app_made_reads_back_as_an_upload():
    """The Word copy puts the dates on a tab stop: the reader takes a tab as the gap before them."""
    resume = app_made_resume()
    text = extract_text(resume_docx(resume, styles("Letter")[0]), "resume.docx")
    assert "\t" in text
    assert_file_reads_back(text, resume)


# ---------------------------------------------------------------- job titles the rules do not know
# (job title, employer) per job. The rules know a title by its last word ('Engineer', 'Analyst') or its rank ('Head
# of'); the app's line starts with the title whatever it is, so its line is read by position.
UNKNOWN_TITLES = {
    "titles": [("Scrum Master", COMPANIES[0]), ("SDE II", COMPANIES[1]),
               ("Software Engineer, Payments", COMPANIES[2]), ("Engineer – Data Platform", COMPANIES[3])],
    # employers that read like a job title ('Bluefin Coach') or like a place ('Rivera, Quinn' as 'Austin, Texas')
    "employers": [("Barista", COMPANIES[0].model_copy(update={"name": "The Lakeside Economist"})),
                  ("Sales Associate", COMPANIES[1].model_copy(update={"name": "Bluefin Coach"})),
                  ("Associate", COMPANIES[2].model_copy(update={"name": "Rivera, Quinn"})),
                  ("Cashier", COMPANIES[3].model_copy(update={"name": "Brooks, Rivera", "location": "New York, NY"}))],
    # titles too long to share a line with their employer: the line breaks before the employer or its place
    "long titles": [("Principal Agile Delivery and Release Train Scrum Master", COMPANIES[0]),
                    ("QA Tester", COMPANIES[1]),
                    ("Lead Barista and Coffee Program Mentor for the Downtown Flagship Store",
                     COMPANIES[2].model_copy(update={"name": "Rivera, Quinn", "location": "New York, NY"})),
                    ("Nurse Practitioner, Pediatric Intensive Care and Emergency Medicine", COMPANIES[3])],
}


@pytest.mark.parametrize("jobs", UNKNOWN_TITLES.values(), ids=UNKNOWN_TITLES.keys())
def test_job_titles_the_rules_do_not_know_read_back_from_the_apps_files(jobs):
    """Each employer comes back with its dates, place and work mode, from the PDF (both papers, the design's sizes,
    the floor and the last two steps) and the Word copy, never a job title in its place."""
    base = app_made_resume()
    profile = base.profile.model_copy(update={"companies": [company for _title, company in jobs]})
    experience = [exp.model_copy(update={"job_title": title}) for exp, (title, _company) in zip(base.experience, jobs)]
    resume = base.model_copy(update={"profile": profile, "experience": experience})
    texts = {f"{paper} step {step}": extract_text(render_resume_pdf(resume, styles(paper)[step])[0], "resume.pdf")
             for paper in ("Letter", "A4") for step in (0, len(LADDER) - 1, len(LADDER), len(LADDER) + 1)}
    texts["Word"] = extract_text(resume_docx(resume, styles("Letter")[0]), "resume.docx")
    if jobs is UNKNOWN_TITLES["long titles"]:  # at the design's sizes the long lines break
        assert "\nAustin, TX · Remote\n" in texts["Letter step 0"]
        assert "\nRivera, Quinn · New York, NY · Hybrid\n" in texts["A4 step 0"]
    for label, text in texts.items():
        assert [c.model_dump() for c in parse_resume_text(text).companies] == [
            company.model_dump() for _title, company in jobs], label


# ---------------------------------------------------------------- the reader's second review
def test_a_bullet_holding_a_range_is_no_employer_in_the_apps_files():
    """A bullet with a range of numbers or years ('from 2000 to 9000', 'the 2019 – 2021 move') reads as a sentence, so
    it is no job line: in the PDF, and in the Word copy, whose bullets get their mark from their List style."""
    base = app_made_resume()
    bullets = ["Scaled the order API from 2000 to 9000 requests per second.",
               "Led the 2019 – 2021 move of 40 services to AWS."]
    experience = [base.experience[0].model_copy(update={"bullets": bullets}), *base.experience[1:]]
    resume = base.model_copy(update={"experience": experience})
    pdf = extract_text(render_resume_pdf(resume, styles("Letter")[0])[0], "resume.pdf")
    word = extract_text(resume_docx(resume, styles("Letter")[0]), "resume.docx")
    for label, text in (("PDF", pdf), ("Word", word)):
        assert "\n• Led the 2019 – 2021 move of 40 services to AWS.\n" in text, label
        assert [c.model_dump() for c in parse_resume_text(text).companies] == [
            c.model_dump() for c in COMPANIES], label


def test_a_job_title_that_wraps_reads_back():
    """A title too long for the line beside its dates (77 characters on A4 at the design's sizes) wraps by words where
    it is drawn: its last word comes on a line of its own, over the rest of the job line. The employer is that rest's,
    never the end of the title."""
    title = "Senior Software Engineer, Payments Infrastructure and Fraud Detection Systems"
    fourth = Company(name="Fourth Coffee", start="Jan 2020", end="11/2021", mode="Onsite", location="Seattle, WA")
    base = app_made_resume()
    profile = base.profile.model_copy(update={"companies": [fourth, *COMPANIES[1:]]})
    experience = [base.experience[0].model_copy(update={"job_title": title}), *base.experience[1:]]
    resume = base.model_copy(update={"profile": profile, "experience": experience})
    texts = {f"{paper} step {step}": extract_text(render_resume_pdf(resume, styles(paper)[step])[0], "resume.pdf")
             for paper in ("Letter", "A4") for step in (0, len(LADDER), len(LADDER) + 1)}
    texts["Word"] = extract_text(resume_docx(resume, styles("Letter")[0]), "resume.docx")
    assert "\nSystems\nFourth Coffee · Seattle, WA · Onsite\n" in texts["A4 step 0"]
    for label, text in texts.items():
        assert [c.model_dump() for c in parse_resume_text(text).companies] == [
            c.model_dump() for c in profile.companies], label


def test_a_certification_with_a_year_and_no_issuer_reads_back_from_the_word_copy():
    """The Word copy puts a certification's dates on a tab stop: in 'Six Sigma Green Belt<TAB>2020' the year after the
    gap is the date it was earned. (The PDF's 'Six Sigma Green Belt 2020' keeps the year in the name, as a year one
    space after a name may be part of it: 'MCSA: Windows Server 2016'.)"""
    six_sigma = Certification(name="Six Sigma Green Belt", year="2020")
    certifications = [six_sigma, *CERTIFICATIONS]
    base = app_made_resume()
    resume = base.model_copy(update={
        "profile": base.profile.model_copy(update={"certifications": certifications}),
        "certifications": certifications})
    text = extract_text(resume_docx(resume, styles("Letter")[0]), "resume.docx")
    assert "\nSix Sigma Green Belt\t2020\n" in text
    assert [c.model_dump() for c in parse_resume_text(text).certifications] == [
        c.model_dump() for c in certifications]


# ---------------------------------------------------------------- the reader's third review: people's Word files
def numbered(paragraph, num_id: str = "1"):
    """Word's own bullet or number on a paragraph (its w:numPr, as the Bullets button gives it); numId 0 is none."""
    num = OxmlElement("w:numPr")
    for tag, value in (("w:ilvl", "0"), ("w:numId", num_id)):
        child = OxmlElement(tag)
        child.set(qn("w:val"), value)
        num.append(child)
    paragraph._p.get_or_add_pPr().append(num)
    return paragraph


def word_upload(document: Document) -> str:
    data = io.BytesIO()
    document.save(data)
    return extract_text(data.getvalue(), "resume.docx")


def word_resume(*paragraphs) -> Document:
    """Taylor Quinn's invented Word resume: the header, then (text, style, numbering) per paragraph."""
    document = Document()
    document.add_paragraph("Taylor Quinn")
    document.add_paragraph("taylor.quinn@example.com | (555) 010-4477 | Seattle, WA")
    for text, style, num_id in paragraphs:
        paragraph = document.add_paragraph(text, style=style)
        if num_id is not None:
            numbered(paragraph, num_id)
    return document


def test_a_word_paragraph_gets_its_bullet_only_where_word_draws_one():
    """A paragraph in a List style is marked '• ' only when Word draws a bullet or number before it: its own
    numbering, or its style's ('List Bullet'). A heading or job line left in List Paragraph with its bullet taken off
    (numId 0), or in List Continue, has none on the page, so it is still a heading or a job line."""
    text = word_upload(word_resume(
        ("EXPERIENCE", None, None), ("Northwind Labs | Data Engineer\t2019 – Present", None, None),
        ("Built the reporting pipeline for the finance team.", "List Paragraph", "1"),
        ("Bluefin Commerce — Data Analyst (contract)\t2016 – 2019", "List Continue", None),
        ("Wrote the weekly sales reports.", "List Bullet", None),
        ("Store Manager\t2013 – 2016", "List Paragraph", "0"), ("Contoso Market, Seattle, WA", "List Paragraph", "0"),
        ("Ran the store.", "List Paragraph", "1"),
        ("EDUCATION", "List Paragraph", "0"), ("State University | B.S. in Computer Science\t2009 – 2013", None, None)))
    assert "\n• Built the reporting pipeline for the finance team.\n• " not in text
    assert "\n• Wrote the weekly sales reports.\n" in text and "\n• Ran the store.\n" in text
    assert "\nEDUCATION\n" in text and "\nStore Manager\t2013 – 2016\n" in text
    profile = parse_resume_text(text)
    assert [(c.name, c.start, c.end) for c in profile.companies] == [
        ("Northwind Labs", "2019", "Present"), ("Bluefin Commerce", "2016", "2019"),
        ("Contoso Market, Seattle, WA", "2013", "2016")]
    assert [(e.school, e.degree) for e in profile.education] == [("State University", "B.S.")]


@pytest.mark.parametrize("style", ["List Paragraph", "List Number"])
def test_section_headings_word_numbers_stay_headings(style):
    """Headings Word bullets or numbers ('1. Work Experience'): a heading is never marked, so the upload is read (it was
    refused for having no employer)."""
    num_id = "1" if style == "List Paragraph" else None
    text = word_upload(word_resume(
        ("WORK EXPERIENCE", style, num_id), ("Bluefin Commerce, Austin, TX\tMar 2021 – Present", None, None),
        ("Store Manager", None, None), ("Ran a team of 14 across two shifts.", "List Paragraph", "1"),
        ("Fourth Coffee, Austin, TX\tJun 2017 – Feb 2021", None, None), ("Shift Supervisor", None, None),
        ("Education", style, num_id), ("Lakeside Community College\t2015 – 2017", None, None),
        ("Associate of Arts, Business", None, None),
        ("Certifications", style, num_id), ("ServSafe Food Handler, National Restaurant Association, 2019", None, None)))
    profile = parse_resume_text(text)
    assert [(c.name, c.start, c.end) for c in profile.companies] == [
        ("Bluefin Commerce, Austin, TX", "Mar 2021", "Present"), ("Fourth Coffee, Austin, TX", "Jun 2017", "Feb 2021")]
    assert [e.school for e in profile.education] == ["Lakeside Community College"]
    assert [c.name for c in profile.certifications] == ["ServSafe Food Handler"]


def test_a_bulleted_degree_and_certifications_line_read_from_a_word_file():
    """Word bullets on a degree under its school and on 'Certifications: ...' in Skills: 'B.S.' (not '• B.S.'), and
    both certifications."""
    profile = parse_resume_text(word_upload(word_resume(
        ("EXPERIENCE", None, None), ("Northwind Labs | Data Engineer\t2019 – Present", None, None),
        ("EDUCATION", None, None), ("State University\t2010 – 2014", None, None),
        ("B.S. in Computer Science", "List Paragraph", "1"), ("SKILLS", None, None),
        ("Certifications: AWS Certified Solutions Architect – Associate, Certified ScrumMaster (CSM)", "List Paragraph",
         "1"))))
    assert [(e.school, e.degree, e.field) for e in profile.education] == [("State University", "B.S.", "Computer Science")]
    assert [c.name for c in profile.certifications] == [
        "AWS Certified Solutions Architect – Associate", "Certified ScrumMaster (CSM)"]


def test_word_styles_are_read_once_and_never_stop_an_upload():
    """The styles are read once per file, not once per paragraph: a file with a thousand styles and thousands of
    paragraphs reads at once (it took seconds, longer with more). A style whose id holds a double quote (python-docx
    keeps it from the name) no longer stops the upload."""
    document = word_resume(("EXPERIENCE", None, None))
    quoted = document.styles.add_style('Job "Line"', WD_STYLE_TYPE.PARAGRAPH)
    document.add_paragraph("Northwind Labs | Data Engineer\t2019 – Present", style=quoted)
    assert [c.name for c in parse_resume_text(word_upload(document)).companies] == ["Northwind Labs"]
    for index in range(1000):
        document.styles.add_style(f"Extra {index}", WD_STYLE_TYPE.PARAGRAPH)
    for _ in range(3000):
        document.add_paragraph("")
    started = time.perf_counter()
    assert "\nNorthwind Labs | Data Engineer\t2019 – Present" in word_upload(document)
    assert time.perf_counter() - started < 1.5


# ---------------------------------------------------------------- the reader's third review: the app's own files
def small_resume(jobs: list[tuple[str, Company]]) -> Resume:
    profile = Profile(name=NAME, contact=CONTACT, companies=[company for _title, company in jobs])
    return Resume(profile=profile, headline="Store Lead", summary="Practical professional.",
                  skills=[SkillGroup(category="Tools", skills=["Point of Sale", "Inventory"])],
                  experience=[ResumeExperience(company_index=i, job_title=title, bullets=["Served 200 guests a day."])
                              for i, (title, _company) in enumerate(jobs)])


def every_render(resume: Resume) -> dict[str, str]:
    texts = {f"{paper} step {step}": extract_text(render_resume_pdf(resume, style)[0], "resume.pdf")
             for paper in ("Letter", "A4") for step, style in enumerate(styles(paper))}
    texts.update({f"Word {paper}": extract_text(resume_docx(resume, styles(paper)[0]), "resume.docx")
                  for paper in ("Letter", "A4")})
    return texts


LONG_TITLE = ("Customer Experience Associate for Premium Accounts, Loyalty Programs and In-Store Events Across Three "
              "Locations")


def test_long_titles_over_their_companies_read_back():
    """The app breaks a long title's line before its company. One such line over an employer named like a role
    ('Chief Tailspin Outfitters', a known limit) no longer turns the section's other long titles into employers; and
    a company with no place or work mode under a broken line keeps its name when the section's broken lines show the
    app's layout. (At the design's sizes the 83-character title wraps, and its last word is read as the employer.)"""
    lakeside = Company(name="Lakeside Unified School District", start="2016", end="2019", location="Paris, France")
    fourth = Company(name="Fourth Coffee", start="2014", end="2016")
    resume = small_resume([
        ("Head of Platform Engineering, Developer Experience, Observability and Site Reliability",
         Company(name="Chief Tailspin Outfitters", start="2019", end="Present", location="London, UK", mode="Remote")),
        (LONG_TITLE, lakeside), ("Sales Associate", fourth)])
    for label, text in every_render(resume).items():
        assert [c.model_dump() for c in parse_resume_text(text).companies[1:]] == [
            lakeside.model_dump(), fourth.model_dump()], label
    northwind = Company(name="Northwind Labs", start="11/2022", end="Present")
    resume = small_resume([
        ("Lead Teacher for Early Childhood Literacy and Bilingual Family Engagement Programs", northwind),
        ("Lead Barista and Coffee Program Mentor for the Downtown Flagship Store",
         Company(name="Fourth Coffee", start="Feb 2020", end="Sep 2022", location="Seattle, WA", mode="Onsite"))])
    for label, text in every_render(resume).items():
        if " step 0" not in label:
            assert parse_resume_text(text).companies[0].model_dump() == northwind.model_dump(), label


def test_the_apps_job_with_one_date_and_a_title_holding_a_work_mode_read_back():
    """A job with only its end date ('Principal · Bluefin Analytics · Onsite   Feb 2022') is read back with that
    date, although the rules know neither part as a title; 'Hybrid Cloud Engineer' gives a job with no work mode none."""
    jobs = [("Data Analyst", Company(name="Northwind Labs", start="Mar 2022", end="Present", location="Austin, TX")),
            ("Principal", Company(name="Bluefin Analytics", start="", end="Feb 2022", mode="Onsite")),
            ("Hybrid Cloud Engineer", Company(name="Contoso Systems", start="2016", end="2019", location="Seattle, WA"))]
    resume = small_resume(jobs)
    for label, text in (("PDF", extract_text(render_resume_pdf(resume, styles("Letter")[0])[0], "resume.pdf")),
                        ("Word", extract_text(resume_docx(resume, styles("Letter")[0]), "resume.docx"))):
        assert [c.model_dump() for c in parse_resume_text(text).companies] == [
            company.model_dump() for _title, company in jobs], label
