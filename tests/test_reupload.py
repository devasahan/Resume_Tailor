"""Re-uploads: a resume the app made, read back by the rule-based reader (Settings > reader "rules").

The lines are what pdfplumber extracts from the app's PDFs (extract_text in app/parsing/resume_parser.py, as uploads are
read). Since the restyle a job line starts with the title and joins its parts with ' · ', with en-dash dates on the
right; a line too long for the page breaks before the company or a place. Resumes made before put the company first,
with ' | ' and hyphens, and some stacked the title on its own line. Every fact the profile holds comes back: the name,
the contact details, each employer (name, dates, work mode, place), each school and each certification. The profile
keeps no job titles (each run writes its own), so a title must never come back as an employer or a place.
"""
import pytest

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
