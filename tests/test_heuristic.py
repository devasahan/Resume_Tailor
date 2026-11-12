from app.parsing.heuristic import (
    _parse_certifications,
    _parse_education,
    _reads_as_title,
    parse_resume_text,
)

from .conftest import SAMPLE_RESUME_TEXT


def experience(*lines):
    """How the rules read these lines under an EXPERIENCE heading: (name, start, end, mode, location) per employer."""
    profile = parse_resume_text("\n".join(["Taylor Quinn", "EXPERIENCE", *lines]))
    return [(c.name, c.start, c.end, c.mode, c.location) for c in profile.companies]


def education(*lines):
    return [(e.school, e.degree, e.field, e.start, e.end) for e in _parse_education(list(lines))]


def certifications(*lines):
    return [(c.name, c.issuer, c.year, c.expires) for c in _parse_certifications(list(lines))]


def test_parses_contact_companies_and_education():
    p = parse_resume_text(SAMPLE_RESUME_TEXT)
    assert p.name == "Jordan Rivera"
    assert p.contact.email == "jordan.rivera@example.com"
    assert p.contact.phone == "(555) 010-2030"
    assert p.contact.linkedin == "linkedin.com/in/jordanrivera"
    assert p.contact.location == "Austin, TX"
    assert [(c.name, c.start, c.end, c.mode) for c in p.companies] == [
        ("Northwind Labs", "Mar 2022", "Present", "Remote"),
        ("Contoso Systems", "Jun 2019", "Feb 2022", "Onsite"),
        ("Fabrikam Inc.", "Jul 2016", "May 2019", "Hybrid"),
    ]
    assert p.companies[1].location == "Dallas, TX"
    assert p.education[0].school == "University of Texas at Austin"
    assert (p.education[0].degree, p.education[0].field, p.education[0].end) == ("B.S.", "Computer Science", "2016")


def test_unknown_text_gives_empty_but_valid_profile():
    p = parse_resume_text("just some words without structure")
    assert p.companies == [] and p.contact.email == ""


def test_education_dates_keep_their_months_and_leave_the_school_name():
    """A real upload showed 'Fontbonne University Aug' with '2016 - 2020': only the years were taken out."""
    from app.parsing.heuristic import _parse_education

    def read(*lines):
        return [(e.school, e.degree, e.field, e.start, e.end) for e in _parse_education(list(lines))]

    assert read("Fontbonne University Aug 2016 – Aug 2020", "Bachelor's Degree in Computer Science") == [
        ("Fontbonne University", "Bachelor's Degree", "Computer Science", "Aug 2016", "Aug 2020")]
    # the dates on their own line under the school
    assert read("Fontbonne University", "Aug 2016 - Aug 2020", "B.S. in Computer Science") == [
        ("Fontbonne University", "B.S.", "Computer Science", "Aug 2016", "Aug 2020")]
    # one date on the degree line is the graduation date
    assert read("University of Texas at Austin", "B.S. in Computer Science, May 2016") == [
        ("University of Texas at Austin", "B.S.", "Computer Science", "", "May 2016")]
    assert read("Fontbonne University, St. Louis, MO   08/2016 – 05/2020") == [
        ("Fontbonne University, St. Louis, MO", "", "", "08/2016", "05/2020")]
    assert read("Fontbonne University Aug 2022 - Present")[0][3:] == ("Aug 2022", "Present")


# ------------------------------------------------------------------ job lines, title first (the app's own layout)
def test_a_job_title_reads_as_one_and_a_company_never_does():
    for title in ("Senior Data Engineer", "Data Engineer II", "Software Engineer 3", "Engineer L5", "Product Owner",
                  "Founder & CEO", "Tech Lead", "Registered Nurse", "ASP.NET Developer", "Software Engineer (Contract)",
                  "Software Engineer (Contract", "Head of Data", "Senior Director of Engineering", "VP Sales",
                  "VP, Engineering", "Senior Manager, Data Engineering", "Chief of Staff", "Member of Technical Staff"):
        assert _reads_as_title(title), title
    for other in ("Northwind Labs", "Bluefin Commerce", "Fabrikam Inc.", "Acme Consulting LLC", "Smith & Associates",
                  "Engineer.io", "Chief", "Principal Financial Group", "Austin, TX", "Remote",
                  # a part holding more than a title: a line split only by commas
                  "Graduate Analyst, Initech, Leeds", "Senior Data Engineer, Acme Ltd, London",
                  "Data Engineer, Initech"):
        assert not _reads_as_title(other), other


def test_title_first_job_lines_give_the_company():
    """The app's job line since the restyle: 'Title · Company · Location · Mode', the dates on the right (pdfplumber
    keeps one space or several before them). An upload's 'Title | Company' is read the same way."""
    assert experience(
        "Senior Data Engineer · Northwind Labs · Austin, TX · Remote   Mar 2022 – Present", "• Built the pipelines.",
        "Data Engineer II · Bluefin Commerce · Onsite Jun 2019 – Feb 2022", "• Moved the reports to Snowflake.",
        "Software Developer · Fabrikam Inc.   2016 – 2019", "• Wrote the order services.",
    ) == [("Northwind Labs", "Mar 2022", "Present", "Remote", "Austin, TX"),
          ("Bluefin Commerce", "Jun 2019", "Feb 2022", "Onsite", ""),
          ("Fabrikam Inc.", "2016", "2019", "", "")]
    assert experience("Senior Software Engineer | Contoso Systems | Seattle, WA   2019 - 2021",
                      "Head of Data · Northwind Labs 2017 – 2019",
                      "VP, Engineering | Bluefin Commerce 2015 - 2017") == [
        ("Contoso Systems", "2019", "2021", "", "Seattle, WA"), ("Northwind Labs", "2017", "2019", "", ""),
        ("Bluefin Commerce", "2015", "2017", "", "")]


def test_a_long_job_line_takes_the_rest_from_the_line_under_it():
    """A job line too long for one line breaks before the company or a place: the dated first line holds the title
    (maybe with the company), the line under it the rest, without a leading '· '."""
    bluefin = ("Bluefin Commerce", "Jun 2019", "Feb 2022", "Onsite", "Dallas, TX")
    after = ["• Moved the reports to Snowflake.", "Software Developer · Fabrikam Inc. · Hybrid   2016 – 2019"]
    fabrikam = ("Fabrikam Inc.", "2016", "2019", "Hybrid", "")
    assert experience("Principal Data Platform Engineer   Jun 2019 – Feb 2022",
                      "Bluefin Commerce · Dallas, TX · Onsite", *after) == [bluefin, fabrikam]
    assert experience("Principal Data Platform Engineer · Bluefin Commerce   Jun 2019 – Feb 2022",
                      "Dallas, TX · Onsite", *after) == [bluefin, fabrikam]
    assert experience("Principal Data Platform Engineer · Bluefin Commerce · Dallas, TX   Jun 2019 – Feb 2022",
                      "Onsite", *after) == [bluefin, fabrikam]
    assert experience("Principal Data Platform Engineer   Jun 2019 – Feb 2022", "Bluefin Commerce", *after)[0] == (
        "Bluefin Commerce", "Jun 2019", "Feb 2022", "", "")
    assert experience("Data Analyst Intern   Jun 2019 – Aug 2019", "Contoso Systems, Remote")[0] == (  # an upload's
        "Contoso Systems", "Jun 2019", "Aug 2019", "Remote", "")
    # a line under a title that is no rest of it: a bullet, a sentence, a sub-heading, another title, a date
    for under in ("• Built the data platform.", "Built the data platform for 20 teams.", "Key achievements:",
                  "Key Responsibilities", "Team: Data Platform", "Data Engineer II", "Northwind Labs   2017 - 2019"):
        assert experience("Senior Data Engineer   2019 - 2021", under)[0] == (
            "Senior Data Engineer", "2019", "2021", "", ""), under
    # under a title and its company, only a place or a work mode is the rest
    assert experience("Senior Data Engineer · Northwind Labs   2019 - 2021", "Data Platform Team") == [
        ("Northwind Labs", "2019", "2021", "", "")]


def test_a_job_line_with_one_date():
    """When only the end (or the start) is known the app prints that date alone: 'Present' is the end."""
    assert experience("Data Engineer · Northwind Labs · Remote   Present", "• Built the pipelines.",
                      "Data Analyst · Bluefin Commerce   Mar 2019") == [
        ("Northwind Labs", "", "Present", "Remote", ""), ("Bluefin Commerce", "", "Mar 2019", "", "")]
    # never a bullet, a line without a job title, or a year inside a name with nothing around it
    assert experience("• Promoted to Lead Engineer | Remote 2020", "Tools | Python, SQL | 2020",
                      "Senior Data Engineer 2020") == []


def test_dates_with_any_dash_are_read_alike():
    for dash in ("-", "–", "—"):
        assert experience(f"Data Engineer · Northwind Labs   Mar 2022 {dash} Present") == [
            ("Northwind Labs", "Mar 2022", "Present", "", "")], dash
        assert education(f"B.S., Computer Science · State University   2014 {dash} 2018") == [
            ("State University", "B.S.", "Computer Science", "2014", "2018")], dash
        row = f"Certified Kubernetes Administrator · The Linux Foundation   Mar 2023 {dash} Mar 2026"
        assert certifications(row) == [
            ("Certified Kubernetes Administrator", "The Linux Foundation", "Mar 2023", "Mar 2026")], dash


def test_comma_only_job_lines_stay_as_they_are():
    """A line split only by commas (a UK CV's 'Title, Company, Town') keeps all of it as the name, and the line under
    it is never read as its rest."""
    assert experience("Senior Data Engineer, Acme Ltd, London    Jan 2020 - Present", "Responsibilities:",
                      "- Own the data platform", "Graduate Analyst, Initech, Leeds    Sep 2014 - May 2016",
                      "Leeds Office") == [
        ("Senior Data Engineer, Acme Ltd, London", "Jan 2020", "Present", "", ""),
        ("Graduate Analyst, Initech, Leeds", "Sep 2014", "May 2016", "", "")]


def test_the_apps_older_layouts_still_read_right():
    """Resumes made before the restyle: the company first, ' | ' and hyphens; stacked, the title on its own line."""
    northwind = ("Northwind Labs", "Mar 2022", "Present", "Remote", "Austin, TX")
    bluefin = ("Bluefin Commerce", "Jun 2019", "Feb 2022", "Onsite", "")
    assert experience("Northwind Labs | Senior Data Engineer | Austin, TX | Remote   Mar 2022 - Present",
                      "• Built the pipelines.", "Bluefin Commerce | Data Engineer II | Onsite Jun 2019 - Feb 2022",
                      "• Moved the reports.") == [northwind, bluefin]
    assert experience("Northwind Labs | Austin, TX | Remote   Mar 2022 - Present", "Senior Data Engineer",
                      "• Built the pipelines.", "Bluefin Commerce | Onsite Jun 2019 - Feb 2022", "Data Engineer II",
                      "• Moved the reports.") == [northwind, bluefin]
    # a company that reads like a title keeps its place, and the stacked title under it stays a title
    assert experience("Acme Consultant | Remote   2019 - 2021", "Data Engineer II", "• Built it.") == [
        ("Acme Consultant", "2019", "2021", "Remote", "")]


def test_a_job_title_is_never_a_section_heading():
    """Job lines start with the title now: 'History Teacher' or 'Experience Designer' must not start a section, on a job
    line or as the headline under the name (the contact line under it stays in the header)."""
    profile = parse_resume_text("""Taylor Quinn
Experience Designer
Austin, TX · taylor.quinn@example.com
EXPERIENCE
Experience Designer · Northwind Labs   2019 – 2021
• Designed the onboarding.
History Teacher   2015 – 2019
Lakeside High School
• Taught three classes.
EDUCATION
B.A., History · State University   2011 – 2015
""")
    assert [(c.name, c.start, c.end) for c in profile.companies] == [
        ("Northwind Labs", "2019", "2021"), ("Lakeside High School", "2015", "2019")]
    assert profile.contact.location == "Austin, TX"
    assert [(e.school, e.degree, e.field) for e in profile.education] == [("State University", "B.A.", "History")]


# ------------------------------------------------------------------ education and certification rows
def test_education_splits_degree_field_and_school():
    """The app's row is 'Degree, Field · School   dates'; the school was read as 'B.S., Computer Science' before."""
    assert education("B.S., Computer Science · State University   2014 – 2018", "Lakeside Community College",
                     "Dean's list") == [("State University", "B.S.", "Computer Science", "2014", "2018"),
                                        ("Lakeside Community College", "", "", "", "")]
    assert education("Bachelor of Science, Computer Science | State University 2018") == [
        ("State University", "Bachelor of Science", "Computer Science", "", "2018")]
    assert education("MBA · Northwind Business School   2016") == [("Northwind Business School", "MBA", "", "", "2016")]
    assert education("State University | B.S. in Computer Science | 2018") == [
        ("State University", "B.S.", "Computer Science", "", "2018")]
    # the part naming a school and no degree is the school
    assert education("Master of Science, School of Engineering · State University   2016") == [
        ("State University", "Master of Science", "School of Engineering", "", "2016")]
    # a degree line under its school is cut at a comma as well as at ' in '
    assert education("State University", "Bachelor of Science, Computer Science   2018") == [
        ("State University", "Bachelor of Science", "Computer Science", "", "2018")]


def test_certification_rows_give_name_issuer_earned_and_expiry():
    """The app's row is 'Name · Issuer   dates' (' | ' in older resumes): the dates no longer join the issuer."""
    assert certifications(
        "AWS Certified Solutions Architect – Associate · Amazon Web Services   Mar 2023 – Mar 2026",
        "Certified ScrumMaster (CSM) · Scrum Alliance 2020",
        "Certified Kubernetes Administrator · The Linux Foundation   Expires Mar 2027",
        "AWS Certified Developer – Associate | Amazon Web Services Mar 2024 - Mar 2027",
        "Oracle Certified Professional, Java SE 11 Developer · Oracle   2020",  # a name keeps its comma
        "Certified Kubernetes Application Developer (CKAD) · The Linux Foundation   Mar 2024 – 03/2026",
        "HashiCorp Certified: Terraform Associate   Mar 2024",  # no issuer: a month and year end no name
    ) == [
        ("AWS Certified Solutions Architect – Associate", "Amazon Web Services", "Mar 2023", "Mar 2026"),
        ("Certified ScrumMaster (CSM)", "Scrum Alliance", "2020", ""),
        ("Certified Kubernetes Administrator", "The Linux Foundation", "", "Mar 2027"),
        ("AWS Certified Developer – Associate", "Amazon Web Services", "Mar 2024", "Mar 2027"),
        ("Oracle Certified Professional, Java SE 11 Developer", "Oracle", "2020", ""),
        ("Certified Kubernetes Application Developer (CKAD)", "The Linux Foundation", "Mar 2024", "03/2026"),
        ("HashiCorp Certified: Terraform Associate", "", "Mar 2024", ""),
    ]
    # a year right after the name is part of it; after a separator it is the date earned
    assert certifications("MCSA: Windows Server 2016", "MCSA: SQL Server 2012 · Microsoft   2018",
                          "HubSpot Inbound Marketing 2020", "CKAD · The Linux Foundation   Expires 03/2026") == [
        ("MCSA: Windows Server 2016", "", "", ""), ("MCSA: SQL Server 2012", "Microsoft", "2018", ""),
        ("HubSpot Inbound Marketing 2020", "", "", ""), ("CKAD", "The Linux Foundation", "", "03/2026")]


def test_skills_rows_and_the_dotted_contact_line_read_as_before():
    """The new header joins the contact line with ' · ' (wrapped between items when long) and the skills rows have a
    label column: the reading is the same as with ' | '."""
    lines = ["Taylor Quinn", "Data Engineer",
             "Austin, TX · (555) 010-4477 · taylor.quinn@example.com",
             "linkedin.com/in/taylorquinn · taylorquinn.dev",
             "SUMMARY", "Data engineer with ten years of pipelines.",
             "SKILLS", "Languages:     Python, SQL, Scala", "Cloud: AWS, Snowflake",
             "EXPERIENCE", "Data Engineer · Northwind Labs · Remote   Mar 2022 – Present", "• Built the pipelines."]
    profile = parse_resume_text("\n".join(lines))
    assert (profile.name, profile.contact.email, profile.contact.phone, profile.contact.linkedin,
            profile.contact.location) == ("Taylor Quinn", "taylor.quinn@example.com", "(555) 010-4477",
                                          "linkedin.com/in/taylorquinn", "Austin, TX")
    assert [c.name for c in profile.companies] == ["Northwind Labs"] and profile.certifications == []
    assert parse_resume_text("\n".join(line.replace(" · ", " | ") for line in lines)) == profile
