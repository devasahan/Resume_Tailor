import time

from app.parsing.heuristic import (
    _is_header,
    _parse_certifications,
    _parse_education,
    _reads_as_title,
    parse_resume_text,
    read_certification_list,
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


# ------------------------------------------------------------------ the restyle's review: what the rules read back
def test_the_apps_job_line_is_read_by_position_whatever_the_title():
    """The app writes 'Title · Company · Place · Mode', the title first: a title the rules do not know ('Scrum
    Master', 'SDE II') is never the company, and the line is split only at ' · ' ('Engineer – Data Platform' stays
    whole). A person's own 'Company · Title' lines keep their company when the lines that show the order plainly (a
    title the rules know, a company mark) put it first; where no line shows it, they read as the app's lines."""
    assert experience(
        "Scrum Master · Northwind Labs · Austin, TX · Remote   Mar 2022 – Present", "• Ran the sprints.",
        "SDE II · Bluefin Commerce · Dallas, TX · Onsite   Jun 2019 – Feb 2022", "• Built the order services.",
        "Software Engineer, Payments · Fabrikam Inc. · Hybrid   2016 – 2019", "• Wrote the payment flows.",
        "Engineer – Data Platform · Contoso Systems   Jan 2014 – May 2016", "• Cleaned the sales data.",
    ) == [("Northwind Labs", "Mar 2022", "Present", "Remote", "Austin, TX"),
          ("Bluefin Commerce", "Jun 2019", "Feb 2022", "Onsite", "Dallas, TX"),
          ("Fabrikam Inc.", "2016", "2019", "Hybrid", ""), ("Contoso Systems", "Jan 2014", "May 2016", "", "")]
    # employers that read like a title or like a place ('Rivera, Quinn' as 'Austin, Texas')
    assert experience(
        "Barista · The Lakeside Economist · London, UK   Mar 2022 – Present", "• Made the coffee.",
        "Sales Associate · Bluefin Coach · New York, NY   2018 – 2020", "• Sold the bags.",
        "Associate · Rivera, Quinn · New York, NY · Onsite   2016 – 2018", "• Drafted the contracts.",
        "Associate · Rivera, Quinn · Onsite   2014 – 2016", "• Drafted the briefs.",
    ) == [("The Lakeside Economist", "Mar 2022", "Present", "", "London, UK"),
          ("Bluefin Coach", "2018", "2020", "", "New York, NY"),
          ("Rivera, Quinn", "2016", "2018", "Onsite", "New York, NY"), ("Rivera, Quinn", "2014", "2016", "Onsite", "")]
    # a line too long for the page breaks before the company or before the place
    assert experience(
        "Principal Agile Delivery and Release Train Scrum Master   Mar 2022 – Present",
        "Northwind Labs · Austin, TX · Remote", "• Ran the release train.",
        "Lead Barista and Coffee Program Mentor · Rivera, Quinn   2016 – 2019", "New York, NY · Hybrid",
        "• Trained the baristas.",
        "Nurse Practitioner, Pediatric Intensive Care and Emergency Medicine   2014 – 2016", "Contoso Systems",
        "• Ran the night shift.",
    ) == [("Northwind Labs", "Mar 2022", "Present", "Remote", "Austin, TX"),
          ("Rivera, Quinn", "2016", "2019", "Hybrid", "New York, NY"), ("Contoso Systems", "2014", "2016", "", "")]
    # a person's own lines: the company first, as the section's ' · ' lines show, over a title the rules do not know too
    assert experience(
        "Northwind Labs · Senior Data Engineer · Austin, TX   2019 – 2021", "• Built it.",
        "Fabrikam Inc. · Barista   2017 – 2019", "• Made coffee.",
        "Bluefin Commerce · Dallas, TX   2015 – 2017", "Scrum Master", "• Ran it.",
        "Contoso Systems   2013 – 2015", "Scrum Master · Remote", "• Ran it.",
        "Senior Data Engineer · Austin, TX   2011 – 2013", "Northwind Labs", "• Built it.",
    ) == [("Northwind Labs", "2019", "2021", "", "Austin, TX"), ("Fabrikam Inc.", "2017", "2019", "", ""),
          ("Bluefin Commerce", "2015", "2017", "", "Dallas, TX"), ("Contoso Systems", "2013", "2015", "", ""),
          ("Northwind Labs", "2011", "2013", "", "Austin, TX")]


def test_a_town_and_state_after_the_school_is_no_degree():
    """'MA' and 'MS' are degrees too: the school's town and state ('Springfield, MA') must not take the degree's
    place, on the school's line or from the degree line under it."""
    assert education("State University – Springfield, MA   2014 – 2018", "Bachelor of Arts in Economics") == [
        ("State University", "Bachelor of Arts", "Economics", "2014", "2018")]
    assert education("Lakeside Community College | Oxford, MS | 2012 – 2014", "Associate of Science, Nursing") == [
        ("Lakeside Community College", "Associate of Science", "Nursing", "2012", "2014")]
    assert education("State University · Springfield, MA · M.A. in History   2016 – 2018") == [
        ("State University", "M.A.", "History", "2016", "2018")]
    assert education("State University | MA, History | 2016 – 2018") == [
        ("State University", "MA", "History", "2016", "2018")]


def test_a_side_label_run_into_the_first_line_still_starts_its_section():
    """Templates with the section labels in a left column: the text puts a label on the line of the entry beside it
    ('EXPERIENCE Senior Data Engineer'). A job title in capitals or after an acronym is still no heading, and neither
    is a dated ' · ' line, the app's job line, whatever its title says."""
    profile = parse_resume_text("""Taylor Quinn
taylor.quinn@example.com | (555) 010-4477
SKILLS Python, SQL, Airflow
EXPERIENCE Senior Data Engineer
Northwind Labs | Austin, TX | Jan 2020 – Present
• Built the pipelines.
Data Engineer
Bluefin Commerce | Dallas, TX | Jun 2016 – Dec 2019
• Moved the reports.
""")
    assert [(c.name, c.start, c.end, c.location) for c in profile.companies] == [
        ("Northwind Labs", "Jan 2020", "Present", "Austin, TX"),
        ("Bluefin Commerce", "Jun 2016", "Dec 2019", "Dallas, TX")]
    lines = ["WORK EXPERIENCE Data Engineer", "EDUCATION   Data Analyst", "QA Tester", "HR Manager",
             "EXPERIENCE DESIGNER", "SDE II · Northwind Labs   2019 – 2021", "Education Aide · Contoso   2019 – 2021"]
    assert [_is_header(line) for line in lines] == ["experience", "education", None, None, None, None, None]


def test_long_runs_of_tabs_or_spaces_are_read_quickly():
    """A run of tabs or spaces is one gap: a crafted line of thousands of them took minutes, as some patterns take
    time growing with the square of a run of white space."""
    started = time.perf_counter()
    for gap in ("\t" * 8000, " " * 8000, " " * 8000, " \t" * 4000):
        profile = parse_resume_text(f"Taylor Quinn\nCERTIFICATIONS\nCertified Kubernetes Administrator{gap}Mar 2023")
        assert [(c.name, c.year) for c in profile.certifications] == [
            ("Certified Kubernetes Administrator", "Mar 2023")]
    assert time.perf_counter() - started < 1.5


def test_a_sentence_naming_a_work_mode_is_a_bullet_not_the_mode():
    """A Word file gives its bullets without their mark: a sentence under 'Title | Company' that mentions remote or
    hybrid work is no work mode. A line of a place and a mode, with a note, still is."""
    assert experience(
        "Senior Data Engineer | Northwind Labs\tJan 2020 – Present", "Built a hybrid cloud platform on AWS.",
        "Data Engineer | Bluefin Commerce\tJun 2016 – Dec 2019", "Gave remote and on-site support to 300 staff.",
        "Support Analyst · Contoso Systems\t2014 – 2016", "Provided on-site support for 40 clinics.",
    ) == [("Northwind Labs", "Jan 2020", "Present", "", ""), ("Bluefin Commerce", "Jun 2016", "Dec 2019", "", ""),
          ("Contoso Systems", "2014", "2016", "", "")]
    assert experience("Senior Data Engineer | Northwind Labs\t2019 – 2021", "Austin, TX | Hybrid (3 days onsite)",
                      "Built the platform.") == [("Northwind Labs", "2019", "2021", "Hybrid", "Austin, TX")]


def test_the_dash_or_word_before_a_certifications_dates_goes_with_them():
    """'CompTIA Security+ - Feb 2021': the dash (or 'Issued') before the dates is no part of the name or the issuer,
    in an upload and in the list pasted on the Profile page."""
    rows = ["CompTIA Security+ - Feb 2021", "Certified ScrumMaster, Lakeside Agile Guild - Jan 2022",
            "AWS Certified Developer – Associate - Jan 2022 - Jan 2025", "CKAD – Mar 2024 – Mar 2027",
            "AWS Certified Cloud Practitioner – Issued Mar 2022", "Six Sigma Green Belt: March 2019"]
    expected = [("CompTIA Security+", "", "Feb 2021", ""),
                ("Certified ScrumMaster", "Lakeside Agile Guild", "Jan 2022", ""),
                ("AWS Certified Developer – Associate", "", "Jan 2022", "Jan 2025"),
                ("CKAD", "", "Mar 2024", "Mar 2027"), ("AWS Certified Cloud Practitioner", "", "Mar 2022", ""),
                ("Six Sigma Green Belt", "", "March 2019", "")]
    assert certifications(*rows) == expected
    assert [(c.name, c.issuer, c.year, c.expires) for c in read_certification_list("\n".join(rows))] == expected
    # a dash inside a name stays, and a bare year right after the name is part of it
    assert certifications("PMP - Northwind Project Institute - 2019", "MCSA: Windows Server 2016") == [
        ("PMP - Northwind Project Institute", "", "2019", ""), ("MCSA: Windows Server 2016", "", "", "")]


def test_a_headline_and_a_town_parted_by_a_tab_give_the_town():
    """A Word header 'Senior Data Engineer<TAB>Austin, TX': the gap parts two items, so the town is the location, not
    the whole line. A town with two spaces after its comma keeps them."""
    for gap in ("\t", "    "):
        profile = parse_resume_text(f"Riley Brooks\nSenior Data Engineer{gap}Austin, TX\n"
                                    "riley.brooks@example.com | (555) 010-9988\nEXPERIENCE\n"
                                    "Data Engineer | Northwind Labs   2020 – 2022\n")
        assert profile.contact.location == "Austin, TX", gap
    assert parse_resume_text("Riley Brooks\nAustin,  TX | riley.brooks@example.com\n").contact.location == "Austin,  TX"


def test_a_company_over_its_titles_keeps_its_own_dates():
    """Some layouts put each company on a line of its own over its title ('Northwind Labs' over 'Senior Data
    Engineer   Jan 2020 – Present'). A job without bullets must not take the next job's company. A layout with the
    company under its title never takes a line above it for the company."""
    jobs = [("Northwind Labs", "Senior Data Engineer   Jan 2020 – Present"),
            ("Bluefin Commerce — Dallas, TX", "Data Engineer   2017 – 2019"),
            ("Contoso Systems", "Junior Analyst   2015 – 2017")]
    expected = [("Northwind Labs", "Jan 2020", "Present", "", ""),
                ("Bluefin Commerce", "2017", "2019", "", "Dallas, TX"), ("Contoso Systems", "2015", "2017", "", "")]
    assert experience(*[line for job in jobs for line in job]) == expected
    assert experience(*[line for job in jobs for line in (*job, "• Built it.")]) == expected
    assert experience(*[line for job in jobs for line in (*job, "Built it in Python.")]) == expected  # Word bullets
    assert experience("Senior Data Engineer   2019 – 2021", "Northwind Labs", "Python, SQL, Airflow",
                      "Data Engineer   2017 – 2019", "Bluefin Commerce", "Junior Analyst   2015 – 2017",
                      "Contoso Systems") == [("Northwind Labs", "2019", "2021", "", ""),
                                             ("Bluefin Commerce", "2017", "2019", "", ""),
                                             ("Contoso Systems", "2015", "2017", "", "")]


# ------------------------------------------------------------------ the reader's second review: people's layouts kept
def test_a_side_label_in_title_case_run_into_a_job_title_starts_the_experience_section():
    """A two-column template puts the section labels in a left column, and the text runs a label into the first line
    beside it. In title case ('Experience Senior Data Engineer') the job-title check took the line for a title, so no
    Experience section was found and the upload was refused for having no employer."""
    for label, title in (("Experience", "Senior Data Engineer"), ("Work Experience", "Data Analyst")):
        profile = parse_resume_text(f"""Jordan Rivera
jordan.rivera@example.com | (555) 010-2233 | Austin, TX
Summary Data engineer with ten years of pipelines for finance teams.
{label} {title}
Northwind Labs | Austin, TX | Jan 2020 – Present
- Built the pipelines for the finance team.
Data Engineer
Bluefin Commerce | Dallas, TX | Jun 2016 – Dec 2019
- Moved the reports to the new warehouse.
Education B.S. in Computer Science
State University | 2012 – 2016
""")
        assert [(c.name, c.start, c.end, c.location) for c in profile.companies] == [
            ("Northwind Labs", "Jan 2020", "Present", "Austin, TX"),
            ("Bluefin Commerce", "Jun 2016", "Dec 2019", "Dallas, TX")], label
        assert (profile.name, profile.contact.location) == ("Jordan Rivera", "Austin, TX")
        assert [e.school for e in profile.education] == ["State University"]


def test_a_company_over_a_title_the_rules_do_not_know_stays_the_company():
    """A person's 'Company   dates' over 'Title': the app's line broken before the company (a title the rules do not
    know over 'Company · Place · Mode') is read only in a section that shows the app's ' · ' lines, so a long company
    over such a title is no longer taken for a title, whatever the gap before the dates. A section whose titles stand
    over their companies still reads such a title over its company, long or short."""
    for gap in ("   ", "\t", " "):
        assert experience(
            f"Lakeside Unified School District{gap}Aug 2016 – Present", "Special Education Paraprofessional",
            "• Supported 12 students.",
            f"Woodgrove Bank of the Pacific Northwest{gap}2011 – 2014", "Bank Teller", "• Opened accounts.",
            f"Wingtip Toys Distribution Center{gap}2009 – 2011", "Forklift Driver", "• Loaded the trucks.",
        ) == [("Lakeside Unified School District", "Aug 2016", "Present", "", ""),
              ("Woodgrove Bank of the Pacific Northwest", "2011", "2014", "", ""),
              ("Wingtip Toys Distribution Center", "2009", "2011", "", "")], repr(gap)
    assert experience("Senior Data Engineer   Jan 2020 – Present", "Bluefin Commerce – New York, NY", "• Built it.",
                      "Clinical Social Worker   2017 – 2019", "Lakeside Unified School District – Onsite",
                      "• Ran the intake.", "Cashier   2015 – 2017", "Woodgrove Bank", "• Ran the till.") == [
        ("Bluefin Commerce", "Jan 2020", "Present", "", "New York, NY"),
        ("Lakeside Unified School District", "2017", "2019", "Onsite", ""), ("Woodgrove Bank", "2015", "2017", "", "")]


def test_company_first_dot_lines_keep_their_company_with_a_title_the_rules_do_not_know():
    """A person's 'Company · Title' lines: a section's ' · ' lines are read in one order, the one its lines that show
    it plainly vote for (a title the rules know, a company mark), so a title the rules do not know ('Scrum Master',
    'Cashier') is not taken for the employer. Dates on a line of their own are never the app's: that line is read by
    what its parts say."""
    assert experience("Northwind Labs · Data Engineer · Austin, TX   2021 – Present", "• Built the pipelines.",
                      "Bluefin Commerce · Scrum Master · Remote   2018 – 2021", "• Ran the release train.",
                      "Contoso Systems · Cashier   2016 – 2018", "• Ran the till.") == [
        ("Northwind Labs", "2021", "Present", "", "Austin, TX"), ("Bluefin Commerce", "2018", "2021", "Remote", ""),
        ("Contoso Systems", "2016", "2018", "", "")]
    assert experience("Northwind Labs · Scrum Master", "2019 – Present", "• Ran the sprints.") == [
        ("Northwind Labs", "2019", "Present", "", "")]


def test_the_apps_employer_named_like_a_place_reads_back_without_a_place():
    """The app's 'Barista · Rivera, Quinn · Onsite' (a title the rules do not know, no place): in a section whose ' · '
    lines read the title first, the part read as a place is the employer. A person's 'Northwind Labs · London,
    England' keeps its company and its place."""
    assert experience("Barista · Rivera, Quinn · Onsite 2014 – 2016", "• Made the coffee.",
                      "Junior Analyst · Contoso Systems 2010 – 2013", "• Wrote the reports.") == [
        ("Rivera, Quinn", "2014", "2016", "Onsite", ""), ("Contoso Systems", "2010", "2013", "", "")]
    for lines in (["Northwind Labs · London, England   2019 – 2021", "• Built it."],
                  ["Northwind Labs · London, England   2019 – 2021", "• Built it.",
                   "Bluefin Commerce · Data Engineer · Dallas, TX   2016 – 2019", "• Moved the reports."]):
        assert experience(*lines)[0] == ("Northwind Labs", "2019", "2021", "", "London, England")


def test_the_apps_employer_named_like_a_role_reads_back():
    """The app's 'Cashier · Bluefin Coach' (a title the rules do not know, an employer that reads as one): a line where
    only the employer reads as a title cannot turn the section around. With a tie, or no line deciding, the section's
    ' · ' lines are read by position. A line the app broke for the page votes for the title first too."""
    junior = ["Junior Analyst · Contoso Systems 2010 – 2013", "• Wrote the reports."]
    assert experience("Cashier · Bluefin Coach · New York, NY 2019 – Present", "• Ran the till.", *junior) == [
        ("Bluefin Coach", "2019", "Present", "", "New York, NY"), ("Contoso Systems", "2010", "2013", "", "")]
    assert experience("Scrum Master · Chief Tailspin Outfitters · Austin, TX · Hybrid 2016 – 2019", "• Ran it.",
                      *junior)[0] == ("Chief Tailspin Outfitters", "2016", "2019", "Hybrid", "Austin, TX")
    title = "Lead Barista and Coffee Program Mentor for the Downtown Flagship Store"
    for broken in ([f"{title} Jul 2022 – 2023", "Fourth Coffee · Seattle, WA · Remote"],
                   [f"{title} · Fourth Coffee Jul 2022 – 2023", "Seattle, WA · Remote"]):
        assert experience(*broken, "• Trained the baristas.",
                          "Translator · Northwind Traders · Toronto, ON · Remote Sep 2020 – Present", "• Wrote it.",
                          "Principal · Bluefin Coach · Seattle, WA · Hybrid 2015 – 2017", "• Ran the school.") == [
            ("Fourth Coffee", "Jul 2022", "2023", "Remote", "Seattle, WA"),
            ("Northwind Traders", "Sep 2020", "Present", "Remote", "Toronto, ON"),
            ("Bluefin Coach", "2015", "2017", "Hybrid", "Seattle, WA")], broken[0]


def test_a_long_run_of_letters_after_a_month_is_read_quickly():
    """'janjanjan...' under EXPERIENCE: a month's name takes at most six more letters ('September'), so the date
    patterns stay quick on a text within the upload limit (it took seconds)."""
    started = time.perf_counter()
    assert parse_resume_text("Taylor Quinn\ntaylor@example.com\nEXPERIENCE\n" + "jan" * 7980).companies == []
    assert time.perf_counter() - started < 1.0
    assert experience("Data Engineer · Northwind Labs   September 2019 – February 2021") == [
        ("Northwind Labs", "September 2019", "February 2021", "", "")]


def test_a_bullet_holding_a_range_is_no_job_line():
    """A bullet's sentence with a range of numbers or years is no employer; a job line with a list mark still is."""
    assert experience("Northwind Labs | Data Engineer   2019 – 2021",
                      "• Scaled the order API from 2000 to 9000 requests per second.",
                      "• Led the 2019 – 2021 move of 40 services to AWS.", "- Cut costs 2020-2021 by a third.") == [
        ("Northwind Labs", "2019", "2021", "", "")]
    assert experience("• Northwind Labs — Data Engineer (2019 – 2021)", "   – Built the pipelines.",
                      "• Bluefin Commerce — Analyst (2016 – 2019)") == [
        ("Northwind Labs", "2019", "2021", "", ""), ("Bluefin Commerce", "2016", "2019", "", "")]


def test_a_town_typed_with_two_spaces_keeps_its_name():
    """Two spaces part two items of the header (a town and its ZIP code, a headline and a town), but not inside one
    place: 'New  York, NY' is the town, not 'York, NY'."""
    for line, town in (("New  York, NY | taylor@example.com", "New  York, NY"),
                       ("taylor@example.com  |  San  Antonio, TX", "San  Antonio, TX"),
                       ("Salt  Lake City, UT · (555) 010-4477", "Salt  Lake City, UT"),
                       ("Austin, TX  78701 | taylor@example.com", "Austin, TX"),
                       ("Senior Data Engineer  Austin, TX", "Austin, TX")):
        assert parse_resume_text(f"Taylor Quinn\n{line}\n").contact.location == town, line
