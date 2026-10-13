from app.parsing.heuristic import parse_resume_text

from .conftest import SAMPLE_RESUME_TEXT


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
