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
