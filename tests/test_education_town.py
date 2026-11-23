"""Education as the reviewer asked: the degree exactly as on the diploma, then its subject ("B.S. in Computer Science"),
the school and its town ("St. Louis, MO"). Both upload readers keep the town, the Profile page's Education box edits
each school, and a resume uploaded again keeps what that box added. Every person and school here is invented."""
import json

import pytest

from app.llm.schema import strict_json_schema
from app.main import carry_education
from app.parsing.heuristic import _parse_education, parse_resume_text
from app.parsing.resume_parser import profile_warnings
from app.render.pdf import degree_text
from app.schemas import Company, Education, Profile

from .test_api import save_profile

TOWN = "Education: add each school's town"


@pytest.mark.parametrize("line,expected", [
    # the app's own new row, read back
    ("B.S. in Computer Science · Tailspin University · St. Louis, MO 2010 - 2014",
     ("Tailspin University", "B.S.", "Computer Science", "St. Louis, MO", "2014")),
    ("Tailspin University | B.S. in Computer Science | 2014",
     ("Tailspin University", "B.S.", "Computer Science", "", "2014")),
    ("Tailspin University – Springfield, MA", ("Tailspin University", "", "", "Springfield, MA", "")),
    ("Tailspin University, St. Louis, MO", ("Tailspin University", "", "", "St. Louis, MO", "")),
    # commas only: cut next to the last word naming a school
    ("Bachelor of Science in Computer Science, Tailspin University, Springfield, MA 2014",
     ("Tailspin University", "Bachelor of Science", "Computer Science", "Springfield, MA", "2014")),
    ("Tailspin University, B.S. in Computer Science", ("Tailspin University", "B.S.", "Computer Science", "", "")),
    ("Master of Science, School of Engineering, Tailspin University",
     ("Tailspin University", "Master of Science", "School of Engineering", "", "")),
    ("MBA, Tailspin School of Business, Boston, MA", ("Tailspin School of Business", "MBA", "", "Boston, MA", "")),
    # a school whose own name holds a comma stays whole
    ("B.S. in Computer Science, University of California, Berkeley",
     ("University of California, Berkeley", "B.S.", "Computer Science", "", "")),
    ("University of California, Berkeley", ("University of California, Berkeley", "", "", "", "")),
])
def test_the_rules_reader_keeps_the_town_and_the_degree(line, expected):
    [entry] = _parse_education([line])
    assert (entry.school, entry.degree, entry.field, entry.location, entry.end) == expected


def test_a_degree_on_its_own_line_joins_the_school_above():
    [entry] = _parse_education(["Tailspin University, St. Louis, MO", "B.S. in Computer Science   2010 - 2014"])
    assert (entry.school, entry.degree, entry.field, entry.location, entry.end) == (
        "Tailspin University", "B.S.", "Computer Science", "St. Louis, MO", "2014")


def test_a_whole_resume_read_by_the_rules_keeps_the_town():
    text = ("Taylor Quinn\ntaylor@example.com\n\nEXPERIENCE\nBackend Engineer\nNorthwind Labs | Jan 2022 - Present\n"
            "- Build the order API in Go for 450 partner stores.\n\nEDUCATION\n"
            "B.S. in Computer Science · Tailspin University · St. Louis, MO\n")
    [entry] = parse_resume_text(text).education
    assert (entry.degree, entry.field, entry.school, entry.location) == (
        "B.S.", "Computer Science", "Tailspin University", "St. Louis, MO")


def test_the_ai_reader_is_asked_for_the_town():
    schema = json.dumps(strict_json_schema(Profile))
    assert '"location"' in json.dumps(strict_json_schema(Education))
    assert "location" in schema
    assert Education.model_validate({"school": "Tailspin University"}).location == ""  # a profile saved before


@pytest.mark.parametrize("degree,field,layout,printed", [
    ("B.S.", "Computer Science", 1, "B.S. in Computer Science"),
    ("Bachelor of Science in", "Biology", 1, "Bachelor of Science in Biology"),
    ("Master of Science", "School of Engineering", 1, "Master of Science, School of Engineering"),
    ("B.S.", "Computer Science", 0, "B.S., Computer Science"),  # a run saved before keeps its look
    ("MBA", "", 1, "MBA"),
    ("", "Computer Science", 1, "Computer Science"),
])
def test_the_degree_is_printed_as_on_the_diploma_then_its_subject(degree, field, layout, printed):
    assert degree_text(Education(school="Tailspin University", degree=degree, field=field), layout) == printed


def test_a_school_without_its_town_gets_a_warning():
    profile = Profile(name="Taylor Quinn", companies=[Company(name="Northwind Labs", start="Jan 2022", end="Present")],
                      education=[Education(school="Tailspin University", degree="B.S.", field="Computer Science")])
    assert any(warning.startswith(TOWN) for warning in profile_warnings(profile))
    profile.education[0].location = "St. Louis, MO"
    assert not any(warning.startswith(TOWN) for warning in profile_warnings(profile))


# ------------------------------------------------------------------ the Profile page's Education box
def education_body(entry: dict, **changes) -> dict:
    """The Save education body: each school as the page shows it, the first one changed, and the schools then."""
    entries = [{key: edu.get(key, "") for key in ("school", "degree", "field", "location", "start", "end")}
               for edu in entry["profile"]["education"]]
    entries[0].update(changes)
    return {"entries": entries, "schools": [edu["school"] for edu in entry["profile"]["education"]]}


def test_the_education_box_saves_the_degree_and_the_town(client):
    entry = save_profile(client)
    body = education_body(entry, degree="  B.S. ", location="Austin,   TX")
    response = client.put(f"/api/resumes/{entry['id']}/education", json=body)
    assert response.status_code == 200, response.text
    [saved] = response.json()["resume"]["profile"]["education"]
    assert (saved["degree"], saved["field"], saved["location"]) == ("B.S.", "Computer Science", "Austin, TX")
    assert client.get("/api/resumes").json()["resumes"][0]["profile"]["education"][0]["location"] == "Austin, TX"
    assert not any(w.startswith(TOWN) for w in response.json()["resume"]["warnings"])


def test_the_education_box_refuses_a_stale_list_an_empty_school_and_another_account(client, user_client):
    entry = save_profile(client)
    url, body = f"/api/resumes/{entry['id']}/education", education_body(entry, location="Austin, TX")
    assert client.put(url, json={**body, "schools": ["Fabrikam College"]}).status_code == 409  # changed in another tab
    assert client.put(url, json={**body, "entries": body["entries"] * 2}).status_code == 409  # one school too many
    assert client.put(url, json=education_body(entry, school="  ")).status_code == 422
    assert user_client.put(url, json=body).status_code == 404  # another account's resume
    assert client.put("/api/resumes/99999/education", json=body).status_code == 404
    assert client.get("/api/resumes").json()["resumes"][0]["profile"]["education"][0]["location"] == ""


# ------------------------------------------------------------------ a resume uploaded again
def person(*education: Education, name: str = "Taylor Quinn") -> Profile:
    return Profile(name=name, companies=[Company(name="Northwind Labs", start="Jan 2022", end="Present")],
                   education=list(education))


def test_a_new_upload_keeps_what_the_box_added_and_what_the_file_says_wins():
    earlier = [person(Education(school="Tailspin University", degree="B.S.", field="Computer Science",
                                location="St. Louis, MO", end="2014"))]
    new = person(Education(school="tailspin  university", degree="Bachelor's Degree", end="2014"))
    assert carry_education(new, earlier) == ["tailspin  university"]
    [edu] = new.education
    assert (edu.degree, edu.field, edu.location) == ("Bachelor's Degree", "Computer Science", "St. Louis, MO")


def test_two_degrees_from_one_school_each_keep_their_own_details():
    earlier = [person(Education(school="Tailspin University", degree="M.S.", field="Data Science", end="2016",
                                location="St. Louis, MO"),
                      Education(school="Tailspin University", degree="B.S.", field="Computer Science", end="2014",
                                location="St. Louis, MO"))]
    new = person(Education(school="Tailspin University", degree="B.S.", end="2014"),
                 Education(school="Tailspin University", degree="M.S.", end="2016"))
    carry_education(new, earlier)
    assert [(e.degree, e.field) for e in new.education] == [("B.S.", "Computer Science"), ("M.S.", "Data Science")]
    assert all(e.location == "St. Louis, MO" for e in new.education)


def test_another_persons_upload_never_gives_its_details():
    earlier = [person(Education(school="Tailspin University", field="Computer Science", location="St. Louis, MO"),
                      name="Avery Lindqvist")]
    new = person(Education(school="Tailspin University", degree="B.S."))
    assert carry_education(new, earlier) == [] and new.education[0].location == ""


def test_an_upload_through_the_api_keeps_the_town_typed_before(client):
    entry = save_profile(client)
    body = education_body(entry, location="Austin, TX")
    assert client.put(f"/api/resumes/{entry['id']}/education", json=body).status_code == 200
    again = save_profile(client, name="again.docx")
    assert again["id"] != entry["id"] and again["profile"]["education"][0]["location"] == "Austin, TX"
