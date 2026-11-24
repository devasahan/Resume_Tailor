"""Education as the reviewer asked: the degree exactly as on the diploma, then its subject ("B.S. in Computer Science"),
the school and its town ("St. Louis, MO"). Both upload readers keep the town, the Profile page's Education box edits
each school, and a resume uploaded again keeps what that box added. Every person and school here is invented."""
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfReader

from app.llm.schema import strict_json_schema
from app.main import carry_education
from app.parsing.heuristic import _parse_education, parse_resume_text
from app.parsing.resume_parser import profile_warnings
from app.render.pdf import degree_text, education_runs
from app.schemas import Company, Education, Profile

from .test_api import docx_bytes, pdf_bytes, save_profile
from .test_api import tailor as tailor_request

TOWN = "Education: add each school's town"
PAGE = Path(__file__).parents[1] / "app/web/templates/profile.html"


def resume_text(education: str) -> str:
    """A resume the rules read, with these lines under EDUCATION."""
    return ("Taylor Quinn\nAustin, TX | taylor.quinn@example.com | (555) 010-2030\n\nEXPERIENCE\n"
            "Northwind Labs | Remote\nMar 2022 - Present\n\nEDUCATION\n" + education + "\n")


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


@pytest.mark.parametrize("line,expected", [
    # the school, its town, then the degree: cut where the degree starts, and the town kept apart
    ("Lakeside University, Seattle, WA, B.S. in Informatics   Jun 2019",
     ("Lakeside University", "B.S.", "Informatics", "Seattle, WA", "Jun 2019")),
    ("Tailspin Institute of Technology, Atlanta, GA, M.S. Computer Science, 2021",
     ("Tailspin Institute of Technology", "M.S. Computer Science", "", "Atlanta, GA", "2021")),
    ("Lakeside University, Springfield, IL, Bachelor of Science in Computer Science",
     ("Lakeside University", "Bachelor of Science", "Computer Science", "Springfield, IL", "")),
    # the rest of the school's name after its comma stays with it
    ("University of Tailspin, Northfield, B.S. in Computer Science, 2018",
     ("University of Tailspin, Northfield", "B.S.", "Computer Science", "", "2018")),
    # a faculty after the degree belongs to the degree, never to the school
    ("Lakeside University, Master of Science, School of Engineering",
     ("Lakeside University", "Master of Science", "School of Engineering", "", "")),
    # a state's code after a town is no degree ('MA', 'MS'), and one after the school is
    ("Lakeside University, Boston, MA, B.S. in Biology", ("Lakeside University", "B.S.", "Biology", "Boston, MA", "")),
    ("Lakeside University, Jackson, MS, BS in Nursing", ("Lakeside University", "BS", "Nursing", "Jackson, MS", "")),
    ("Lakeside University, Seattle, WA, BS in Informatics",
     ("Lakeside University", "BS", "Informatics", "Seattle, WA", "")),
    ("Lakeside University, MA, Economics", ("Lakeside University", "MA", "Economics", "", "")),
    # no comma with a degree right after it: the line stays whole
    ("Lakeside University, School of Engineering", ("Lakeside University, School of Engineering", "", "", "", "")),
])
def test_a_school_first_comma_line_is_cut_where_its_degree_starts(line, expected):
    [entry] = _parse_education([line])
    assert (entry.school, entry.degree, entry.field, entry.location, entry.end) == expected


def test_a_whole_resume_prints_a_school_first_comma_line_right():
    text = resume_text("Lakeside University, Seattle, WA, B.S. in Informatics\tJun 2019")
    [entry] = parse_resume_text(text).education
    assert "".join(run.text for run in education_runs(entry, 1)) == (
        "B.S. in Informatics · Lakeside University · Seattle, WA")


@pytest.mark.parametrize("line,school,town", [
    # a degree written before the school is never its town: the real town after the school is
    ("BBA, Finance | Lakeside University | Austin, TX | 2016", "Lakeside University", "Austin, TX"),
    ("Computer Science, BS | Lakeside University | Springfield, IL | 2020", "Lakeside University", "Springfield, IL"),
    ("MFA, Creative Writing | Lakeside University | Iowa City, IA", "Lakeside University", "Iowa City, IA"),
    ("BSN, Nursing · Lakeside University · Springfield, IL", "Lakeside University", "Springfield, IL"),
    ("Juris Doctor, Cum Laude | Tailspin School of Law | Cambridge, MA", "Tailspin School of Law", "Cambridge, MA"),
    # nor is one after the school; the town is the last place after it, one with a state's code first
    ("Lakeside University | BBA, Finance | 2016", "Lakeside University", ""),
    ("Lakeside University | Computer Science, BS", "Lakeside University", ""),
    ("Tailspin School of Law | Juris Doctor, Cum Laude", "Tailspin School of Law", ""),
    ("Lakeside University | MFA, Creative Writing | Iowa City, IA", "Lakeside University", "Iowa City, IA"),
    ("Lakeside University | Computer Science, Mathematics | Springfield, IL", "Lakeside University", "Springfield, IL"),
    ("Lakeside University | Springfield, IL | Computer Science, Mathematics", "Lakeside University", "Springfield, IL"),
    ("Lakeside Community College | Oxford, MS | 2014", "Lakeside Community College", "Oxford, MS"),
    ("Tailspin University | Mumbai, India", "Tailspin University", "Mumbai, India"),
])
def test_the_town_is_a_place_after_the_school_and_never_a_degree(line, school, town):
    [entry] = _parse_education([line])
    assert (entry.school, entry.location) == (school, town)
    profile = Profile(name="Taylor Quinn", companies=[Company(name="Northwind Labs", start="Jan 2022", end="Present")],
                      education=[entry])
    assert any(warning.startswith(TOWN) for warning in profile_warnings(profile)) == (not town)  # asked for it


@pytest.mark.parametrize("line,town", [
    # a town written with its province, state or country, then majors shaped like a place: the town is the real place
    ("Lakeside University | Toronto, Ontario | B.Sc. | Computer Science, Statistics | 2016", "Toronto, Ontario"),
    ("Lakeside University | London, United Kingdom | BA | Economics, Politics | 2015", "London, United Kingdom"),
    ("Tailspin University | Springfield, Illinois | B.S. | Biology, Chemistry | 2014", "Springfield, Illinois"),
    ("Tailspin University | Mumbai, India | Bachelor of Engineering | Computer Science, Honours | 2015",
     "Mumbai, India"),
    ("Lakeside University | Paris, France | MSc | Finance, Economics", "Paris, France"),
    ("Lakeside University | B.Sc. | Computer Science, Statistics | Toronto, Ontario | 2016", "Toronto, Ontario"),
    # a subject ending in two capitals is no state's code
    ("Lakeside University | Springfield, IL | B.S. | Computer Science, AI | 2020", "Springfield, IL"),
    ("Lakeside University | Austin, TX | MBA | Finance, IT | 2019", "Austin, TX"),
    ("Lakeside University | Austin, TX | Information Systems, HR | 2019", "Austin, TX"),
    ("Lakeside University | Toronto, ON | B.Sc. | 2016", "Toronto, ON"),
    ("Lakeside University | London, UK | MSc | 2016", "London, UK"),
    # no real place after the school: no town, so the town warning asks for it
    ("Lakeside University | B.S. | Computer Science, AI | 2020", ""),
    ("Lakeside University | B.Sc. | Computer Science, Statistics | 2016", ""),
    ("Lakeside University | Information Systems, HR | 2019", ""),
    ("Lakeside University, Computer Science, AI", ""),
])
def test_the_town_is_a_real_place_never_a_major(line, town):
    [entry] = _parse_education([line])
    assert entry.location == town
    profile = Profile(name="Taylor Quinn", companies=[Company(name="Northwind Labs", start="Jan 2022", end="Present")],
                      education=[entry])
    assert any(warning.startswith(TOWN) for warning in profile_warnings(profile)) == (not town)  # asked for it


def test_a_degree_above_its_school_joins_it():
    """The degree line first, then the school's: one row, never a row without its school."""
    [entry] = _parse_education(["Bachelor of Science in Computer Science   2016 - 2020",
                                "Lakeside University, Springfield, IL"])
    assert (entry.school, entry.degree, entry.field, entry.location, entry.start, entry.end) == (
        "Lakeside University", "Bachelor of Science", "Computer Science", "Springfield, IL", "2016", "2020")
    text = resume_text("Bachelor of Science in Computer Science\t2016 - 2020\nLakeside University, Springfield, IL")
    assert [(e.school, e.degree, e.location, e.end) for e in parse_resume_text(text).education] == [
        ("Lakeside University", "Bachelor of Science", "Springfield, IL", "2020")]
    # a school line with its own degree is that degree's row only: the degree above keeps no school, as before
    assert [(e.school, e.degree) for e in _parse_education(["M.S. in Data Science   2022",
                                                             "Lakeside University | B.S. in Biology | 2016"])] == [
        ("", "M.S."), ("Lakeside University", "B.S.")]
    # every degree above its own school: each one joins the school under it
    assert [(e.school, e.degree, e.end) for e in _parse_education([
        "M.S. in Data Science   2018", "Tailspin University", "B.S. in Computer Science   2016",
        "Lakeside University"])] == [("Tailspin University", "M.S.", "2018"), ("Lakeside University", "B.S.", "2016")]
    # a degree under its school, as before
    assert [(e.school, e.degree) for e in _parse_education(["Lakeside University", "B.S. in Computer Science, 2016",
                                                             "Tailspin University", "M.S. in Data Science"])] == [
        ("Lakeside University", "B.S."), ("Tailspin University", "M.S.")]
    # the degree's dates on their own line between it and its school
    assert [(e.school, e.degree, e.end) for e in _parse_education([
        "Bachelor of Science in Computer Science", "2016 - 2020", "Lakeside University, Springfield, IL"])] == [
        ("Lakeside University", "Bachelor of Science", "2020")]


# A school written without a school word ('Fabrikam Tech', as people write a school's short name) is no school line to
# the rules, so it must never let a degree take another school's name: the rows are read as before (45b617c)
WRONG_SCHOOL = [
    # each school over its degree, the first one without a school word: that line is the first degree's school
    (["Fabrikam Tech, Atlanta, GA", "M.S. in Computer Science   2018", "Lakeside University, Springfield, IL",
      "B.S. in Computer Science   2016"],
     [("", "M.S.", "", "2018"), ("Lakeside University", "B.S.", "Springfield, IL", "2016")]),
    (["Contoso State", "MBA   2020", "Tailspin University", "Bachelor of Arts in Economics   2014"],
     [("", "MBA", "", "2020"), ("Tailspin University", "Bachelor of Arts", "", "2014")]),
    # one line per school: a school line with its own degree is that degree's row only
    (["M.S. in Computer Science, Fabrikam Tech, 2018", "B.S. in Computer Science, Lakeside University, 2016"],
     [("", "M.S.", "", "2018"), ("Lakeside University", "B.S.", "", "2016")]),
    # a degree line holding its own school before the degree waits for no school line
    (["Fabrikam Tech | M.S. Computer Science | 2018", "Lakeside University", "B.S. Computer Science, 2016"],
     [("", "Fabrikam Tech | M.S. Computer Science", "", "2018"),
      ("Lakeside University", "B.S. Computer Science", "", "2016")]),
    # each degree over its school: a degree takes only the school line right under it
    (["B.S. in Computer Science   2014", "Fabrikam Tech, Atlanta, GA", "M.S. in Computer Science   2016",
      "Lakeside University, Springfield, IL"],
     [("", "B.S.", "", "2014"), ("Lakeside University", "", "Springfield, IL", "")]),
    # a Word table's cells, one per line: degree, school, year
    (["M.S. in Computer Science", "Fabrikam Tech", "2018", "B.S. in Computer Science", "Lakeside University", "2016"],
     [("", "M.S.", "", "2018"), ("Lakeside University", "", "", "2016")]),
]


@pytest.mark.parametrize("lines,expected", WRONG_SCHOOL)
def test_a_degree_never_takes_the_name_of_another_school(lines, expected):
    assert [(e.school, e.degree, e.location, e.end) for e in _parse_education(lines)] == expected


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_a_pdf_and_a_word_upload_keep_each_degree_with_its_own_school(client, kind):
    text = resume_text("Fabrikam Tech, Atlanta, GA\nM.S. in Computer Science   2018\n"
                       "Lakeside University, Springfield, IL\nB.S. in Computer Science   2016")
    data = pdf_bytes(text) if kind == "pdf" else docx_bytes(text)
    response = client.post("/api/resumes", files={"file": (f"base.{kind}", data)})
    assert response.status_code == 200, response.text
    assert [(e["school"], e["degree"], e["location"], e["end"]) for e in response.json()["resume"]["profile"][
        "education"]] == [("", "M.S.", "", "2018"), ("Lakeside University", "B.S.", "Springfield, IL", "2016")]


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


NOT_A_SUBJECT = [
    # a school or a faculty anywhere in the field
    ("MBA", "Graduate School of Business", "MBA, Graduate School of Business"),
    ("M.S.", "Northwind School of Engineering", "M.S., Northwind School of Engineering"),
    ("MBA", "The Tailspin School", "MBA, The Tailspin School"),
    # honors, a grade, a minor, a major or a concentration
    ("Bachelor of Science", "Magna Cum Laude", "Bachelor of Science, Magna Cum Laude"),
    ("B.S.", "Minor in Mathematics", "B.S., Minor in Mathematics"),
    ("B.A.", "Cum Laude", "B.A., Cum Laude"),
    ("MBA", "Concentration in Finance", "MBA, Concentration in Finance"),
    ("B.S.", "GPA: 3.8/4.0", "B.S., GPA: 3.8/4.0"),
    ("Bachelor of Engineering", "First Class Honours", "Bachelor of Engineering, First Class Honours"),
    ("B.A.", "Dean's List", "B.A., Dean's List"),
    ("B.S.", "With Distinction", "B.S., With Distinction"),
    ("B.A.", "Major in Economics", "B.A., Major in Economics"),
]
SUBJECT_ROWS = [
    # a subject, then a minor after a comma: still 'in'
    ("Bachelor of Arts", "Economics, Minor in Mathematics", "Bachelor of Arts in Economics, Minor in Mathematics"),
    # the degree ends with 'in' or 'of', in any letter case
    ("Bachelor of Science In", "Biology", "Bachelor of Science In Biology"),
    ("BACHELOR OF SCIENCE IN", "NURSING", "BACHELOR OF SCIENCE IN NURSING"),
    ("Master of", "Fine Arts", "Master of Fine Arts"),
    ("Bachelor of Science in", "Science", "Bachelor of Science in Science"),  # as written, never a dangling 'in'
    # the degree already names its field: printed once
    ("B.S. Computer Science", "Computer Science", "B.S. Computer Science"),
    ("Bachelor of Science in Computer Science", "computer science", "Bachelor of Science in Computer Science"),
]


@pytest.mark.parametrize("degree,field,printed", NOT_A_SUBJECT + SUBJECT_ROWS)
def test_in_joins_the_degree_only_to_a_subject(degree, field, printed):
    edu = Education(school="Tailspin University", degree=degree, field=field)
    assert degree_text(edu, 1) == printed
    assert degree_text(edu, 0) == f"{degree}, {field}"  # a run saved before keeps its look


DEGREE_HARNESS = r"""
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const code = page.slice(page.indexOf('// The degree as a new resume prints it'),
                        page.indexOf('// Each school as new resumes print it'));
const degreeText = new Function(code + '\nreturn degreeText;')();
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
console.log(JSON.stringify(cases.map(([degree, field]) => degreeText({ degree, field }))));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="runs the Profile page's own script with Node.js")
def test_the_profile_page_prints_the_degree_as_the_resume_does(tmp_path):
    """The page's degreeText (the card's Education line, the 'New resumes print' after Save) follows degree_text."""
    cases = [(degree, field) for degree, field, _ in NOT_A_SUBJECT + SUBJECT_ROWS] + [
        ("B.S.", "Computer Science"), ("Master of Science", "School of Engineering"), ("MBA", ""),
        ("", "Computer Science"), ("B.S.", "Computer Science (Honors)"), ("Ph.D.", "Physics")]
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "degree.js").write_text(DEGREE_HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(tmp_path / "degree.js"), str(PAGE), str(tmp_path / "cases.json")],
                         capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == [
        degree_text(Education(school="Tailspin University", degree=degree, field=field), 1) for degree, field in cases]


ROW_HARNESS = r"""
const fs = require('fs');
const page = fs.readFileSync(process.argv[2], 'utf8');
const code = page.slice(page.indexOf('// The degree as a new resume prints it'),
                        page.indexOf('// Each school as new resumes print it'));
const educationRow = new Function(code + '\nreturn educationRow;')();
const rows = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
console.log(JSON.stringify(rows.map(([school, degree, field, location]) =>
  educationRow({ school, degree, field, location }))));
"""
ROWS = [  # (school, degree, field, town)
    ("Tailspin University", "B.S.", "Computer Science", "St. Louis, MO"),
    ("", "B.S.", "Computer Science", "Austin, TX"),  # read without its school
    ("", "B.S.", "Computer Science", ""),
    ("Tailspin University", "", "", "St. Louis, MO"),
    ("Tailspin University", "MBA", "", ""),
    ("Tailspin University", "", "", ""),
    ("", "", "Computer Science", "Austin, TX"),
    ("", "", "", "Austin, TX"),
    ("Tailspin University", "Master of Science", "School of Engineering", "Boston, MA"),
    ("", "", "", ""),
]


@pytest.mark.skipif(shutil.which("node") is None, reason="runs the Profile page's own script with Node.js")
def test_the_profile_page_prints_each_row_as_the_resume_does(tmp_path):
    """The 'New resumes print' after Save (the page's educationRow) is the row the PDF and the Word copy print."""
    (tmp_path / "rows.json").write_text(json.dumps(ROWS), encoding="utf-8")
    (tmp_path / "row.js").write_text(ROW_HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(tmp_path / "row.js"), str(PAGE), str(tmp_path / "rows.json")],
                         capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == ["".join(piece.text for piece in education_runs(Education(
        school=school, degree=degree, field=field, location=town), 1)) for school, degree, field, town in ROWS]
    assert "'Saved. New resumes print: ' + saved.map(educationRow).join('; ')" in PAGE.read_text(encoding="utf-8")


def test_a_row_without_its_school_prints_no_empty_place(client):
    """A row read without its school, given only its town in the Education box: the PDF and the Word copy print what
    the page says ('B.S. in Computer Science · Austin, TX'), never an empty place between two dots."""
    entry = save_profile(client, text=resume_text("B.S. in Computer Science   2012 - 2016"))
    url = f"/api/resumes/{entry['id']}/education"
    assert client.put(url, json=education_body(entry, location="Austin, TX")).status_code == 200
    run = tailor_request(client)
    assert run.status_code == 200, run.text
    run_id = run.json()["run_id"]
    pdf = client.get(f"/api/runs/{run_id}/resume.pdf")
    lines = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf.content)).pages).splitlines()
    [printed] = [line for line in lines if "Computer Science" in line and "2016" in line]
    assert printed.startswith("B.S. in Computer Science · Austin, TX ") and "· ·" not in printed
    word = Document(io.BytesIO(client.get(f"/api/runs/{run_id}/resume.docx").content))
    assert [p.text for p in word.paragraphs if "Computer Science" in p.text and "2016" in p.text] == [
        "B.S. in Computer Science · Austin, TX\t2012 – 2016"]


def test_a_school_without_its_town_gets_a_warning():
    profile = Profile(name="Taylor Quinn", companies=[Company(name="Northwind Labs", start="Jan 2022", end="Present")],
                      education=[Education(school="Tailspin University", degree="B.S.", field="Computer Science")])
    assert any(warning.startswith(TOWN) for warning in profile_warnings(profile))
    profile.education[0].location = "St. Louis, MO"
    assert not any(warning.startswith(TOWN) for warning in profile_warnings(profile))


# ------------------------------------------------------------------ the Profile page's Education box
BOX_FIELDS = ("school", "degree", "field", "location", "start", "end")


def education_body(entry: dict, **changes) -> dict:
    """The Save education body: each school as the page shows it, the first one changed, and each one as loaded."""
    loaded = [{key: edu.get(key, "") for key in BOX_FIELDS} for edu in entry["profile"]["education"]]
    entries = [dict(edu) for edu in loaded]
    entries[0].update(changes)
    return {"entries": entries, "loaded": loaded}


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
    other = [{**body["loaded"][0], "school": "Fabrikam College"}]  # the schools changed in another tab
    assert client.put(url, json={**body, "loaded": other}).status_code == 409
    assert client.put(url, json={**body, "entries": body["entries"] * 2}).status_code == 409  # one school too many
    assert client.put(url, json=education_body(entry, school="  ")).status_code == 422
    assert user_client.put(url, json=body).status_code == 404  # another account's resume
    assert client.put("/api/resumes/99999/education", json=body).status_code == 404
    assert client.get("/api/resumes").json()["resumes"][0]["profile"]["education"][0]["location"] == ""


def test_a_save_from_a_stale_tab_never_undoes_what_another_tab_saved(client):
    """Two tabs show the same resume: one saves the town, then the other (opened before) saves the degree."""
    entry = save_profile(client)
    url = f"/api/resumes/{entry['id']}/education"
    first, second = education_body(entry, location="Austin, TX"), education_body(entry, degree="Bachelor of Science")
    assert client.put(url, json=first).status_code == 200
    response = client.put(url, json=second)
    assert response.status_code == 409 and "another tab" in response.json()["detail"]
    saved = client.get("/api/resumes").json()["resumes"][0]["profile"]["education"][0]
    assert (saved["degree"], saved["location"]) == ("B.S.", "Austin, TX")
    page = PAGE.read_text(encoding="utf-8")
    assert "body: { entries, loaded }" in page  # the page sends each school as it loaded it


def test_a_row_read_without_its_school_can_be_saved(client):
    """A row whose school was empty when the page loaded it is saved as it is; only a school cleared is refused."""
    entry = save_profile(client, text=resume_text("B.S. in Computer Science   2012 - 2016"))
    assert [(e["school"], e["degree"]) for e in entry["profile"]["education"]] == [("", "B.S.")]
    url = f"/api/resumes/{entry['id']}/education"
    response = client.put(url, json=education_body(entry, location="Austin, TX"))
    assert response.status_code == 200, response.text
    assert response.json()["resume"]["profile"]["education"][0]["location"] == "Austin, TX"
    # the degree line above its school: one row, saved like any other
    entry = save_profile(client, text=resume_text("Bachelor of Science in Computer Science\t2016 - 2020\n"
                                                  "Lakeside University, Springfield, IL"))
    assert [e["school"] for e in entry["profile"]["education"]] == ["Lakeside University"]
    url = f"/api/resumes/{entry['id']}/education"
    assert client.put(url, json=education_body(entry, degree="B.S.")).status_code == 200
    assert client.put(url, json=education_body(client.get("/api/resumes").json()["resumes"][0],
                                               school="")).status_code == 422  # a school cleared


def test_the_education_box_saves_what_the_readers_store(client):
    """A long school line the rules keep whole, and more schools than the old 20: the box saves them. The page has the
    same limits, so it says what is too long instead of a technical error."""
    from app.main import MAX_EDUCATION, MAX_EDUCATION_CHARS, MAX_SCHOOL_CHARS

    assert (MAX_EDUCATION, MAX_SCHOOL_CHARS, MAX_EDUCATION_CHARS) == (50, 500, 300)
    long_line = ("Tailspin University: coursework in Data Structures, Algorithms, Operating Systems, Computer "
                 "Networks, Databases, Machine Learning, Distributed Systems, Compilers, Computer Graphics, Computer "
                 "Security, Cloud Computing")
    entry = save_profile(client, text=resume_text(long_line))
    assert len(entry["profile"]["education"][0]["school"]) > 200
    url = f"/api/resumes/{entry['id']}/education"
    assert client.put(url, json=education_body(entry, location="St. Louis, MO")).status_code == 200
    entry = client.get("/api/resumes").json()["resumes"][0]
    assert client.put(url, json=education_body(entry, school="x" * MAX_SCHOOL_CHARS,
                                               field="y" * MAX_EDUCATION_CHARS)).status_code == 200
    entry = client.get("/api/resumes").json()["resumes"][0]
    assert client.put(url, json=education_body(entry, school="x" * (MAX_SCHOOL_CHARS + 1))).status_code == 422
    assert client.put(url, json=education_body(entry, end="2" * (MAX_EDUCATION_CHARS + 1))).status_code == 422
    # more schools than the box ever took (20), and more than it takes (50): the upload keeps 50
    for count, kept in ((25, 25), (MAX_EDUCATION + 5, MAX_EDUCATION)):
        lines = "\n".join(f"Lakeside University – Campus {number}" for number in range(1, count + 1))
        entry = save_profile(client, text=resume_text(lines))
        assert len(entry["profile"]["education"]) == kept
        body = education_body(entry, location="Austin, TX")
        response = client.put(f"/api/resumes/{entry['id']}/education", json=body)
        assert response.status_code == 200, response.text
    page = PAGE.read_text(encoding="utf-8")
    assert (f"const MAX_EDUCATION = {MAX_EDUCATION}, MAX_SCHOOL_CHARS = {MAX_SCHOOL_CHARS}, "
            f"MAX_EDUCATION_CHARS = {MAX_EDUCATION_CHARS};") in page
    assert "maxlength: key === 'school' ? MAX_SCHOOL_CHARS : MAX_EDUCATION_CHARS" in page
    assert "then press Save education again" in page and "rows.length > MAX_EDUCATION" in page


# ------------------------------------------------------------------ a resume uploaded again
def person(*education: Education, name: str = "Taylor Quinn") -> Profile:
    return Profile(name=name, companies=[Company(name="Northwind Labs", start="Jan 2022", end="Present")],
                   education=list(education))


def test_a_new_upload_keeps_what_the_box_added_and_what_the_file_says_wins():
    earlier = [person(Education(school="Tailspin University", degree="B.S.", field="Computer Science",
                                location="St. Louis, MO", end="2014"))]
    new = person(Education(school="tailspin  university", degree="BS", end="May 2014"))  # the same degree and year
    assert carry_education(new, earlier) == ["tailspin  university"]
    [edu] = new.education
    assert (edu.degree, edu.field, edu.location) == ("BS", "Computer Science", "St. Louis, MO")
    new = person(Education(school="Tailspin University", degree="B.S.", field="Data Science",
                           location="Springfield, MO", end="2014"))
    assert carry_education(new, earlier) == [] and (new.education[0].field, new.education[0].location) == (
        "Data Science", "Springfield, MO")


def test_a_degree_never_takes_what_was_typed_for_another_degree():
    """A master's listed above the bachelor's that the box filled in: only the bachelor's gets its field and town, also
    when the file names neither degree (the years tell them apart). Two different degrees never pair."""
    earlier = [person(Education(school="Lakeside University", degree="B.S.", field="Computer Science",
                                location="Springfield, IL", end="2014"))]
    new = person(Education(school="Lakeside University", degree="M.S.", end="2016"),
                 Education(school="Lakeside University", degree="B.S.", end="2014"))
    assert carry_education(new, earlier) == ["Lakeside University"]
    assert [(e.degree, e.field, e.location) for e in new.education] == [
        ("M.S.", "", ""), ("B.S.", "Computer Science", "Springfield, IL")]
    new = person(Education(school="Lakeside University", end="2016"),
                 Education(school="Lakeside University", end="2014"))
    carry_education(new, earlier)
    assert [(e.degree, e.field, e.location) for e in new.education] == [
        ("", "", ""), ("B.S.", "Computer Science", "Springfield, IL")]
    new = person(Education(school="Lakeside University", degree="MBA"))
    assert carry_education(new, earlier) == [] and new.education[0].location == ""


def test_a_field_goes_only_with_its_own_degree():
    """The box split the file's 'B.S. Computer Science' into 'B.S.' and 'Computer Science': the same file uploaded
    again keeps the town, and its degree is never given the field a second time."""
    earlier = [person(Education(school="Tailspin University", degree="B.S.", field="Computer Science",
                                location="St. Louis, MO", end="2014"))]
    new = person(Education(school="Tailspin University", degree="B.S. Computer Science", end="2014"))
    assert carry_education(new, earlier) == ["Tailspin University"]
    [edu] = new.education
    assert (edu.degree, edu.field, edu.location) == ("B.S. Computer Science", "", "St. Louis, MO")
    assert degree_text(edu, 1) == "B.S. Computer Science"
    # a field typed without its degree never joins the degree the file names
    earlier[0].education[0].degree = ""
    new = person(Education(school="Tailspin University", degree="B.A.", end="2014"))
    carry_education(new, earlier)
    assert (new.education[0].field, new.education[0].location) == ("", "St. Louis, MO")
    # a file without the degree gets the degree and its field together
    earlier[0].education[0].degree = "B.S."
    new = person(Education(school="Tailspin University", end="2014"))
    carry_education(new, earlier)
    assert (new.education[0].degree, new.education[0].field) == ("B.S.", "Computer Science")


def test_the_same_file_uploaded_again_after_the_box_split_its_degree(client):
    entry = save_profile(client, text=resume_text("B.S. Computer Science | Tailspin University | 2014"))
    assert entry["profile"]["education"][0]["degree"] == "B.S. Computer Science"
    body = education_body(entry, degree="B.S.", field="Computer Science", location="St. Louis, MO")
    assert client.put(f"/api/resumes/{entry['id']}/education", json=body).status_code == 200
    again = save_profile(client, text=resume_text("B.S. Computer Science | Tailspin University | 2014"))
    [edu] = again["profile"]["education"]
    assert (edu["degree"], edu["field"], edu["location"]) == ("B.S. Computer Science", "", "St. Louis, MO")


@pytest.mark.parametrize("typed,read", [
    # the degree written as on the diploma, as the box asks
    ({"degree": "B.S.", "field": "Computer Science", "end": "2014"},
     {"degree": "Bachelor of Science", "field": "Computer Science", "end": "2014"}),
    # the file's 'B.S. Computer Science' split by the box: the field is not given to it a second time
    ({"degree": "Bachelor of Science", "field": "Computer Science", "end": "2014"},
     {"degree": "B.S. Computer Science", "end": "2014"}),
    # the End typed with a word before the date
    ({"degree": "M.S.", "field": "Data Science", "end": "Expected May 2027"},
     {"degree": "M.S.", "field": "Data Science", "end": "May 2027"}),
])
def test_a_typed_town_is_kept_when_the_degree_or_the_end_was_rewritten(typed, read):
    """The same school and graduation year upload again: the town typed in the Education box is kept, whatever the
    box did to the degree or the End. What the new file says wins, and the field goes only with its own degree."""
    earlier = [person(Education(school="Lakeside University", location="Springfield, IL", **typed))]
    new = person(Education(school="Lakeside University", **read))
    assert carry_education(new, earlier) == ["Lakeside University"]
    [edu] = new.education
    assert (edu.degree, edu.field, edu.end, edu.location) == (
        read["degree"], read.get("field", ""), read["end"], "Springfield, IL")


def test_two_degrees_of_one_school_each_keep_their_typed_town_after_a_rewrite():
    earlier = [person(Education(school="Lakeside University", degree="M.S.", field="Data Science",
                                location="Springfield, IL", end="Expected May 2027"),
                      Education(school="Lakeside University", degree="B.S.", field="Computer Science",
                                location="Springfield, IL", end="2022"))]
    new = person(Education(school="Lakeside University", degree="Master of Science", end="May 2027"),
                 Education(school="Lakeside University", degree="Bachelor of Science", end="May 2022"))
    carry_education(new, earlier)
    assert [(e.degree, e.field, e.location) for e in new.education] == [
        ("Master of Science", "", "Springfield, IL"), ("Bachelor of Science", "", "Springfield, IL")]
    # a B.A. and a B.S. of the same year, listed in the other order: each keeps its own field
    earlier = [person(Education(school="Lakeside University", degree="B.A.", field="Economics", end="2014",
                                location="Springfield, IL"),
                      Education(school="Lakeside University", degree="B.S.", field="Mathematics", end="2014",
                                location="Springfield, IL"))]
    new = person(Education(school="Lakeside University", degree="B.S.", end="2014"),
                 Education(school="Lakeside University", degree="B.A.", end="2014"))
    carry_education(new, earlier)
    assert [(e.degree, e.field, e.location) for e in new.education] == [
        ("B.S.", "Mathematics", "Springfield, IL"), ("B.A.", "Economics", "Springfield, IL")]


def test_a_typed_town_finds_its_school_by_name_and_never_goes_to_another_school():
    """The school's name is compared whatever its case, punctuation, 'The' and '&' or 'and'; another school, or a
    school-less row whose degree is another one, never gets the town."""
    earlier = [person(Education(school="The Fabrikam College of Arts & Sciences", degree="B.A.", end="2012",
                                location="Springfield, IL"),
                      Education(school="St. Contoso's College", degree="B.S.", end="2014",
                                location="Austin, TX"),
                      Education(school="", degree="B.S.", end="2016", location="Boston, MA"))]
    new = person(Education(school="fabrikam college of arts and sciences", degree="B.A.", end="2012"),
                 Education(school="St Contosos College", degree="B.S.", end="2014"),
                 Education(school="Fabrikam College of Arts", degree="B.A.", end="2012"),
                 Education(school="St. Contoso's University", degree="B.S.", end="2014"),
                 Education(school="", degree="M.S.", end="2016"))
    carry_education(new, earlier)
    assert [e.location for e in new.education] == ["Springfield, IL", "Austin, TX", "", "", ""]


def test_the_same_file_uploaded_again_keeps_the_town_typed_with_the_degree_as_on_the_diploma(client):
    line = "Bachelor of Science in Computer Science | Lakeside University | 2014"
    entry = save_profile(client, text=resume_text(line))
    body = education_body(entry, degree="B.S.", location="Springfield, IL")
    assert client.put(f"/api/resumes/{entry['id']}/education", json=body).status_code == 200
    again = save_profile(client, text=resume_text(line))
    [edu] = again["profile"]["education"]
    assert (edu["degree"], edu["field"], edu["location"]) == (
        "Bachelor of Science", "Computer Science", "Springfield, IL")
    assert not any(w.startswith(TOWN) for w in again["warnings"])


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
