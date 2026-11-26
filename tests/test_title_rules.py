"""The title rules of prompt v39, as the review left them: the line under the name, the summary's first words and the
current job's title read the same whatever form the title's level, hyphens or extra parts take, and the current job is
the one marked 'Present' even when it is not listed first; older titles in the target's field under another form of its
words ('Nursing', 'Cybersecurity', 'iOS') or under a rank ('VP of Engineering') cost no rewrite pass; a career of two
jobs is asked for one title path; a special title's variant never reaches the headline; the role-level note quotes what
the summary reads; the demo's titles are real ones. Every person, employer and product here is invented."""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient, _title_ladder
from app.pipeline import validate
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, revise_user, tailor_system
from app.schemas import Company, Contact, Education, ExperienceOut, Profile

from .test_title_and_language import AI_POST, posting, tailored, titled
from .test_validate import make_output
from .test_writing_quality import five_jobs


def opened(profile, title: str, summary: str):
    """A resume whose current job has this title and whose summary is this one, after same_title."""
    resume = validate.build_resume(profile, make_output(headline="Engineer", summary=summary))
    resume.experience[0].job_title = title
    validate.same_title(resume)
    return resume


# ------------------------------------------------------------------ a level in any form, written once
@pytest.mark.parametrize("title,summary,expected", [
    ("Software Engineer 2", "Software Engineer 2 with 5+ years of web work.",
     "Software Engineer 2 with 5+ years of web work."),
    ("Software Engineer Level 2", "Software Engineer Level 2 with 5+ years.",
     "Software Engineer Level 2 with 5+ years."),
    ("Software Engineer II", "Software Engineer 2 with 5+ years.", "Software Engineer II with 5+ years."),
    ("Software Engineer 2", "Software Engineer level 2 with 5+ years.", "Software Engineer 2 with 5+ years."),
    ("Software Engineer 2", "Software Engineer 2, building Python services for 5+ years.",
     "Software Engineer 2, building Python services for 5+ years."),
    ("SDE 2", "SDE 2 with 5+ years of AWS services.", "SDE 2 with 5+ years of AWS services."),
    ("Software Engineer II", "software engineer ii with 5+ years.", "Software Engineer II with 5+ years."),
    ("Data Engineer L4", "data engineer l4 with 6+ years.", "Data Engineer L4 with 6+ years."),
    # a figure or a span of time after the role is never a level
    ("Software Engineer", "Software Engineer 3 years into a platform rebuild.",
     "Software Engineer 3 years into a platform rebuild."),
    ("Senior Data Engineer", "Senior data engineer 5+ years in fintech.", "Senior Data Engineer 5+ years in fintech."),
    ("Software Engineer", "Software Engineer 2.5 years into a platform rebuild.",
     "Software Engineer 2.5 years into a platform rebuild."),
    ("Software Engineer", "Software Engineer 2019-2024 at Contoso.", "Software Engineer 2019-2024 at Contoso."),
])
def test_a_level_in_any_form_is_written_once(profile, title, summary, expected):
    resume = opened(profile, title, summary)
    assert resume.headline == title and resume.summary == expected
    validate.same_title(resume)  # build() runs it again on a fix round's draft
    assert resume.summary == expected


# ------------------------------------------------------------------ hyphens, joins and an article
@pytest.mark.parametrize("title,summary", [
    ("Senior Full-Stack Engineer", "Senior full stack engineer with 8+ years of web work."),
    ("Senior Full-Stack Engineer", "Senior fullstack engineer with 8+ years of web work."),
    ("Senior Full Stack Engineer", "Senior fullstack engineer with 8+ years of web work."),
    ("Senior Frontend Engineer", "Senior front-end engineer with 8+ years of web work."),
    ("Senior Back-End Engineer", "Senior backend engineer with 8+ years of web work."),
    ("Senior Back End Engineer", "Senior backend engineer with 8+ years of web work."),
    ("Senior DevOps Engineer", "Senior dev-ops engineer with 8+ years of web work."),
    ("Senior AI Platform Engineer", "A senior AI platform engineer with 8+ years of web work."),
    ("Senior AI Platform Engineer", "An AI platform engineer with 8+ years of web work."),
])
def test_the_summary_takes_the_title_whatever_its_hyphens_joins_or_article(profile, title, summary):
    resume = opened(profile, title, summary)
    assert resume.headline == title
    assert resume.summary == f"{title} with 8+ years of web work."


@pytest.mark.parametrize("summary", [
    "Analytics engineering lead with 8+ years of web work.",  # a longer word is not the title
    "A data platform run for 40 teams over 8+ years.",  # an article before other words: left alone
])
def test_an_opening_that_is_not_the_title_stays(profile, summary):
    assert opened(profile, "Senior Analytics Engineer", summary).summary == summary


# ------------------------------------------------------------------ what follows the title, on the job line too
@pytest.mark.parametrize("title,expected", [
    ("Senior Software Engineer, Payments", "Senior Software Engineer"),
    ("Senior Data Engineer (Remote)", "Senior Data Engineer"),
    ("Senior Data Engineer - Payments", "Senior Data Engineer"),
    ("Senior Data Engineer | Python | SQL", "Senior Data Engineer"),
    ("Senior Manager, Customer Success", "Senior Customer Success Manager"),  # a bare role noun takes its function
    ("Manager, Customer Success, EMEA", "Customer Success Manager"),
    ("Manager, Lead Generation", "Lead Generation Manager"),  # 'Lead' starts the function here, not a level
    ("Senior Manager, Lead Generation", "Senior Lead Generation Manager"),
    ("Software Engineer, Senior", "Senior Software Engineer"),  # a level after the role keeps its place in the title
    ("Software Engineer (Senior)", "Senior Software Engineer"),
    ("Software Engineer II (Backend)", "Software Engineer II"),
    ("Engineer II, Platform", "Platform Engineer II"),
    ("Software Engineer, Level 2", "Software Engineer Level 2"),
    ("Software Engineer - II", "Software Engineer II"),  # never a level lost: it would read as a step down
    ("Data Engineer | Senior | Python", "Senior Data Engineer"),
    ("Vice President, Engineering", "Vice President of Engineering"),
    ("Head, Data", "Head of Data"),
    ("Tech Lead", "Tech Lead"),
])
def test_the_job_line_the_headline_and_the_summary_read_the_same(profile, title, expected):
    resume = opened(profile, title, f"{title.lower()} with 9+ years.")
    assert resume.headline == resume.experience[0].job_title == expected
    assert resume.summary == f"{expected} with 9+ years."
    validate.same_title(resume)  # build() runs it again on a fix round's draft
    assert resume.headline == resume.experience[0].job_title == expected
    assert resume.summary == f"{expected} with 9+ years."


def test_a_summary_that_goes_on_after_the_role_keeps_its_words(profile):
    summary = "Senior Data Engineer, payments and fraud platforms, with 9+ years of Python work."
    resume = opened(profile, "Senior Data Engineer, Payments", summary)
    assert resume.experience[0].job_title == "Senior Data Engineer" and resume.summary == summary


def level_after_a_comma(sections):
    sections["experience"][0]["job_title"] = "AI Platform Engineer, Senior"
    sections["experience"][1]["job_title"] = "Senior AI Platform Engineer"


def test_a_level_after_a_comma_stays_on_the_job_line_without_a_rewrite_pass(router, db, registry):
    _client, result = tailored(router, db, change=level_after_a_comma, registry=registry)
    resume = result.resume
    assert result.retries == 0  # never a step down from the job before it
    assert resume.headline == resume.experience[0].job_title == "Senior AI Platform Engineer"
    assert resume.summary.startswith("Senior AI Platform Engineer with ")


# ------------------------------------------------------------------ the role-level note quotes what stays
def _level_drafted(profile, latest, summary, target="Software Engineer III"):
    out = make_output(headline=target, summary=summary)
    out.jd_analysis.target_role = target
    resume = validate.build_resume(profile, out)
    resume.experience[0].job_title = latest
    validate.short_role_headline(resume)  # build()'s order
    notes = validate.role_level_fixes(resume, out.jd_analysis)
    validate.same_title(resume)
    return resume, notes


@pytest.mark.parametrize("latest,summary,said,opening", [
    ("Senior Software Engineer", "Software Engineer III with 7+ years.", "Senior Software Engineer",
     "Senior Software Engineer with"),
    ("Software Engineer II", "Software Engineer III with 7+ years.", "Software Engineer II",
     "Software Engineer II with"),
    ("Software Engineer", "Software Engineer III with 7+ years.", "Software Engineer", "Software Engineer with"),
    # the summary opens with something else: same_title leaves it, and so does the note
    ("Backend Developer", "Software Engineer III with 7+ years.", "Software Engineer", "Software Engineer with"),
    # only a later mention had the level: the opening was the model's own
    ("Senior Software Engineer", "Senior Software Engineer with 7+ years, aiming at a Software Engineer III role.",
     "Software Engineer", "Senior Software Engineer with"),
])
def test_the_level_note_names_what_the_summary_reads(profile, latest, summary, said, opening):
    resume, notes = _level_drafted(profile, latest, summary)
    assert resume.summary.startswith(opening)
    assert (f"Summary: wrote '{said}' instead of the posting's level 'Software Engineer III', which is not your title."
            in notes)


def test_a_level_written_after_the_title_counts_as_the_candidates(profile):
    # 'Software Engineer II (Backend)' has the posting's level: nothing is taken out, and no note says so
    resume, notes = _level_drafted(profile, "Software Engineer II (Backend)", "Software Engineer II with 7+ years.",
                                   target="Software Engineer II")
    assert notes == [] and resume.summary == "Software Engineer II with 7+ years."


# ------------------------------------------------------------------ a special title's variant
class SpecialVariant(LLMClient):
    """A 'Senior Voice AI Engineer' post (common title 'AI Engineer'): the first draft's current title names the
    specialty in a form the exact special title misses, and its summary opens with `opening`."""

    def __init__(self, inner, title, opening):
        self.inner, self.title, self.opening, self.prompts = inner, title, opening, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            result.data["jd_analysis"].update(target_role="Senior Voice AI Engineer", general_role="AI Engineer")
            sections = result.data["sections"]
            sections["experience"][0]["job_title"] = self.title
            sections["summary"] = self.opening + " with " + sections["summary"].split(" with ", 1)[1]
        return result


@pytest.mark.parametrize("opening", ["Senior AI Engineer", "Senior Voice and Conversational AI Engineer"])
def test_a_special_titles_variant_never_reaches_the_headline_without_the_rewrite_pass(router, db, registry, opening):
    client = SpecialVariant(FakeClient(registry.get("demo-concise")), "Senior Voice and Conversational AI Engineer",
                            opening)
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=five_jobs(), profile_id=None, jd=AI_POST,
                                    model_id="demo-concise", time_limit=10))  # too short for the rewrite pass
    assert result.status == "ok" and result.retries == 0
    assert result.resume.headline == result.resume.experience[0].job_title == "Senior AI Engineer"
    assert result.resume.summary.startswith("Senior AI Engineer with ")
    assert "voice" not in (result.resume.headline + result.resume.summary).lower()


def test_a_special_titles_variant_keeps_its_level_number_and_never_steps_down(profile):
    analysis = posting("Senior Voice AI Engineer", "AI Engineer")
    resume = validate.build_resume(profile, make_output(
        headline="Voice AI Engineer", summary="Voice and conversational AI engineer II with 6+ years of speech work."))
    resume.experience[0].job_title = "Voice and Conversational AI Engineer II"
    resume.experience[1].job_title = "AI Engineer II"
    validate.general_role_fixes(resume, analysis)
    assert validate.fix_title_regressions(resume) == []
    validate.same_title(resume)
    assert resume.headline == resume.experience[0].job_title == "AI Engineer II"
    assert resume.summary == "AI Engineer II with 6+ years of speech work."
    assert validate.specialty_problems(resume, analysis) == []
    assert not any("step down" in problem for problem in validate.title_growth_problems(resume))


# ------------------------------------------------------------------ in the field under another form of its words
@pytest.mark.parametrize("titles,target,general", [
    (["Registered Nurse", "Certified Nursing Assistant", "Patient Care Technician"], "Registered Nurse",
     "Registered Nurse"),
    (["Senior Accountant", "Accounting Associate", "Accounting Clerk"], "Senior Accountant", "Accountant"),
    (["Senior Recruiter", "Recruiting Coordinator", "HR Assistant"], "Senior Recruiter", "Recruiter"),
    (["Senior Security Engineer", "Cybersecurity Analyst", "IT Support Specialist"], "Senior Security Engineer",
     "Security Engineer"),
    (["Senior Mobile Engineer", "iOS Engineer", "Junior iOS Engineer"], "Senior Mobile Engineer", "Mobile Engineer"),
    (["Senior AI Engineer", "NLP Engineer", "Software Engineer"], "Senior AI Engineer", "AI Engineer"),
    (["Senior AI Engineer", "Deep Learning Engineer", "Software Engineer"], "Senior AI Engineer", "AI Engineer"),
    (["Senior AI Engineer", "Computer Vision Engineer", "Software Engineer"], "Senior AI Engineer", "AI Engineer"),
    (["Pharmacist", "Pharmacy Technician", "Pharmacy Assistant"], "Pharmacist", "Pharmacist"),
    # a rank names no field ('VP', 'Member', 'Chief'): the common title repeats the posting's
    (["VP of Engineering", "Director of Engineering", "Engineering Manager"], "VP of Engineering", "VP of Engineering"),
    (["Member of Technical Staff", "Senior Software Engineer", "Software Engineer"], "Member of Technical Staff",
     "Member of Technical Staff"),
    (["Chief Technology Officer", "VP of Engineering", "Director of Engineering"], "Chief Technology Officer",
     "Chief Technology Officer"),
])
def test_an_older_title_in_the_field_under_another_form_of_its_words_is_fine(titles, target, general):
    assert validate.title_path_problems(titled(titles), posting(target, general)) == ([], False)


@pytest.mark.parametrize("titles,target,general", [
    (["Registered Nurse", "Patient Care Technician", "Hospital Volunteer"], "Registered Nurse", "Registered Nurse"),
    (["Senior Product Manager", "Production Engineer", "Software Engineer"], "Senior Product Manager",
     "Product Manager"),
    (["Senior Recruiter", "IT Support Specialist", "Help Desk Technician"], "Senior Recruiter", "Recruiter"),
    (["IT Manager", "Audit Associate", "Accounting Clerk"], "IT Manager", "IT Manager"),  # 'it' is never in 'audit'
])
def test_an_older_title_outside_the_field_is_still_a_strong_problem(titles, target, general):
    problems, strong = validate.title_path_problems(titled(titles), posting(target, general))
    assert strong and len(problems) == 1


MOBILE_POST = ("Senior Mobile Engineer at Fabrikam. You will build our iOS and Android apps in Swift and Kotlin. "
               "Required: Swift, Kotlin, iOS, Android.")


def ios_path(sections):
    for exp, title in zip(sections["experience"], ["Senior Mobile Engineer", "iOS Engineer", "Junior iOS Engineer"]):
        exp["job_title"] = title


def test_an_ios_path_into_a_mobile_post_costs_no_rewrite_pass(router, db, registry):
    client, result = tailored(router, db, jd=MOBILE_POST, person=five_jobs(3), change=ios_path, registry=registry)
    assert result.retries == 0 and client.fix_round() == ""
    assert [e.job_title for e in result.resume.experience] == [
        "Senior Mobile Engineer", "iOS Engineer", "Junior iOS Engineer"]


# ------------------------------------------------------------------ a career of two jobs
def test_the_prompt_asks_for_one_path_whatever_the_number_of_jobs():
    text = tailor_system("estimate")
    assert "is already in that field, even when it is the only older job" in text
    assert "the jobs before that one step back toward a plain title, one broader title at a time" in text
    assert "the general title" not in text  # `general_role` is the target's own common title


def plain_only_older_title(sections):
    sections["experience"][1]["job_title"] = "Software Engineer"


def test_a_two_job_career_is_asked_for_one_title_in_the_field(router, db, registry):
    client, result = tailored(router, db, person=five_jobs(2), change=plain_only_older_title, registry=registry)
    asked = client.fix_round().split("Revise the previous draft")[0]
    assert result.retries == 1 and "the job before the current one in the target's field" in asked
    assert "plain title" not in asked  # no clause that points the other way
    assert [e.job_title for e in result.resume.experience] == ["Senior AI Platform Engineer", "AI Platform Engineer"]


def test_a_longer_career_is_asked_for_the_whole_path(router, db, registry):
    def plain_older_titles(sections):
        for exp, title in zip(sections["experience"][1:], ["Senior Software Engineer", "Software Engineer II",
                                                          "Software Engineer", "Junior Software Engineer"]):
            exp["job_title"] = title

    client, result = tailored(router, db, change=plain_older_titles, registry=registry)
    asked = client.fix_round()
    assert result.retries == 1
    assert ("the job before the current one in the target's field; the jobs before it step back toward a plain title, "
            "the oldest of them plain titles") in asked


# ------------------------------------------------------------------ the demo's titles are real ones
@pytest.mark.parametrize("prefix,base,ladder", [
    ("Senior", "AI Platform Engineer", ["Senior AI Platform Engineer", "AI Platform Engineer", "Platform Engineer",
                                        "Software Engineer", "Junior Software Engineer"]),  # the owner's path
    ("Senior", "Machine Learning Engineer", ["Senior Machine Learning Engineer", "Machine Learning Engineer",
                                             "Data Engineer", "Software Engineer", "Junior Software Engineer"]),
    ("", "Front End Developer", ["Front End Developer", "Associate Front End Developer", "Associate Software Developer",
                                 "Junior Software Developer", "Junior Software Developer"]),
    ("Senior", "Data and Analytics Engineer", ["Senior Data and Analytics Engineer", "Data and Analytics Engineer",
                                               "Analytics Engineer", "Software Engineer", "Junior Software Engineer"]),
    ("", "Data & Analytics Engineer", ["Data & Analytics Engineer", "Analytics Engineer", "Associate Software Engineer",
                                       "Junior Software Engineer", "Junior Software Engineer"]),
    ("Senior", "Business Intelligence Analyst", [
        "Senior Business Intelligence Analyst", "Business Intelligence Analyst",
        "Associate Business Intelligence Analyst", "Junior Business Intelligence Analyst",
        "Junior Business Intelligence Analyst"]),
    ("", "Senior iOS Developer", ["Senior iOS Developer", "iOS Developer", "Software Developer",
                                  "Associate Software Developer", "Junior Software Developer"]),
])
def test_the_demos_titles_are_whole_titles(prefix, base, ladder):
    assert _title_ladder(prefix, base, 5) == ladder


@pytest.mark.parametrize("post", [
    "Senior Machine Learning Engineer", "Machine Learning Engineer", "Senior Front End Developer",
    "Senior Full Stack Developer", "Full Stack Engineer", "Senior Customer Success Manager",
    "Senior Quality Assurance Engineer", "Senior Computer Vision Engineer", "Site Reliability Engineer",
    "Senior Supply Chain Analyst", "Senior Human Resources Manager", "Senior Data and Analytics Engineer",
])
def test_no_demo_title_is_a_piece_of_one(post):
    from app.llm.fake_client import guess_role

    prefix, base = guess_role(post + " at Contoso.")
    for title in _title_ladder(prefix, base, 7):
        first = next(word for word in title.split() if word not in ("Senior", "Associate", "Junior"))
        assert first not in {"and", "&", "Learning", "End", "Stack", "Success", "Assurance", "Vision", "Reliability",
                             "Chain", "Resources"}, (post, title)  # never a piece of a field's name


# ------------------------------------------------------------------ the job held now, listed second
def present_second(n: int = 3) -> Profile:
    """Northwind Labs, a contract that started later and ended, is listed first; Bluefin Commerce, the job held now
    ('Present'), second."""
    companies = [Company(name="Northwind Labs", start="Jan 2026", end="Jun 2026", mode="Remote"),
                 Company(name="Bluefin Commerce", start="Mar 2021", end="Present", mode="Remote"),
                 Company(name="Contoso", start="Jul 2017", end="Feb 2021", mode="Onsite"),
                 Company(name="Fabrikam", start="Jun 2015", end="Jun 2017", mode="Onsite")][:n]
    return Profile(name="Taylor Quinn", contact=Contact(email="taylor@example.com", location="Austin, TX"),
                   companies=companies, education=[Education(school="Tailspin University", degree="B.S.",
                                                             field="Computer Science", end="2015")])


def present_second_resume(titles, summary="Senior AI platform engineer with 9+ years of platform work."):
    out = make_output(n=1, headline="AI Engineer", summary=summary)
    out.sections.experience = [ExperienceOut(company_index=i, job_title=title, bullets=["Built things", "Fixed things"])
                               for i, title in enumerate(titles)]
    return validate.build_resume(present_second(len(titles)), out)


def test_the_headline_and_the_summary_take_the_title_of_the_job_held_now():
    resume = present_second_resume(["Senior AI Platform Engineer", "AI Platform Engineer", "Platform Engineer"])
    validate.same_title(resume)
    assert validate.current_job(resume) is resume.experience[1]
    assert resume.headline == resume.experience[1].job_title == "AI Platform Engineer"
    assert resume.summary == "AI Platform Engineer with 9+ years of platform work."


def test_the_job_before_the_current_one_is_the_one_listed_after_it():
    analysis = posting()
    problems, strong = validate.title_path_problems(present_second_resume(
        ["Senior AI Platform Engineer", "Senior AI Platform Engineer", "Senior Software Engineer"]), analysis)
    assert strong and problems == [(
        "'Senior Software Engineer' at Contoso, the job before the current one, is not in the field of 'AI Platform "
        "Engineer', and no older title is either.")]
    problems, strong = validate.title_path_problems(present_second_resume(
        ["Senior AI Platform Engineer", "Senior AI Platform Engineer", "Senior Software Engineer",
         "Platform Engineer"]), analysis)
    assert not strong and problems[0].startswith("'Senior Software Engineer' at Contoso, the job before the current")
    # the current job listed last: no job before it to check
    resume = present_second_resume(["Senior Software Engineer", "Senior AI Platform Engineer"])
    assert validate.title_path_problems(resume, analysis) == ([], False)


def test_the_level_check_and_the_note_follow_the_job_held_now():
    analysis = posting("Software Engineer II", "Software Engineer")
    resume = present_second_resume(["Software Engineer II", "Senior Software Engineer", "Software Engineer"],
                                   summary="Software Engineer II with 7+ years of web work.")
    notes = validate.role_level_fixes(resume, analysis)
    validate.same_title(resume)
    assert resume.headline == "Senior Software Engineer"
    assert resume.summary == "Senior Software Engineer with 7+ years of web work."
    assert notes == [("Summary: wrote 'Senior Software Engineer' instead of the posting's level 'Software Engineer "
                      "II', which is not your title.")]


def test_a_job_listed_above_the_one_held_now_is_raised_not_the_current_title():
    resume = present_second_resume(["AI Platform Engineer", "Senior AI Platform Engineer", "Platform Engineer"])
    notes = validate.fix_title_regressions(resume)
    assert [e.job_title for e in resume.experience] == [
        "Senior AI Platform Engineer", "Senior AI Platform Engineer", "Platform Engineer"]
    assert notes == [("Northwind Labs: title changed from 'AI Platform Engineer' to 'Senior AI Platform Engineer' so "
                      "the career path never steps down.")]


def test_a_run_whose_present_job_is_listed_second_gives_it_the_posts_title(router, db):
    _client, result = tailored(router, db, person=present_second())
    resume = result.resume
    current = validate.current_job(resume)
    assert resume.profile.companies[current.company_index].name == "Bluefin Commerce"
    assert resume.headline == current.job_title == "Senior AI Platform Engineer"
    assert resume.summary.startswith("Senior AI Platform Engineer with ")
    assert [e.job_title for e in resume.experience] == [  # Contoso ended in Feb 2021: a title of its time
        "Senior AI Platform Engineer", "Senior AI Platform Engineer", "ML Platform Engineer"]
    assert validate.title_path_problems(resume, result.analysis) == ([], False)


def test_the_demo_gives_the_posts_title_to_the_job_held_now():
    assert _title_ladder("Senior", "AI Platform Engineer", 4, current=1) == [
        "Senior AI Platform Engineer", "Senior AI Platform Engineer", "AI Platform Engineer", "Platform Engineer"]
    assert _title_ladder("Senior", "AI Platform Engineer", 3) == _title_ladder("Senior", "AI Platform Engineer", 3, 0)


def test_the_prompt_gives_the_posts_title_to_the_job_held_now():
    text = tailor_system("estimate")
    held = 'the job marked "Present (current role)" in <employers>, else the most recent'
    assert f"HEADLINE: the current job's title in your plan ({held}), word for word" in text
    assert "the line under the name, the summary's first words and the current job read the same" in text
    assert f"the current job's title ({held}) is `general_role` at the posting's level" in text
    assert "unless the current job's title in your plan has that level" in text
    assert "Check that the headline is the current job's title word for word" in text
    assert "the most recent job's title" not in text and "the job before the most recent one" not in text
    fix = revise_user(AI_POST, five_jobs().companies, "", analysis=posting(), plan=[], draft="{}", problems="(none)")
    assert "the headline and the summary's first words stay the current job's title, word for word" in fix
