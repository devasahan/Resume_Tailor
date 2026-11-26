"""The usual title of a field (prompt v40): a 'Senior AI Developer' post gets 'Senior AI Engineer' under the name, at
the start of the summary and as the current job's title, because Engineer is the usual noun in AI, machine learning,
data, cloud, platform, DevOps, MLOps, site reliability and security work. Where Developer is itself the usual noun
('Software Developer', 'Full Stack Developer', 'iOS Developer', 'Business Developer') the title stays as the post writes
it, and phrases that are not titles ('AI developer tools', 'Developer Advocate') never change. The cover letter, History
and the job description file keep the post's own title. Every person, employer and product here is invented."""
import asyncio
import io

import pytest
from pypdf import PdfReader

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import languages, validate
from app.pipeline.tailor import run_tailor
from app.prompts import PROMPT_VERSION, tailor_system

from .test_title_and_language import BIZDEV_POST, posting
from .test_validate import make_output
from .test_writing_quality import five_jobs

AI_DEV_POST = (
    "Senior AI Developer at Contoso Learning. You will build the platform behind our AI features: LLM evaluation, "
    "retrieval-augmented generation (RAG), agent tool calling and observability. Required: Python, LLM evaluation, "
    "RAG, Kubernetes, PostgreSQL, CI/CD, AWS. Nice to have: HashiCorp Vault, pytest, Terraform."
)


# ------------------------------------------------------------------ the usual title of a field
@pytest.mark.parametrize("role,usual", [
    ("Senior AI Developer", "Senior AI Engineer"),
    ("AI Developer", "AI Engineer"),
    ("AI Developer II", "AI Engineer II"),
    ("Senior AI Developer L5", "Senior AI Engineer L5"),
    ("AI Developer Level 2", "AI Engineer Level 2"),
    ("AI Developer Intern", "AI Engineer Intern"),
    ("Lead AI/ML Developer", "Lead AI/ML Engineer"),
    ("Sr. AI Developer", "Sr. AI Engineer"),
    ("Mid-Level AI Developer", "Mid-Level AI Engineer"),
    ("AI Programmer", "AI Engineer"),
    ("Machine Learning Developer", "Machine Learning Engineer"),
    ("Machine-Learning Developer", "Machine-Learning Engineer"),
    ("Deep Learning Developer", "Deep Learning Engineer"),
    ("Computer Vision Developer", "Computer Vision Engineer"),
    ("Generative AI Developer", "Generative AI Engineer"),
    ("GenAI Developer", "GenAI Engineer"),
    ("LLM Developer", "LLM Engineer"),
    ("NLP Developer", "NLP Engineer"),
    ("AIOps Developer", "AIOps Engineer"),
    ("MLOps Developer", "MLOps Engineer"),
    ("Data Developer", "Data Engineer"),
    ("Big Data Developer", "Big Data Engineer"),
    ("AI Platform Developer", "AI Platform Engineer"),
    ("Staff Data Platform Developer", "Staff Data Platform Engineer"),
    ("Data and AI Developer", "Data and AI Engineer"),
    ("AI & Data Developer", "AI & Data Engineer"),
    ("AI-ML Developer", "AI-ML Engineer"),
    ("Cloud Developer", "Cloud Engineer"),
    ("Infrastructure Developer", "Infrastructure Engineer"),
    ("DevOps Developer", "DevOps Engineer"),
    ("Site Reliability Developer", "Site Reliability Engineer"),
    ("Cybersecurity Developer", "Cybersecurity Engineer"),
])
def test_a_developer_title_of_an_engineer_field_takes_engineer(role, usual):
    assert validate.usual_title(role) == usual


@pytest.mark.parametrize("role", [
    "Software Developer", "Web Developer", "Front End Developer", "Back-End Developer", "Full Stack Developer",
    "Mobile Developer", "iOS Developer", "Android Developer", "Game Developer", "Embedded Developer", "BI Developer",
    "ETL Developer", "SQL Developer", "Database Developer", "Salesforce Developer", "Salesforce Platform Developer",
    ".NET Developer", "Java Developer", "Python Developer", "Business Developer", "Real Estate Developer",
    "Instructional Developer", "Curriculum Developer",
    # a field Developer is usual in, or a word that is no field, before the noun
    "AI Software Developer", "Full Stack AI Developer", "Senior Voice AI Developer",
    # no field at all, more than one person, or not a title
    "Developer", "Senior Developer", "AI Developers", "AI Developer Advocate", "AI Developer-in-Residence",
    "Developer, AI Platform", "AI Developer/Engineer", "Data and Developer",
    "AI Engineer", "Senior AI Platform Engineer", "",
])
def test_a_title_where_developer_is_usual_stays(role):
    assert validate.usual_title(role) == role


def test_the_noun_keeps_the_letter_case_it_had():
    assert validate.usual_title("senior ai developer") == "senior ai engineer"
    assert validate.usual_title("SENIOR AI DEVELOPER") == "SENIOR AI ENGINEER"
    assert validate.usual_title("Senior AI developer") == "Senior AI engineer"
    assert validate.usual_title("  Senior AI Developer  ") == "Senior AI Engineer"


# ------------------------------------------------------------------ the common title of the post
@pytest.mark.parametrize("target,general,common,words", [
    # whatever the model offers, the post's own field with the usual noun
    ("Senior AI Developer", "AI Developer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer", "AI Engineer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer", "Software Engineer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer", "Software Developer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer", "", ("AI Developer", "AI Engineer"), ""),  # the post's own title when the model gave none
    ("AI Developer II", "AI Developer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI/ML Developer", "AI/ML Developer", ("AI/ML Developer", "AI/ML Engineer"), ""),
    ("Senior AI/ML Developer", "Machine Learning Engineer", ("AI/ML Developer", "AI/ML Engineer"), ""),
    ("AI Platform Developer", "AI Developer", ("AI Platform Developer", "AI Platform Engineer"), ""),
    ("Developer, AI Platform", "AI Developer", ("AI Platform Developer", "AI Platform Engineer"), ""),
    ("Data and AI Developer", "AI Developer", ("Data and AI Developer", "Data and AI Engineer"), ""),
    # a niche specialty still goes, and the noun becomes the usual one
    ("Senior Voice AI Developer", "AI Developer", ("Voice AI Developer", "AI Engineer"), "Voice"),
    ("Senior Voice AI Developer", "AI Engineer", ("Voice AI Developer", "AI Engineer"), "Voice"),
])
def test_a_developer_post_of_an_engineer_field_has_the_usual_common_title(target, general, common, words):
    assert validate.common_title(posting(target, general)) == common
    assert validate.specialty(posting(target, general)) == words


@pytest.mark.parametrize("target,general", [
    ("AI Platform Engineer", "AI Developer"),  # a noun is never a specialty
    ("Senior AI Engineer", "AI Developer"),
    ("Senior Software Engineer", "Software Developer"),
    ("Senior Software Developer", "Software Developer"),
    ("Senior Software Developer", "Software Engineer"),
    ("Senior iOS Developer", "iOS Developer"),
    ("Senior iOS Developer", "iOS Engineer"),
    ("Senior Full Stack Developer", "Full Stack Developer"),
    ("Senior Full Stack Developer", "Full Stack Engineer"),
    ("Business Developer", "Business Developer"),
    ("Business Developer", ""),
    ("Senior AI Platform Engineer", "AI Platform Engineer"),
])
def test_a_post_whose_noun_is_already_the_usual_one_keeps_its_title(target, general):
    assert validate.common_title(posting(target, general)) == ("", "")
    assert validate.specialty(posting(target, general)) == ""


# ------------------------------------------------------------------ the draft, as build() fixes it
def drafted(profile, analysis, summary, titles, headline="AI Developer"):
    """A draft with this summary and these job titles (the current job first), after build()'s title steps, and their
    notes."""
    resume = validate.build_resume(profile, make_output(headline=headline, summary=summary))
    for exp, title in zip(resume.experience, titles):
        exp.job_title = title
    validate.short_role_headline(resume)  # build()'s order
    validate.general_role_fixes(resume, analysis)
    notes = validate.fix_title_regressions(resume)
    notes += validate.role_level_fixes(resume, analysis)
    validate.same_title(resume)
    return resume, notes


def test_a_senior_ai_developer_post_reads_senior_ai_engineer_in_the_three_places(profile):
    analysis = posting("Senior AI Developer", "AI Developer")
    resume, notes = drafted(profile, analysis, (
        "Senior AI Developer with 9+ years of Python work. As a senior AI developer at Northwind Labs, built AI "
        "developer tools for 40 teams."), ["Senior AI Developer", "AI Developer", "Software Developer"],
        headline="Senior AI Developer")
    assert resume.headline == resume.experience[0].job_title == "Senior AI Engineer"
    assert resume.summary == ("Senior AI Engineer with 9+ years of Python work. As a senior AI engineer at Northwind "
                              "Labs, built AI developer tools for 40 teams.")  # 'AI developer tools' is no title
    assert [exp.job_title for exp in resume.experience] == ["Senior AI Engineer", "AI Engineer", "Software Developer"]
    assert notes == [] and validate.specialty_problems(resume, analysis) == []
    assert validate.title_path_problems(resume, analysis) == ([], False)
    validate.general_role_fixes(resume, analysis)  # build() runs again on the rewrite pass's draft: nothing moves
    validate.same_title(resume)
    assert resume.headline == "Senior AI Engineer" and resume.summary.startswith("Senior AI Engineer with 9+ years")


@pytest.mark.parametrize("summary,expected", [
    ("Senior AI Developer with 9+ years of Python work.", "Senior AI Engineer with 9+ years of Python work."),
    ("A senior AI developer with 9+ years of Python work.", "Senior AI Engineer with 9+ years of Python work."),
    ("AI Developer with 9+ years of Python work.", "Senior AI Engineer with 9+ years of Python work."),
    ("Senior AI developer, 9+ years into Python work.", "Senior AI Engineer, 9+ years into Python work."),
    # later in the summary, a role the person held: after 'a' or 'an', before punctuation or a small word
    ("Python work for 9+ years, the last 3 as an AI/ML developer at Northwind Labs.",
     "Python work for 9+ years, the last 3 as an AI/ML engineer at Northwind Labs."),
    ("Python work for 9+ years. Started as a data developer.", "Python work for 9+ years. Started as a data engineer."),
    ("Python work for 9+ years as a senior AI developer who ships weekly.",
     "Python work for 9+ years as a senior AI engineer who ships weekly."),
    # phrases that are not a title stay
    ("AI developer tools lead with 9+ years of Python work.", "AI developer tools lead with 9+ years of Python work."),
    ("AI Developer Advocate with 9+ years of Python work.", "AI Developer Advocate with 9+ years of Python work."),
    ("AI developer experience work for 9+ years.", "AI developer experience work for 9+ years."),
    ("Built an AI developer platform for 40 teams.", "Built an AI developer platform for 40 teams."),
    ("Leads an AI developer and data platform team.", "Leads an AI developer and data platform team."),
    ("Built AI developer tools for 40 teams.", "Built AI developer tools for 40 teams."),
    ("Mentored a software developer at Contoso.", "Mentored a software developer at Contoso."),
    ("Coached an AI developer advocate at Contoso.", "Coached an AI developer advocate at Contoso."),
    ("Hired 4 AI developers for the platform team.", "Hired 4 AI developers for the platform team."),
])
def test_the_summary_takes_the_usual_noun_only_where_it_names_a_role(profile, summary, expected):
    resume, notes = drafted(profile, posting("Senior AI Developer", "AI Developer"), summary,
                            ["Senior AI Developer", "AI Developer", "Software Developer"])
    assert resume.summary == expected and notes == []


@pytest.mark.parametrize("written,expected", [
    ("Senior AI Developer", "Senior AI Engineer"),
    ("AI Developer, Platform", "AI Engineer, Platform"),
    ("AI Developer (Remote)", "AI Engineer (Remote)"),
    ("AI Developer | LLM Apps", "AI Engineer | LLM Apps"),
    ("AI Developer - Payments", "AI Engineer - Payments"),
    ("AI Developer II", "AI Engineer II"),
    ("Data Developer", "Data Engineer"),
    ("Machine Learning Developer", "Machine Learning Engineer"),
    ("AI Developer Advocate", "AI Developer Advocate"),
    ("Developer Relations Lead", "Developer Relations Lead"),
    ("Software Developer", "Software Developer"),
    ("Front End Developer", "Front End Developer"),
    ("Developer, AI Platform", "Developer, AI Platform"),
])
def test_an_older_job_title_takes_the_usual_noun_when_it_ends_on_the_role(profile, written, expected):
    resume = validate.build_resume(profile, make_output(headline="AI Developer", summary="Built Python services."))
    resume.experience[1].job_title = written
    validate.general_role_fixes(resume, posting("Senior AI Developer", "AI Developer"))
    assert resume.experience[1].job_title == expected


def test_a_niche_developer_post_takes_the_common_title_and_the_usual_noun(profile):
    analysis = posting("Senior Voice AI Developer", "AI Developer")
    resume, notes = drafted(profile, analysis, (
        "Voice AI Developer with 9+ years of Python work. As a Voice AI Developer, built Voice AI developer tools."),
        ["Senior Voice AI Developer", "AI Developer", "Software Developer"], headline="Voice AI Developer")
    assert resume.headline == resume.experience[0].job_title == "Senior AI Engineer"
    assert resume.summary == ("Senior AI Engineer with 9+ years of Python work. As an AI Engineer, built Voice AI "
                              "developer tools.")  # a phrase that is no title stays, specialty and all
    assert [exp.job_title for exp in resume.experience] == ["Senior AI Engineer", "AI Engineer", "Software Developer"]
    assert notes == [] and validate.specialty_problems(resume, analysis) == ["summary"]


def test_a_niche_developer_titles_variant_takes_the_common_title_too(profile):
    analysis = posting("Senior Voice AI Developer", "AI Developer")
    resume, notes = drafted(profile, analysis, ("Senior Voice and Conversational AI Developer with 6+ years of speech "
                                                "work."),
                            ["Senior Voice and Conversational AI Developer", "AI Developer", "Software Developer"])
    assert resume.headline == resume.experience[0].job_title == "Senior AI Engineer"
    assert resume.summary == "Senior AI Engineer with 6+ years of speech work."
    assert [exp.job_title for exp in resume.experience] == ["Senior AI Engineer", "AI Engineer", "Software Developer"]
    assert notes == [] and validate.specialty_problems(resume, analysis) == []


def test_a_common_engineer_post_leaves_developer_titles_alone(profile):
    resume, notes = drafted(profile, posting("Senior AI Platform Engineer", "AI Platform Engineer"),
                            "Senior AI Platform Engineer with 9+ years of Python work.",
                            ["Senior AI Platform Engineer", "AI Developer", "Software Developer"])
    assert [exp.job_title for exp in resume.experience] == [
        "Senior AI Platform Engineer", "AI Developer", "Software Developer"]
    assert resume.headline == "Senior AI Platform Engineer" and notes == []


# ------------------------------------------------------------------ the post's level number
def test_the_posts_level_number_goes_from_its_usual_title_too(profile):
    analysis = posting("AI Developer II", "AI Developer")
    resume, notes = drafted(profile, analysis, "AI Developer II with 7+ years of Python work.",
                            ["Senior AI Developer", "AI Developer", "Software Developer"], headline="AI Developer II")
    assert resume.headline == resume.experience[0].job_title == "Senior AI Engineer"
    assert resume.summary == "Senior AI Engineer with 7+ years of Python work."
    assert notes == [("Summary: wrote 'Senior AI Engineer' instead of the posting's level 'AI Developer II', which "
                      "is not your title.")]


def test_the_level_note_names_the_form_the_summary_holds(profile):
    analysis = posting("AI Developer II", "AI Developer")
    resume, notes = drafted(profile, analysis, (
        "Senior AI Developer with 7+ years of Python work, the last 3 as an AI Developer II at Northwind Labs."),
        ["Senior AI Developer", "AI Developer", "Software Developer"])
    assert resume.summary == ("Senior AI Engineer with 7+ years of Python work, the last 3 as an AI Engineer at "
                              "Northwind Labs.")
    assert notes == [("Summary: wrote 'AI Engineer' instead of the posting's level 'AI Developer II', which is not "
                      "your title.")]


def test_a_current_title_with_the_posts_level_keeps_it(profile):
    analysis = posting("AI Developer II", "AI Developer")
    resume, notes = drafted(profile, analysis, "AI Developer II with 7+ years of Python work.",
                            ["AI Developer II", "AI Developer", "Software Developer"])
    assert resume.headline == resume.experience[0].job_title == "AI Engineer II"
    assert resume.summary == "AI Engineer II with 7+ years of Python work." and notes == []


# ------------------------------------------------------------------ whole runs
def test_a_senior_ai_developer_run_with_a_letter_keeps_the_posts_title_outside_the_resume(client):
    assert client.post("/api/profile", json=five_jobs().model_dump()).status_code == 200
    run = client.post("/api/tailor", json={"jd": AI_DEV_POST, "model": "demo-concise", "cover_letter": True,
                                           "company_name": "Contoso Learning"}).json()
    assert run["status"] == "ok" and run["retries"] == 0  # no rewrite pass
    resume = run["resume"]
    assert [exp["job_title"] for exp in resume["experience"]] == [
        "Senior AI Engineer", "AI Engineer", "Software Engineer", "Associate Software Engineer",
        "Junior Software Engineer"]
    assert resume["headline"] == "Senior AI Engineer" and resume["summary"].startswith("Senior AI Engineer with ")
    assert run["analysis"]["target_role"] == "Senior AI Developer" and run["analysis"]["general_role"] == "AI Engineer"
    assert run["cover_letter"]["paragraphs"][0].startswith(
        "I am writing to apply for the Senior AI Developer position at Contoso Learning.")
    run_id = run["run_id"]
    pdf = client.get(f"/api/runs/{run_id}/resume.pdf?download=1")
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf.content)).pages)
    assert "Senior AI Engineer" in text and "Developer" not in text
    letter = client.get(f"/api/runs/{run_id}/cover.pdf").content
    assert "Senior AI Developer position" in " ".join(
        " ".join(page.extract_text() for page in PdfReader(io.BytesIO(letter)).pages).split())
    assert client.get("/api/runs").json()["runs"][0]["target_role"] == "Senior AI Developer"  # History
    assert [row["role"] for row in client.get("/api/downloads").json()["downloads"]] == ["Senior AI Developer"]
    assert client.get(f"/api/runs/{run_id}/jd.txt").text.splitlines()[0] == (
        "Company: Contoso Learning | Job title: Senior AI Developer")


class KeepsDeveloper(LLMClient):
    """A model that writes the post's noun everywhere: 'AI Developer' as the common title, under the name, at the
    start of the summary and in the job titles."""

    def __init__(self, inner):
        self.inner = inner

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output":
            result.data["jd_analysis"].update(target_role="Senior AI Developer", general_role="AI Developer")
            sections = result.data["sections"]
            sections["headline"] = "Senior AI Developer"
            sections["summary"] = "Senior AI Developer with " + sections["summary"].split(" with ", 1)[1]
            titles = ["Senior AI Developer", "AI Developer", "Software Developer", "Associate Software Developer",
                      "Junior Software Developer"]
            for exp, title in zip(sections["experience"], titles):
                exp["job_title"] = title
        return result


def test_a_model_that_keeps_ai_developer_still_ends_with_engineer(router, db, registry):
    router.overrides["demo-concise"] = KeepsDeveloper(FakeClient(registry.get("demo-concise")))
    result = asyncio.run(run_tailor(router, db, profile=five_jobs(), profile_id=None, jd=AI_DEV_POST,
                                    model_id="demo-concise", save=False))
    assert result.status == "ok" and result.retries == 0
    resume = result.resume
    assert resume.headline == resume.experience[0].job_title == "Senior AI Engineer"
    assert resume.summary.startswith("Senior AI Engineer with ")
    assert [exp.job_title for exp in resume.experience] == [
        "Senior AI Engineer", "AI Engineer", "Software Developer", "Associate Software Developer",
        "Junior Software Developer"]  # Developer is the usual noun of software work
    assert result.analysis.target_role == "Senior AI Developer"


SAME_POSTS = [  # (post, the title it keeps)
    (("Senior Front End Developer at Contoso. You will build the React checkout in TypeScript. Required: React, "
      "TypeScript, CSS, Jest."), "Senior Front End Developer"),
    (("Senior Full Stack Developer at Fabrikam. You will build the order pages and their APIs in React and "
      "Node.js. Required: React, Node.js, PostgreSQL, AWS."), "Senior Full Stack Developer"),
    (("Senior iOS Developer at Fabrikam. You will build our iPhone app in Swift and SwiftUI. Required: Swift, iOS, "
      "Xcode, CI/CD."), "Senior iOS Developer"),
    (BIZDEV_POST, "Business Developer"),
    (("Senior AI Platform Engineer at Contoso Learning. You will build the platform behind our AI features. "
      "Required: Python, Kubernetes, PostgreSQL, CI/CD, AWS."), "Senior AI Platform Engineer"),
]


@pytest.mark.parametrize("jd,title", SAME_POSTS)
def test_the_demo_keeps_a_title_whose_noun_is_already_usual(router, db, jd, title):
    result = asyncio.run(run_tailor(router, db, profile=five_jobs(3), profile_id=None, jd=jd,
                                    model_id="demo-concise", save=False))
    assert result.status == "ok"
    assert result.resume.headline == result.resume.experience[0].job_title == title
    assert result.resume.summary.startswith(title + " with ")
    assert validate.common_title(result.analysis) == ("", "")


# ------------------------------------------------------------------ the prompt, the language rule
def test_the_prompt_names_the_usual_noun_of_each_field():
    text = tailor_system("estimate")
    assert "the target role title as the posting writes it" in text
    assert ("The role's noun is part of the common title too: in AI, machine learning, data, cloud, platform, "
            'infrastructure, DevOps, MLOps, site reliability and security the usual title says "Engineer", whatever '
            'noun the posting uses, so a "Developer" or "Programmer" title there is a special one ("AI Engineer" for '
            '"AI Developer", "Machine Learning Engineer" for "Machine Learning Developer", "Data Engineer" for "Data '
            'Developer"); where "Developer" is the usual noun the title stays as it is ("Software Developer", "Web '
            'Developer", "Frontend Developer", "Full Stack Developer", "iOS Developer", "Salesforce '
            'Developer").') in text
    assert '"Backend Engineer" for "Payments Backend Engineer", "AI Engineer" for "Senior AI Developer");' in text
    assert '(for a "Senior Voice AI Engineer" or a "Senior AI Developer" post: "Senior AI Engineer")' in text
    assert PROMPT_VERSION == "v40"


@pytest.mark.parametrize("target,general,usual", [
    ("Senior AI Developer", "AI Engineer", "Python"),
    ("Senior AI Developer", "AI Developer", "Python"),
    ("Data Developer", "Data Engineer", "Python"),
    ("Business Developer", "Business Developer", None),
    ("Real Estate Developer", "", None),
])
def test_the_usual_title_changes_no_language(target, general, usual):
    assert languages.usual_language(posting(target, general)) == usual


# ------------------------------------------------------------------ long crafted text
LONG_TEXTS = [
    "ai " * 5000 + "x", "ai/" * 5000, "data and " * 3000, "a ai " * 3000, "an " + "ai " * 5000 + "developer tools",
    "a " + "senior " * 3000 + "x", "ai" + " " * 5000 + "x", ("ai" + " " * 50) * 200 + "developer x",
    "a " + "ai - " * 3000 + "x", "a " + "ai / " * 3000 + "developer x", "a " + "ai and " * 3000 + "developer x",
    "a " + "machine learning " * 2000 + "developer x", " ".join(["as a ai developer x"] * 1000),
    "A " + "AI-" * 5000 + "Developer tools", "a " + "ai &" * 5000 + " developer x", "a ai/ml " * 3000,
    "a " + "ai ops " * 3000 + "x", "a " + "data ops " * 3000 + "developer x", "big data " * 3000 + "developer x",
    "a " + "cyber security " * 3000 + "developer tools", "a " + "llm ops & " * 3000 + "developer x",
    ("a ai " + "ai " * 50 + "x ") * 200,
]


@pytest.mark.parametrize("text", LONG_TEXTS)
def test_long_crafted_titles_and_summaries_are_read_in_time(profile, text):
    """Every pattern here is read in linear time: a pattern that backtracked without end would hang this test."""
    assert validate.usual_title(text) == text
    validate.common_title(posting(text, text))
    analysis = posting("Senior AI Developer", "AI Developer")
    resume = validate.build_resume(profile, make_output(headline="AI Developer", summary="Built Python services."))
    resume.summary, resume.experience[1].job_title = text, text[:80]
    validate.general_role_fixes(resume, analysis)
    validate.role_level_fixes(resume, posting("AI Developer II", "AI Developer"))
    validate.same_title(resume)


def test_a_long_title_of_fields_is_read_whole():
    assert validate.usual_title("ai " * 5000 + "developer") == "ai " * 5000 + "engineer"
    assert validate.usual_title("AI/" * 5000 + "ML Developer II") == "AI/" * 5000 + "ML Engineer II"
