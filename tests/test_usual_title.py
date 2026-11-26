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
    ("AI Developer Apprentice", "AI Engineer Apprentice"),
    ("Lead AI/ML Developer", "Lead AI/ML Engineer"),
    ("Sr. AI Developer", "Sr. AI Engineer"),
    ("Mid-Level AI Developer", "Mid-Level AI Engineer"),
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
    # Programmer, the usual noun at a game studio
    "AI Programmer", "Senior AI Programmer", "Platform Programmer", "Infrastructure Programmer", "Data Programmer",
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
    # an answer too long to be a title counts as none: the post's own title
    ("Senior AI Developer", "Artificial Intelligence / Machine Learning Engineer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer", "Generative AI and LLM Engineer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer", "Machine Learning / AI Developer", ("AI Developer", "AI Engineer"), ""),
    # what the short title leaves out, or a word the post never says, sets no title apart
    ("Senior AI Developer (Python)", "AI Engineer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer | Python | SQL", "AI Developer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer - Salesforce Einstein", "AI Engineer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI Developer", "Chatbot Developer", ("AI Developer", "AI Engineer"), ""),
    ("AI Developer II", "AI Developer", ("AI Developer", "AI Engineer"), ""),
    ("Senior AI/ML Developer", "AI/ML Developer", ("AI/ML Developer", "AI/ML Engineer"), ""),
    ("Senior AI/ML Developer", "Machine Learning Engineer", ("AI/ML Developer", "AI/ML Engineer"), ""),
    ("AI Platform Developer", "AI Developer", ("AI Platform Developer", "AI Platform Engineer"), ""),
    ("Developer, AI Platform", "AI Developer", ("AI Platform Developer", "AI Platform Engineer"), ""),
    ("Data and AI Developer", "AI Developer", ("Data and AI Developer", "Data and AI Engineer"), ""),
    # 'Big' and 'Cyber' start the fields 'Big Data' and 'Cyber Security': no specialty
    ("Senior Big Data Developer", "Data Engineer", ("Big Data Developer", "Big Data Engineer"), ""),
    ("Senior Cyber Security Developer", "Security Engineer", ("Cyber Security Developer", "Cyber Security Engineer"),
     ""),
    # the model's Engineer on the post's own words, when only the noun sets them apart
    ("Senior Applied AI Developer", "Applied AI Engineer", ("Applied AI Developer", "Applied AI Engineer"), ""),
    ("Senior Applied AI Developer", "applied ai engineer", ("Applied AI Developer", "Applied AI Engineer"), ""),
    ("LLM Application Developer", "LLM Application Engineer", ("LLM Application Developer", "LLM Application Engineer"),
     ""),
    ("Senior Applied AI Platform Developer", "Applied AI Engineer",
     ("Applied AI Platform Developer", "Applied AI Platform Engineer"), ""),
    # a niche specialty still goes, and the noun becomes the usual one
    ("Senior Voice AI Developer", "AI Developer", ("Voice AI Developer", "AI Engineer"), "Voice"),
    ("Senior Voice AI Developer", "AI Engineer", ("Voice AI Developer", "AI Engineer"), "Voice"),
    ("Senior Generative AI Developer", "AI Engineer", ("Generative AI Developer", "AI Engineer"), "Generative"),
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
    ("Senior Full Stack Developer", "Full Stack Web and Mobile Application Developer"),  # too long: the post's own
    ("Business Developer", "Business Developer"),
    ("Business Developer", ""),
    ("Senior AI Platform Engineer", "AI Platform Engineer"),
    ("Senior Big Data Engineer", "Data Engineer"),
    ("Senior Cyber Security Engineer", "Security Engineer"),
    # a game studio's title, whatever the model offers: Programmer is the usual noun there
    ("Senior AI Programmer", "Gameplay Programmer"),
    ("Senior AI Programmer", "Game AI Programmer"),
    ("Senior AI Programmer", "AI Programmer"),
    ("Senior AI Programmer", "AI Engineer"),
    ("Senior Applied AI Programmer", "Applied AI Engineer"),
    ("Senior Platform Programmer", "Platform Programmer"),
    # the product the short title leaves out, which the model's common title keeps with Developer
    ("Platform Developer (Salesforce)", "Salesforce Developer"),
    ("Platform Developer - ServiceNow", "ServiceNow Developer"),
    ("Platform Developer, Salesforce", "Salesforce Developer"),
    ("Platform Developer II (Salesforce)", "Salesforce Platform Developer"),
    # where Developer is the usual noun, the model's Engineer changes nothing
    ("AI Software Developer", "AI Software Engineer"),
    ("Full Stack AI Developer", "Full Stack AI Engineer"),
    ("Blockchain Developer", "Blockchain Engineer"),
    ("Salesforce Data Cloud Developer", "Salesforce Data Cloud Engineer"),
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
    ("Senior AI Developer / Tech Lead", "Senior AI Engineer / Tech Lead"),
    ("AI Developer/Team Lead", "AI Engineer/Team Lead"),
    ("AI Developer & Team Lead", "AI Engineer & Team Lead"),
    ("AI Developer and Team Lead", "AI Engineer and Team Lead"),
    ("AI Developer: LLM Apps", "AI Engineer: LLM Apps"),
    ("AI Developer [Contract]", "AI Engineer [Contract]"),
    ("AI Developer Lead", "AI Engineer Lead"),
    ("AI Developer Apprentice", "AI Engineer Apprentice"),
    ("AI Developer/Engineer", "AI Developer/Engineer"),  # never 'AI Engineer/Engineer'
    ("AI Developer & Engineer", "AI Developer & Engineer"),
    ("AI Developer and Engineer", "AI Developer and Engineer"),
    ("AI Developer Leadership Program", "AI Developer Leadership Program"),
    ("AI Developer Relations Lead", "AI Developer Relations Lead"),
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


@pytest.mark.parametrize("general", ["AI Developer", "AI Engineer"])
@pytest.mark.parametrize("current,expected", [
    ("Senior AI Developer / Tech Lead", "Senior AI Engineer / Tech Lead"),
    ("Senior AI Developer & Team Lead", "Senior AI Engineer & Team Lead"),
    ("Senior AI Developer and Team Lead", "Senior AI Engineer and Team Lead"),
])
def test_a_current_title_with_a_second_part_takes_the_usual_noun_under_the_name(profile, general, current, expected):
    resume, notes = drafted(profile, posting("Senior AI Developer", general), "Senior AI Developer with 9+ years.",
                            [current, "AI Developer", "Software Developer"], headline="Senior AI Developer")
    assert resume.headline == resume.experience[0].job_title == expected
    assert resume.summary == "Senior AI Engineer with 9+ years." and notes == []


@pytest.mark.parametrize("current", ["Senior Applied AI Engineer", "Senior Applied AI Developer"])
def test_a_common_title_with_only_another_noun_gives_engineer_in_the_three_places(profile, current):
    """The model's 'Applied AI Engineer' for an 'Applied AI Developer' post: the post's own words with Engineer,
    whether the draft's titles took it or kept Developer."""
    analysis = posting("Senior Applied AI Developer", "Applied AI Engineer")
    resume, notes = drafted(profile, analysis, (
        "Senior Applied AI Developer with 8+ years of Python work; built Applied AI developer tools for 40 teams."),
        [current, "AI Developer", "Software Engineer"], headline=current)
    assert resume.headline == resume.experience[0].job_title == "Senior Applied AI Engineer"
    assert resume.summary == ("Senior Applied AI Engineer with 8+ years of Python work; built Applied AI developer "
                              "tools for 40 teams.")  # a phrase that is no title stays
    assert [exp.job_title for exp in resume.experience] == [
        "Senior Applied AI Engineer", "AI Engineer", "Software Engineer"]
    assert notes == [] and validate.specialty_problems(resume, analysis) == []


@pytest.mark.parametrize("general", ["Artificial Intelligence / Machine Learning Engineer",
                                     "Generative AI and LLM Engineer"])
def test_a_common_title_too_long_to_be_one_leaves_the_posts_usual_title(profile, general):
    analysis = posting("Senior AI Developer", general)
    resume, notes = drafted(profile, analysis, "Senior AI Developer with 9+ years of Python work.",
                            ["Senior AI Developer", "AI Developer", "Software Engineer"],
                            headline="Senior AI Developer")
    assert resume.headline == resume.experience[0].job_title == "Senior AI Engineer"
    assert resume.summary == "Senior AI Engineer with 9+ years of Python work."
    assert [exp.job_title for exp in resume.experience] == ["Senior AI Engineer", "AI Engineer", "Software Engineer"]
    assert notes == []


def test_a_game_studios_programmer_post_keeps_its_titles(profile):
    """Programmer is the usual noun at a game studio: an 'AI Programmer' post keeps it under the name, in the summary
    and in the job titles."""
    analysis = posting("Senior AI Programmer", "Gameplay Programmer")
    resume, notes = drafted(profile, analysis, (
        "Senior AI Programmer with 8+ years building NPC behavior trees in C++ and Unreal."),
        ["Senior AI Programmer", "AI Programmer", "Gameplay Programmer"], headline="Senior AI Programmer")
    assert resume.headline == resume.experience[0].job_title == "Senior AI Programmer"
    assert resume.summary == "Senior AI Programmer with 8+ years building NPC behavior trees in C++ and Unreal."
    assert [exp.job_title for exp in resume.experience] == [
        "Senior AI Programmer", "AI Programmer", "Gameplay Programmer"]
    assert notes == []


def test_a_salesforce_platform_post_keeps_its_titles_and_asks_no_rewrite(profile):
    """'Platform Developer (Salesforce)', for which the model keeps 'Salesforce Developer': Developer is the usual noun
    of Salesforce work, so the titles stay and the path asks no rewrite toward 'Platform Engineer'."""
    analysis = posting("Platform Developer (Salesforce)", "Salesforce Developer")
    resume, notes = drafted(profile, analysis, "Senior Salesforce Developer with 6+ years of Apex work.",
                            ["Senior Salesforce Developer", "Salesforce Developer", "Salesforce Administrator"],
                            headline="Salesforce Developer")
    assert resume.headline == resume.experience[0].job_title == "Senior Salesforce Developer"
    assert resume.summary == "Senior Salesforce Developer with 6+ years of Apex work."
    assert [exp.job_title for exp in resume.experience] == [
        "Senior Salesforce Developer", "Salesforce Developer", "Salesforce Administrator"]
    assert notes == [] and validate.title_path_problems(resume, analysis) == ([], False)


def test_big_data_is_a_field_never_a_specialty(profile):
    analysis = posting("Senior Big Data Developer", "Data Engineer")
    resume, notes = drafted(profile, analysis, "Senior Big Data Engineer with 7+ years of big data pipelines on Spark.",
                            ["Senior Big Data Engineer", "Data Developer", "Software Developer"],
                            headline="Senior Big Data Engineer")
    assert resume.headline == resume.experience[0].job_title == "Senior Big Data Engineer"
    assert resume.summary == "Senior Big Data Engineer with 7+ years of big data pipelines on Spark."
    assert [exp.job_title for exp in resume.experience] == [
        "Senior Big Data Engineer", "Data Engineer", "Software Developer"]
    assert notes == [] and validate.specialty_problems(resume, analysis) == []


# ------------------------------------------------------------------ the noun step: only a post whose own noun changed
def test_a_niche_engineer_post_leaves_older_developer_titles_alone(profile):
    """A 'Voice AI Engineer' post changes no older title or summary phrase that a common 'AI Engineer' post keeps."""
    resume, notes = drafted(profile, posting("Senior Voice AI Engineer", "AI Engineer"),
                            "Senior AI Engineer with 8+ years, the first 3 as a data developer at Fabrikam.",
                            ["Senior AI Engineer", "Data Developer", "Cloud Developer"])
    assert [exp.job_title for exp in resume.experience] == ["Senior AI Engineer", "Data Developer", "Cloud Developer"]
    assert resume.summary == "Senior AI Engineer with 8+ years, the first 3 as a data developer at Fabrikam."
    assert notes == []


def test_a_niche_full_stack_developer_post_leaves_an_older_data_developer_alone(profile):
    resume, notes = drafted(profile, posting("Senior Payments Full Stack Developer", "Full Stack Developer"),
                            "Senior Full Stack Developer with 8+ years.",
                            ["Senior Full Stack Developer", "Full Stack Developer", "Data Developer"])
    assert [exp.job_title for exp in resume.experience] == [
        "Senior Full Stack Developer", "Full Stack Developer", "Data Developer"]
    assert notes == []


def test_a_niche_engineer_post_never_writes_its_special_title_into_an_older_job(profile):
    analysis = posting("Senior Computer Vision Engineer", "Machine Learning Engineer")
    resume, notes = drafted(profile, analysis, "Senior Machine Learning Engineer with 8+ years.",
                            ["Senior Machine Learning Engineer", "Computer Vision Developer", "Software Engineer"])
    assert resume.experience[1].job_title == "Computer Vision Developer"  # never 'Computer Vision Engineer'
    assert notes == []


@pytest.mark.parametrize("general", ["AI Developer Intern", "AI Engineer Intern"])
def test_a_post_with_a_word_after_its_noun_still_takes_the_usual_noun(profile, general):
    resume, notes = drafted(profile, posting("AI Developer Intern", general),
                            "AI Developer Intern with Python and PyTorch coursework.",
                            ["AI Developer Intern", "Data Developer", "Software Developer"])
    assert resume.headline == resume.experience[0].job_title == "AI Engineer Intern"
    assert resume.summary == "AI Engineer Intern with Python and PyTorch coursework."
    assert [exp.job_title for exp in resume.experience] == ["AI Engineer Intern", "Data Engineer", "Software Developer"]
    assert notes == []


def test_an_ai_developer_lead_post_gives_a_lead_ai_engineer(profile):
    resume, notes = drafted(profile, posting("AI Developer Lead", "AI Engineer"),
                            "Lead AI Developer with 9+ years of Python work.",
                            ["Lead AI Developer", "AI Developer", "Software Developer"])
    assert resume.headline == resume.experience[0].job_title == "Lead AI Engineer"
    assert resume.summary == "Lead AI Engineer with 9+ years of Python work."
    assert [exp.job_title for exp in resume.experience] == ["Lead AI Engineer", "AI Engineer", "Software Developer"]
    assert notes == []


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
    """A model that writes the post's noun everywhere: 'AI Developer' (or `general`) as the common title, and
    `current` under the name, at the start of the summary and as the current job's title."""

    def __init__(self, inner, general="AI Developer", current="Senior AI Developer"):
        self.inner, self.general, self.current = inner, general, current

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output":
            result.data["jd_analysis"].update(target_role="Senior AI Developer", general_role=self.general)
            sections = result.data["sections"]
            sections["headline"] = self.current
            sections["summary"] = self.current + " with " + sections["summary"].split(" with ", 1)[1]
            titles = [self.current, "AI Developer", "Software Developer", "Associate Software Developer",
                      "Junior Software Developer"]
            for exp, title in zip(sections["experience"], titles):
                exp["job_title"] = title
        return result


@pytest.mark.parametrize("general,current,usual", [
    ("AI Developer", "Senior AI Developer", "Senior AI Engineer"),
    # five words only for the spaced slash: an answer too long to be a title counts as none
    ("Machine Learning / AI Developer", "Senior Machine Learning / AI Developer",
     "Senior Machine Learning / AI Engineer"),
])
def test_a_model_that_keeps_ai_developer_still_ends_with_engineer(router, db, registry, general, current, usual):
    router.overrides["demo-concise"] = KeepsDeveloper(FakeClient(registry.get("demo-concise")), general, current)
    result = asyncio.run(run_tailor(router, db, profile=five_jobs(), profile_id=None, jd=AI_DEV_POST,
                                    model_id="demo-concise", save=False))
    assert result.status == "ok" and result.retries == 0
    resume = result.resume
    assert resume.headline == resume.experience[0].job_title == usual
    assert resume.summary.startswith(usual + " with ")
    assert [exp.job_title for exp in resume.experience] == [
        usual, "AI Engineer", "Software Developer", "Associate Software Developer",
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
            'infrastructure, DevOps, MLOps, site reliability and security the usual title says "Engineer", so a '
            '"Developer" title there is a special one ("AI Engineer" for "AI Developer", "Machine Learning Engineer" '
            'for "Machine Learning Developer", "Data Engineer" for "Data Developer"); where "Developer" is the usual '
            'noun the title stays as it is ("Software Developer", "Web Developer", "Frontend Developer", "Full Stack '
            'Developer", "iOS Developer", "Salesforce Developer", "Salesforce Platform Developer", "ServiceNow '
            'Developer", "Power Platform Developer"), and so does a "Programmer" title, the usual noun at a game '
            'studio ("AI Programmer", "Gameplay Programmer").') in text
    assert "whatever noun the posting uses" not in text
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
    ("a ai " + "ai " * 50 + "x ") * 200, "applied " * 3000 + "ai developer", "power" + " " * 5000 + "x",
    "a " * 5000 + "developer", "Platform Developer (" + "Salesforce " * 2000 + ")",
]


@pytest.mark.parametrize("text", LONG_TEXTS)
def test_long_crafted_titles_and_summaries_are_read_in_time(profile, text):
    """Every pattern here is read in linear time: a pattern that backtracked without end would hang this test."""
    assert validate.usual_title(text) == text
    validate.common_title(posting(text, text))
    validate.common_title(posting(text, "Applied AI Engineer"))  # the model's noun on the post's own words
    validate.common_title(posting(text, "Salesforce Developer"))  # a word the post's short title leaves out
    analysis = posting("Senior AI Developer", "AI Developer")
    resume = validate.build_resume(profile, make_output(headline="AI Developer", summary="Built Python services."))
    resume.summary, resume.experience[1].job_title = text, text[:80]
    validate.general_role_fixes(resume, analysis)
    validate.role_level_fixes(resume, posting("AI Developer II", "AI Developer"))
    validate.same_title(resume)


def test_a_long_title_of_fields_is_read_whole():
    assert validate.usual_title("ai " * 5000 + "developer") == "ai " * 5000 + "engineer"
    assert validate.usual_title("AI/" * 5000 + "ML Developer II") == "AI/" * 5000 + "ML Engineer II"
