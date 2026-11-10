"""The US style / EU style spelling rules, case by case: the job post's keywords keep the post's spelling and are never
reported, product names and code keep theirs, the role always follows the style, a '- ' list item starts a sentence,
an EU letter says CV for the document, and an EU file is British English inside too."""
import asyncio
import io
import zipfile

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import spelling, validate
from app.pipeline.tailor import run_tailor
from app.schemas import CoverLetter, JDAnalysis, SkillGroup

from .conftest import SAMPLE_JD
from .test_api import save_profile
from .test_validate import make_output


def posting(target_role="Senior Data Engineer", keywords=(), must=(), company="", general_role=""):
    return JDAnalysis(company=company, target_role=target_role, general_role=general_role, seniority="Senior",
                      domain="Data", must_have_skills=list(must), nice_to_have_skills=[], keywords=list(keywords),
                      key_responsibilities=["Build pipelines"])


def spelled(profile, style, analysis, names=("", ""), **sections):
    out = make_output(**sections)
    out.jd_analysis = analysis
    resume = validate.build_resume(profile, out)
    validate.spell_resume(resume, style, analysis, names)
    return resume


def letter_in(style, resume, analysis, *paragraphs, names=("", "")):
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[*paragraphs, "Thank you."])
    validate.spell_letter(letter, style, validate.known_names(resume, analysis, *names),
                          validate.skill_terms(resume, analysis), validate.posting_terms(analysis),
                          validate.role_terms(resume, analysis))
    return letter


# ------------------------------------------------------------------ the post's keywords are never reported
@pytest.mark.parametrize("style,keywords,model_wrote,post_says", [
    ("eu", ["Data Modeling", "Query Optimization"], "data modelling and query optimisation",
     "data modeling and query optimization"),  # a US post, its keywords in Title Case
    ("us", ["Data Modelling", "Data Catalogue", "Customer Behaviour"], "data modeling, a data catalog and customer behavior",
     "data modelling, a data catalogue and customer behaviour"),  # a UK post
])
def test_title_case_keywords_keep_the_posts_spelling_and_get_no_note(profile, style, keywords, model_wrote, post_says):
    analysis = posting(keywords=keywords, must=keywords[:1])
    resume = spelled(profile, style, analysis,
                     summary=f"I own {model_wrote} for a warehouse used by 38 analysts, and I organize reviews.")
    organise = "organise" if style == "eu" else "organize"
    assert resume.summary == f"I own {post_says} for a warehouse used by 38 analysts, and I {organise} reviews."
    letter = letter_in(style, resume, analysis, f"At Northwind Labs I own {model_wrote} for 38 analysts.")
    assert letter.paragraphs[0] == f"At Northwind Labs I own {post_says} for 38 analysts."
    assert validate.spelling_problems(resume, letter, style, analysis) == []


def test_a_word_in_neither_the_style_nor_the_posts_spelling_is_still_reported(profile):
    analysis = posting(keywords=["Data Modeling"])  # a US post and a US style resume
    resume = spelled(profile, "us", analysis)
    resume.summary = "I own data modelling for 38 analysts and lead the organisation's reviews."  # the user's edit
    assert validate.spelling_problems(resume, None, "us", analysis) == ["modelling", "organisation"]


class Rewrites(LLMClient):
    """The demo model, with the analysis a real model writes (the post's keywords in Title Case) and British words in
    the cover letter."""

    def __init__(self, inner):
        self.inner = inner

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output":
            analysis = result.data["jd_analysis"]
            analysis["must_have_skills"] = ["Data Modeling", "Query Optimization", "Python", "SQL", "Airflow"]
            analysis["keywords"] = ["Data Modeling", "Query Optimization", "Python", "SQL", "Snowflake", "ETL"]
        elif kwargs["schema_name"] == "cover_letter":
            result.data["paragraphs"][1] = (
                "At Northwind Labs I own the data modelling and the query optimisation for a Snowflake warehouse that "
                "38 analysts use every day, and I organised the on-call rotation.")
        return result


def test_an_eu_run_for_a_us_post_with_title_case_keywords_has_no_false_spelling_note(router, db, registry, profile):
    jd = ("Senior Data Engineer, Northbeam Data (New York). You will own Data Modeling and Query Optimization for our "
          "Snowflake warehouse, build ETL pipelines in Python, SQL and Airflow, and work with stakeholders to optimize "
          "and analyze our data. Must have: Data Modeling, Query Optimization, Python, SQL, Airflow, Snowflake.")
    router.overrides["demo-concise"] = Rewrites(FakeClient(registry.get("demo-concise")))
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    cover_letter=True, company_name="Northbeam Data", spelling="eu"))
    assert result.status == "ok"
    assert "the data modeling and the query optimization for a Snowflake" in result.cover_letter.paragraphs[1]
    assert "I organised the on-call rotation" in result.cover_letter.paragraphs[1]
    assert not any(w.startswith("Words in American spelling") for w in result.ats_warnings)
    assert any(w.startswith("This job post uses American spelling") for w in result.ats_warnings)


# ------------------------------------------------------------------ product names keep their owner's spelling
def test_a_data_modeller_role_never_renames_the_data_modeling_tools(profile):
    analysis = posting(target_role="Senior Data Modeller")
    tools = [SkillGroup(category="Tools", skills=["erwin Data Modeler", "Toad Data Modeler"])]
    resume = spelled(profile, "eu", analysis, headline="Data Modeller", skills=tools,
                     summary="Data Modeller with 8 years of dimensional modelling in erwin Data Modeler and Toad Data "
                             "Modeler for finance warehouses.")
    resume.experience[0].job_title = "Senior Data Modeler"
    resume.experience[0].bullets = [("Designed 120 dimensional models in erwin Data Modeler and Oracle SQL Developer "
                                     "Data Modeler for the finance warehouse.")]
    validate.spell_resume(resume, "eu", analysis)
    assert resume.experience[0].job_title == "Senior Data Modeller"  # the job follows the style
    assert resume.summary == ("Data Modeller with 8 years of dimensional modelling in erwin Data Modeler and Toad Data "
                              "Modeler for finance warehouses.")
    assert resume.experience[0].bullets[0] == ("Designed 120 dimensional models in erwin Data Modeler and Oracle SQL "
                                               "Developer Data Modeler for the finance warehouse.")
    assert resume.skills[0].skills == ["erwin Data Modeler", "Toad Data Modeler"]
    letter = letter_in("eu", resume, analysis, "As a Data Modeller I use erwin Data Modeler and SQL Developer Data "
                                               "Modeler every day.")
    assert letter.paragraphs[0] == "As a Data Modeller I use erwin Data Modeler and SQL Developer Data Modeler every day."
    resume.experience[1].bullets = ["Kept erwin Data Modeler models in Git."]  # the user restores the name
    assert validate.spelling_problems(resume, letter, "eu", analysis) == []


@pytest.mark.parametrize("style", ["eu", "us"])
def test_a_posts_keyword_never_renames_a_product_that_holds_it(profile, style):
    analysis = posting(keywords=["data catalogue"])  # a UK post
    governance = [SkillGroup(category="Governance", skills=["AWS Glue Data Catalog", "Collibra Data Catalog",
                                                            "Azure Data Catalog", "Data Catalog"])]
    resume = spelled(profile, style, analysis, skills=governance)
    resume.experience[0].bullets = [("Rolled out the AWS Glue Data Catalog to 40 teams, cutting time-to-find-data from "
                                     "3 days to 4 hours."), "Built a data catalog for 40 teams."]
    validate.spell_resume(resume, style, analysis)
    assert resume.skills[0].skills == ["AWS Glue Data Catalog", "Collibra Data Catalog", "Azure Data Catalog",
                                       "Data Catalogue"]  # the generic phrase is the post's keyword
    assert resume.experience[0].bullets == [
        "Rolled out the AWS Glue Data Catalog to 40 teams, cutting time-to-find-data from 3 days to 4 hours.",
        "Built a data catalogue for 40 teams."]


@pytest.mark.parametrize("keyword,skill", [
    ("Data Modeller", "erwin Data Modeler"), ("Data Modeller", "Toad Data Modeler"), ("data centre", "Jira Data Center"),
    ("data centre", "Confluence Data Center"), ("bundle analyser", "Webpack Bundle Analyzer"),
    ("catalogue", "Unity Catalog"), ("user behaviour", "Splunk User Behavior Analytics"), ("organisations", "AWS Organizations"),
])
@pytest.mark.parametrize("style", ["eu", "us"])
def test_a_posts_keyword_never_renames_a_product_in_the_skills(style, keyword, skill):
    assert spelling.convert(skill, style, title=True, exact=[keyword]) == skill


@pytest.mark.parametrize("style,text,title,keyword,expected", [
    ("eu", "Fibre broadband rollout for 2M homes.", False, "fiber broadband", "Fiber broadband rollout for 2M homes."),
    ("eu", "Fiber Optics", True, "fibre optics", "Fibre Optics"),  # a skill
    ("us", "Tuned React Fiber rendering for 2M users.", False, "fibre", "Tuned React Fiber rendering for 2M users."),
])
def test_a_word_that_is_often_a_name_takes_the_posts_spelling_unless_it_is_inside_one(style, text, title, keyword,
                                                                                       expected):
    assert spelling.convert(text, style, title=title, exact=[keyword]) == expected


@pytest.mark.parametrize("skill", [
    "IBM ILOG CPLEX Optimization Studio", "AWS Cost Optimization Hub", "Salesforce Marketing Cloud Personalization",
    "Splunk User Behavior Analytics (UBA)", "Google Maps Route Optimization API", "Google Cloud Armor",
    "SAP Organizational Management (OM)", "Oracle Labor Distribution",
])
def test_a_products_name_after_a_brand_keeps_its_spelling(skill):
    assert spelling.convert(skill, "eu", title=True) == skill
    assert spelling.convert(f"Built 14 models in {skill} for the planning team.", "eu", terms=[skill]) == (
        f"Built 14 models in {skill} for the planning team.")


@pytest.mark.parametrize("skill,expected", [
    ("Snowflake Data Modeling", "Snowflake Data Modelling"), ("Tableau Visualization", "Tableau Visualisation"),
    ("Spark Optimizations", "Spark Optimisations"), ("Tableau Visualizations", "Tableau Visualisations"),
    ("Salesforce Customizations", "Salesforce Customisations"), ("AWS Organizations", "AWS Organizations"),
])
def test_an_activity_after_a_brand_follows_the_style(skill, expected):
    assert spelling.convert(skill, "eu", title=True) == expected


@pytest.mark.parametrize("group,skills,expected", [
    ("Cloud and Infrastructure (AWS, Azure, VMware)", ["Data Center Migration", "Data Center Operations"],
     ["Data Centre Migration", "Data Centre Operations"]),
    ("Cloud (AWS)", ["Data Centers"], ["Data Centres"]),
    ("Frontend (React)", ["Color Systems", "Performance Optimizations"], ["Colour Systems", "Performance Optimisations"]),
    ("Salesforce", ["Flows", "Customizations", "Data Visualizations"], ["Flows", "Customisations", "Data Visualisations"]),
    ("Adobe Creative Cloud", ["Premiere Pro", "Color Grading", "Color Correction"],
     ["Premiere Pro", "Colour Grading", "Colour Correction"]),
    ("Azure", ["Azure Migrate", "Data Center Migration", "Security Center", "Data Catalog"],
     ["Azure Migrate", "Data Centre Migration", "Security Center", "Data Catalog"]),
    ("AWS", ["Organizations", "Organizations (SCPs)", "Compute Optimizer", "Personalize", "Cost Optimization"],
     ["Organizations", "Organizations (SCPs)", "Compute Optimizer", "Personalize", "Cost Optimisation"]),
])
def test_in_a_group_named_after_a_brand_only_a_products_short_name_keeps_its_spelling(group, skills, expected):
    brand = spelling.names_brand(group)
    assert [spelling.convert(skill, "eu", title=True, brand=brand) for skill in skills] == expected


# ------------------------------------------------------------------ the role follows the style
def test_the_role_follows_the_style_even_where_it_is_one_of_the_posts_keywords(profile):
    analysis = posting(target_role="Senior Data Modeler", keywords=["Data Modeler", "dimensional modeling", "erwin"])
    resume = spelled(profile, "eu", analysis, headline="Data Modeler",
                     summary="Data Modeler with 8 years of dimensional modeling in erwin.")
    assert resume.headline == "Data Modeller"
    assert resume.summary == "Data Modeller with 8 years of dimensional modeling in erwin."  # the keyword stays
    resume.headline = "Data Modeler"  # the user writes the post's own keyword back: no note asks to undo it
    assert validate.spelling_problems(resume, None, "eu", analysis) == []

    analysis = posting(target_role="Data Center Technician", keywords=["data center"])
    resume = spelled(profile, "eu", analysis, headline="Data Center Technician",
                     summary="Data Center Technician with 6 years of rack and stack work in our data center.")
    resume.experience[0].job_title = "Data Center Technician"
    validate.spell_resume(resume, "eu", analysis)
    assert resume.headline == resume.experience[0].job_title == "Data Centre Technician"
    assert resume.summary == "Data Centre Technician with 6 years of rack and stack work in our data center."


def test_the_hiring_companys_first_word_never_holds_back_the_role_or_the_skills(profile):
    analysis = posting(target_role="Senior Analog Design Engineer", company="Analog Devices, Inc.")
    resume = spelled(profile, "eu", analysis, ("Analog Devices, Inc.", ""), headline="Analog Design Engineer",
                     skills=[SkillGroup(category="Core", skills=["Analog Circuit Design", "SPICE"])],
                     summary="Analog design engineer with 8 years of analog front-end work for automotive sensors.")
    resume.experience[0].job_title = "Senior Analog Design Engineer"
    validate.spell_resume(resume, "eu", analysis, ("Analog Devices, Inc.", ""))
    assert resume.headline == "Analogue Design Engineer"
    assert resume.experience[0].job_title == "Senior Analogue Design Engineer"
    assert resume.skills[0].skills == ["Analogue Circuit Design", "SPICE"]
    assert resume.summary == "Analogue design engineer with 8 years of analogue front-end work for automotive sensors."
    letter = letter_in("eu", resume, analysis, "Analog Devices' sensors and Analog's Limerick team are why I apply.",
                       names=("Analog Devices, Inc.", ""))
    assert letter.paragraphs[0] == "Analog Devices' sensors and Analog's Limerick team are why I apply."


def test_a_one_word_name_keeps_its_spelling_where_it_is_the_name(profile):
    keep = validate.known_names(validate.build_resume(profile, make_output()), None, "Color", "Avery Gray")
    assert spelling.convert("Engineer at Color", "eu", keep, title=True, roles=["Engineer at Color"]) == (
        "Engineer at Color")
    assert spelling.convert("At Color, Ms. Gray optimized color pipelines.", "eu", keep) == (
        "At Color, Ms. Gray optimised colour pipelines.")
    assert spelling.convert("Gray Box Testing", "eu", keep, title=True) == "Grey Box Testing"  # a skill


# ------------------------------------------------------------------ where a sentence starts
def test_a_posts_dash_list_starts_sentences_so_its_spelling_is_found():
    uk = ("Data Analyst - Harbour & Finch (Leeds, hybrid)\n\nWhat you'll do:\n- Analyse sales and stock data for 40 "
          "stores with SQL and Python\n- Visualise weekly trends in Power BI\n- Organise our reporting catalogue\n"
          "- Optimise SQL queries in our Snowflake warehouse\n")
    assert spelling.detect(uk) == "eu"
    assert spelling.detect(uk.replace("\n- ", "\n– ")) == "eu" and spelling.detect(uk.replace("\n- ", "\no ")) == "eu"
    us = uk.replace("Analyse", "Analyze").replace("Visualise", "Visualize").replace("Organise", "Organize").replace(
        "Optimise", "Optimize")
    assert spelling.detect(us) == "us"
    # a dash inside a line is no list: 'Harbour' and 'Colour' are names there
    assert spelling.detect("Role - Harbour & Finch\nWe use Harbour tools and Colour Systems.") == ""
    assert spelling.convert("Northwind - Optimized Systems Ltd built it.", "eu") == (
        "Northwind - Optimized Systems Ltd built it.")


def test_a_us_run_for_a_uk_post_with_dash_bullets_suggests_eu_style(router, db, profile):
    jd = ("Senior Data Engineer\n\nWhat you will do\n- Optimise data models in Snowflake for 40 analysts\n"
          "- Prioritise data quality across the warehouse\n- Analyse pipeline failures and fix their causes\n"
          "- Build ETL pipelines with Python, SQL, Airflow and Spark on AWS\n")
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    spelling="us"))
    assert any(w.startswith("This job post uses British spelling, but this resume uses US style")
               for w in result.ats_warnings)


@pytest.mark.parametrize("style,given,expected", [
    ("eu", "Cut p95 latency from 820 to 648 ms. Optimized the Spark jobs that feed 40 dashboards.",
     "Cut p95 latency from 820 to 648 ms. Optimised the Spark jobs that feed 40 dashboards."),
    ("eu", "Brought p95 latency down from 820 ms to 640 ms. Organizing the on-call rotation also cut pages by half.",
     "Brought p95 latency down from 820 ms to 640 ms. Organising the on-call rotation also cut pages by half."),
    ("us", "Cut p95 latency from 820 to 648 ms. Optimised the Spark jobs that feed 40 dashboards.",
     "Cut p95 latency from 820 to 648 ms. Optimized the Spark jobs that feed 40 dashboards."),
    ("eu", "Our team ranked 1st. Organized the quarterly hackathon for 200 engineers.",
     "Our team ranked 1st. Organised the quarterly hackathon for 200 engineers."),
    ("eu", "Thank you, Ms. Gray. Organized teams matter.", "Thank you, Ms. Gray. Organised teams matter."),
    ("eu", "Duties:\n- Optimized the warehouse\n1) Organized the team", "Duties:\n- Optimised the warehouse\n1) Organised the team"),
])
def test_a_sentence_after_ms_or_1st_or_in_a_list_is_a_new_sentence(style, given, expected):
    assert spelling.convert(given, style) == expected


def test_a_name_after_a_title_is_still_a_name():
    assert spelling.convert("I met Dr. Gray at the summit.", "eu") == "I met Dr. Gray at the summit."
    assert spelling.convert("Moved to St. Louis. Organized the team.", "eu") == "Moved to St. Louis. Organised the team."


# ------------------------------------------------------------------ code keeps its spelling, English does not
@pytest.mark.parametrize("style,given,expected", [
    ("eu", "Built ranking models that make personalized recommendations for 2M shoppers.",
     "Built ranking models that make personalised recommendations for 2M shoppers."),
    ("eu", "I help leaders make organizational decisions with data.", "I help leaders make organisational decisions with data."),
    ("eu", "Worked with finance to make standardized reports for 12 regions.",
     "Worked with finance to make standardised reports for 12 regions."),
    ("us", "Built ranking models that make personalised recommendations for 2M shoppers.",
     "Built ranking models that make personalized recommendations for 2M shoppers."),
    ("eu", "Partnered with 3 partner organization(s) on data sharing.", "Partnered with 3 partner organisation(s) on data sharing."),
    ("eu", "Built visualization(s) for the board.", "Built visualisation(s) for the board."),
    ("eu", "Moved the team to an api-centered design.", "Moved the team to an api-centred design."),
    ("eu", "Cut payload size by 40% with json-serialized events.", "Cut payload size by 40% with json-serialised events."),
    ("eu", "Owned build/optimization work for the CI pipeline.", "Owned build/optimisation work for the CI pipeline."),
    ("eu", "Led test/optimization cycles for the checkout flow.", "Led test/optimisation cycles for the checkout flow."),
    ("eu", "Owned analysis/modeling/visualization for 38 analysts.", "Owned analysis/modelling/visualisation for 38 analysts."),
])
def test_english_that_looks_like_code_follows_the_style(style, given, expected):
    assert spelling.convert(given, style) == expected
    assert spelling.other_style_words(given, style) != []  # and a user's own such word gets the note


@pytest.mark.parametrize("text", [
    "Ran make optimize in CI.", "Ran `make optimize` in CI.", "Ran scripts/optimize nightly.",
    "Moved src/normalize, utils/normalize and app/utils/normalize into one module.", "Used json-normalize.",
    "Used sanitize-html and color-loader.", "Replaced apply() with normalize(df).",
])
def test_code_paths_and_make_targets_keep_their_spelling(text):
    assert spelling.convert(text, "eu") == text


# ------------------------------------------------------------------ the letter: CV and the sign-off
@pytest.mark.parametrize("given,expected", [
    ("I have attached my resume and writing samples.", "I have attached my CV and writing samples."),
    ("I have attached my resume and project portfolio.", "I have attached my CV and project portfolio."),
    ("Please find my resume and data portfolio attached.", "Please find my CV and data portfolio attached."),
    ("I attached my resume and two writing samples.", "I attached my CV and two writing samples."),
    ("As the resume shows, I have led data teams for 8 years.", "As the CV shows, I have led data teams for 8 years."),
    ("As this resume shows, I have led data teams for six years.", "As this CV shows, I have led data teams for six years."),
    ("The resume has more detail on each project, and the resume and portfolio links are below.",
     "The CV has more detail on each project, and the CV and portfolio links are below."),
    ("Please find my resume/CV attached.", None), ("My resume (CV) is attached.", None),
    ("My resume or CV is attached.", None), ("I built my resume to job matching service.", None),
    ("I led the resume and cover letter generator project.", None),
])
def test_an_eu_letter_says_cv_for_the_document_and_never_cv_twice(given, expected):
    assert spelling.cv_wording(given, "eu") == (expected or given)
    assert spelling.cv_wording(given, "us") == given


@pytest.mark.parametrize("closing", ["Very truly yours,", "Truly yours,"])
def test_more_american_sign_offs_become_kind_regards(closing):
    assert spelling.sign_off(closing, "eu") == "Kind regards," and spelling.sign_off(closing, "us") == closing


# ------------------------------------------------------------------ the files
def _text_language(docx: bytes) -> str:
    styles = zipfile.ZipFile(io.BytesIO(docx)).read("word/styles.xml").decode()
    return styles.split("<w:lang ", 1)[1].split('w:val="', 1)[1].split('"', 1)[0]


def test_eu_word_files_proof_their_text_as_british_english(client):
    save_profile(client)
    languages = {}
    for style in ("eu", "us"):
        run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "spelling": style}).json()
        client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})
        languages[style] = [_text_language(client.get(f"/api/runs/{run['run_id']}/{name}.docx").content)
                            for name in ("resume", "cover")]
    assert languages == {"eu": ["en-GB", "en-GB"], "us": ["en-US", "en-US"]}
