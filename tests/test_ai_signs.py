"""The four signs a recruiter reads as an AI-written CV, and what the app does about each: dashes used as
punctuation, the wrong spelling for the market (the US style / EU style buttons), percentages without their base, and
vague results ("improved operational efficiency")."""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import spelling, validate
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, cover_system, tailor_system
from app.schemas import CoverLetter

from .conftest import SAMPLE_JD
from .test_api import save_profile
from .test_validate import make_output


def tailor(router, db, profile, jd, model="demo-concise", **kw):
    return asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id=model, **kw))


class Injects(LLMClient):
    """Changes the first draft's sections; the fix round's answer (the demo's own text) comes back unchanged."""

    def __init__(self, inner, change):
        self.inner, self.change, self.prompts = inner, change, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            self.change(result.data["sections"])
        return result


def all_text(result) -> str:
    resume = result.resume
    parts = [resume.headline, resume.summary, *(s for g in resume.skills for s in g.skills),
             *(e.job_title for e in resume.experience), *(b for e in resume.experience for b in e.bullets)]
    if result.cover_letter:
        parts += result.cover_letter.paragraphs
    return "\n".join(parts)


def edit_body(run: dict, **changes) -> dict:
    resume = run["resume"]
    body = {"headline": resume["headline"], "summary": resume["summary"], "skills": resume["skills"],
            "experience": [{"company_index": e["company_index"], "job_title": e["job_title"], "bullets": e["bullets"]}
                           for e in resume["experience"]]}
    body.update(changes)
    return body


# ------------------------------------------------------------------ 1. dashes
@pytest.mark.parametrize("given,expected", [
    ("X — aside — Y.", "X, aside, Y."), ("17.8%—saving", "17.8%, saving"), ("Airflow--no", "Airflow, no"),
    ("Airflow -- no", "Airflow, no"), ("Airflow --- no", "Airflow, no"), ("it – on time.", "it, on time."),
    ("it - on time.", "it, on time."), ("results —", "results"), ("— Senior", "Senior"),
    ("Done, — fast", "Done, fast"), ("Done —.", "Done."), ("Fixed it: — all", "Fixed it: all"),
    ("2019 - 2021", "2019-2021"), ("Q1 – Q3", "Q1-Q3"), ("FY23 — FY24", "FY23-FY24"),
    # a number before the dash and another after it is no range unless both look alike
    ("Cut failed nightly runs from 101 to 83 — 17.8% fewer — with staged rollouts.",
     "Cut failed nightly runs from 101 to 83, 17.8% fewer, with staged rollouts."),
    ("Saved $41,300 — 17.8% of the annual AWS bill — by rightsizing EC2.",
     "Saved $41,300, 17.8% of the annual AWS bill, by rightsizing EC2."),
    ("Led the move to Kubernetes in 2022 — 14 services in 3 months.", "Led the move to Kubernetes in 2022, 14 services in 3 months."),
    # ranges between like values lose their spaces
    ("Handled 10K – 50K events per second.", "Handled 10K-50K events per second."),
    ("Closed deals of $5M – $10M each.", "Closed deals of $5M-$10M each."), ("Grew revenue 5% – 10% a quarter.", "Grew revenue 5%-10% a quarter."),
    ("Upgraded from React v16 – v18.", "Upgraded from React v16-v18."), ("Owned Sev1 – Sev3 triage.", "Owned Sev1-Sev3 triage."),
    ("Supported Tier 1 – Tier 3 escalations.", "Supported Tier 1 to Tier 3 escalations."),
    ("Ran it (2021 – present) from Jan – Mar 2023.", "Ran it (2021-present) from Jan-Mar 2023."),
    ("Tracked p50 – p99 latency.", "Tracked p50-p99 latency."),
    # non-breaking spaces, and punctuation already in place
    ("Built the Airflow pipeline\u00a0– so 38 teams got data by 7am.", "Built the Airflow pipeline, so 38 teams got data by 7am."),
    ("Shipped on time. — Then cut costs.", "Shipped on time. Then cut costs."),
    ("Did it scale? — Yes, to 83K users.", "Did it scale? Yes, to 83K users."),
])
def test_a_dash_used_as_punctuation_becomes_a_comma(given, expected):
    assert validate.fix_dashes(given) == expected and validate.fix_dashes(expected) == expected


@pytest.mark.parametrize("text", [
    "event-driven", "end-to-end", "5-10", "COVID-19", "GPT-4", "x86-64", "Python- and SQL-based", "a -12% swing",
    "ran plan --refresh-only daily", "New York–London", "AI-/ML-based", "pre- and post-processing", "Python 3.11-3.13",
    "Added a dry-run mode (--dry-run) to the deploy CLI.", 'Added a `--dry-run` flag and a "--force" option (--dry-run, --verbose).',
])
def test_hyphens_ranges_and_flags_stay(text):
    assert validate.fix_dashes(text) == text


def test_flags_and_ranges_typed_in_the_editor_get_no_dash_note(profile):
    out = make_output()
    out.sections.experience[0].bullets = ["Ran terraform plan --refresh-only before every apply in 2019 - 2021."]
    assert validate.dash_problems(validate.build_resume(profile, out, from_model=False)) == []


def test_odd_text_never_slows_or_breaks_the_dash_fix():
    import time

    started = time.perf_counter()
    assert validate.fix_dashes("x" + " " * 32000 + "— y") == "x, y"
    assert time.perf_counter() - started < 0.5
    assert validate.clean_prose("a\x001\x00 — b") == "a1, b"  # a NUL from a broken answer is not a crash


def test_pounds_euros_and_arrows_survive_cleaning():
    assert validate.clean_prose("Cut spend by 17.8% (£41,000 a year) and €12,400; p95 by 37.8% (820 → 510 ms).") == (
        "Cut spend by 17.8% (£41,000 a year) and €12,400; p95 by 37.8% (820 to 510 ms).")


def test_the_models_summary_and_bullets_lose_their_dashes(profile):
    out = make_output(summary="Data Engineer — Python and SQL -- for product teams.", headline="Data Engineer - Payments")
    out.sections.experience[0].bullets = [("Rebuilt the Airflow pipeline — cutting failed runs by 17.8% (from 101 to 83 "
                                           "a month) - with staged rollouts.")]
    resume = validate.build_resume(profile, out)
    assert resume.summary == "Data Engineer, Python and SQL, for product teams."
    assert resume.experience[0].bullets[0] == ("Rebuilt the Airflow pipeline, cutting failed runs by 17.8% (from 101 "
                                               "to 83 a month), with staged rollouts.")
    assert resume.headline == "Data Engineer - Payments"  # not a sentence: the headline rule shortens it later
    assert validate.dash_problems(resume) == []


def test_a_cover_letter_loses_its_dashes():
    _greeting, paragraphs, _closing = validate.clean_cover_letter(
        "Dear Hiring Manager,", ["I build data pipelines — reliable ones.", "Thanks -- I would like to talk."], "Sincerely,")
    assert paragraphs == ["I build data pipelines, reliable ones.", "Thanks, I would like to talk."]


def test_dashes_in_a_first_draft_cost_no_extra_call(router, db, registry, profile, jd):
    def dashes(sections):
        sections["summary"] = sections["summary"].replace(", ", " — ", 1)
        bullets = sections["experience"][0]["bullets"]
        bullets[1] = bullets[1].rstrip(".") + " -- every week."

    client = Injects(FakeClient(registry.get("demo-concise")), dashes)
    router.overrides["demo-concise"] = client
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and result.retries == 0 and len(client.prompts) == 1
    assert "—" not in all_text(result) and "--" not in all_text(result) and validate.dash_problems(result.resume) == []


def test_a_dash_typed_in_the_editor_stays_and_gets_a_note(client):
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()
    experience = edit_body(run)["experience"]
    experience[0]["bullets"][1] = "Moved the reports to Airflow - every week on time for the finance team."
    edited = client.put(f"/api/runs/{run['run_id']}/resume", json=edit_body(run, experience=experience)).json()
    assert edited["resume"]["experience"][0]["bullets"][1] == experience[0]["bullets"][1]  # the user's own words
    assert any(w.startswith("Dashes used as punctuation: 'reports to Airflow - every week on'") for w in edited["ats_warnings"])


def test_the_prompts_ask_for_full_stops_and_commas_instead_of_dashes():
    for text in (tailor_system("estimate"), cover_system()):
        assert "Never use a dash as punctuation" in text and "full stop" in text and "event-driven" in text


# ------------------------------------------------------------------ 2. US style / EU style spelling
def test_the_word_lists_never_change_words_that_are_right_in_both_styles():
    for word in ("advertise", "compromise", "enterprise", "exercise", "expertise", "promise", "revise", "supervise",
                 "surprise", "otherwise", "size", "prize", "emphasis", "synthesis", "hypotheses", "specialist",
                 "criticism", "nonstandard", "rigorous", "honorary", "program", "license", "practice", "artifact",
                 "meter", "inquiry", "dialog", "analyses"):
        assert word not in spelling.US_TO_EU, word
    for word in ("analyses", "advertise", "promise", "enterprise", "exercise", "surprise", "towards"):
        assert word not in spelling.EU_TO_US, word
    assert not set(spelling.US_TO_EU) & set(spelling.EU_TO_US)  # every word belongs to one style
    for us, eu in spelling.US_TO_EU.items():
        assert eu.endswith("yses") or eu == "towards" or spelling.EU_TO_US[eu] == us


@pytest.mark.parametrize("style,given,expected,kw", [
    ("eu", "Optimized onboarding, reorganized teams around behavior data for organizations, moving toward one catalog.",
     "Optimised onboarding, reorganised teams around behaviour data for organisations, moving towards one catalogue.", {}),
    ("eu", "Led Data Modeling and Query Optimization work in Unity Catalog and Google Optimize at the Kennedy Space Center.",
     "Led Data Modelling and Query Optimisation work in Unity Catalog and Google Optimize at the Kennedy Space Center.",
     {"terms": ["Data Modeling", "Query Optimization"]}),
    ("eu", "Analyzed churn, analyzes cohorts and wrote the analyses.", "Analysed churn, analyses cohorts and wrote the analyses.", {}),
    ("us", "Optimised the programme, analysed churn and wrote the analyses for the data centre on behaviour and the licence.",
     "Optimized the program, analyzed churn and wrote the analyses for the data center on behavior and the license.", {}),
    ("eu", "OPTIMIZATION in iOS with React Fiber for the Department of Labor; lowercase fiber and labor.",
     "OPTIMIZATION in iOS with React Fiber for the Department of Labor; lowercase fibre and labour.", {}),
    ("eu", "Built Color Health dashboards at Behavior Labs, optimizing color contrast.",
     "Built Color Health dashboards at Behavior Labs, optimising colour contrast.", {"keep": ["Color Health", "Behavior Labs"]}),
])
def test_a_sentence_follows_the_style_and_names_keep_their_spelling(style, given, expected, kw):
    assert spelling.convert(given, style, **kw) == expected
    assert spelling.convert(expected, style, **kw) == expected  # once is enough


@pytest.mark.parametrize("skill,expected", [
    ("Search Engine Optimization (SEO)", "Search Engine Optimisation (SEO)"), ("Data Visualization", "Data Visualisation"),
    ("Containerization", "Containerisation"), ("Unity Catalog", "Unity Catalog"), ("AWS Glue Data Catalog", "AWS Glue Data Catalog"),
    ("Webpack Bundle Analyzer", "Webpack Bundle Analyzer"), ("Azure Security Center", "Azure Security Center"),
    ("AWS Organizations", "AWS Organizations"),
])
def test_a_skill_follows_the_style_unless_it_is_a_products_name(skill, expected):
    assert spelling.convert(skill, "eu", title=True) == expected


def test_the_posts_own_keywords_keep_its_spelling_and_the_role_follows_the_style(profile):
    """An ATS matches the post's keywords letter for letter, so they stay as the post spells them; the role the
    resume aims at follows the style everywhere, in the headline and inside sentences."""
    out = make_output(headline="Behavioral Scientist",
                      summary="Senior Behavioral Scientist with deep data modeling and behavior analysis experience.")
    out.jd_analysis.target_role = "Senior Behavioral Scientist"
    out.jd_analysis.keywords = ["data modeling"]
    resume = validate.build_resume(profile, out)
    validate.spell_resume(resume, "eu", out.jd_analysis)
    assert resume.headline == "Behavioural Scientist"
    assert resume.summary == "Senior Behavioural Scientist with deep data modeling and behaviour analysis experience."


@pytest.mark.parametrize("text,title", [
    ("Used scipy.optimize, pg_catalog, --color-primary and src/utils/normalize daily.", False),
    ("Tuned scikit-optimize searches.", False), ("webpack-bundle-analyzer", True),
])
def test_code_and_package_names_keep_their_spelling(text, title):
    assert spelling.convert(text, "eu", title=title) == text


@pytest.mark.parametrize("us,eu", [
    ("Productionized", "Productionised"), ("annualized", "annualised"), ("dockerized", "dockerised"),
    ("unfavorable", "unfavourable"), ("grayed", "greyed"), ("acknowledgments", "acknowledgements"),
    ("labelers", "labellers"), ("centerpiece", "centrepiece"), ("featurization", "featurisation"),
])
def test_more_resume_words_follow_the_style_both_ways(us, eu):
    assert spelling.convert(us, "eu") == eu and spelling.convert(eu, "us") == us


@pytest.mark.parametrize("skill", ["Collibra Data Catalog", "Glue Data Catalog", "SPSS Modeler", "Jira Data Center"])
def test_a_products_name_in_the_skills_keeps_its_spelling(skill):
    assert spelling.convert(skill, "eu", title=True) == skill


def test_the_hiring_companys_short_name_keeps_its_spelling(profile):
    resume = validate.build_resume(profile, make_output())
    analysis = make_output().jd_analysis.model_copy(update={"company": "Color Health, Inc."})
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "Color Health's mission is why I am applying, and I optimized lab color workflows.", "I would like to talk."])
    validate.spell_letter(letter, "eu", validate.known_names(resume, analysis), validate.skill_terms(resume, analysis))
    assert letter.paragraphs[0] == "Color Health's mission is why I am applying, and I optimised lab colour workflows."


def test_the_role_level_fix_works_in_either_spelling(profile):
    out = make_output(summary="Data Centre Technician II with 6 years of hands-on experience in Python and SQL.")
    out.jd_analysis.target_role = "Data Center Technician II"
    resume = validate.build_resume(profile, out)
    assert validate.role_level_fixes(resume, out.jd_analysis)
    assert resume.summary.startswith("Data Centre Technician with 6 years")
    assert validate.title_level("Data Centre Technician")[0] == validate.title_level("Data Center Technician")[0]


def test_keywords_with_punctuation_match_in_either_spelling():
    assert validate.mentions("led data modelling/warehousing", "data modeling/warehousing")
    assert validate.mentions("SEO (search engine optimisation) work", "SEO (search engine optimization)")
    assert validate.mentions("the Organisation's data strategy", "Organization's data strategy")


def test_an_eu_style_file_is_titled_cv_inside_too(client):
    import io

    from docx import Document
    from pypdf import PdfReader

    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "spelling": "eu"}).json()
    pdf = client.get(f"/api/runs/{run['run_id']}/resume.pdf").content
    docx = client.get(f"/api/runs/{run['run_id']}/resume.docx").content
    assert PdfReader(io.BytesIO(pdf)).metadata.title == "Jordan Rivera - CV"
    assert Document(io.BytesIO(docx)).core_properties.title == "Jordan Rivera - CV"
    assert PdfReader(io.BytesIO(pdf)).trailer["/Root"]["/Lang"] == "en-GB"  # British English, also for screen readers
    assert Document(io.BytesIO(docx)).core_properties.language == "en-GB"


def test_keywords_match_in_either_spelling():
    assert validate.mentions("data modelling work", "Data Modeling") and validate.mentions("data centres", "data center")
    assert validate.mentions("query optimisations", "query optimization")
    assert validate.mentions("graduate programme", "Graduate Program")


def test_eu_spelling_never_turns_a_posting_skill_into_an_added_one_or_hides_a_copied_sentence(profile):
    jd = "We need Data Visualization and Optimization of costs. You will optimize the data warehouse for analysts."
    out = make_output()
    out.sections.skills[0].skills = ["Data Visualisation", "Optimisation"]
    out.sections.experience[0].bullets = ["You will optimise the data warehouse for analysts at Northwind."]
    resume = validate.build_resume(profile, out)
    flags = validate.claim_flags(resume, jd, "")
    assert not any(f.text in ("Data Visualisation", "Optimisation") and "Related skill" in f.reason for f in flags)
    assert validate.copied_posting_spans(resume, jd, out.jd_analysis)  # the copied sentence is still found
    assert validate.plain_word_problems(validate.build_resume(profile, make_output(summary="Utilised Spark.")))


def test_an_eu_style_run_is_british_everywhere_and_scores_like_a_us_one(client):
    save_profile(client)
    runs = {}
    for style in ("us", "eu"):
        run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "spelling": style}).json()
        client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})
        runs[style] = client.get(f"/api/runs/{run['run_id']}").json()
    us, eu = runs["us"], runs["eu"]
    assert us["spelling"] == "us" and eu["spelling"] == "eu"  # kept with the run
    assert "organizations" in us["resume"]["summary"] and "organisations" in eu["resume"]["summary"]
    assert us["cover_letter"]["closing"] == "Sincerely," and eu["cover_letter"]["closing"] == "Kind regards,"
    assert eu["coverage"]["score"] == us["coverage"]["score"]  # the keyword match does not depend on the style
    names = {style: client.get(f"/api/runs/{runs[style]['run_id']}/resume.pdf?download=1").headers["content-disposition"]
             for style in runs}
    assert 'filename="Jordan_Rivera_Resume.pdf"' in names["us"] and 'filename="Jordan_Rivera_CV.pdf"' in names["eu"]
    assert client.post("/api/tailor", json={"jd": SAMPLE_JD, "spelling": "uk"}).status_code == 422


class Buzzy(LLMClient):
    """A first draft with two buzzwords, so the fix round runs; it answers in the demo's American spelling."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            result.data["sections"]["experience"][0]["bullets"][1] = (
                "Leveraged Python to build a seamless reporting flow, so managers saw organization-wide numbers")
        return result


def test_the_fix_round_and_the_shorter_draft_come_out_british_too(router, db, registry, profile, jd):
    client = Buzzy(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    fixed = tailor(router, db, profile, jd, spelling="eu")
    assert fixed.status == "ok" and client.calls == 2
    from app.schemas import Company, Profile

    jobs = Profile(name="Jordan Rivera", companies=[
        Company(name=f"Employer {i}", start=f"Jan {2023 - 3 * i}", end="Present" if i == 0 else f"Dec {2025 - 3 * i}")
        for i in range(9)])
    shorter = tailor(router, db, jobs, jd, model="demo-detailed", spelling="eu")
    assert "The model shortened the bullets to fit two pages." in shorter.notes
    for result in (fixed, shorter):
        assert spelling.other_style_words(all_text(result), "eu") == []
        assert "organisation" in all_text(result).lower() or "optimis" in all_text(result).lower()


def test_an_older_run_has_no_style_and_reads_as_us():
    from app.schemas import TailorResult

    assert TailorResult.model_validate({"requested_model": "m"}).spelling == ""


def test_the_prompts_name_the_style_and_its_sign_off():
    us, eu = tailor_system("estimate", "us"), tailor_system("estimate", "eu")
    assert "SPELLING (US style)" in us and "optimize, organization" in us and "SPELLING (EU style)" not in us
    assert "SPELLING (EU style)" in eu and "optimise, organisation" in eu and "Unity Catalog" in eu
    assert '"Kind regards,"' in cover_system("eu") and '"Sincerely,"' in cover_system("us")
    assert "call the resume a CV" in cover_system("eu") and "{{" not in eu + cover_system("eu")


def test_the_tailor_page_has_the_two_style_buttons_with_us_first(client):
    page = client.get("/tailor").text
    assert 'id="spell-us" type="button" class="on" aria-pressed="true">US style' in page
    assert 'id="spell-eu" type="button" aria-pressed="false">EU style' in page


def test_an_american_word_typed_into_an_eu_resume_gets_a_note(client):
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise", "spelling": "eu"}).json()
    run = client.get(f"/api/runs/{run['run_id']}").json()
    edited = client.put(f"/api/runs/{run['run_id']}/resume",
                        json=edit_body(run, summary=run["resume"]["summary"] + " Known for cost optimization.")).json()
    assert "cost optimization" in edited["resume"]["summary"]  # the user's own words stay
    assert any(w.startswith("Words in American spelling in this EU style document: optimization")
               for w in edited["ats_warnings"])


def test_a_letter_calls_the_resume_a_cv_and_signs_off_the_eu_way():
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "I attached my resume. The work resumed in May, when our team built the resume parser."])
    validate.spell_letter(letter, "eu", [], [])
    assert letter.paragraphs == ["I attached my CV. The work resumed in May, when our team built the resume parser."]
    assert letter.closing == "Kind regards,"


def test_a_us_letter_never_turns_cv_into_resume():
    """CV also means computer vision or cross-validation, and American readers know the word."""
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Yours sincerely,", paragraphs=[
        "At Northwind Labs I led the CV team, and your CV platform is where my CV work fits next."])
    validate.spell_letter(letter, "us", [], [])
    assert letter.paragraphs == ["At Northwind Labs I led the CV team, and your CV platform is where my CV work fits next."]
    assert letter.closing == "Sincerely,"


def test_a_british_job_post_with_us_style_gets_a_note(client):
    save_profile(client)
    jd = SAMPLE_JD + " You will optimise our organisation's data and the behaviour of its pipelines."
    run = client.post("/api/tailor", json={"jd": jd, "model": "demo-concise"}).json()
    full = client.get(f"/api/runs/{run['run_id']}").json()
    assert any(w.startswith("This job post uses British spelling, but this resume uses US style") for w in full["ats_warnings"])
    eu = client.post("/api/tailor", json={"jd": jd, "model": "demo-concise", "spelling": "eu"}).json()
    assert not any("This job post uses" in w for w in client.get(f"/api/runs/{eu['run_id']}").json()["ats_warnings"])


# ------------------------------------------------------------------ 3. percentages with their base
@pytest.mark.parametrize("text", [
    "Built the service for 83K players, cutting queue time by 22% with retries.", "Cut costs by 17.8% by rightsizing EC2.",
    "Made checkout 21% faster.", "Revenue grew 12.6% year over year.", "Saw a 9.4% drop in tickets.",
    "For $2.3M in orders, cutting churn by 12.6%.", "Brought 21% more orders.", "Sales were up 9.4%.",
    "Cut costs by 17.8% ($41,300 a year) and latency by 21%.", "Cut tags by 23.4% (GA4 tags).",
    "Cut AWS costs by 17.8% and p95 latency by 21% (from 820 to 648 ms) for 4.6K merchants.",
    "Cut p95 latency by 21% and failed jobs by 31.7%, from 41 to 28 a week.", "Cut latency by 21% and saved $41,300 a year.",
    "Cut costs by 17.8% (in 2 quarters).", "Cut costs by 17.8% (41 services).",
])
def test_a_percentage_without_its_base_is_found(text):
    assert validate.unbased_percentages(text)


@pytest.mark.parametrize("text", [
    "Cut AWS costs by 17.8% ($41,000 a year).", "Cut queue time by 22% (from 41 to 32 seconds).",
    "Cut queue time by 22%, from 41 to 32 seconds.", "Cut queue time from 41 to 32 seconds (22%).",
    "Saving $41,300 a year, a 17.8% cut.", "Cut runtime by 23.4% to 11 minutes.", "Kept 99.9% uptime.",
    "Closed 95% of tickets.", "Raised test coverage from 61% to 87%.", "Cut errors to 0.3%.",
    "Lifted conversion by 2.3% (from 3.1% to 3.2%).", "Made builds 21% faster (from 14 to 11 minutes).",
    "Kept the payments platform up 99.95% of the time for 4.6K merchants.",
    "Launched the CSV export, used by 38% of active users within a quarter.",
    "Cut cloud spend by 17.8% across 41 services in 3 regions ($41,300 a year).", "Cut spend by 17.8%, saving $41,300 a year.",
    "Cut p95 latency by 37.8% (820 to 510 ms) with Redis.",
])
def test_a_percentage_with_its_base_or_a_level_is_fine(text):
    assert validate.unbased_percentages(text) == []


@pytest.mark.parametrize("text,wrong", [
    ("Cut queue time by 22% (from 41 to 38 seconds).", True), ("Cut reporting by 30% (from 8 hours to 2 hours).", True),
    ("Grew users by 21% (from 380 to 460).", False), ("Cut spend by 17.8% (from $232K to $191K a year).", False),
    ("Cut reporting by 75% (from 8 hours to 2 hours).", False), ("Cut it by 21% (from 2 minutes to 41 seconds).", True),
    ("Cut it by 65.8% (from 2 minutes to 41 seconds).", False),  # units are compared on one scale
    ("Cut p95 latency by 31.7% (from 1.2s to 820ms) with Redis caching.", False),
    ("Cut nightly storage by 44.7% (from 1.7TB to 940GB) with Parquet.", False),
    ("Cut build time by 65.8% (from 2h to 41m) with remote caching.", False),
    ("Grew ARR by 31.7% (from $4.1M to $5.4M) from 2021 to 2023.", False),
    # only the base beside the percentage is checked, never another 'from ... to ...' in the sentence
    ("Grew the data team from 3 to 7 engineers while cutting on-call pages by 41.4% (from 87 to 51 a month).", False),
    ("Upgraded Postgres from 11 to 15, cutting p95 query time by 23.4% (from 47 to 36 ms).", False),
    ("From 2019 to 2022 at Contoso, I cut cloud spend by 17.8% ($41,300 a year).", False),
    ("Kept 99.9% uptime while moving 41 services from 3 to 2 regions.", False),
])
def test_a_base_must_agree_with_its_percentage(text, wrong):
    assert bool(validate.base_mismatches(text)) is wrong


class Bare(LLMClient):
    """Takes the base off `count` first bullets of the first draft."""

    def __init__(self, inner, count):
        self.inner, self.count, self.prompts = inner, count, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            import re

            for job in result.data["sections"]["experience"][: self.count]:  # its base in brackets, or in words
                job["bullets"][0] = re.sub(r",? \(?from \d+ to \d+ [a-z ]+\)?,?", "", job["bullets"][0])
        return result


@pytest.mark.parametrize("count,calls", [(2, 2), (1, 1)])
def test_bare_percentages_are_given_their_base_in_the_one_fix_round(router, db, registry, profile, jd, count, calls):
    client = Bare(FakeClient(registry.get("demo-concise")), count)
    router.overrides["demo-concise"] = client
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and len(client.prompts) == calls  # one bare percentage is not worth a model call
    if count == 2:
        assert "A percentage means little without its base" in client.prompts[1]
        assert validate.unbased_percentage_problems(result.resume) == []
    else:
        assert any(w.startswith("Percentages without their base: 'by 17.8%' (Northwind Labs)") for w in result.ats_warnings)


def test_with_only_the_users_numbers_a_bare_percentage_is_only_noted(router, db, registry, profile, jd):
    db.set_settings({"metrics_mode": "hints_only"})

    def bare(sections):
        for job in sections["experience"][:2]:
            job["bullets"][0] = job["bullets"][0].rstrip(".") + ", cutting costs by 17.8%."

    client = Injects(FakeClient(registry.get("demo-concise")), bare)
    router.overrides["demo-concise"] = client
    result = tailor(router, db, profile, jd)
    assert len(client.prompts) == 1  # never a fix round for a base the model may not invent
    assert any(w.startswith("Percentages without their base") for w in result.ats_warnings)


def test_a_cover_letter_percentage_without_its_base_is_noted(profile):
    resume = validate.build_resume(profile, make_output())
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,",
                         paragraphs=["At Northwind Labs I cut cloud costs by 17.8%.", "I would like to talk."])
    assert validate.unbased_percentage_problems(resume, letter) == ["'by 17.8%' (cover letter paragraph 1)"]


def test_the_prompts_ask_for_each_percentages_base():
    estimate, words = tailor_system("estimate"), tailor_system("hints_only")
    assert "by 41.2%, from 85 to 50" in estimate and "by 17.8%, or $41,000 a year" in estimate
    assert "The base must agree with the percentage" in estimate and "never use numbers ending in 0 or 5" in estimate
    assert "never add a base yourself" in words and "by 22%" not in words
    assert "keep its base with it" in cover_system()


def test_the_companys_own_figure_from_the_post_needs_no_base_in_a_letter(profile):
    resume = validate.build_resume(profile, make_output())
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,",
                         paragraphs=["Your team's 40% growth last year is why I applied.", "I would like to talk."])
    assert validate.unbased_percentage_problems(resume, letter, "We grew 40% last year.") == []
    assert validate.unbased_percentage_problems(resume, letter, "We are growing fast.") == [
        "'40% growth' (cover letter paragraph 1)"]


@pytest.mark.parametrize("token,expected", [("$41,000", False), ("$40,000", True), ("$3,000", True), ("$41K", False),
                                            ("$41,300", False), ("$7,000", True), ("41,000", True)])
def test_whole_thousands_of_money_are_judged_like_k(token, expected):
    assert validate.is_round_figure(token) is expected


# ------------------------------------------------------------------ 4. vague results
@pytest.mark.parametrize("text", [
    "Improved operational efficiency across teams", "Increased decision making", "Enhanced cross-functional collaboration",
    "Drove alignment with sales", "Boosted team productivity", "improving overall performance", "Streamlined processes",
    "streamlined onboarding processes", "data-driven decision making", "operational excellence",
    "Delivered measurable results", "added value for clients", "Optimised business outcomes",
    "improved organisational effectiveness", "increased visibility",
])
def test_a_vague_result_is_found(text):
    assert validate._plain_problems([("Northwind Labs", text)])


@pytest.mark.parametrize("text", [
    "Streamlined Airflow reporting by replacing spreadsheet hand-offs", "Improved Spark processes by caching",
    "Increased delivery speed by 23.4% (from 47 to 36 minutes)", "improved query performance from 820 to 310 ms",
    "cut energy efficiency losses", "raised developer productivity", "visibility into failed payments",
    "better patient outcomes", "customer lifetime value",
    "Improved the performance of the checkout API by adding Redis caching.", "Built event-driven performance tests in k6.",
    "Set up data-driven performance reviews.", "Rolled out AI-driven productivity tools.", "Ran data-driven engagement campaigns.",
    "Modeled money as value objects in the domain.", "Improved team productivity by 18.3% (from 11 to 13 tickets a sprint).",
])
def test_a_specific_result_is_not_vague(text):
    assert validate._plain_problems([("Northwind Labs", text)]) == []


def test_a_verb_before_the_posts_own_phrase_is_still_vague():
    places = [("Northwind Labs", ("Improved operational excellence for the finance team, and drove data-driven decision "
                                  "making for sales. Brought operational excellence to on-call."))]
    assert validate._plain_problems(places, ["operational excellence", "data-driven decision making"]) == [
        "'Improved operational excellence' (Northwind Labs)", "'drove data-driven decision making' (Northwind Labs)"]
    assert validate._plain_problems([("X", "A results-driven engineer.")], ["results-driven"]) == ["'results-driven' (X)"]


def test_a_posting_term_is_the_keyword_it_asks_for_but_a_verb_makes_it_vague(profile):
    out = make_output(summary="Data Engineer who brings operational excellence to Python and SQL teams.")
    out.sections.experience[0].bullets = ["Improved operational efficiency for the finance team every quarter."]
    out.jd_analysis.keywords = ["operational excellence", "operational efficiency"]
    resume = validate.build_resume(profile, out)
    assert validate.plain_word_problems(resume, out.jd_analysis) == ["'Improved operational efficiency' (Northwind Labs)"]


class Vague(LLMClient):
    def __init__(self, inner, count):
        self.inner, self.count, self.prompts = inner, count, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            phrases = ["Improved operational efficiency", "Increased decision making"][: self.count]
            for job, phrase in zip(result.data["sections"]["experience"], phrases):
                job["bullets"][1] = f"{phrase} for the product teams by moving weekly reports into shared dashboards."
        return result


@pytest.mark.parametrize("count,calls", [(2, 2), (1, 1)])
def test_vague_results_are_rewritten_in_the_one_fix_round(router, db, registry, profile, jd, count, calls):
    client = Vague(FakeClient(registry.get("demo-concise")), count)
    router.overrides["demo-concise"] = client
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and len(client.prompts) == calls
    if count == 2:
        assert "Replace a vague result" in client.prompts[1] and validate.plain_word_problems(result.resume) == []
    else:
        assert any("'Improved operational efficiency' (Northwind Labs)" in w for w in result.ats_warnings)


def test_the_demo_writes_no_vague_results_dashes_or_bare_percentages(router, db, profile, jd):
    for model in ("demo-concise", "demo-detailed"):
        result = tailor(router, db, profile, jd, model=model, cover_letter=True)
        plain = validate.plain_word_problems(result.resume, result.analysis)
        plain += validate.cover_plain_problems(result.cover_letter, result.analysis)
        assert [p for p in plain if "'-ing' clauses" not in p] == []  # the detailed demo's long bullets chain '-ing'
        assert validate.base_mismatch_problems(result.resume, result.cover_letter) == []
        assert validate.dash_problems(result.resume, result.cover_letter) == []
        assert validate.unbased_percentage_problems(result.resume, result.cover_letter) == []


def test_the_prompts_name_the_recruiters_examples():
    for text in (tailor_system("estimate"), cover_system()):
        assert '"improved operational efficiency"' in text and '"increased decision-making"' in text
        assert 'such as "utilise"' in text and "leverage, utilize, spearhead, seamless, robust" in text


# ------------------------------------------------------------------ a second review: dashes
@pytest.mark.parametrize("given,expected", [
    # a dash after 'from A to B', after punctuation, or before a new point is punctuation, never a range
    ("Brought monthly AWS spend from $18,600 to $12,400 — $74,400 saved a year.",
     "Brought monthly AWS spend from $18,600 to $12,400, $74,400 saved a year."),
    ("Cut failed nightly runs from 101 to 83 — 18 fewer each month.",
     "Cut failed nightly runs from 101 to 83, 18 fewer each month."),
    ("Cut costs by 22% — 9% more than planned.", "Cut costs by 22%, 9% more than planned."),
    ("Launched the analytics portal in 2023 — now used by 40 teams.",
     "Launched the analytics portal in 2023, now used by 40 teams."),
    ("Built the fraud model in 2022 — today it screens 3M payments a day.",
     "Built the fraud model in 2022, today it screens 3M payments a day."),
    ("I joined Northwind in 2019 — in 2022 I took over the data platform.",
     "I joined Northwind in 2019, in 2022 I took over the data platform."),
    ("We moved to Snowflake in 2021. — 2022 brought the dbt rewrite.",
     "We moved to Snowflake in 2021. 2022 brought the dbt rewrite."),
    ("Cut report pages from 120 to 45, — 75 fewer to read.", "Cut report pages from 120 to 45, 75 fewer to read."),
    # ranges, also with a unit on one end, money of two sizes, version numbers and two-word ends
    ("Grew revenue 10 – 20% a quarter.", "Grew revenue 10-20% a quarter."),
    ("Made dashboard queries 3 – 5x faster.", "Made dashboard queries 3-5x faster."),
    ("Closed deals of $500K – $2M each.", "Closed deals of $500K-$2M each."),
    ("Supported Python 3.8 – 3.12 in CI.", "Supported Python 3.8-3.12 in CI."),
    ("At Northwind (Jan 2021 – Mar 2024), I rebuilt the pipeline.",
     "At Northwind (Jan 2021 to Mar 2024), I rebuilt the pipeline."),
    ("Ran planning from Q1 2023 – Q2 2024.", "Ran planning from Q1 2023 to Q2 2024."),
    ("Handled escalations (Tier 1 – Tier 3) for 40 accounts.", "Handled escalations (Tier 1 to Tier 3) for 40 accounts."),
    ("Ran it (2021 – now) for finance.", "Ran it (2021-now) for finance."),
    # flags after a dash, doubled dashes, list markers, line breaks, thin spaces and the minus sign
    ("Gave on-call a preview mode — --dry-run — before every Terraform apply.",
     "Gave on-call a preview mode, --dry-run, before every Terraform apply."),
    ("Led the migration — — of 14 services.", "Led the migration, of 14 services."),
    ("Airflow —- no", "Airflow, no"), ("Airflow ---- no", "Airflow, no"), ("- Led the data team", "Led the data team"),
    ("(— see below)", "(see below)"),
    ("I'd bring three things:\n- Airflow DAGs that run on time\n- dbt models finance trusts",
     "I'd bring three things: Airflow DAGs that run on time, dbt models finance trusts"),
    ("I build pipelines.\nI also mentor.", "I build pipelines. I also mentor."),
    ("Rebuilt the billing pipeline\u2009—\u2009cutting failed runs by 18%.",
     "Rebuilt the billing pipeline, cutting failed runs by 18%."),
    ("Turned net revenue retention from \u22122% to +6% in two quarters.",
     "Turned net revenue retention from -2% to +6% in two quarters."),
])
def test_a_dash_before_a_new_point_is_a_comma_and_a_range_goes_up(given, expected):
    assert validate.clean_prose(given) == expected and validate.clean_prose(expected) == expected


def test_an_em_dash_typed_in_the_editor_is_noted_never_glued_into_one_word(profile):
    out = make_output()
    out.sections.experience[0].bullets = [
        "Rebuilt the billing pipeline—cutting failed runs by 17.8% (from 101 to 83 a month).",
        "Grew revenue 10 - 20% a quarter for three product lines.",
    ]
    resume = validate.build_resume(profile, out, from_model=False)
    assert resume.experience[0].bullets[0].startswith("Rebuilt the billing pipeline - cutting failed runs")
    assert validate.dash_problems(resume) == ["'the billing pipeline - cutting failed runs' (Northwind Labs)"]


def test_many_dashes_take_linear_time():
    import time

    started = time.perf_counter()
    assert validate.fix_dashes("a — " * 20000).count(",") == 19999
    assert validate.fix_dashes("| --- " * 10000).startswith("|, |")
    assert time.perf_counter() - started < 1.5


# ------------------------------------------------------------------ a second review: vague results
@pytest.mark.parametrize("text,vague", [
    ("Results-driven Data Engineer with 7+ years of Python and SQL pipelines.", True),  # a cliche, whatever follows
    ("Drove alignment between B2B sales and product teams.", True),  # B2B, Q3, S3, EC2, GPT-4 and years are no figures
    ("Improved operational efficiency for the Q3 close.", True),
    ("Enhanced cross-functional collaboration with the S3 and EC2 platform teams.", True),
    ("Improved team productivity with GPT-4 coding assistants.", True),
    ("Drove alignment on the 2024 roadmap with sales and finance.", True),
    ("Improved operational efficiency for the finance team, and migrated 14 Airflow DAGs to Dagster.", True),
    ("Improved the overall efficiency of our operations.", True),  # 'of' a generic owner is still vague
    ("Boosted the overall performance of the business.", True),
    ("Enhanced the productivity of the engineering organization.", True),
    ("Drove alignment based on shared OKRs with sales.", True),
    ("Improved the performance of the checkout API with Redis caching.", False),
    ("Improved team productivity by 18.3% (from 11 to 13 tickets a sprint).", False),
    ("Added value objects for money and currency in the billing domain.", False),  # technical terms
    ("Added value-based pricing tiers for enterprise customers.", False),
    ("Optimized performance-critical code paths in the C++ order router.", False),
    ("Improved performance monitoring for the checkout service with Datadog APM.", False),
])
def test_a_vague_result_needs_a_real_figure_in_its_own_clause(text, vague):
    assert bool(validate._plain_problems([("bullet", text)])) is vague


# ------------------------------------------------------------------ a second review: percentages and their bases
@pytest.mark.parametrize("text", [
    "Lifted trial-to-paid conversion by 18.2% (from 22% to 26%).",
    "Raised test coverage by 42.6% (from 61% to 87%).",
    "Lifted conversion by 18.2%, from 22% to 26%.",
    "Cut API latency by 21% (p95 from 820 to 648 ms) with Redis caching.",
    "Cut queue time by 22% (down from 41 to 32 seconds).",
    "Cut AWS spend by 17.7% ($41,300 a year, from $232K to $191K).",
    "Cut p95 latency by 21% (41 services, from 820 to 648 ms).",
    "Cut build time from 14 to 11 minutes (21% faster) with remote caching.",
    "Cut p95 latency from 820 to 648 ms (a 21% drop) with Redis.",
    "Lifted trial-to-paid conversion by 12.9%, from 3.1% to 3.5%.",
    "Lifted trial-to-paid conversion by 12.9% (3.1% to 3.5%).",
    "Raised trial-to-paid conversion from 3.1% to 3.5%, a 12.9% lift.",
    "Cut AWS spend by 17.8%, saving the company $41,300 a year.",
    "Cut AWS spend by 17.8%, a saving of $41,300 a year.",
    "Cut AWS spend by 17.8%, which saved $41,300 a year.",
    "Cut AWS spend by 17.8% for $41,300 in annual savings.",
    "Cut p95 latency by 21% for 4.6K merchants (from 820 to 648 ms).",
    "Saved $41.3K a year on AWS, a 17.8% cut.",
    "Cut p95 latency by 31.7% (from 1.2 seconds to 820 milliseconds) with Redis.",
    "Grew monthly active users by 22.4% (from 980K to 1.2 million).",
    "Grew GMV by 22.4% (from £980m to £1.2bn).",
    "Cut build time by 48.8% (from 1h 20m to 41m).",
    "Made report generation 140% faster (from 12 to 5 minutes).",
    "Kept checkout up 99.95% through Black Friday traffic.",
    "Cut errors to 0.3% and raised test coverage from 61% to 87%.",
])
def test_a_base_written_in_any_usual_way_counts_and_agrees(text):
    assert validate.unbased_percentages(text) == [] and validate.base_mismatches(text) == []


@pytest.mark.parametrize("text,found", [
    ("Cut latency by 21% and saved $41,300.", ["by 21%"]),  # the saving is its own claim, not the latency's base
    ("Cut AWS spend by 17.8% ($41,300 a year) and made nightly jobs 21% faster.", ["21% faster"]),
    ("Cut queue time by 22%, from 41 to 32 seconds, and made builds 21% faster.", ["21% faster"]),
    ("Rightsized EC2 fleets, delivering a 17.8% cost reduction.", ["17.8% cost reduction"]),
    ("Cut AWS spend 17.8% by rightsizing EC2 and RDS.", ["Cut AWS spend 17.8%"]),
    ("Reduced p95 latency 21% with Redis caching.", ["Reduced p95 latency 21%"]),
    ("Delivered 12.6% year-on-year growth in the EMEA region.", ["12.6% year-on-year growth"]),
    ("Achieved a reduction of 17.8% in monthly AWS spend.", ["reduction of 17.8%"]),
    ("Saved 17.8% on monthly AWS spend by rightsizing.", ["Saved 17.8%"]),
    ("Grew enterprise ARR by 31.7% from 2021 to 2023 by adding usage-based pricing.", ["by 31.7%"]),  # years: no base
    ("Grew enterprise ARR by 31.7% (2021 to 2023).", ["by 31.7%"]),
])
def test_a_percentage_without_its_own_base_is_found_however_it_is_worded(text, found):
    assert validate.unbased_percentages(text) == found


def test_only_the_posts_own_change_figures_count_as_the_companys(profile):
    jd = "Up to 20% travel. Benefits: a 4% 401(k) match and a 15% bonus target. We grew 40% last year."
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "At Northwind Labs I cut nightly ETL runtime by 20% by moving the largest Airflow jobs to Spark.",
        "Your team grew revenue 40% last year, and I would like to help it grow again.",
    ])
    resume = validate.build_resume(profile, make_output())
    assert validate.unbased_percentage_problems(resume, letter, jd) == ["'by 20%' (cover letter paragraph 1)"]


@pytest.mark.parametrize("token,round_figure", [
    ("$7K", True), ("$7,000", True), ("$3M", True), ("$3,000,000", True), ("$40,000", True),
    ("$41.3M", False), ("$41,300,000", False), ("$2,400,000", False), ("$41,300", False),
])
def test_money_reads_as_round_or_not_the_same_with_or_without_a_suffix(token, round_figure):
    assert validate.is_round_figure(token) is round_figure


def test_digits_inside_names_are_not_figures():
    assert not validate.has_figure("Moved billing jobs to EC2 spot instances and S3 lifecycle rules.")
    assert validate._numbers_in("Ran Microsoft 365 for Fortune 500 retailers, cutting p95 by 21.3% (from 820 to "
                                "645.6 ms) on Postgres 15. Built 40 pipelines.") == ["21.3%", "820", "645.6", "40"]


def test_huge_numbers_never_crash_or_stall_the_figure_checks():
    import time

    started = time.perf_counter()
    validate.unbased_percentages("9" * 4000 + "% faster")
    assert validate.is_round_figure("1" * 5000) is False and validate.has_figure("1" * 5000)
    assert time.perf_counter() - started < 0.5


# ------------------------------------------------------------------ a second review: spelling
def _posting(target_role="Senior Data Engineer", keywords=()):
    from app.schemas import JDAnalysis

    return JDAnalysis(target_role=target_role, seniority="Senior", domain="Data", must_have_skills=list(keywords),
                      nice_to_have_skills=[], keywords=list(keywords), key_responsibilities=["Build pipelines"])


@pytest.mark.parametrize("style,model_wrote,post_says", [
    ("eu", "data modelling and query optimisation", "data modeling and query optimization"),  # a US post
    ("us", "data modeling and query optimization", "data modelling and query optimisation"),  # a UK post
])
def test_the_posts_keywords_take_the_posts_spelling_whatever_the_model_wrote(profile, style, model_wrote, post_says):
    keywords = ["data modeling", "query optimization"] if style == "eu" else ["data modelling", "query optimisation"]
    out = make_output(summary=f"I own {model_wrote} for a warehouse used by 38 analysts, and I organize reviews.")
    out.jd_analysis = _posting(keywords=keywords)
    resume = validate.build_resume(profile, out)
    validate.spell_resume(resume, style, out.jd_analysis)
    organise = "organise" if style == "eu" else "organize"
    assert resume.summary == f"I own {post_says} for a warehouse used by 38 analysts, and I {organise} reviews."
    assert validate.spelling_problems(resume, None, style, out.jd_analysis) == []


def test_a_job_title_follows_the_style_even_where_it_reads_like_a_products_name(profile):
    out = make_output(summary="Data Modeler with 8 years of dimensional modeling. As a Senior Data Modeler I organized "
                              "140 metrics.", headline="Data Modeler")
    out.sections.experience[0].job_title = "Senior Data Modeler"
    out.jd_analysis = _posting(target_role="Senior Data Modeler")
    resume = validate.build_resume(profile, out)
    validate.spell_resume(resume, "eu", out.jd_analysis)
    assert resume.experience[0].job_title == "Senior Data Modeller"
    assert resume.summary == ("Data Modeller with 8 years of dimensional modelling. As a Senior Data Modeller I "
                              "organised 140 metrics.")


@pytest.mark.parametrize("group,skill,expected", [
    ("AWS", "Organizations", "Organizations"), ("AWS", "Compute Optimizer", "Compute Optimizer"),
    ("AWS", "Personalize", "Personalize"), ("Azure", "Security Center", "Security Center"),
    ("Cloud", "AWS (Organizations, IAM, Control Tower)", "AWS (Organizations, IAM, Control Tower)"),
    ("Cloud", "AWS: Organizations, Compute Optimizer", "AWS: Organizations, Compute Optimizer"),
    ("Endpoints", "System Center Configuration Manager (SCCM)", "System Center Configuration Manager (SCCM)"),
    ("Data", "Spark Catalyst Optimizer", "Spark Catalyst Optimizer"),
    ("Data", "Snowflake Data Modeling", "Snowflake Data Modelling"),  # the activity follows the style
    ("BI", "Tableau Visualization", "Tableau Visualisation"), ("Data", "Data Catalog", "Data Catalogue"),
])
def test_a_brands_products_keep_their_names_and_activities_follow_the_style(group, skill, expected):
    assert spelling.convert(skill, "eu", title=True, brand=spelling.names_brand(group)) == expected


@pytest.mark.parametrize("given,expected", [
    ("Implemented the OAuth 2.0 /authorize and /token endpoints.", None),
    ("Internationalized the checkout with @angular/localize.", "Internationalised the checkout with @angular/localize."),
    ("Replaced pandas apply() with normalize(df) for speed.", None),
    ("Sanitized user HTML with sanitize-html.", "Sanitised user HTML with sanitize-html."),
    ("Ran npm run optimize in CI.", None),
    ("Owned SQL/modeling/visualization for 38 analysts.", "Owned SQL/modelling/visualisation for 38 analysts."),
])
def test_endpoints_calls_scripts_and_packages_keep_their_spelling(given, expected):
    assert spelling.convert(given, "eu") == (expected or given)


def test_all_caps_keywords_match_in_either_spelling():
    assert validate.mentions("Led data modelling and query optimisation.", "DATA MODELING")
    assert validate.mentions("Led data modelling and query optimisation.", "QUERY OPTIMIZATION")


def test_the_hiring_managers_name_keeps_its_spelling_after_a_title(profile):
    resume = validate.build_resume(profile, make_output())
    eu_keep = validate.known_names(resume, None, "Northbeam Data", "Avery Gray")
    text = "Thank you, Ms. Gray, for reading this. I would value Dr. Gray's view on the gray-box tests."
    assert spelling.convert(text, "eu", eu_keep) == text.replace("gray-box", "grey-box")
    us_keep = validate.known_names(resume, None, "Northbeam Data", "Sam Grey")
    assert spelling.convert("As Mr. Grey said, the grey-box tests matter.", "us", us_keep) == (
        "As Mr. Grey said, the gray-box tests matter.")


def test_a_one_word_name_keeps_only_its_capitalised_form():
    text = "Color's mission is why I apply. I built color-coded dashboards and led a data center move."
    assert spelling.convert(text, "eu", ["Color", "Center"]) == (
        "Color's mission is why I apply. I built colour-coded dashboards and led a data centre move.")


def test_the_hiring_companys_first_word_is_its_name_too(profile):
    keep = validate.known_names(validate.build_resume(profile, make_output()), None, "Color Health, Inc.", "")
    assert spelling.convert("Color's mission is why I apply.", "eu", keep) == "Color's mission is why I apply."


def test_a_uk_post_naming_us_products_reads_as_british():
    assert spelling.detect("London. You will optimise our AWS Organizations set-up and Azure Security Center alerts, "
                           "organise on-call, and model behaviour.") == "eu"


@pytest.mark.parametrize("given,expected", [
    ("Please find my updated resume attached.", "Please find my updated CV attached."),
    ("As my resume demonstrates, I build pipelines.", "As my CV demonstrates, I build pipelines."),
    ("My resume reflects 8 years of work.", "My CV reflects 8 years of work."),
    ("I built the resume and application data models.", None),
    ("I wrote the resume to job matching service.", None),
    ("I built my resume parser in Python.", None),
])
def test_an_eu_letter_says_cv_only_where_it_means_the_document(given, expected):
    assert spelling.cv_wording(given, "eu") == (expected or given)


@pytest.mark.parametrize("closing", ["Respectfully yours,", "Yours truly,"])
def test_american_sign_offs_become_kind_regards_in_eu_style(closing):
    assert spelling.sign_off(closing, "eu") == "Kind regards," and spelling.sign_off(closing, "us") == closing


@pytest.mark.parametrize("us,eu", [
    ("hospitalization", "hospitalisation"), ("immunization", "immunisation"), ("privatization", "privatisation"),
    ("externalized", "externalised"), ("relabeling", "relabelling"), ("mislabeling", "mislabelling"),
    ("trialed", "trialled"), ("overanalyzed", "overanalysed"), ("endeavoring", "endeavouring"),
    ("marshaling", "marshalling"),
])
def test_more_words_follow_the_style_both_ways(us, eu):
    assert spelling.convert(us, "eu") == eu and spelling.convert(eu, "us") == us


@pytest.mark.parametrize("eu,us", [("benefitting", "benefiting"), ("centimetres", "centimeters"),
                                   ("millimetres", "millimeters")])
def test_british_only_words_become_american(eu, us):
    assert spelling.convert(eu, "us") == us


# ------------------------------------------------------------------ cases from sentences written from the rules alone
def test_an_ellipsis_or_an_emoji_leaves_no_trace():
    assert validate.clean_prose("I automated reconciliations, accruals and more… — saving the team 20 hours a month.") == (
        "I automated reconciliations, accruals and more... saving the team 20 hours a month.")
    assert validate.clean_prose("✔️ Cut month-end close from 9 to 5 days") == "Cut month-end close from 9 to 5 days"


@pytest.mark.parametrize("text,unbased,mismatched", [
    ("Moved CI builds from 41 to 32 minutes and drove a 30% drop in flaky test reruns.", ["30% drop"], []),
    ("Cut invoice processing by 40% (from $6.20 to 3 minutes per invoice) with UiPath bots.", [], []),
    ("Investigated a 6% decline rate on Stripe card payments.", [], []),  # a rate is a level
    ("Grew organic signups by 1,200% after launching programmatic SEO pages.", ["by 1,200%"], []),
    ("Grew signups by 1,200% (from 100 to 1,300 a week).", [], []),
])
def test_a_base_belongs_to_its_own_clause_and_figures_with_commas_count(text, unbased, mismatched):
    assert validate.unbased_percentages(text) == unbased and validate.base_mismatches(text) == mismatched


# ------------------------------------------------------------------ a special posting title is never copied
def _analysis(target_role, general_role=""):
    from app.schemas import JDAnalysis

    return JDAnalysis(target_role=target_role, general_role=general_role, seniority="Senior", domain="AI",
                      must_have_skills=["Python"], nice_to_have_skills=[], keywords=["Python"],
                      key_responsibilities=["Build voice agents"])


def test_a_special_posting_title_becomes_its_common_title_everywhere_but_the_bullets(profile):
    analysis = _analysis("Senior Voice AI Engineer", "AI Engineer")
    out = make_output(headline="Voice AI Engineer", summary="Voice AI Engineer with 7+ years of experience building "
                      "Python services. As a Voice AI Engineer I shipped 4 agents.")
    out.sections.experience[0].job_title = "Senior Voice AI Engineer"
    out.sections.experience[0].bullets = ["Built voice agents in Python for 3 call centers.", "Ran the voice AI rollout."]
    resume = validate.build_resume(profile, out)
    validate.short_role_headline(resume)
    validate.general_role_fixes(resume, analysis)
    assert resume.headline == "AI Engineer"
    assert resume.summary == ("AI Engineer with 7+ years of experience building Python services. As an AI Engineer I "
                              "shipped 4 agents.")
    assert [exp.job_title for exp in resume.experience] == ["Senior AI Engineer", "Engineer 1", "Engineer 2"]
    assert resume.experience[0].bullets[1] == "Ran the voice AI rollout."  # the specialty stays in the bullets
    assert validate.specialty_problems(resume, analysis) == []
    resume.summary += " I build voice agents."
    assert validate.specialty_problems(resume, analysis) == ["summary"]


@pytest.mark.parametrize("target,general", [("Senior Data Engineer", "Data Engineer"), ("AI Engineer II", "AI Engineer"),
                                            ("Voice AI Engineer", ""), ("Data Engineer", "Data Engineer")])
def test_a_common_posting_title_or_an_older_run_changes_nothing(profile, target, general):
    out = make_output(headline="Data Engineer", summary="Senior Data Engineer with 7+ years of Python and SQL.")
    resume = validate.build_resume(profile, out)
    validate.general_role_fixes(resume, _analysis(target, general))
    assert resume.headline == "Data Engineer" and resume.summary == "Senior Data Engineer with 7+ years of Python and SQL."
    assert validate.common_title(_analysis(target, general)) == ("", "") and validate.specialty(_analysis(target, general)) == ""


@pytest.mark.parametrize("target,general", [
    ("AI Platform Engineer", "AI Engineer"), ("AI Platform Engineer", "Platform Engineer"),
    ("AI Platform Engineer", "Software Engineer"), ("Senior Data Platform Engineer", "Data Engineer"),
    ("Cloud Security Engineer", "Security Engineer"),
])
def test_a_title_of_broad_fields_is_a_common_one_whatever_the_model_offers(target, general):
    assert validate.common_title(_analysis(target, general)) == ("", "") and validate.specialty(_analysis(target, general)) == ""


def test_an_ai_platform_engineer_post_keeps_its_title_and_its_ai(profile):
    analysis = _analysis("Senior AI Platform Engineer", "AI Engineer")
    out = make_output(headline="AI Engineer", summary="AI Platform Engineer with 7+ years of AI and ML platform work on "
                      "Kubernetes and Ray.")
    out.sections.experience[0].job_title = "Senior AI Platform Engineer"
    resume = validate.build_resume(profile, out)
    validate.short_role_headline(resume)
    validate.general_role_fixes(resume, analysis)
    assert resume.headline == "AI Platform Engineer"  # a draft's shorter headline gets the posting's title back
    assert resume.summary == "AI Platform Engineer with 7+ years of AI and ML platform work on Kubernetes and Ray."
    assert resume.experience[0].job_title == "Senior AI Platform Engineer"
    assert validate.specialty_problems(resume, analysis) == []  # 'AI' is a field, never a specialty to take out


@pytest.mark.parametrize("target,general,special,words", [
    ("Voice AI Platform Engineer", "AI Platform Engineer", ("Voice AI Platform Engineer", "AI Platform Engineer"),
     "Voice"),
    ("Payments Backend Engineer", "Backend Engineer", ("Payments Backend Engineer", "Backend Engineer"), "Payments"),
])
def test_a_niche_specialty_still_makes_a_title_special(target, general, special, words):
    assert validate.common_title(_analysis(target, general)) == special
    assert validate.specialty(_analysis(target, general)) == words


class SpecialPost(LLMClient):
    """The first draft copies a special posting title ('Voice AI Engineer') and names its specialty in the summary."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            result.data["jd_analysis"].update(target_role="Voice AI Engineer", general_role="AI Engineer")
            sections = result.data["sections"]
            sections["headline"] = "Voice AI Engineer"
            rest = sections["summary"].split(" with ", 1)[1].replace("delivering ", "delivering voice agents, ", 1)
            sections["summary"] = "Voice AI Engineer with " + rest
            sections["experience"][0]["job_title"] = "Senior Voice AI Engineer"
        return result


def test_a_run_never_copies_a_special_title_and_keeps_its_specialty_out_of_the_summary(router, db, registry, profile, jd):
    client = SpecialPost(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = tailor(router, db, profile, jd)
    assert result.status == "ok" and result.retries == 1  # one fix round, for the summary's 'voice'
    assert "the posting's specialty 'Voice'" in client.prompts[1] and "COMMON TITLE" in client.prompts[1]
    assert result.resume.headline == "AI Engineer" and "voice" not in result.resume.summary.lower()
    assert not any("Voice" in exp.job_title for exp in result.resume.experience)


def test_the_prompt_asks_for_the_common_title():
    text = tailor_system("estimate")
    assert "COMMON TITLE: some posting titles are special ones" in text
    assert '"AI Engineer" for "Voice AI Engineer" or "Conversational AI Engineer"' in text
    assert "show it only in the bullets" in text and "never a special title" in text
    assert ('"AI Platform Engineer", "Data Platform Engineer" and "Cloud Security Engineer" are common titles and are '
            "used as they are") in text
    assert '"AI Platform Engineer" for "Senior AI Platform Engineer"' in text
