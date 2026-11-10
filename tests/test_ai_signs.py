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
])
def test_a_dash_used_as_punctuation_becomes_a_comma(given, expected):
    assert validate.fix_dashes(given) == expected and validate.fix_dashes(expected) == expected


@pytest.mark.parametrize("text", [
    "event-driven", "end-to-end", "5-10", "COVID-19", "GPT-4", "x86-64", "Python- and SQL-based", "a -12% swing",
    "ran plan --refresh-only daily", "New York–London",
])
def test_hyphens_ranges_and_flags_stay(text):
    assert validate.fix_dashes(text) == text


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
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=["I attached my resume. The work resumed in May."],
                         closing="Sincerely,")
    validate.spell_letter(letter, "eu", [], [])
    assert letter.paragraphs == ["I attached my CV. The work resumed in May."] and letter.closing == "Kind regards,"
    validate.spell_letter(letter, "us", [], [])
    assert letter.paragraphs == ["I attached my resume. The work resumed in May."] and letter.closing == "Kind regards,"
    assert spelling.sign_off("Yours sincerely,", "us") == "Sincerely,"


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
])
def test_a_percentage_without_its_base_is_found(text):
    assert validate.unbased_percentages(text)


@pytest.mark.parametrize("text", [
    "Cut AWS costs by 17.8% ($41,000 a year).", "Cut queue time by 22% (from 41 to 32 seconds).",
    "Cut queue time by 22%, from 41 to 32 seconds.", "Cut queue time from 41 to 32 seconds (22%).",
    "Saving $41,300 a year, a 17.8% cut.", "Cut runtime by 23.4% to 11 minutes.", "Kept 99.9% uptime.",
    "Closed 95% of tickets.", "Raised test coverage from 61% to 87%.", "Cut errors to 0.3%.",
    "Lifted conversion by 2.3% (from 3.1% to 3.2%).", "Made builds 21% faster (from 14 to 11 minutes).",
])
def test_a_percentage_with_its_base_or_a_level_is_fine(text):
    assert validate.unbased_percentages(text) == []


@pytest.mark.parametrize("text,wrong", [
    ("Cut queue time by 22% (from 41 to 38 seconds).", True), ("Cut reporting by 30% (from 8 hours to 2 hours).", True),
    ("Grew users by 21% (from 380 to 460).", False), ("Cut spend by 17.8% (from $232K to $191K a year).", False),
    ("Cut reporting by 75% (from 8 hours to 2 hours).", False), ("Cut it by 21% (from 2 minutes to 41 seconds).", False),
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

            for job in result.data["sections"]["experience"][: self.count]:
                job["bullets"][0] = re.sub(r" \(from \d+ to \d+ [a-z ]+\)", "", job["bullets"][0])
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
    assert "by 22% (from 41 to 32 seconds)" in estimate and "($41,000 a year)" in estimate
    assert "The base must agree with the percentage" in estimate and "never use numbers ending in 0 or 5" in estimate
    assert "never add a base yourself" in words and "by 22%" not in words
    assert "keep its base with it" in cover_system()


@pytest.mark.parametrize("token,expected", [("$41,000", False), ("$40,000", True), ("$3,000", True), ("$41K", False)])
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
    assert validate._VAGUE_RESULT.search(text)


@pytest.mark.parametrize("text", [
    "Streamlined Airflow reporting by replacing spreadsheet hand-offs", "Improved Spark processes by caching",
    "Increased delivery speed by 23.4% (from 47 to 36 minutes)", "improved query performance from 820 to 310 ms",
    "cut energy efficiency losses", "raised developer productivity", "visibility into failed payments",
    "better patient outcomes", "customer lifetime value",
])
def test_a_specific_result_is_not_vague(text):
    assert not validate._VAGUE_RESULT.search(text)


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

