"""The checks that ask for the one rewrite pass, the admin's warnings and the prompt, held to prompt v38: the person's
own figures stay theirs once the resume writes them its way ('40 percent' is their '40%'), and only figures they gave
whole are theirs ('$12 million' never makes a '2M' theirs), a result led or joined the ways the prompt asks counts as a
result (work done with others 'to build' something names none), figures per job follow the METRICS rule (a draft
within it costs nothing), the summary is 30-40 words, runs saved before the one-page rule keep their own limits, a
posting term keeps its number word, a two-page plan is said, a bullet the app leaves in the other tense is named, and
the rewrite pass keeps only the first draft's notes that are still true. Every person, employer and product here is
invented."""
import asyncio
import re
from datetime import date
from pathlib import Path

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import rules, tailor, validate
from app.pipeline.fit import styles
from app.pipeline.page_fit import design_style
from app.prompts import (
    RETRY_MARKER,
    SHORTEN_NOTE,
    TWO_PAGES_LINE,
    revise_user,
    tailor_system,
    tailor_user,
)
from app.schemas import (
    Company,
    Contact,
    CoverLetter,
    Education,
    FitStyle,
    JDAnalysis,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
    TailorResult,
)

from .conftest import SAMPLE_JD
from .test_reviewer_rules import analysis, resume_of

SYSTEM_MD = (Path(__file__).parent.parent / "app" / "prompts" / "system.md").read_text(encoding="utf-8")


class Rewrites(LLMClient):
    """The demo model, with each draft's sections (never the fix round's) changed by `change`; every request is
    kept."""

    def __init__(self, inner, change=None):
        self.inner, self.change, self.prompts = inner, change, []

    async def generate_structured(self, **kwargs):
        self.prompts.append((kwargs["schema_name"], kwargs["user"]))
        result = await self.inner.generate_structured(**kwargs)
        if self.change and kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            self.change(result.data["sections"])
        return result

    def fix_round(self) -> str:
        """What the one rewrite pass was asked to fix ("" when it was not asked for)."""
        asked = [user for schema, user in self.prompts if schema == "tailor_revision"]
        return asked[0].split(RETRY_MARKER + ": ", 1)[1].split("\n\nRevise the previous draft", 1)[0] if asked else ""


class Stubborn(Rewrites):
    """The demo, whose rewrite pass changes nothing (a model that ignores what the fix round asks)."""

    async def generate_structured(self, **kwargs):
        result = await super().generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_revision":
            result.data = {"headline": "", "summary": "", "titles": [], "bullets": [], "skill_changes": []}
        return result


def tailored(router, db, registry, profile, change, hints: str = "", kind=Rewrites):
    client = kind(FakeClient(registry.get("demo-concise")), change)
    router.overrides["demo-concise"] = client
    result = asyncio.run(tailor.run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD, hints=hints,
                                           model_id="demo-concise", save=False, time_limit=0))
    assert result.status == "ok", result.error
    return client, result


def career(spans: list[tuple[str, str]]) -> Profile:
    return Profile(name="Taylor Quinn", contact=Contact(email="taylor@example.com", location="Austin, TX"),
                   companies=[Company(name=f"Employer {chr(65 + i)}", start=start, end=end, mode="Remote")
                              for i, (start, end) in enumerate(spans)],
                   education=[Education(school="Tailspin University", degree="B.S.", field="Computer Science",
                                        end="2014")])


def jobs_resume(counts: list[int], more: list[int]) -> Resume:
    """One job per count, most recent first: each opens with a figure and has `more` further figure bullets."""
    profile = career([(f"Jan {2024 - 2 * i}", "Present" if i == 0 else f"Dec {2025 - 2 * i}")
                      for i in range(len(counts))])
    return Resume(profile=profile, headline="Backend Engineer", summary="", skills=[], experience=[
        ResumeExperience(company_index=i, job_title="Backend Engineer",
                         bullets=[FIGURE_FIRST] + [FIGURE_MORE] * min(m, n - 1) + [IN_WORDS] * max(0, n - 1 - m))
        for i, (n, m) in enumerate(zip(counts, more))])


FIGURE_FIRST = "Built the order API for 83K stores in Go, cutting sync errors from 41 to 6 a week"
FIGURE_MORE = "Moved billing to PostgreSQL with nightly checks, saving $41,300 a year in support time"
IN_WORDS = "Rebuilt the deploy pipeline in GitHub Actions, letting product ship pricing tests without a deploy"


# ------------------------------------------------------------------ the person's figures, written the resume's way
HINTS = "At Northwind Labs I cut cloud costs by 40 percent and saved $5 million a year."
HINT_BULLET = ("Cut cloud costs by 40 percent by moving Spark batch jobs on AWS to spot instances, saving $5 million a "
               "year.")


@pytest.mark.parametrize("hints,bullet,written", [
    (HINTS, HINT_BULLET, ("Cut cloud costs by 40% by moving Spark batch jobs on AWS to spot instances, saving $5M a "
                         "year.")),
    ("Cut churn 30 percent; onboarding 85 percent faster", "Cut churn 30 percent with onboarding 85 percent faster",
     "Cut churn 30% with onboarding 85% faster"),
    ("Cut errors 40 % and costs 25 %", "Cut errors 40 % and costs 25 % in Go", "Cut errors 40% and costs 25% in Go"),
    ("Saved $50 thousand for 20 thousand users", "Saved $50 thousand a year for 20 thousand users",
     "Saved $50K a year for 20K users"),
])
def test_the_persons_figures_stay_theirs_once_written_the_resumes_way(profile, hints, bullet, written):
    resume = resume_of(profile, [[bullet]])
    rules.fix_numbers(resume, analysis())  # as build() writes every figure
    assert resume.experience[0].bullets[0] == written
    assert validate.figure_style_problems(resume, hints) == []  # no 'round guesses' sent to the rewrite pass
    assert not [f for f in validate.claim_flags(resume, "", hints) if f.level == "verify" and f.section != "summary"]
    assert validate.figure_style_problems(resume, "") != []  # without the hints: estimates, as before


def test_the_persons_figures_in_the_letter_stay_theirs():
    paragraphs = rules.letter_fixes([HINT_BULLET.replace("Cut", "At Northwind Labs I cut")], analysis(), "",
                                    ["Northwind Labs"])
    assert "40%" in paragraphs[0] and "$5M" in paragraphs[0]
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=paragraphs, closing="Sincerely,")
    assert validate.cover_flags(letter, HINTS) == []
    assert [f.reason for f in validate.cover_flags(letter, "")] == [
        "Contains '40%', which is not in your hints. Only keep figures you can back up."]


# a figure is theirs only when they gave it whole: never because its digits sit inside a longer figure of theirs
@pytest.mark.parametrize("hints,bullet,estimate,rounded", [
    ("Grew ARR to $12 million.", "Grew ARR to $12 million with a Go onboarding service for 2M users.", "2M", []),
    ("Raised conversion 115 percent.", "Raised conversion 115 percent and cut churn 15% with a React onboarding flow.",
     "15%", ["15%"]),
    ("Served 25 thousand stores.", "Built the order API in Go for 25 thousand stores, cutting sync errors 5K a week.",
     "5K", []),
    # their unit words, which the resume writes short: '5 million dollars' is their '$5M', '3 times' their '3x'
    ("Cut costs by 5 million dollars.", "Cut costs by $5M by moving batch jobs to spot instances on AWS.", "", []),
    ("Made checkout 3 times faster.", "Made the checkout API 3x faster with Redis caching in Go.", "", []),
])
def test_a_figure_is_the_persons_only_when_they_gave_it_whole(profile, hints, bullet, estimate, rounded):
    resume = resume_of(profile, [[bullet]])
    rules.fix_numbers(resume, analysis())  # as build() writes every figure
    flags = [f.reason for f in validate.claim_flags(resume, "", hints)
             if f.level == "verify" and f.section.startswith("experience")]
    reason = f"'{estimate}' is an estimate, not a number you gave. Replace it with your real figure or remove it."
    assert flags == ([reason] if estimate else [])
    assert validate.figure_style_problems(resume, hints) == rounded  # a round '15%' is a guess, '115%' theirs


def test_the_letter_trusts_only_figures_given_whole():
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "At Northwind Labs I grew ARR to $12M with an onboarding service in Go for 2M users."])
    assert [f.reason for f in validate.cover_flags(letter, "Grew ARR to $12 million.")] == [
        "Contains '2M', which is not in your hints. Only keep figures you can back up."]
    letter.paragraphs = ["At Northwind Labs I cut costs by $5M and made checkout 3x faster with Redis caching."]
    assert validate.cover_flags(letter, "Cut costs by 5 million dollars and made checkout 3 times faster.") == []


@pytest.mark.parametrize("mode", ["estimate", "hints_only"])
def test_a_hint_copied_as_given_costs_no_rewrite_and_no_flag(router, db, registry, profile, mode):
    db.set_settings({"metrics_mode": mode})

    def copy(sections):
        sections["experience"][0]["bullets"][1] = HINT_BULLET

    client, result = tailored(router, db, registry, profile, copy, hints=HINTS)
    assert result.retries == 0 and "round guesses" not in client.fix_round()
    assert any("40%" in b and "$5M" in b for b in result.resume.experience[0].bullets)
    assert not any(f.level == "verify" and ("40%" in f.reason or "$5M" in f.reason) for f in result.flags)
    assert not any(w.startswith("These figures look rounded") for w in result.ats_warnings)


# the person's project notes write their figures in words; the draft writes them as figures, as the prompt asks
SCALE_NOTES = ("Saved three million dollars a year on cloud spend. Handled two billion events a day for twenty "
               "thousand users.")
SCALE_BULLETS = ["Cut cloud spend by $3M a year by moving the Spark batch jobs on AWS to spot instances",
                 "Serve 2B events a day to 20K users from the Go edge tier, cutting pages for the on-call team"]


def test_the_persons_figures_in_words_in_their_notes_stay_theirs(profile):
    assert validate.note_figures(SCALE_NOTES) == {"3m", "2b", "20k"}
    resume = resume_of(profile, [list(SCALE_BULLETS)], summary="Platform engineer at Northwind Labs.")
    rules.fix_numbers(resume, analysis())
    assert resume.experience[0].bullets == SCALE_BULLETS
    assert not [f.reason for f in validate.claim_flags(resume, "", "", SCALE_NOTES) if f.level == "verify"]
    assert validate.figure_style_problems(resume, "", SCALE_NOTES) == []  # no 'round guesses' for the rewrite pass
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "At Northwind Labs I cut cloud spend by $3M a year and served 2B events a day to 20K users."])
    assert validate.cover_flags(letter, "", SCALE_NOTES) == []
    assert validate.figure_style_problems(resume, "") != []  # without the notes: estimates, as before


def test_figures_from_the_notes_cost_no_rewrite(router, db, registry, profile):
    profile.companies[0].projects = SCALE_NOTES

    def write(sections):
        sections["experience"][0]["bullets"][:2] = SCALE_BULLETS

    client, result = tailored(router, db, registry, profile, write)
    assert result.retries == 0 and "round guesses" not in client.fix_round()
    assert result.resume.experience[0].bullets[:2] == SCALE_BULLETS  # the person's real figures are kept
    assert not any(f.level == "verify" and ("$3M" in f.reason or "20K" in f.reason) for f in result.flags)
    assert not any(w.startswith("These figures look rounded") for w in result.ats_warnings)


# ------------------------------------------------------------------ a result led or joined the ways the prompt asks
STRONG = re.findall(r'^\s*Strong \([^)]*\): "(.*)"$', SYSTEM_MD, re.MULTILINE)
RESULTS_IN_WORDS = [
    ("Brought release checks from a manual Friday checklist to an automated GitHub Actions gate with pytest and "
     "contract tests."),
    "Reduced manual reconciliation for the finance team by automating the nightly invoice match in Python and Airflow.",
    ("Shortened partner onboarding by replacing emailed spreadsheets with a self-serve React portal and validated "
     "PostgreSQL imports."),
    "Eliminated duplicate charges at checkout by moving payment retries into an idempotent Go worker with Redis locks.",
    ("Increased trial-to-paid conversion for small business accounts by rebuilding the React onboarding flow in "
     "TypeScript."),
    ("Built the claims ingestion pipeline in Airflow and Snowflake to give analysts same-day data for every new "
     "provider contract."),
    ("Moved the search index to Elasticsearch with per-tenant shards, ending the timeouts enterprise customers hit at "
     "month end."),
    ("Rewrote the billing export in dbt with tested models, replacing three hand-maintained SQL scripts the finance "
     "team distrusted."),
    "Reduce manual reconciliation for the finance team by automating the nightly invoice match in Python.",  # present
    "Sped up the partner search API with Redis caching and query batching in Go for the mobile app.",
    "Rebuilt the order export in Python and dbt and halved the support load for the finance analysts.",
]


def test_the_prompts_own_strong_examples_are_found():
    assert len(STRONG) == 3


@pytest.mark.parametrize("bullet", STRONG + RESULTS_IN_WORDS)
def test_a_result_led_or_joined_as_the_prompt_asks_names_a_result(profile, bullet):
    resume = resume_of(profile, [[bullet]])
    assert [why for _company, text, why in validate.weak_bullets(resume) if text == bullet] == []


# work done with others 'to build' something, a compound such as 'End-to-end' and a past form used as an adjective
# ('the improved onboarding flow') name no result: 'to' counts only before a verb that says what changed
NO_RESULT = [
    "Partnered with the analytics team to build dbt models in Snowflake for the quarterly revenue reports.",
    "Worked with the platform engineers to migrate the nightly Airflow jobs from cron servers to AWS.",
    "Collaborated with designers to develop the onboarding screens in SwiftUI for the iOS app.",
    "End-to-end ownership of the checkout service in Node.js and PostgreSQL on AWS.",
    "Rolled out the improved onboarding flow in React to all regions with LaunchDarkly.",
]


@pytest.mark.parametrize("bullet", [
    "Mentored junior engineers on Python best practices and code review standards during daily pairing sessions.",
    "Moved the nightly reporting jobs to Kubernetes with Helm charts and Argo CD for the analytics team.",
    "Maintained the Airflow DAGs and Snowflake models behind the finance data mart for the month-end close.",
    "Worked to build the Terraform modules for the shared staging and production environments on AWS.",
    *NO_RESULT,
])
def test_a_bullet_that_names_no_result_is_still_found(profile, bullet):
    resume = resume_of(profile, [[bullet]])
    assert [why for _company, text, why in validate.weak_bullets(resume) if text == bullet] == ["names no result"]


def test_two_bullets_that_name_no_result_ask_for_the_rewrite(router, db, registry, profile):
    def work_only(sections):
        jobs = sections["experience"]
        jobs[1]["bullets"][1], jobs[2]["bullets"][1] = NO_RESULT[:2]

    client, result = tailored(router, db, registry, profile, work_only)
    asked = client.fix_round()
    assert result.retries == 1 and "These bullets do not say how the work was done or what changed" in asked
    assert '"Partnered with the analytics team to build dbt ..." (Contoso Systems: names no result)' in asked
    assert '"Worked with the platform engineers to migrate the ..." (Fabrikam: names no result)' in asked
    final = [b for exp in result.resume.experience for b in exp.bullets]
    assert not set(NO_RESULT[:2]) & set(final)  # the rewrite replaced them


def test_two_result_led_bullets_cost_no_rewrite(router, db, registry, profile):
    def lead(sections):
        jobs = sections["experience"]
        jobs[1]["bullets"][1] = ("Reduced manual reconciliation for the finance team by automating the nightly invoice "
                                 "match in Python and SQL.")
        jobs[2]["bullets"][1] = ("Shortened partner onboarding by replacing emailed spreadsheets with validated SQL "
                                 "imports and a shared Tableau dashboard.")

    client, result = tailored(router, db, registry, profile, lead)
    assert result.retries == 0 and client.fix_round() == ""
    assert validate.weak_bullets(result.resume) == []


# ------------------------------------------------------------------ figures per job, as the METRICS rule says
@pytest.mark.parametrize("counts", [[5, 3, 3], [4, 3, 2, 1], [4, 2, 1, 1, 1], [3, 2, 1, 1, 1, 1],
                                    [6, 5, 4, 4, 3, 2, 2, 2], [3, 2, 2, 1, 1], [3, 2, 1, 2, 1], [5, 2, 1, 1, 1],
                                    [6, 3, 3], [3, 1, 1, 1, 1, 1, 1]])
def test_a_draft_at_the_metrics_rules_allowance_has_no_density_problem(counts):
    # the current job: its first bullet and 2 more with figures; every other job: its first and at most 1 more,
    # older or short roles usually none ([5, 2, 1, 1, 1]: the plan of five stays of about three years, about half
    # for the current job)
    usual = jobs_resume(counts, [2, 1] + [0] * (len(counts) - 2))
    most = jobs_resume(counts, [2] + [1] * (len(counts) - 1))
    assert validate.metric_density_problems(usual) == validate.metric_density_problems(most) == []


def test_too_many_figures_are_still_found():
    resume = jobs_resume([5, 3, 3], [3, 2, 0])  # 4 and 3 bullets with figures
    assert validate.metric_density_problems(resume) == [
        "Employer A: 4 of 5 bullets have figures", "Employer B: 3 of 3 bullets have figures"]
    resume = jobs_resume([4, 2, 2, 2], [2, 1, 1, 1])  # every older job's every bullet has one: its first and 1 more
    assert validate.metric_density_problems(resume) == []  # within the rule
    resume.experience[0].bullets[3] = FIGURE_MORE  # one more than the current job may have
    assert validate.metric_density_problems(resume) == ["Employer A: 4 of 4 bullets have figures"]


def test_the_share_counts_only_figures_beyond_the_rule():
    resume = jobs_resume([3, 3, 3], [2, 2, 2])  # every older job one figure bullet over its limit
    assert validate.metric_density_problems(resume) == [
        "Employer B: 3 of 3 bullets have figures", "Employer C: 3 of 3 bullets have figures",
        ("2 of 2 bullets have figures, not counting each job's first or up to 2 more in the current job and 1 more in "
         "every other job")]


FIVE_JOBS = [  # each first bullet with its size and result, the current job 2 more figures, jobs 2 and 3 one more
    [("Design the Airflow ETL platform behind claims analytics, loading 1.7TB of daily events into Snowflake for 38 "
      "analysts."),
     "Mentor 14 data engineers through code reviews on Spark jobs, cutting review rounds from 16 to 9 per release.",
     "Build dbt models with data quality tests in SQL, catching 17 broken loads a quarter before the morning reports."],
    [("Built the Python ingestion service for 4.6K store feeds on AWS Lambda, cutting failed loads from 87 to 31 a "
      "month."),
     ("Moved legacy SQL Server reports to Snowflake with tested models, saving analysts 12 hours of manual exports a "
      "week.")],
    ["Rebuilt the Spark batch jobs for 23 regional offices on AWS EMR, bringing the nightly run from 6.4 to 2.9 hours.",
     "Added CI/CD for the data pipelines with Jenkins and Docker, catching 19 broken transforms in the first quarter."],
    [("Wrote the Python and SQL ETL jobs for the marketing mart on PostgreSQL, joining 3.2M campaign rows a night for "
      "analysts.")],
    [("Automated the weekly finance extracts in Python and SQL for 41 store managers, ending a 2-day manual "
      "spreadsheet merge.")],
]
FIVE_TITLES = ["Senior Data Engineer", "Data Engineer III", "Data Engineer II", "Data Engineer", "Junior Data Engineer"]


def test_a_five_job_draft_within_the_rule_costs_no_rewrite(router, db, registry):
    year = date.today().year  # five stays of three years (the plan: 5 / 2 / 1 / 1 / 1), written as 3 / 2 / 2 / 1 / 1
    person = career([(f"Jan {year - 3 * i - 3}", "Present" if i == 0 else f"Dec {year - 3 * i - 1}")
                     for i in range(5)])

    def within_the_rule(sections):
        for exp, bullets, title in zip(sections["experience"], FIVE_JOBS, FIVE_TITLES):
            exp["bullets"], exp["job_title"] = list(bullets), title

    client, result = tailored(router, db, registry, person, within_the_rule)
    assert result.retries == 0 and client.fix_round() == ""
    assert [exp.bullets for exp in result.resume.experience] == FIVE_JOBS  # 3 / 2 / 2 / 1 / 1, kept
    assert not any(w.startswith("Many bullets carry figures") for w in result.ats_warnings)


def test_in_hints_only_mode_the_persons_own_figures_do_not_count():
    resume = jobs_resume([5, 3], [4, 1])  # every bullet of the current job has a figure
    own = "Order API for 83K stores, sync errors from 41 to 6 a week. Billing saves $41,300 a year."
    assert validate.metric_density_problems(resume)[0] == "Employer A: 5 of 5 bullets have figures"
    assert validate.metric_density_problems(resume, own) == []  # every one is the person's own
    resume.experience[0].bullets[1:] = [f"Moved the ledger to PostgreSQL, catching {n} regressions a quarter"
                                        for n in (17, 23, 29, 31)]
    assert validate.metric_density_problems(resume, own) == ["Employer A: 4 of 5 bullets have figures"]


class Dense(Rewrites):
    """A figure in every bullet of the first draft."""

    def __init__(self, inner):
        super().__init__(inner, self.dense)

    @staticmethod
    def dense(sections):
        figures = iter(["23.4%", "17.8%", "38", "21%", "1.7TB", "12.6%", "47", "8.3%", "19%", "26.1%", "13.7%"] * 3)
        for exp in sections["experience"]:
            exp["bullets"] = [f"Improved delivery of reporting stream {'abcdefgh'[i]} by {next(figures)} by reworking "
                              "its batch schedule and adding checks to every load" for i in range(len(exp["bullets"]))]


RULE = "at most 3 bullets with figures in the current job and 2 in every other job"


def test_every_density_text_says_the_metrics_rules_numbers(router, db, registry, profile):
    client = Dense(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(tailor.run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                           model_id="demo-concise", save=False, time_limit=0))
    asked = client.fix_round()
    assert "Too many bullets carry numbers" in asked and RULE in asked
    assert "2 in 5" not in asked and "at most 2 bullets per employer" not in asked
    dense = TailorResult(requested_model="demo-concise", resume=jobs_resume([5, 3, 3], [4, 2, 0]),
                         analysis=result.analysis, page_target=1, metrics_mode="estimate",
                         style=design_style("Letter"))
    tailor.run_checks(dense, SAMPLE_JD)
    warning = next(w for w in dense.ats_warnings if w.startswith("Many bullets carry figures"))
    assert RULE in warning and "at most 2 per role" not in warning


def test_the_first_bullet_fix_says_the_metrics_rules_numbers(router, db, registry, profile):
    def no_figure_first(sections):
        for job in sections["experience"]:
            job["bullets"][0] = ("Rebuilt the order export behind the finance dashboards with Python and SQL, so "
                                 "month-end numbers matched the ledger without manual fixes.")

    client, _result = tailored(router, db, registry, profile, no_figure_first)
    asked = client.fix_round()
    assert "These jobs open without a measured result" in asked and RULE in asked
    assert "at most 2 bullets with figures" not in asked


# ------------------------------------------------------------------ a summary of 30-40 words
SENTENCE_1 = ("Senior data engineer with hands-on experience delivering ETL and Python work for product and operations "
              "organizations.")
SENTENCE_2 = ("At Northwind Labs, built the evaluation suite behind its claims pipeline, which replays 3,142 real "
              "nightly runs before every release.")
SENTENCE_3 = "Also designed the warehouse schema and migrations behind its billing reports."
LONG_3 = ("Earlier, moved the nightly reporting jobs for 38 analysts from hand-run SQL scripts to dbt models with "
          "automated data quality tests, clear owners for each source table and alerts for every failed load.")
FOUR_LINES = f"{SENTENCE_1} {SENTENCE_2} {SENTENCE_3}"  # 47 words: 4 lines at the design's sizes


def test_the_prompt_asks_for_a_summary_that_fits_three_lines():
    text = tailor_system("estimate")
    assert "SUMMARY: 2-3 sentences, 30-40 words (three lines at most)" in text
    assert "Rewrite the summary if it is over 3 sentences or 40 words" in text and "48 words" not in text


def test_the_summary_warnings_say_30_to_40_words(profile):
    resume = resume_of(profile, [["Built the Go order API for 450 partner stores"]], summary=" ".join(["word"] * 61))
    assert "Summary is long (61 words; aim for 30-40, 2-3 lines)." in validate.ats_lint(resume)
    resume.summary = "Backend engineer."
    assert "Summary is short (2 words; aim for 30-40)." in validate.ats_lint(resume)


def after_the_demos_first_sentence(*sentences: str):
    """The demo's summary: its own first sentence (the role and the posting's terms), then these."""
    def change(sections):
        sections["summary"] = " ".join([sections["summary"].split(". ")[0] + ".", *sentences])
    return change


def test_a_four_line_summary_alone_costs_no_rewrite(router, db, registry, profile):
    client, result = tailored(router, db, registry, profile, after_the_demos_first_sentence(SENTENCE_2, SENTENCE_3))
    assert rules.long_text(result.resume)[1] == 4
    assert result.retries == 0 and client.fix_round() == ""
    words = validate.word_count(result.resume.summary)
    assert f"The summary takes 4 lines ({words} words): aim for 2-3 lines, 30-40 words" in " ".join(
        result.ats_warnings)


def test_a_four_line_summary_is_named_with_other_fixes(router, db, registry, profile):
    def long_and_weak(sections):
        after_the_demos_first_sentence(SENTENCE_2, SENTENCE_3)(sections)
        jobs = sections["experience"]
        jobs[1]["bullets"][1] = ("Used Airflow and SQL to load the nightly finance reports, giving product teams "
                                 "fresh numbers before each stand-up.")
        jobs[2]["bullets"][1] = ("Supported the analysts with Spark jobs that refreshed the shared dashboards, "
                                 "cutting manual exports for finance.")

    client, _result = tailored(router, db, registry, profile, long_and_weak)
    found = re.search(r"The summary takes 4 lines: it has (\d+) words; cut it to 40 or fewer", client.fix_round())
    assert found and int(found.group(1)) > 40


def test_a_five_line_summary_alone_asks_for_the_rewrite(router, db, registry, profile):
    client, result = tailored(router, db, registry, profile, after_the_demos_first_sentence(SENTENCE_2, LONG_3))
    assert result.retries == 1
    assert re.match(r"The summary takes 5 lines: it has \d+ words; cut it to 40 or fewer", client.fix_round())


def test_the_other_fix_texts_say_30_to_40_words(router, db, registry, profile):
    def thin(sections):
        sections["summary"] = "Data engineer. Builds pipelines."

    client, _result = tailored(router, db, registry, profile, thin)
    asked = client.fix_round()
    assert "2-3 flowing sentences of 30-40 words" in asked and "30-48" not in asked


# ------------------------------------------------------------------ runs saved before the one-page rule (prompt v36)
OLD_SUMMARY = ("Senior Data Engineer with 9+ years of experience building healthcare and retail analytics platforms, "
               "mostly Python and SQL pipelines on AWS and Snowflake. Designs Airflow-orchestrated ETL pipelines, "
               "Spark batch jobs and tested dbt models that stay reliable as data volumes grow. Has made reporting "
               "faster and more trustworthy for analysts through data quality checks and safer CI/CD releases. Brings "
               "mentoring and close work with stakeholders from intake to delivery.")
OLD_BULLETS = [
    [("Design and run the Airflow ETL pipelines in Python behind claims analytics, loading 1.7TB of daily events into "
      "Snowflake for 38 analysts and cutting late reports from 41 to 6 a month."),
     ("Build Spark batch jobs on AWS EMR with partitioned Parquet outputs and tested SQL transforms, letting finance "
      "close month-end reporting two days sooner without manual fixes."),
     ("Mentor four data engineers through code reviews and pairing on Airflow and dbt conventions, cutting review "
      "rounds and helping new hires ship their first pipeline in their first month.")],
    [("Built the Python ingestion service for retail point-of-sale feeds on AWS Lambda and SQS, processing 4.6K store "
      "files a night and cutting failed loads from 87 to 76 a quarter."),
     ("Moved legacy SQL Server reports into Snowflake with dbt models and automated data quality tests, so store "
      "managers saw trusted sales numbers each morning instead of weekly exports."),
     ("Set up CI/CD for data pipelines with GitHub Actions and Docker, adding schema checks before each deploy so "
      "broken transforms stopped reaching the shared reporting warehouse.")],
    [("Implemented Python and SQL ETL jobs for the marketing data mart on PostgreSQL, joining campaign and order data "
      "for 23 regional offices and cutting manual spreadsheet work for analysts."),
     ("Wrote monitoring and alerting for nightly batch jobs with simple dashboards, catching failed loads before the "
      "morning reports and feeding recurring fixes into sprint planning with stakeholders.")],
]
OLD_SKILLS = [
    SkillGroup(category="Languages", skills=["Python", "SQL", "Scala", "Bash", "PySpark", "Java", "YAML"]),
    SkillGroup(category="Data Engineering", skills=["Airflow", "Spark", "dbt", "Snowflake", "ETL", "Data Modeling",
                                                    "Kafka"]),
    SkillGroup(category="Cloud and Infrastructure", skills=["AWS", "Docker", "Terraform", "AWS Lambda", "Amazon SQS",
                                                            "Amazon EMR", "Kubernetes"]),
    SkillGroup(category="Practices", skills=["CI/CD", "Agile", "Data Quality Testing", "Code Review", "Mentoring",
                                             "Monitoring", "Git"]),
]
OLD_ANALYSIS = JDAnalysis(target_role="Senior Data Engineer", seniority="Senior", domain="Data",
                          must_have_skills=["Python", "SQL", "Airflow", "Spark", "AWS", "Snowflake", "ETL pipelines"],
                          nice_to_have_skills=["Docker", "CI/CD", "Agile"],
                          keywords=["Python", "SQL", "Airflow", "Spark", "AWS", "Snowflake", "ETL", "CI/CD", "Docker",
                                    "Agile", "mentor", "stakeholders"])


def old_run(profile, page_target: int = 0) -> TailorResult:
    titles = ["Senior Data Engineer", "Data Engineer II", "Data Engineer"]
    resume = Resume(profile=profile, headline="Data Engineer", summary=OLD_SUMMARY, skills=OLD_SKILLS,
                    experience=[ResumeExperience(company_index=i, job_title=title, bullets=bullets)
                                for i, (title, bullets) in enumerate(zip(titles, OLD_BULLETS))])
    return TailorResult(requested_model="demo-concise", resume=resume, analysis=OLD_ANALYSIS, jd=SAMPLE_JD,
                        prompt_version="v36", metrics_mode="estimate", spelling="us", page_target=page_target,
                        style=FitStyle(font_pt=9.5) if page_target == 0 else design_style("Letter"))


def test_an_old_run_keeps_its_own_limits_after_an_edit(profile):
    result = old_run(profile)
    tailor.run_checks(result, SAMPLE_JD)  # as a hand edit, a certification tick or a new cover letter runs it
    assert result.ats_warnings == []
    new = old_run(profile, page_target=1)  # the same text in a one-page run breaks the one-page rules
    tailor.run_checks(new, SAMPLE_JD)
    assert any(w.startswith("Summary is long") for w in new.ats_warnings)
    assert any(w.startswith("The summary takes") for w in new.ats_warnings)


def test_an_old_runs_lint_is_the_v36_one(profile):
    resume = old_run(profile).resume
    assert validate.ats_lint(resume, page_target=0) == []
    resume.summary = " ".join(["word"] * 96)
    resume.experience[0].bullets[0] = " ".join(["word"] * 37)
    resume.experience[2].bullets = resume.experience[2].bullets[:1]
    resume.skills = resume.skills[:1]
    assert validate.ats_lint(resume, page_target=0) == [
        "Summary is long (96 words; aim for 60-85).", "7 skills listed (aim for roughly 24-32).",
        "Northwind Labs: a bullet is 37 words long.",
        "Fabrikam: 1 bullets (aim for 3-8, the most in recent roles)."]


def test_a_new_runs_lines_are_measured_at_its_own_style(profile):
    resume = resume_of(profile, [["Built the Go order API for 450 partner stores"]], summary=FOUR_LINES)
    smaller = next(style for style in styles("Letter", 1) if style.font_pt < design_style("Letter").font_pt)
    at_design = tailor.reviewer_warnings(resume, design_style("Letter"))
    assert any(w.startswith("The summary takes 4 lines") for w in at_design)
    assert not any(w.startswith("The summary takes") for w in tailor.reviewer_warnings(resume, smaller))  # 3 there


# ------------------------------------------------------------------ a posting term keeps its number word
@pytest.mark.parametrize("term,bullet", [
    ("three lines of defense", ("Mapped 41 payment controls to the three lines of defense model in Archer, cutting "
                                "audit findings from 17 to 6 a year.")),
    ("five whys", ("Ran five whys reviews in Jira after each Sev-1 outage, cutting repeat incidents from 23 to 9 a "
                   "quarter.")),
])
def test_the_number_words_warning_leaves_a_posting_term_alone(profile, term, bullet):
    terms = analysis(must=[term], keywords=[term])
    resume = resume_of(profile, [[bullet]], summary="Risk analyst with 9+ years of experience.")
    assert rules.fix_numbers(resume, terms) == [] and resume.experience[0].bullets[0] == bullet  # kept on purpose
    assert rules.number_words(resume, terms) == []
    assert not any(w.startswith("Numbers written as words") for w in tailor.reviewer_warnings(resume, analysis=terms))
    assert rules.number_words(resume) == [bullet]  # without the posting's terms: listed, as before


def test_the_number_words_warning_lists_what_fix_numbers_changes(profile):
    resume = resume_of(profile, [["Mapped 41 controls to the three lines of defense model, cutting findings to 6",
                                  "Led nine teams through the yearly audit of the payments ledger"]],
                       summary="Risk analyst who ran five whys reviews after each outage.",
                       skills=[SkillGroup(category="Practices", skills=["five whys"])])
    terms = analysis(keywords=["three lines of defense"])
    listed = rules.number_words(resume, terms)
    rules.fix_numbers(resume, terms)
    assert listed == ["Led nine teams through the yearly audit of the payments ledger"]  # the only text it changes
    assert resume.experience[0].bullets[1].startswith("Led 9 teams") and rules.number_words(resume, terms) == []


def test_the_warning_and_the_letter_keep_what_fix_numbers_keeps(profile):
    terms = analysis(keywords=["three lines of defense"])
    bullets = ["Ran five whys reviews after each outage, cutting repeat incidents from 23 to 9 a quarter",
               "Mapped 41 controls to the three lines of defense model for seven teams at Northwind Labs"]
    resume = resume_of(profile, [list(bullets)], skills=[SkillGroup(category="Practices", skills=["five whys"])])
    listed = rules.number_words(resume, terms)
    rules.fix_numbers(resume, terms)
    fixed = resume.experience[0].bullets
    assert fixed == [bullets[0], "Mapped 41 controls to the three lines of defense model for 7 teams at Northwind Labs"]
    assert listed == [old for old, new in zip(bullets, fixed) if old != new]  # what fix_numbers changes, no more
    assert rules.letter_fixes(bullets, terms, "", ["Northwind Labs"], ["five whys"]) == fixed  # the letter alike


# ------------------------------------------------------------------ a two-page plan is said
def test_the_prompt_says_one_page_or_two_when_planned(profile):
    system = tailor_system()
    assert ("a one-page resume (about 450 words), or two pages (about 900 words) when <employers> says so"
            in system)
    assert "fits on one page (or on two when <employers> says so)" in system
    two = tailor_user(SAMPLE_JD, profile.companies, "", bullets=[6, 5, 4], pages=2)
    one = tailor_user(SAMPLE_JD, profile.companies, "", bullets=[5, 3, 3])
    employers = two.split("<employers", 1)[1].split("</employers>", 1)[0]
    assert "planned for two pages (about 900 words)" in employers
    assert "two pages" not in one
    fix = revise_user(SAMPLE_JD, profile.companies, "", bullets=[6, 5, 4], pages=2, analysis=analysis(), plan=[],
                      draft="Headline: x", problems="(none)")
    assert fix.startswith(two.split("<candidate_facts>", 1)[0])  # the same opening: the provider reuses its cache


def test_every_prompt_of_a_two_page_plan_says_so(router, db, registry):
    """The draft, the fix round and the shorter draft (no time limit, a draft longer than its plan)."""
    person = career([(f"Jan {2023 - 3 * i}", "Present" if i == 0 else f"Dec {2025 - 3 * i}") for i in range(9)])
    client = Rewrites(FakeClient(registry.get("demo-detailed")))
    router.overrides["demo-detailed"] = client
    result = asyncio.run(tailor.run_tailor(router, db, profile=person, profile_id=None, jd=SAMPLE_JD,
                                           model_id="demo-detailed", save=False, time_limit=0))
    assert result.page_target == 2 and "The model shortened the resume to fit two pages." in result.notes
    prompts = [user for schema, user in client.prompts if schema in ("tailor_output", "tailor_revision")]
    assert any(SHORTEN_NOTE in user for user in prompts) and any(RETRY_MARKER in user for user in prompts)
    assert all(TWO_PAGES_LINE in user.split("</employers>", 1)[0] for user in prompts)


# ------------------------------------------------------------------ the rewrite pass keeps only notes still true
def test_first_draft_notes_about_bullets_go_with_the_rewrite(profile):
    draft = resume_of(profile, [["Build the Go order API"], ["Built the ledger"], ["Built the reports"]],
                      summary="Backend engineer with 9+ years.")
    draft.experience[1].job_title = "Data Engineer II"
    fixed = draft.model_copy(deep=True)
    notes = [
        ("Contoso Systems: title changed from 'Data Engineer' to 'Data Engineer II' so the career path never steps "
         "down."),
        "Northwind Labs: wrote 1 bullet(s) in the present tense, as the current job is written.",
        "Contoso Systems: wrote 2 bullet(s) in the past tense, as every earlier job is written.",
        ("Fabrikam: removed the closing 'working closely with product teams' from a bullet: it said who the work was "
         "done with, not how it was done or what changed."),
        "Summary: removed 'Brings strong judgment.' (self-praise, or more than 3 sentences).",
        "Headline: wrote 'Data Engineer' instead of the posting's level 'Data Engineer II', which is not your title.",
        "Skills: left out what no bullet shows: Jira.",
    ]
    # nothing rewritten: a job's tense note stays true while none of its bullets changed
    assert tailor.first_draft_notes(notes, draft, fixed) == [notes[0], notes[1], notes[2], notes[4], notes[5]]
    fixed.experience[1].job_title, fixed.summary = "Senior Data Engineer", "Data engineer with 9+ years."
    fixed.experience[1].bullets = ["Built the ledger service in Go"]  # the rewrite changed a Contoso Systems bullet
    assert tailor.first_draft_notes(notes, draft, fixed) == [notes[1], notes[5]]


def test_the_rewrite_pass_drops_the_first_drafts_tense_notes(router, db, registry, profile):
    def slips(sections):
        jobs = sections["experience"]
        jobs[0]["bullets"][1] = ("Modeled and optimised the nightly claims loads in dbt and Airflow, so analysts get "
                                 "fresh numbers by 7 am every weekday.")
        jobs[1]["bullets"][1] = ("Design and run the order-sync service in Go with idempotent retries, so partner "
                                 "integrations stayed stable through traffic spikes.")
        jobs[1]["bullets"][2] = ("Used Airflow and SQL to load the nightly finance reports, giving product teams fresh "
                                 "numbers before each stand-up.")
        jobs[2]["bullets"][1] = ("Supported the analysts with Spark jobs that refreshed the shared dashboards, cutting "
                                 "manual exports for finance.")
        sections["summary"] += " Brings strong judgment to every project."

    client, result = tailored(router, db, registry, profile, slips)
    assert result.retries == 1 and "open with a weak verb" in client.fix_round()
    final = [b for exp in result.resume.experience for b in exp.bullets]
    assert not any(b.startswith(("Model and optimize", "Designed and ran")) for b in final)  # the rewrite replaced them
    assert not any("bullet(s) in the" in n for n in result.notes)
    assert any(n.startswith("Summary: removed 'Brings strong judgment") for n in result.notes)  # the summary stays


# the demo's own current-job bullet, written in the past: the app puts it in the present, as the demo writes it
PAST_SLIP = ("Built Spark automation for recurring operational work, replacing manual hand-offs with reliable runs the "
             "team no longer has to watch.")
WEAK_OPENINGS = [
    ("Used Airflow and SQL to load the nightly finance reports, giving product teams fresh numbers before each "
     "stand-up."),
    "Supported the analysts with Spark jobs that refreshed the shared dashboards, cutting manual exports for finance.",
]


def test_a_tense_note_stays_while_the_rewrite_leaves_its_job_alone(router, db, registry, profile):
    def slip_and_weak(sections):
        jobs = sections["experience"]
        jobs[0]["bullets"][3] = PAST_SLIP
        jobs[1]["bullets"][1], jobs[2]["bullets"][1] = WEAK_OPENINGS

    client, result = tailored(router, db, registry, profile, slip_and_weak)
    assert result.retries == 1 and "open with a weak verb" in client.fix_round()
    assert "Build" + PAST_SLIP.removeprefix("Built") in result.resume.experience[0].bullets  # the rewrite kept it
    assert [n for n in result.notes if "bullet(s) in the" in n] == [
        "Northwind Labs: wrote 1 bullet(s) in the present tense, as the current job is written."]


# ------------------------------------------------------------------ a bullet left in the other tense is named
# the app cannot put these right safely: a second verb in the same tense follows the opening ('... and added ...'), or
# a word that is as often a noun ('Track') has no object it can see; the rewrite pass and the admin are told
CURRENT_SLIP = ("Rebuilt the order sync service in Go and added retries, replacing manual re-runs the team no longer "
                "has to watch.")
OLDER_SLIP = "Track cloud costs in Datadog and flag waste weekly, cutting idle compute spend for the platform team."
TENSE_FIX = ("Write the current job's bullets in the present tense and the others in the past, changing only the "
             "verbs")


def in_the_other_tense(sections):
    jobs = sections["experience"]
    jobs[0]["bullets"][3], jobs[1]["bullets"][2] = CURRENT_SLIP, OLDER_SLIP


def tense_warning(result) -> str:
    return next((w for w in result.ats_warnings if w.startswith("Bullets not in their job's tense")), "")


def test_two_bullets_left_in_the_other_tense_ask_for_the_rewrite(router, db, registry, profile):
    client, result = tailored(router, db, registry, profile, in_the_other_tense)
    asked = client.fix_round()
    assert result.retries == 1 and asked.startswith("These bullets are not in their job's tense: ")  # on their own
    assert (f"'{CURRENT_SLIP}' (Northwind Labs, the current job); '{OLDER_SLIP}' (Contoso Systems)" in asked
            and TENSE_FIX in asked)
    final = [b for exp in result.resume.experience for b in exp.bullets]
    assert CURRENT_SLIP not in final and OLDER_SLIP not in final and tense_warning(result) == ""


def test_one_bullet_left_in_the_other_tense_is_named_with_other_fixes(router, db, registry, profile):
    def one(sections):
        sections["experience"][0]["bullets"][3] = CURRENT_SLIP

    client, result = tailored(router, db, registry, profile, one)
    assert result.retries == 0 and client.fix_round() == ""  # one alone costs no rewrite ...
    assert f"'{CURRENT_SLIP[:60]}...' (Northwind Labs)" in tense_warning(result)  # ... and the admin is told

    def one_and_weak(sections):
        one(sections)
        jobs = sections["experience"]
        jobs[1]["bullets"][1], jobs[2]["bullets"][1] = WEAK_OPENINGS

    client, _result = tailored(router, db, registry, profile, one_and_weak)
    asked = client.fix_round()
    assert "open with a weak verb" in asked
    assert f"These bullets are not in their job's tense: '{CURRENT_SLIP}' (Northwind Labs, the current job)" in asked


def test_bullets_the_rewrite_leaves_in_the_other_tense_are_named(router, db, registry, profile):
    """The rewrite pass changed nothing (or was skipped at the time limit): the Format notes still say so."""
    _client, result = tailored(router, db, registry, profile, in_the_other_tense, kind=Stubborn)
    final = [b for exp in result.resume.experience for b in exp.bullets]
    assert result.retries == 1 and CURRENT_SLIP in final and OLDER_SLIP in final
    warning = tense_warning(result)
    assert (f"'{CURRENT_SLIP[:60]}...' (Northwind Labs); '{OLDER_SLIP[:60]}...' (Contoso Systems)" in warning
            and 'the present tense ("Build", "Lead") and the others in the past ("Built", "Led")' in warning)


def test_only_a_run_since_the_one_page_rule_is_told_about_the_tense(profile):
    old, new = old_run(profile), old_run(profile, page_target=1)
    for result in (old, new):
        result.resume.experience[0].bullets[1] = CURRENT_SLIP
        tailor.run_checks(result, SAMPLE_JD)  # a hand edit, a certification tick or a new cover letter
    assert tense_warning(old) == "" and CURRENT_SLIP[:60] in tense_warning(new)
