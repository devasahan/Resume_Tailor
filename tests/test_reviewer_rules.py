"""The reviewer's rules for a one-page resume that code keeps for free (app/pipeline/rules.py, numerals.py): numbers as
numerals written one way, official tool names, a short summary without praise, at most 3 skill rows of what the bullets
show; and the checks that ask for the one rewrite pass (weak openings, 3-line bullets, a long summary, praise, "so"
endings, a current job without figures). Every person, employer and product here is invented."""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import rules, validate
from app.pipeline.numerals import figure_format, normal, numerals
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER
from app.schemas import Company, JDAnalysis, Profile, SkillGroup

from .test_validate import make_output


def analysis(must=(), nice=(), keywords=()) -> JDAnalysis:
    return JDAnalysis(target_role="Backend Engineer", seniority="Senior", domain="", must_have_skills=list(must),
                      nice_to_have_skills=list(nice), keywords=list(keywords))


def resume_of(profile, bullets: list[list[str]], summary: str = "Backend engineer with 9+ years of experience.",
              skills: list[SkillGroup] | None = None):
    resume = validate.build_resume(profile, make_output())
    for exp, texts in zip(resume.experience, bullets):
        exp.bullets = list(texts)
    resume.summary = summary
    if skills is not None:
        resume.skills = skills
    return resume


# ------------------------------------------------------------------ numbers as numerals, one way
@pytest.mark.parametrize("text,expected", [
    ("Rolled out the portal to seven international schools", "Rolled out the portal to 7 international schools"),
    ("Cut onboarding to under five days", "Cut onboarding to under 5 days"),
    ("Coached twenty-five engineers and twenty five analysts", "Coached 25 engineers and 25 analysts"),
    ("Shipped one or two releases a week", "Shipped 1 or 2 releases a week"),
    ("Led one engineer and the first release", "Led one engineer and the first release"),  # 'one' alone, an ordinal
    ("Built two-factor login and one-on-one coaching", "Built two-factor login and one-on-one coaching"),  # compounds
    ("Moved Two Sigma data to Three.js charts", "Moved Two Sigma data to Three.js charts"),  # names
    ("Ran one to one mentoring with zero downtime", "Ran one to one mentoring with zero downtime"),
    ("Often tuned the canine-care app", "Often tuned the canine-care app"),  # never inside a word
])
def test_number_words_become_numerals(text, expected):
    assert numerals(text) == expected


def test_a_posting_term_keeps_its_number_word():
    assert numerals("Set up three lines of defense for three teams", keep=["three lines of defense"]) == (
        "Set up three lines of defense for 3 teams")


@pytest.mark.parametrize("text,expected", [
    ("for 500k+ users", "for 500K+ users"),
    ("saving $1.2m and $3b", "saving $1.2M and $3B"),
    ("saving $2 million a year", "saving $2M a year"),
    ("for 12 thousand stores", "for 12K stores"),
    ("cut errors by 38 percent, then 12 per cent", "cut errors by 38%, then 12%"),
    ("a 38 % drop", "a 38% drop"),
    ("p95 of 684 ms over 1.7 TB", "p95 of 684ms over 1.7TB"),
    ("ran 5km and 5kg and 1.2m", "ran 5km and 5kg and 1.2m"),  # units, and an 'm' without a currency sign
])
def test_every_figure_is_written_one_way(text, expected):
    assert figure_format(text) == expected


def test_normal_does_both():
    assert normal("seven schools saved $2 million") == "7 schools saved $2M"


def test_fix_numbers_changes_the_text_and_keeps_names_and_posting_terms(profile):
    resume = resume_of(profile, [["Coached seven engineers through three lines of defense reviews"]],
                       summary="Backend engineer who led nine teams through a $2 million migration.")
    resume.profile.companies[0].name = "Nine Yards"  # capitalised anyway: never changed
    assert rules.fix_numbers(resume, analysis(keywords=["three lines of defense"])) == []
    assert resume.summary == "Backend engineer who led 9 teams through a $2M migration."
    assert resume.experience[0].bullets[0] == "Coached 7 engineers through three lines of defense reviews"


def test_number_words_left_by_a_hand_edit_are_listed(profile):
    resume = resume_of(profile, [["Cut costs by 38 percent"], ["Moved 12 services"]], summary="Led five teams.")
    assert rules.number_words(resume) == ["Led five teams.", "Cut costs by 38 percent"]


def test_the_persons_seven_in_hints_and_notes_is_their_7(profile):
    resume = resume_of(profile, [["Rolled out the portal to 7 schools in 2 terms"]])
    flags = validate.claim_flags(resume, "", "Rolled it out to seven schools in two terms")
    assert not any(f.level == "verify" and f.section.startswith("experience") for f in flags)
    assert "7" in validate.note_figures("Portal for seven schools")


# ------------------------------------------------------------------ official names
def test_tools_get_their_official_names_everywhere(profile):
    resume = resume_of(profile, [["Built Postgres replicas with Pytest checks on Github Actions and NodeJS"]],
                       summary="Backend engineer who ships Typescript services on Postgres.",
                       skills=[SkillGroup(category="Data", skills=["Postgres", "PostgreSQL", "Pytest"])])
    assert rules.official_names(resume) == []
    assert resume.experience[0].bullets[0] == (
        "Built PostgreSQL replicas with pytest checks on GitHub Actions and Node.js")
    assert resume.summary == "Backend engineer who ships TypeScript services on PostgreSQL."
    assert resume.skills[0].skills == ["PostgreSQL", "pytest"]  # one way, then de-duplicated


def test_links_paths_and_addresses_keep_their_spelling(profile):
    text = "Moved github.com/acme/api and postgres://db.internal to Github"
    resume = resume_of(profile, [[text]])
    rules.official_names(resume)
    assert resume.experience[0].bullets[0] == "Moved github.com/acme/api and postgres://db.internal to GitHub"


def test_the_postings_own_form_wins(profile):
    resume = resume_of(profile, [["Tuned PostgreSQL queries and Postgres indexes"]])
    rules.official_names(resume, analysis(must=["Postgres"]), "We run Postgres on AWS.")
    assert resume.experience[0].bullets[0] == "Tuned Postgres queries and Postgres indexes"


def test_the_letter_gets_the_same_names_and_numbers():
    assert rules.letter_fixes(["I moved seven Postgres clusters at Nine Yards."], analysis(), "",
                              ["Nine Yards"]) == ["I moved 7 PostgreSQL clusters at Nine Yards."]


# ------------------------------------------------------------------ the summary
def test_self_praise_and_a_fourth_sentence_leave_the_summary(profile):
    resume = resume_of(profile, [["Built the Go order API for 450 partner stores"]], summary=(
        "Backend engineer with 9+ years in commerce. Focuses on partner APIs, cutting failed syncs from 41 to 6 a "
        "week. Brings strong judgment to every project. Known for calm incident work."))
    notes = rules.summary_trim(resume, analysis(must=["Go"]))
    assert resume.summary == ("Backend engineer with 9+ years in commerce. Focuses on partner APIs, cutting failed "
                              "syncs from 41 to 6 a week.")
    assert notes and notes[0].startswith("Summary: removed 'Brings strong judgment")


def test_a_sentence_holding_a_must_haves_only_mention_stays(profile):
    summary = ("Backend engineer with 9+ years in commerce. Focuses on partner APIs in Go. Brings proven Kubernetes "
               "leadership to platform teams.")
    resume = resume_of(profile, [["Built the Go order API for 450 partner stores"]], summary=summary)
    assert rules.summary_trim(resume, analysis(must=["Kubernetes"])) == [] and resume.summary == summary


def test_two_sentences_always_stay(profile):
    summary = "Backend engineer with 9+ years in commerce. Passionate about clean code."
    resume = resume_of(profile, [["Built the Go order API"]], summary=summary)
    assert rules.summary_trim(resume) == [] and resume.summary == summary
    assert rules.self_praise(summary) == ["Passionate about clean code."]


def test_a_summary_naming_more_than_three_skills_is_found(profile):
    resume = resume_of(profile, [["Built the Go order API"]],
                       summary="Backend engineer using Python, SQL, Airflow and Spark for data teams.",
                       skills=[SkillGroup(category="Core", skills=["Python", "SQL", "Airflow", "Spark", "Docker"])])
    assert rules.summary_skill_items(resume) == ["Python", "SQL", "Airflow", "Spark"]


# ------------------------------------------------------------------ the skills
def test_skills_keep_what_the_bullets_show_and_the_postings_terms_first(profile):
    resume = resume_of(profile, [["Built Airflow loads in Python and SQL"], ["Moved reports to dbt"]], skills=[
        SkillGroup(category="Tools", skills=["dbt", "Jira", "Airflow"]),
        SkillGroup(category="Languages", skills=["SQL", "Python", "Rust"])])
    notes = rules.shape_skills(resume, analysis(must=["Python"], nice=["Kafka", "Airflow"], keywords=["Rust"]))
    assert [(g.category, g.skills) for g in resume.skills] == [
        ("Languages", ["Python", "Rust", "SQL"]),  # the must-have first; Rust is a posting term, shown or not
        ("Tools", ["Airflow", "dbt"])]
    assert notes == ["Skills: left out what no bullet shows: Jira."]


def test_a_fourth_row_joins_the_third_and_the_rest_is_named(profile):
    shown = "Built it with " + ", ".join(f"Tool{i}" for i in range(30))
    resume = resume_of(profile, [[shown]], skills=[
        SkillGroup(category=f"Row {r}", skills=[f"Tool{r * 6 + i}" for i in range(6)]) for r in range(5)])
    notes = rules.shape_skills(resume, analysis(keywords=["Tool29"]))
    # the row with the posting's term first (its term leading), then the model's order; 18 items at most
    assert [g.category for g in resume.skills] == ["Row 4", "Row 0", "Row 1"]
    assert resume.skills[0].skills[0] == "Tool29" and sum(len(g.skills) for g in resume.skills) == 18
    assert notes == ["Skills: kept 3 rows, so left out Tool12, Tool13, Tool14, Tool15, Tool16, Tool17, Tool18, Tool19 and 4 more."]


# ------------------------------------------------------------------ the checks
@pytest.mark.parametrize("bullet,weak", [
    ("Used Claude with tool calling to answer 500K+ support questions", True),
    ("Supported 38 analysts with nightly Airflow loads", True),
    ("Run evaluation suites that replay real scenarios before each release", True),
    ("Helped the team move to Kubernetes", True),
    ("Handled on-call for the payments API", True),
    ("Support tickets fell from 97 to 57 a week after the help center launch", False),  # a noun
    ("Use cases for the assistant were drafted with product", False),
    ("Built the evaluation suite behind every release", False),
])
def test_weak_openings_are_found_and_nouns_are_not(profile, bullet, weak):
    resume = resume_of(profile, [[bullet]])
    assert (rules.weak_openers(resume) == [bullet]) is weak


def test_so_endings_more_than_one_in_a_job(profile):
    many = ["Rebuilt the order sync in Go, so partners saw stock within a minute",
            "Moved reports to dbt with tests, so that month-end closed on time",
            "So far the only owner of billing, rebuilt its retries"]  # an early 'so' is no ending
    resume = resume_of(profile, [many, ["Rebuilt the ledger, so finance closed on time"]])
    assert rules.so_endings(resume) == [("Northwind Labs", many[:2])]


def test_long_bullets_and_a_long_summary_are_measured_in_lines(profile):
    two_lines = ("Rebuilt the order sync service behind the partner REST API in Go, adding retries and rate limits for "
                 "partners")
    three_lines = two_lines + (" and a nightly reconciliation job that compares every order with the ledger and "
                               "opens tickets for each mismatch it finds")
    resume = resume_of(profile, [[two_lines, three_lines]], summary=" ".join(["Backend engineer"] * 40) + ".")
    long, summary_lines = rules.long_text(resume)
    assert long == [three_lines] and summary_lines >= 4


def test_the_current_job_needs_two_figures(profile):
    resume = resume_of(profile, [["Built the Go order API for 450 partner stores", "Moved reports to dbt"],
                                 ["Cut costs by 12.4%", "Saved $41,300 a year"]])
    assert rules.current_job_figures(resume) == ("Northwind Labs", 1)
    resume.experience[0].bullets[1] = "Moved 38 reports to dbt"
    assert rules.current_job_figures(resume) is None


def test_a_tip_when_the_current_jobs_notes_give_no_numbers():
    person = Profile(name="Taylor Quinn", companies=[
        Company(name="Northwind Labs", start="Jan 2024", end="Present", projects="Order API rebuild"),
        Company(name="Bluefin Commerce", start="Jan 2020", end="Dec 2023", projects="Served 450 stores, 12 regions")])
    tip = rules.notes_figure_tip(person)
    assert tip and tip.startswith("Tip: add numbers for Northwind Labs to its project notes")
    person.companies[0].projects = "Order API for 450 partner stores, catching 17 regressions a quarter"
    assert rules.notes_figure_tip(person) is None


# ------------------------------------------------------------------ the one rewrite pass
class WeakDraft(LLMClient):
    """The first draft opens two bullets with weak verbs and praises the candidate; the fix round is the demo's own."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            jobs = result.data["sections"]["experience"]
            jobs[1]["bullets"][1] = ("Used Airflow and SQL to load the nightly finance reports, giving product teams "
                                     "fresh numbers before each stand-up.")
            jobs[2]["bullets"][1] = ("Supported the analysts with Spark jobs that refreshed the shared dashboards, "
                                     "cutting manual exports for finance.")
        return result


def test_two_weak_openings_ask_for_the_rewrite_pass(router, db, registry, profile, jd):
    client = WeakDraft(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    save=False))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    assert "These bullets open with a weak verb: 'Used Airflow and SQL" in client.prompts[1]
    assert "in its job's tense" in client.prompts[1]


class OneWeak(WeakDraft):
    async def generate_structured(self, **kwargs):
        result = await super().generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output":
            jobs = result.data["sections"]["experience"]
            jobs[2]["bullets"][1] = ("Refreshed the shared dashboards with Spark jobs every morning, cutting manual "
                                     "exports for the finance analysts.")
        return result


def test_one_weak_opening_alone_costs_no_rewrite_pass(router, db, registry, profile, jd):
    client = OneWeak(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    save=False))
    assert result.status == "ok" and result.retries == 0 and len(client.prompts) == 1
