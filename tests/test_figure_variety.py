"""Figures written in varied shapes. A recruiter reads the same "by 23.4% (from 47 to 36 minutes)" in every bullet as a
template, a sign of AI-written text, so about half of the results give only the real numbers, the others their base
in words, and a base in brackets appears at most once in a resume (and once in a letter). Every figure keeps its
context: only a percentage is ever dropped, and only where its real numbers stay beside it."""
import asyncio
import re
from collections import Counter

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, cover_system, tailor_system
from app.schemas import Company, CoverLetter, Profile

from .conftest import SAMPLE_JD
from .test_ai_signs import edit_body
from .test_api import save_profile
from .test_validate import make_output

NO_FIGURE = "Wrote the on-call runbook for the billing team so new engineers could handle payment incidents alone."
# A made-up resume in the shape the person sent: every result reads 'by N% (BASE)', and one level ends in '.0'
EXAMPLE = [
    [("Rebuilt the deployment pipeline on GitHub Actions and Argo CD, cutting deployment pipeline runtime by 23.4% "
      "(from 47 to 36 minutes) while supporting 42 application engineers."),
     ("Moved shared platform components to contract tests in Pact, improving test coverage from 62.0% to 89.0% across "
      "shared platform components.")],
    [("Built a self-serve help center in Zendesk Guide for 2.3K merchants, which reduced support tickets by 41.2% "
      "(from 85 to 50 weekly)."),
     ("Added Redis caching to the listing search API, reducing average response latency by 28.5% (from 350ms to "
      "250ms) for high-traffic listings.")],
    [("Moved batch reporting jobs from on-demand EC2 to Spot instances, cutting hosting costs by 18.2% ($24,000 a "
      "year)."), NO_FIGURE],
]
EXPECTED = [
    [EXAMPLE[0][0],  # the first base in brackets stays
     ("Moved shared platform components to contract tests in Pact, improving test coverage from 62% to 89% across "
      "shared platform components.")],
    [("Built a self-serve help center in Zendesk Guide for 2.3K merchants, which reduced support tickets from 85 to "
      "50 weekly."),  # the real numbers alone
     ("Added Redis caching to the listing search API, reducing average response latency by 28.5%, from 350ms to "
      "250ms, for high-traffic listings.")],  # the base in words
    ["Moved batch reporting jobs from on-demand EC2 to Spot instances, cutting hosting costs by $24,000 a year.",
     NO_FIGURE],
]
# The first base in brackets of a document, which stays; and a later one that becomes the real numbers alone, so the
# next rewrite after it takes the other shape
FIRST = "Cut nightly runtime by 23.4% (from 47 to 36 minutes) by moving the largest Airflow jobs to Spark."
REAL = "Cut support tickets by 41.2% (from 85 to 50 weekly) with a help center."
REAL_AFTER = "Cut support tickets from 85 to 50 weekly with a help center."
_NUMBER = re.compile(r"(?<![\w.,])[$€£]?(\d[\d,]*(?:\.\d+)?)(\s?%)?")
BRACKET_BASE = re.compile(r"\d%\)?\s*\((?:from\s|[$€£\d])")  # 'by 22% (from 41 ...', '17.8% ($41,000 a year)'


def run(router, db, profile, jd=SAMPLE_JD, model="demo-concise", **kw):
    return asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id=model, **kw))


def resume_with(profile, *jobs):
    out = make_output()
    for exp, bullets in zip(out.sections.experience, jobs):
        exp.bullets = list(bullets)
    return validate.build_resume(profile, out)


def texts_of(resume) -> list[str]:
    return [resume.summary, *(bullet for exp in resume.experience for bullet in exp.bullets)]


def bracketed(texts) -> int:
    """How many percentages in the texts have their base in brackets."""
    return sum(base.kind == "bracket" for text in texts for sentence in validate._SENTENCE_END.split(text)
               for _match, base in validate._change_percentages(sentence))


def numbers(texts) -> tuple[Counter, Counter]:
    """(the numbers that are no percentage, the percentages), by value: '1,200' and '1200', '62.0%' and '62%' are one."""
    found = [(validate._canonical(value), bool(percent)) for text in texts for value, percent in _NUMBER.findall(text)]
    return Counter(v for v, percent in found if not percent), Counter(v for v, percent in found if percent)


def assert_sound(before: list[str], after: list[str], allowed: str = "") -> None:
    """What vary_figures promises: no number changes value, no value before or after and no amount goes, only
    percentages are dropped; no percentage loses its base or gets one that disagrees; a second run changes nothing."""
    (plain_before, percents_before), (plain_after, percents_after) = numbers(before), numbers(after)
    assert plain_after == plain_before and percents_after <= percents_before
    assert [u for text in after for u in validate.unbased_percentages(text)] == [
        u for text in before for u in validate.unbased_percentages(text)]
    assert Counter(m for text in after for m in validate.base_mismatches(text)) <= Counter(
        m for text in before for m in validate.base_mismatches(text))
    assert validate._varied(after, allowed) == after


# ------------------------------------------------------------------ the resume the person sent, made up
def test_a_resume_of_bracketed_bases_gets_varied_shapes_and_keeps_every_figure(profile):
    resume = resume_with(profile, *EXAMPLE)
    before = texts_of(resume)
    changed = validate.vary_figures(resume)
    assert [exp.bullets for exp in resume.experience] == EXPECTED
    assert changed == [EXPECTED[0][1], *EXPECTED[1], EXPECTED[2][0]]
    assert bracketed(texts_of(resume)) == 1 and bracketed(before) == 5 - 1  # the levels '62.0% to 89.0%' need none
    assert validate.unbased_percentage_problems(resume) == validate.base_mismatch_problems(resume) == []
    assert_sound(before, texts_of(resume))
    assert validate.vary_figures(resume) == []  # a second run changes nothing


# ------------------------------------------------------------------ each rule
@pytest.mark.parametrize("given,expected", [
    # the real numbers alone, for 'by P%' right before a bracket that holds a pair or an amount
    ([FIRST, REAL], [FIRST, REAL_AFTER]),
    ([FIRST, "Cut queue time by 23.4% (47 to 36 minutes) for 83K players."],
     [FIRST, "Cut queue time from 47 to 36 minutes for 83K players."]),
    ([FIRST, "Cut hosting costs by 18.2% ($24,300 a year)."], [FIRST, "Cut hosting costs by $24,300 a year."]),
    ([FIRST, "Cut hosting costs by 18.2% (€24,300 a year)."], [FIRST, "Cut hosting costs by €24,300 a year."]),
    ([FIRST, "Cut manual QA effort by 37.5% (12 hours a week) with Playwright suites."],
     [FIRST, "Cut manual QA effort by 12 hours a week with Playwright suites."]),
    ([FIRST, "Lifted trial-to-paid conversion by 12.9% (from 3.1% to 3.5%) with onboarding emails."],
     [FIRST, "Lifted trial-to-paid conversion from 3.1% to 3.5% with onboarding emails."]),
    # the base in words where the real numbers alone would not read naturally; a comma closes the clause only when
    # the sentence goes on with a word
    ([FIRST, "Made nightly builds 21% faster (from 14 to 11 minutes) with remote caching."],
     [FIRST, "Made nightly builds 21% faster, from 14 to 11 minutes, with remote caching."]),
    ([FIRST, "Cut false alerts by 37.2% (1,200 fewer alerts a month) with tuned thresholds."],
     [FIRST, "Cut false alerts by 37.2%, or 1,200 fewer alerts a month, with tuned thresholds."]),
    ([FIRST, "Moved batch jobs to Spot, cutting EC2 spend by 17.8% (saving $41,300 a year)."],
     [FIRST, "Moved batch jobs to Spot, cutting EC2 spend by 17.8%, saving $41,300 a year."]),
    ([FIRST, "Cut API latency by 21% (p95: 820 to 648 ms) with Redis caching."],  # a labelled pair
     [FIRST, "Cut API latency by 21%, p95 from 820 to 648 ms, with Redis caching."]),
    ([FIRST, "Cut queue time by 22% (down from 41 to 32 seconds) for 83K players."],
     [FIRST, "Cut queue time by 22%, down from 41 to 32 seconds, for 83K players."]),
    ([FIRST, "Cut cloud spend by 17.8% across 41 services in 3 regions ($41,300 a year)."],
     [FIRST, "Cut cloud spend by 17.8% across 41 services in 3 regions, or $41,300 a year."]),
    # the two shapes take turns: real numbers, then words
    ([FIRST, REAL, "Cut deploy time by 23.4% (from 47 to 36 minutes) while supporting 42 engineers."],
     [FIRST, REAL_AFTER, "Cut deploy time by 23.4%, from 47 to 36 minutes, while supporting 42 engineers."]),
    ([FIRST, REAL, "Cut deploy time by 23.4% (from 47 to 36 minutes)."],
     [FIRST, REAL_AFTER, "Cut deploy time by 23.4%, from 47 to 36 minutes."]),
    ([FIRST, REAL, "Cut deploy time by 23.4% (from 47 to 36 minutes), so releases shipped daily."],
     [FIRST, REAL_AFTER, "Cut deploy time by 23.4%, from 47 to 36 minutes, so releases shipped daily."]),
    ([FIRST, REAL, "Cut deploy time by 23.4% (from 47 to 36 minutes); releases shipped daily."],
     [FIRST, REAL_AFTER, "Cut deploy time by 23.4%, from 47 to 36 minutes; releases shipped daily."]),
    ([FIRST, REAL, "Cut licence costs by 17.8% (£41,300 a year) for the finance team."],
     [FIRST, REAL_AFTER, "Cut licence costs by 17.8%, or £41,300 a year, for the finance team."]),
    ([FIRST, REAL, "Cut deploy time by 23.4% (from 47 to 36 minutes).", "Cut hosting costs by 18.2% ($24,300 a year)."],
     [FIRST, REAL_AFTER, "Cut deploy time by 23.4%, from 47 to 36 minutes.", "Cut hosting costs by $24,300 a year."]),
])
def test_each_later_base_in_brackets_becomes_the_real_numbers_or_words(given, expected):
    varied = validate._varied(given, "")
    assert varied == expected and bracketed(varied) == 1
    assert_sound(given, varied)


@pytest.mark.parametrize("text", [
    "Added single sign-on (SSO) and role-based access control (RBAC) to the admin console for 38 teams.",
    "Tracked checkout latency (p95) and errors in Grafana, with retries on queues (e.g. Kafka).",
    "Grew enterprise ARR by 31.7% (2021 to 2023) with usage-based pricing.",  # years: no base, not touched
    "Cut p95 latency by 21% (41 services, from 820 to 648 ms) with Redis caching.",  # reads naturally in no shape
    "Saved $41,300 a year on AWS (EC2 and RDS rightsizing), a 17.8% cut in spend.",
    "Raised availability to 99.95% and kept error rates at 0.05% and 1.05% through peak.",
])
def test_brackets_that_are_no_base_or_cannot_be_reworded_stay(text):
    assert validate._varied([FIRST, text], "") == [FIRST, text]


def test_a_rewrite_that_would_chain_three_ing_clauses_leaves_the_bracket():
    text = ("Moved batch jobs to Spot, cutting EC2 spend by 17.8% (freeing 12 hours a week), and lowering on-call load "
            "for 38 engineers.")
    assert validate._varied([FIRST, text], "") == [FIRST, text]  # ', freeing' would make a third '-ing' clause
    assert validate.cover_plain_problems(CoverLetter(greeting="Hi,", closing="Thanks,", paragraphs=[text, "Bye."])) == []


@pytest.mark.parametrize("given,expected", [
    # a bracket-free shape used before becomes the real numbers alone where that reads naturally
    (["Cut weekly support tickets by 41.2%, from 85 to 50, with a help center.",
      "Cut nightly runtime by 23.4%, from 47 to 36 minutes, while adding 14 jobs."],
     ["Cut weekly support tickets by 41.2%, from 85 to 50, with a help center.",
      "Cut nightly runtime from 47 to 36 minutes, while adding 14 jobs."]),
    (["Brought p95 latency from 820 to 648ms, a 21% drop, with Redis caching.",
      "Cut queue time from 41 to 32 seconds, a 22% drop for 83K players."],
     ["Brought p95 latency from 820 to 648ms, a 21% drop, with Redis caching.",
      "Cut queue time from 41 to 32 seconds for 83K players."]),
    (["Reduced cloud spend by 17.8%, or $41,300 a year.", "Reduced EC2 spend by 12.6%, or $23,700 a year."],
     ["Reduced cloud spend by 17.8%, or $41,300 a year.", "Reduced EC2 spend by $23,700 a year."]),
    # ... otherwise it stays
    (["Cut false alerts by 37.2%, or 1,200 fewer alerts a month.", "Cut false pages by 41.4%, or 36 fewer pages a month."],
     None),
    (["Brought p95 latency from 820 to 648ms, a 21% drop, with Redis caching.",
      "Cut handling time from 41 to 32 minutes, a 22% drop in wait time for agents."], None),
    (["Made nightly builds 21% faster, from 14 to 11 minutes.", "Made test runs 23.4% faster, from 47 to 36 minutes."],
     None),
    # two different shapes both stay
    (["Cut weekly support tickets by 41.2%, from 85 to 50.", "Reduced cloud spend by 17.8%, or $41,300 a year."], None),
])
def test_a_shape_used_twice_becomes_the_real_numbers_where_it_reads_naturally(given, expected):
    varied = validate._varied(given, "")
    assert varied == (expected or given)
    assert_sound(given, varied)


def test_the_candidates_own_figures_are_never_dropped():
    hints = "Cut weekly support tickets by 41.2% at Contoso. Reduced cloud spend by 17.8 percent."
    given = [FIRST, REAL, "Reduced cloud spend by 17.8% ($41,300 a year).",
             "Cut weekly support tickets by 41.2%, from 85 to 50.", "Cut weekly tickets by 41.2%, from 85 to 50 again."]
    varied = validate._varied(given, hints)
    assert varied == [FIRST, "Cut support tickets by 41.2%, from 85 to 50 weekly, with a help center.",
                      "Reduced cloud spend by 17.8%, or $41,300 a year.", *given[3:]]
    assert_sound(given, varied, hints)
    assert validate._varied(given, "")[1] == REAL_AFTER  # an estimate may lose its percentage


def test_a_percentage_whose_base_disagrees_is_never_dropped():
    """Dropping it would hide the mismatch from the checks and the fix round, and leave odd text ('Cut Kafka costs from
    14 to 6 clusters')."""
    given = [FIRST, REAL, "Cut Kafka costs by 17.8% (from 14 to 6 clusters) with tiered storage.",
             "Cut queue time by 22%, from 41 to 38 seconds.", "Cut wait time by 23.4%, from 41 to 38 seconds."]
    varied = validate._varied(given, "")
    assert varied == [FIRST, REAL_AFTER, "Cut Kafka costs by 17.8%, from 14 to 6 clusters, with tiered storage.",
                      *given[3:]]
    assert [m for text in varied for m in validate.base_mismatches(text)] == [
        "from 14 to 6", "from 41 to 38 seconds", "from 41 to 38 seconds"]


def test_a_percentage_loses_its_point_zero_unless_the_candidate_wrote_it():
    given = ["Raised test coverage from 62.0% to 89.0% and kept uptime at 99.0%.",
             "Kept error rates at 10.05% and 1.05%, and cut 1,200.0% of nothing by 12.00% (from 50 to 44 a week)."]
    assert validate._varied(given, "") == [
        "Raised test coverage from 62% to 89% and kept uptime at 99%.",
        "Kept error rates at 10.05% and 1.05%, and cut 1,200% of nothing by 12% (from 50 to 44 a week)."]
    assert validate._varied(given[:1], "My coverage went from 62.0% to 89%.") == [
        "Raised test coverage from 62.0% to 89% and kept uptime at 99%."]


MONEY_NO_CHANGE = [  # a total, a budget, a level or a comparison: none is the size of the change
    "Exceeded the FY23 sales quota by 18% ($2.36M against a $2M target) with 14 enterprise deals.",
    "Grew the mid-market book of business by 31% ($4.1M ARR) through expansion plays.",
    "Lowered freight spend by 11.3% (€1.2M budget) by tendering 14 lanes to regional carriers.",
    "Beat the annual savings target by 12% (£1.12M vs £1M) through supplier consolidation.",
    "Grew ARR by 24.3% ($12.4M) through usage-based pricing.",  # money alone may be the new level
]


@pytest.mark.parametrize("text", MONEY_NO_CHANGE)
@pytest.mark.parametrize("before", [[FIRST], [FIRST, REAL]])  # its turn for the real numbers alone, then for words
def test_money_that_is_no_amount_with_its_period_stays_in_its_bracket(before, text):
    """'by $2.36M against a $2M target' or 'by 31%, or $4.1M ARR' would misstate the result."""
    assert validate._varied([*before, text], "")[-1] == text


@pytest.mark.parametrize("amount", ["$24,300 a year", "€41,300 per year", "£12K a month", "$41,300 annually"])
def test_money_with_its_period_is_the_amount_of_the_change(amount):
    text = f"Cut hosting costs by 18.2% ({amount}) with Spot instances."
    assert validate._varied([FIRST, text], "")[1] == f"Cut hosting costs by {amount} with Spot instances."
    assert validate._varied([FIRST, REAL, text], "")[2] == (
        f"Cut hosting costs by 18.2%, or {amount}, with Spot instances.")


def test_money_that_is_no_amount_stays_in_a_letter_and_in_words():
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        ("At Northwind Labs I grew the self-serve channel by 31% ($4.1M ARR) and exceeded my FY21 quota by 18% "
         "($2.36M against a $2M target)."), "I would welcome a conversation about the revenue team."])
    assert validate.vary_letter_figures(letter) == []
    given = ["Reduced cloud spend by 17.8%, or $41,300 a year.",
             "Grew the partner book by 31%, or $4.1M ARR, through expansion plays."]  # the same shape twice
    assert validate._varied(given, "") == given


@pytest.mark.parametrize("given", [
    ["Closed $2.4M in new ARR in FY24, a 38% increase over FY23.",
     "Grew the partner channel to $900K in annual bookings, a 45% increase, with co-selling playbooks."],
    ["Moved logs to S3 Glacier at $12K a year, 64% cheaper.",
     "Renegotiated the courier contract at £310K a year, 18% cheaper."],
])
def test_a_percentage_after_money_is_never_dropped(given):
    """The money may be a level or a price ('to $900K', 'at £310K a year'), not the change: without its percentage
    the bullet would state no result at all."""
    assert validate._varied(given, "") == given


@pytest.mark.parametrize("text", [
    "Cut queue time from 41 to 32 seconds, a 22% drop and a 9-point rise in NPS.",
    "Shortened the close from 9 to 6 days, a 33.3% drop and the shortest in the group.",
    "Cut checkout time from 95 to 71 seconds, a 25.3% drop, and the fastest in the company.",
])
def test_an_apposition_that_and_goes_on_with_is_never_dropped(text):
    """'Cut queue time from 41 to 32 seconds and a 9-point rise in NPS' would say the NPS rise was cut."""
    given = ["Brought p95 latency from 847 to 669ms, a 21% drop, with Redis caching.", text]
    assert validate._varied(given, "") == given


def test_a_clause_that_already_says_from_gets_its_base_in_words():
    """The real numbers alone would read 'from organic search from 2,513 to 4,121'."""
    text = "Grew traffic from organic search by 64% (from 2,513 to 4,121 visits a month) with an SEO hub."
    assert validate._varied([FIRST, text], "")[1] == (
        "Grew traffic from organic search by 64%, from 2,513 to 4,121 visits a month, with an SEO hub.")
    text = "Grew sign-ups from organic search for 1,213 stores by 64% (from 2,513 to 4,121 a month) with an SEO hub."
    assert validate._varied([FIRST, text], "")[1] == (  # the comma of '1,213' starts no clause
        "Grew sign-ups from organic search for 1,213 stores by 64%, from 2,513 to 4,121 a month, with an SEO hub.")
    given = ["Cut weekly support tickets by 41.2%, from 97 to 57.",
             "Grew traffic from organic search by 64.1%, from 2,513 to 4,121 visits a month."]  # the same shape twice
    assert validate._varied(given, "") == given


@pytest.mark.parametrize("hints", ["Cut support tickets by 41,2 % at Contoso (my own number).",
                                   "Cut support tickets by 41,2% at Contoso."])
def test_the_candidates_percentage_with_a_decimal_comma_is_never_dropped(hints):
    assert validate._varied([FIRST, REAL], hints)[1] == (
        "Cut support tickets by 41.2%, from 85 to 50 weekly, with a help center.")
    # a comma before three digits stays a thousands separator: 4,120% is not the draft's 4.12%
    given = [FIRST, "Cut support tickets by 4.12% (from 97 to 93 weekly) with a help center."]
    assert validate._varied(given, "Grew sign-ups by 4,120% in 2021.")[1] == (
        "Cut support tickets from 97 to 93 weekly with a help center.")


@pytest.mark.parametrize("period", ["per week", "a week", "each month"])
def test_a_period_after_the_bracket_stays_with_its_base(period):
    text = f"Reduced inbound tickets by 41.2% (from 97 to 57) {period} with a chatbot."
    assert validate._varied([FIRST, REAL, text], "")[2] == (
        f"Reduced inbound tickets by 41.2%, from 97 to 57 {period}, with a chatbot.")
    assert validate._varied([FIRST, text], "")[1] == f"Reduced inbound tickets from 97 to 57 {period} with a chatbot."
    for words in ("a year later", "each quarter of 2023"):  # no period of the base
        later = f"Reduced inbound tickets by 41.2% (from 97 to 57) {words} with a chatbot."
        assert validate._varied([FIRST, REAL, later], "")[2] == (
            f"Reduced inbound tickets by 41.2%, from 97 to 57, {words} with a chatbot.")


@pytest.mark.parametrize("given,expected", [
    ("Raised 90-day retention by 22.6% (up from 31% to 38%) with lifecycle emails.",
     "Raised 90-day retention by 22.6%, up from 31% to 38%, with lifecycle emails."),
    ("Cut API latency by 21% (p95 latency from 847 to 669 ms) with Redis caching.",
     "Cut API latency by 21%, p95 latency from 847 to 669 ms, with Redis caching."),
])
def test_a_labelled_or_up_and_down_base_is_written_in_words(given, expected):
    assert validate._varied([FIRST, given], "") == [FIRST, expected]


def test_a_letters_ing_clauses_are_counted_per_sentence():
    """As cover_plain_problems reads a letter: two '-ing' clauses in the sentence before do not keep a second base
    in brackets; in one sentence joined by ';' a third would be a chain, so that bracket stays."""
    first = ("At Northwind Labs I cut deployment time by 23.4% (from 47 to 36 minutes), working with 9 product teams, "
             "and coaching 4 new engineers")
    later = "later I reduced EC2 spend by 17.8% (freeing 12 hours a week) with Spot instances."
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,",
                         paragraphs=[f"{first}. {later[0].upper()}{later[1:]}", "Thank you for your time."])
    validate.vary_letter_figures(letter)
    assert letter.paragraphs[0] == (
        f"{first}. Later I reduced EC2 spend by 17.8%, freeing 12 hours a week, with Spot instances.")
    assert bracketed(letter.paragraphs) == 1 and validate.cover_plain_problems(letter) == []
    joined = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,",
                         paragraphs=[f"{first}; {later}", "Thank you for your time."])
    assert validate.vary_letter_figures(joined) == []


def test_a_cover_letter_keeps_its_own_first_bracket_whatever_the_resume_holds(profile):
    resume = resume_with(profile, *EXAMPLE)
    validate.vary_figures(resume)
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "At Northwind Labs I cut nightly runtime by 23.4% (from 47 to 36 minutes) by moving the largest jobs to Spark.",
        ("At Contoso Systems I cut support tickets by 41.2% (from 85 to 50 a week) with a help center, and cloud spend "
         "by 17.8% ($41,300 a year)."),
        "I would welcome a conversation about the platform team."])
    before = list(letter.paragraphs)
    assert validate.vary_letter_figures(letter) == [
        ("At Contoso Systems I cut support tickets from 85 to 50 a week with a help center, and cloud spend by 17.8%, "
         "or $41,300 a year.")]
    assert letter.paragraphs[0] == before[0] and bracketed(letter.paragraphs) == 1
    assert validate.unbased_percentage_problems(resume, letter) == validate.base_mismatch_problems(resume, letter) == []
    assert_sound(before, letter.paragraphs)


# ------------------------------------------------------------------ the validator reads every shape
@pytest.mark.parametrize("text", [
    "Cut false alerts by 37.2%, or 1,200 fewer alerts a month.",
    "Cut manual QA effort by 37.5%, or 12 hours a week, with Playwright suites.",
    "Delivered a 22% reduction in queue time, from 41 to 32 seconds.",
    "Cut cloud spend by 17.8% across 41 services, or $41,300 a year.",
    "Cut p95 latency by 21% for 4,600 merchants, from 820 to 648 ms.",
    "Cut API latency by 21%, p95 from 820 to 648 ms, with Redis caching.",
    "Cut MTTR by 31.7%, MTTR from 41 to 28 minutes.",
    "Reduced cloud spend by 17.8%, or €41,300 a year.",
    "Cut weekly support tickets by 41.2%, from 97 to 57.",  # the prompt's own examples
    "Brought p95 latency from 847 to 669ms, a 21% drop.",
    "Reduced cloud spend by 17.8%, or $41,000 a year.",
    "Cut API latency by 21%, p95 latency from 847 to 669 ms.",  # a label with a noun, or with 'down'
    "Cut API latency by 21%, p95 down from 847 to 669 ms.",
    "Cut monthly churn by 12.2%, down from 4.1% to 3.6%, with a save flow.",
])
def test_the_validator_accepts_every_shape_the_fixer_writes_and_the_prompt_teaches(text):
    assert validate.unbased_percentages(text) == [] and validate.base_mismatches(text) == []


@pytest.mark.parametrize("text,found", [
    ("Cut cloud spend by 17.8% across 41 services.", ["by 17.8%"]),
    ("Cut checkout latency by 21% for EU shoppers, protecting $2.3M in revenue.", ["by 21%"]),
    ("Cut queue time by 22% and moved builds, from 41 to 32 seconds.", ["by 22%"]),
    ("Grew ARR by 31.7% in EMEA, from 2021 to 2023.", ["by 31.7%"]),
    ("Cut false alerts by 37.2%, or more.", ["by 37.2%"]),
    ("Cut costs by 17.8% for clients, apart from 3 to 4 vendors.", ["by 17.8%"]),
    # a span of years is no base, with a comma after it too
    ("Grew ARR by 31.7% in EMEA, from 2021 to 2023, with 3 new channel partners.", ["by 31.7%"]),
    ("Grew enterprise ARR by 31.7% as Head of Sales, from 2021 to 2023, by adding usage-based pricing.", ["by 31.7%"]),
    ("Grew enterprise ARR by 31.7% from 2021 to 2023, by adding usage-based pricing.", ["by 31.7%"]),
    # a comparison is no base
    ("Lifted trial-to-paid conversion by 12.9%, or 3 times more than the 2022 campaign.", ["by 12.9%"]),
    ("Raised retention by 21% for SMB accounts, or 2 points more than the target.", ["by 21%"]),
    ("Raised retention by 21%, or 4x more than the previous quarter.", ["by 21%"]),
    ("Raised retention by 21%, or 2 more than planned.", ["by 21%"]),
])
def test_a_bare_percentage_is_still_found(text, found):
    assert validate.unbased_percentages(text) == found
    assert validate.base_mismatches(text) == []  # never a false 'does not match its base', which costs a fix round


def test_a_span_of_years_in_a_later_bracket_is_left_alone():
    text = "Grew enterprise ARR by 31.7% (from 2021 to 2023, EMEA) with usage-based pricing."
    assert validate._varied([FIRST, text], "") == [FIRST, text]


def test_a_base_in_words_must_still_agree():
    assert validate.base_mismatches("Delivered a 22% reduction in queue time, from 41 to 38 seconds.") == [
        "from 41 to 38 seconds"]
    assert validate.base_mismatches("Cut queue time by 22%, from 41 to 38 seconds.") == ["from 41 to 38 seconds"]
    assert validate.base_mismatches("Cut API latency by 21%, p95 latency from 820 to 701 ms.") == [
        "p95 latency from 820 to 701 ms"]


@pytest.mark.parametrize("text", [
    "Cut p95 latency from 820 to 648 ms, a 21% drop, p99 from 1,900 to 1,200 ms.",
    ("Cut hosting costs from $24.1K to $19.7K a month, an 18.3% drop after moving to spot instances, from 14 to 6 "
     "clusters."),
    "Cut matchmaking queue time from 41 to 32 seconds, a 22% drop for 83K players, from 2,100 to 2,600 matches a day.",
    "Consolidated Kafka from 14 to 6 clusters, a 22% drop in queue time, from 41 to 32 seconds.",
])
def test_a_second_result_after_a_percentage_is_not_taken_for_its_base(text):
    """A noun or adjective percentage whose pair after it (after a label or a few words) disagrees takes the agreeing
    pair right before it."""
    assert validate.unbased_percentages(text) == validate.base_mismatches(text) == []


def test_a_percentage_that_agrees_with_neither_pair_is_still_reported():
    assert validate.base_mismatches("Cut p95 latency from 820 to 701 ms, a 21% drop, p99 from 1,900 to 1,200 ms.") == [
        "p99 from 1,900 to 1,200 ms"]


def test_the_posts_own_figure_is_the_posts_with_or_without_its_point_zero(profile):
    resume = validate.build_resume(profile, make_output())
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "I was glad to read about your 38% revenue growth year over year.", "I would like to talk."])
    jd = SAMPLE_JD + " Our platform grew revenue 38.0% year over year, and we are hiring to keep up."
    assert validate.unbased_percentage_problems(resume, letter, jd) == []


# ------------------------------------------------------------------ what the model and the person are told
def test_the_prompts_teach_varied_shapes_not_brackets():
    estimate, words, letter = tailor_system("estimate"), tailor_system("hints_only"), cover_system()
    for text in (estimate, words, letter):
        assert not BRACKET_BASE.search(text)
    assert "give only the real numbers" in estimate and "Never give two figures the same shape" in estimate
    assert "put a base in brackets at most once in the whole resume" in estimate and 'never ".0"' in estimate
    for example in ("cut pipeline runtime from 47 to 36 minutes", "saving $41,000 a year in cloud spend",
                    "freeing 12 hours a week for 38 analysts", "cut weekly support tickets by 41.2%, from 97 to 57",
                    "brought p95 latency from 847 to 669ms, a 21% drop",
                    "reduced cloud spend by 17.8%, or $41,000 a year",
                    "83K concurrent players in Java on AWS, cutting queue-to-lobby time from 41 to 32 seconds"):
        assert example in estimate
    assert "about 2 of every 5 bullets" in estimate and "never use numbers ending in 0 or 5" in estimate  # still there
    assert '"by 18%, or $41K a year"' in words
    assert "in brackets at most once in the letter" in letter and "as the real numbers alone" in letter


@pytest.mark.parametrize("mode", ["estimate", "hints_only"])
def test_the_prompts_examples_follow_its_own_figure_rules(mode):
    """A model copies the examples it is shown: none may end in 0 or 5 ('from 85 to 50' would cost a fix round for
    round figures) or have a base that disagrees with its percentage."""
    rule = next(line for line in tailor_system(mode).splitlines() if line.startswith("METRICS:"))
    examples = [example for example in re.findall(r'"([^"]*)"', rule) if re.search(r"\d", example)]
    assert len(examples) >= (12 if mode == "estimate" else 1)
    for example in examples:
        assert [figure for figure in validate._numbers_in(example) if validate.is_round_figure(figure)] == [], example
        assert validate.base_mismatches(example) == [], example


class Rewrites(LLMClient):
    """The demo, with each job's first bullet changed in the first draft (`first`) and in the fix round's answer
    (`fixes`, by employer); every prompt is kept."""

    def __init__(self, inner, first=None, fixes=None, letter=None):
        self.inner, self.first, self.fixes, self.letter, self.prompts = inner, first or {}, fixes or {}, letter, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output":
            for job in result.data["sections"]["experience"]:
                job["bullets"][0] = self.first.get(job["company_index"], job["bullets"][0])
        elif kwargs["schema_name"] == "tailor_revision":
            result.data["bullets"] = [{"company_index": index, "bullet_index": 0, "text": text}
                                      for index, text in self.fixes.items()]
        elif kwargs["schema_name"] == "cover_letter" and self.letter:
            result.data["paragraphs"] = self.letter
        return result


def problems_in(prompt: str) -> str:
    """What the fix round asks to change."""
    return prompt.split(f"{RETRY_MARKER}: ", 1)[1].split("\n\nRevise the previous draft", 1)[0]


BARE = {0: "Rebuilt the Airflow pipeline for 38 internal teams, cutting failed nightly runs by 17.8% with staged "
           "rollouts and clear ownership.",
        1: "Launched self-service reporting for 12.4K active users, so the time to an answer dropped by 21% once shared "
           "dashboards replaced spreadsheets."}
BRACKETS = {0: "Rebuilt the Airflow pipeline for 38 internal teams, cutting failed nightly runs by 17.8% (from 101 to 83 "
               "a month) with staged rollouts.",
            1: "Launched self-service reporting for 12.4K active users, cutting the time to an answer by 21% (from 62 to "
               "49 hours) with shared dashboards.",
            2: "Moved order processing for 613 stores onto a tested setup, cutting missing-order tickets by 12.6% (from "
               "87 to 76 a week) within two quarters."}


def test_the_fix_round_asks_for_bases_in_words_or_the_real_numbers(router, db, registry, profile):
    client = Rewrites(FakeClient(registry.get("demo-concise")), first=BARE)
    router.overrides["demo-concise"] = client
    result = run(router, db, profile)
    assert result.status == "ok" and len(client.prompts) == 2
    problems = problems_in(client.prompts[1])
    assert "These have none: 'by 17.8%' (Northwind Labs); 'by 21%' (Contoso Systems)" in problems
    assert "in words" in problems and "or give the real numbers alone" in problems and "no base in brackets" in problems
    assert not BRACKET_BASE.search(problems)


def test_the_first_bullet_example_and_the_warning_give_no_bracketed_base(router, db, registry, profile):
    client = Rewrites(FakeClient(registry.get("demo-concise")), first={i: NO_FIGURE for i in range(3)})
    router.overrides["demo-concise"] = client
    run(router, db, profile)
    problems = problems_in(client.prompts[1])
    assert "cutting queue-to-lobby time from 41 to 32 seconds" in problems and not BRACKET_BASE.search(problems)
    router.overrides["demo-concise"] = Rewrites(FakeClient(registry.get("demo-concise")), first={0: BARE[0]})
    warning = next(w for w in run(router, db, profile).ats_warnings if w.startswith("Percentages without their base"))
    assert "'by 17.8%, or $41,000 a year'" in warning and "real numbers alone" in warning
    assert not BRACKET_BASE.search(warning)


# ------------------------------------------------------------------ every AI-written draft, end to end
def career(jobs: int) -> Profile:
    return Profile(name="Jordan Rivera", companies=[
        Company(name=f"Employer {i}", start=f"Jan {2023 - 3 * i}", end="Present" if i == 0 else f"Dec {2025 - 3 * i}")
        for i in range(jobs)])


@pytest.mark.parametrize("model,jobs", [("demo-concise", 3), ("demo-detailed", 3), ("demo-concise", 9)])
def test_a_demo_resume_prints_at_most_one_base_in_brackets(router, db, model, jobs):
    result = run(router, db, career(jobs), model=model, cover_letter=True)
    texts = texts_of(result.resume)
    assert result.status == "ok" and bracketed(texts) == 1 and bracketed(result.cover_letter.paragraphs) <= 1
    assert validate.unbased_percentage_problems(result.resume, result.cover_letter) == []
    assert validate.base_mismatch_problems(result.resume, result.cover_letter) == []
    assert validate._varied(texts, f"{result.hints}\n{result.facts}") == texts  # nothing left to change


def test_the_demo_shows_the_mix_itself(router, db, profile):
    leads = [exp.bullets[0] for exp in run(router, db, profile).resume.experience]
    assert "by 17.8% (from 101 to 83 a month)" in leads[0]  # its one base in brackets
    assert "by 21%, from 62 to 49 hours," in leads[1]  # a base in words
    assert "fell from 87 to 76 a week" in leads[2] and "%" not in leads[2]  # the real numbers alone


def test_a_first_draft_with_brackets_everywhere_is_fixed_at_no_cost(router, db, registry, profile):
    client = Rewrites(FakeClient(registry.get("demo-concise")), first=BRACKETS)
    router.overrides["demo-concise"] = client
    result = run(router, db, profile)
    assert result.status == "ok" and len(client.prompts) == 1 and result.retries == 0
    assert [exp.bullets[0] for exp in result.resume.experience] == [
        BRACKETS[0],
        ("Launched self-service reporting for 12.4K active users, cutting the time to an answer from 62 to 49 hours "
         "with shared dashboards."),
        ("Moved order processing for 613 stores onto a tested setup, cutting missing-order tickets by 12.6%, from 87 "
         "to 76 a week, within two quarters.")]


def test_the_fix_rounds_answer_is_varied_too(router, db, registry, profile):
    client = Rewrites(FakeClient(registry.get("demo-concise")), first={**BARE, 2: BRACKETS[2]},
                      fixes={0: BRACKETS[0], 1: BRACKETS[1]})
    router.overrides["demo-concise"] = client
    result = run(router, db, profile)
    assert result.status == "ok" and len(client.prompts) == 2  # the two bare percentages cost the fix round
    assert bracketed(texts_of(result.resume)) == 1 and validate.unbased_percentage_problems(result.resume) == []
    assert result.resume.experience[0].bullets[0] == BRACKETS[0]  # now the first base in brackets


def test_the_shorter_draft_is_varied_too(router, db, registry, profile):
    client = Rewrites(FakeClient(registry.get("demo-detailed")), first=BRACKETS)
    router.overrides["demo-detailed"] = client
    result = run(router, db, profile, model="demo-detailed")
    assert "The model shortened the bullets to fit two pages." in result.notes
    assert bracketed(texts_of(result.resume)) == 1 and validate.unbased_percentage_problems(result.resume) == []


def test_an_ai_written_cover_letter_is_varied(router, db, registry, profile):
    letter = ["At Northwind Labs I cut nightly runtime by 23.4% (from 47 to 36 minutes) by moving jobs to Spark.",
              "At Contoso Systems I cut support tickets by 41.2% (from 85 to 50 a week) with a help center.",
              "I would welcome a conversation about the data platform team."]
    router.overrides["demo-concise"] = Rewrites(FakeClient(registry.get("demo-concise")), letter=letter)
    result = run(router, db, profile, cover_letter=True)
    assert result.cover_letter.paragraphs == [
        letter[0], "At Contoso Systems I cut support tickets from 85 to 50 a week with a help center.", letter[2]]


@pytest.mark.parametrize("hints", ["At Contoso Systems I cut weekly support tickets by 41.2%.",
                                   "At Contoso Systems I cut weekly support tickets by 41,2 %."])
def test_a_run_keeps_the_candidates_own_figure_in_the_resume_and_the_letter(router, db, registry, profile, hints):
    """The hints reach both fixers: a later bracket holding the candidate's own 41.2% gets its base in words, never
    the real numbers alone."""
    bullet = ("Built the self-serve help center for 4.6K merchants, cutting weekly support tickets by 41.2% (from 97 "
              "to 57) within two quarters.")
    letter = ["At Northwind Labs I cut nightly runtime by 23.4% (from 47 to 36 minutes) by moving jobs to Spark.",
              "At Contoso Systems I cut weekly support tickets by 41.2% (from 97 to 57) with a help center.",
              "I would welcome a conversation about the data platform team."]
    router.overrides["demo-concise"] = Rewrites(FakeClient(registry.get("demo-concise")), first={0: FIRST, 1: bullet},
                                                letter=letter)
    result = run(router, db, profile, hints=hints, cover_letter=True)
    assert result.status == "ok" and result.resume.experience[1].bullets[0] == (
        "Built the self-serve help center for 4.6K merchants, cutting weekly support tickets by 41.2%, from 97 to 57, "
        "within two quarters.")
    assert result.cover_letter.paragraphs[1] == (
        "At Contoso Systems I cut weekly support tickets by 41.2%, from 97 to 57, with a help center.")


def test_a_letter_quoting_the_posts_figure_keeps_it_as_written(router, db, registry, profile):
    jd = SAMPLE_JD + " Our platform grew revenue 38.0% year over year, and we are hiring to keep up."
    letter = [("I was glad to read about your 38.0% revenue growth year over year, and I would like to help the data "
               "platform keep up with it."),
              "At Northwind Labs I cut nightly runtime from 47 to 36 minutes by moving jobs to Spark.",
              "I would welcome a conversation about the data platform team."]
    router.overrides["demo-concise"] = Rewrites(FakeClient(registry.get("demo-concise")), letter=letter)
    result = run(router, db, profile, jd=jd, cover_letter=True)
    assert result.status == "ok" and result.cover_letter.paragraphs[0] == letter[0]
    assert not [w for w in result.ats_warnings if w.startswith("Percentages without their base")]


@pytest.mark.parametrize("jobs", [2, 3, 4])
def test_the_demos_detailed_bullets_stay_short_enough(router, db, jobs):
    result = run(router, db, career(jobs), model="demo-detailed")
    assert result.status == "ok" and not [w for w in result.ats_warnings if "words long" in w]


def test_text_the_person_edits_is_kept_as_typed(client):
    save_profile(client)
    run_body = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()
    experience = edit_body(run_body)["experience"]
    experience[1]["bullets"][0] = BRACKETS[1]
    experience[2]["bullets"][0] = BRACKETS[2]
    edited = client.put(f"/api/runs/{run_body['run_id']}/resume", json=edit_body(run_body, experience=experience)).json()
    assert [e["bullets"][0] for e in edited["resume"]["experience"][1:]] == [BRACKETS[1], BRACKETS[2]]
