"""The dash rules in detail: which dashes are punctuation (a comma) and which join a range, after a third review
(now/today with more words, 'from A to B' with a range as B, decimals, values the number reader cannot read, list
items, doubled dashes, counts that look like years, more date forms, abbreviations, and a minus sign that starts a
bullet)."""
import time

import pytest

from app.pipeline import validate
from app.schemas import CoverLetter

from .test_validate import make_output


def cleaned(given: str) -> str:
    """The model's text after the dash fix; cleaning it again changes nothing."""
    result = validate.clean_prose(given)
    assert validate.clean_prose(result) == result
    return result


# ------------------------------------------------------------------ now, today, current
@pytest.mark.parametrize("given,expected", [
    ("I started at Northwind as a junior analyst in 2019 — today, I lead a team of six data engineers.",
     "I started at Northwind as a junior analyst in 2019, today, I lead a team of six data engineers."),
    ("I joined Northwind in 2019 — now, as a staff engineer, I own its data platform.",
     "I joined Northwind in 2019, now, as a staff engineer, I own its data platform."),
    ("I own the platform since 2019 – now; the team has grown to 12.",
     "I own the platform since 2019, now; the team has grown to 12."),
    # a range when the clause ends with the word: nothing after it, a closing bracket or a full stop
    ("At Northwind (2019 – today), I rebuilt the pipeline.", "At Northwind (2019-today), I rebuilt the pipeline."),
    ("Ran the data platform 2021 – now.", "Ran the data platform 2021-now."),
    ("Ran the data platform (Mar 2022 – current).", "Ran the data platform (Mar 2022 to current)."),
    ("Ran the data platform 2021 - now", "Ran the data platform 2021-now"),
])
def test_now_or_today_before_more_words_is_no_range_end(given, expected):
    assert cleaned(given) == expected


def test_a_typed_dash_before_today_and_more_words_gets_a_note(profile):
    out = make_output()
    out.sections.experience[0].bullets = [
        "I started at Northwind as a junior analyst in 2019 - today, I lead a team of six data engineers.",
        "Ran the data platform (2021 - now) for 40 analysts across finance and sales.",
    ]
    resume = validate.build_resume(profile, out, from_model=False)
    assert validate.dash_problems(resume) == ["'analyst in 2019 - today, I lead' (Northwind Labs)"]


# ------------------------------------------------------------------ from A to B
@pytest.mark.parametrize("given,expected", [
    # B is a bare number whose unit comes after the dash: the range test decides
    ("Cut MTTR from 4 hours to 30 – 45 minutes with better runbooks.",
     "Cut MTTR from 4 hours to 30-45 minutes with better runbooks."),
    ("Moved deploys from weekly to 2 – 3 times a day with GitHub Actions.",
     "Moved deploys from weekly to 2-3 times a day with GitHub Actions."),
    ("Reduced onboarding from 3 weeks to 4 - 5 days with a self-serve sandbox.",
     "Reduced onboarding from 3 weeks to 4-5 days with a self-serve sandbox."),
    ("Cut build time from 40 to 8 – 10 minutes.", "Cut build time from 40 to 8-10 minutes."),
    # B with a currency, a unit or a year ends the pair; so does a C that is the change from A to B, or a year
    ("Brought monthly AWS spend from $18,600 to $12,400 – $74,400 saved a year.",
     "Brought monthly AWS spend from $18,600 to $12,400, $74,400 saved a year."),
    ("Cut report pages from 120 to 45 – 75 fewer to read.", "Cut report pages from 120 to 45, 75 fewer to read."),
    ("Grew the on-call rota from 10 to 13 – 30% more cover.", "Grew the on-call rota from 10 to 13, 30% more cover."),
    ("Cut failed nightly runs from 101 to 83 – 18 fewer each month.",
     "Cut failed nightly runs from 101 to 83, 18 fewer each month."),
    ("Moved the warehouse from 2019 to 2021 – 2022 brought the dbt rewrite.",
     "Moved the warehouse from 2019 to 2021, 2022 brought the dbt rewrite."),
    ("Cut failed runs from 120 to 45 – 2022 brought more.", "Cut failed runs from 120 to 45, 2022 brought more."),
])
def test_a_range_can_be_the_end_of_from_a_to_b(given, expected):
    assert cleaned(given) == expected


# ------------------------------------------------------------------ decimals
@pytest.mark.parametrize("given,expected", [
    ("Saved analysts 2 – 2.5 hours a day by automating the reconciliation.",
     "Saved analysts 2-2.5 hours a day by automating the reconciliation."),
    ("Cut page load time to 1.5 – 2 seconds on mobile.", "Cut page load time to 1.5-2 seconds on mobile."),
    ("Kept model AUC at 0.85 – 0.9 across all regions.", "Kept model AUC at 0.85-0.9 across all regions."),
    ("Upgraded React 16.8 – 18 across 40 apps.", "Upgraded React 16.8-18 across 40 apps."),
    ("Grew revenue 2.5 – 3.5% a quarter for three product lines.",
     "Grew revenue 2.5-3.5% a quarter for three product lines."),
    ("Raised 1.5 – 2M in seed funding.", "Raised 1.5-2M in seed funding."),
    # versions go up part by part
    ("Supported Python 3.8 – 3.12 in CI.", "Supported Python 3.8-3.12 in CI."),
    ("Supported Go 1.9 – 1.10 in CI.", "Supported Go 1.9-1.10 in CI."),
    # going down is punctuation
    ("Priced it at $1.5 – $1.25 was the floor.", "Priced it at $1.5, $1.25 was the floor."),
    ("Cut p95 latency 820ms – 510ms by caching.", "Cut p95 latency 820ms, 510ms by caching."),
])
def test_decimal_ranges_compare_by_value(given, expected):
    assert cleaned(given) == expected


# ------------------------------------------------------------------ values of the same shape, two-word labels
@pytest.mark.parametrize("given,expected", [
    ("Upgraded 12 EKS clusters v1.24 – v1.28 with zero downtime.",
     "Upgraded 12 EKS clusters v1.24-v1.28 with zero downtime."),
    ("Billed contractors at $80/hr – $120/hr for the migration.",
     "Billed contractors at $80/hr-$120/hr for the migration."),
    ("Covered on-call from 9:00 – 17:00 CET.", "Covered on-call from 9:00-17:00 CET."),
    ("Ranked #1 – #3 in the regional sales league.", "Ranked #1-#3 in the regional sales league."),
    ("Owned Sev-1 – Sev-3 incidents.", "Owned Sev-1-Sev-3 incidents."),
    ("Led the team at Kestrel (11/2019 – 02/2021).", "Led the team at Kestrel (11/2019-02/2021)."),
    ("Managed budgets of EUR 2M – EUR 5M for the data platform.",
     "Managed budgets of EUR 2M to EUR 5M for the data platform."),
    ("Made the FY23 Q1 – FY23 Q3 forecast with finance.", "Made the FY23 Q1 to FY23 Q3 forecast with finance."),
    ("Covered the 7am – 7pm help desk shift for 1.2K staff.", "Covered the 7am-7pm help desk shift for 1.2K staff."),
    # going down, or not alike, is punctuation
    ("Ranked #3 – #1 by the end of Q4.", "Ranked #3, #1 by the end of Q4."),
    ("Moved stand-up from 10:00 – 9:30 was earlier.", "Moved stand-up from 10:00, 9:30 was earlier."),
    ("Retired v2.1 – v1.9 was kept for one client.", "Retired v2.1, v1.9 was kept for one client."),
    ("Cut budgets from EUR 5M – EUR 2M in one year.", "Cut budgets from EUR 5M, EUR 2M in one year."),
    ("Covered the 9pm – 6am shift.", "Covered the 9pm, 6am shift."),
    ("Sold B2B – B2C pricing.", "Sold B2B, B2C pricing."),
])
def test_like_values_the_number_reader_cannot_read_still_make_a_range(given, expected):
    assert cleaned(given) == expected


def test_typed_ranges_of_any_shape_get_no_dash_note(profile):
    out = make_output()
    out.sections.experience[0].bullets = [
        "Cut MTTR from 4 hours to 30 - 45 minutes with better runbooks.",
        "Saved analysts 2 - 2.5 hours a day by automating the reconciliation.",
        "Upgraded 12 EKS clusters v1.24 - v1.28 with zero downtime.",
        "Managed budgets of EUR 2M - EUR 5M for the data platform.",
        "Scaled the API to 2000 - 5000 requests per second.",
        "At Northwind (Jan. 2021 - Mar. 2024), I rebuilt the pipeline.",
        "Covered the 7am - 7pm help desk shift for 1.2K staff across 4 offices.",
    ]
    resume = validate.build_resume(profile, out, from_model=False)
    assert resume.experience[0].bullets == out.sections.experience[0].bullets  # the user's own words
    assert validate.dash_problems(resume) == []


# ------------------------------------------------------------------ list items, doubled dashes
@pytest.mark.parametrize("given,expected", [
    ("What I'd bring in my first 90 days:\n- Cut AWS spend by 18%\n- 40% faster deploys with GitHub Actions",
     "What I'd bring in my first 90 days: Cut AWS spend by 18%, 40% faster deploys with GitHub Actions"),
    ("What I'd bring:\n- Cut AWS spend by 18%\n- 40% faster", "What I'd bring: Cut AWS spend by 18%, 40% faster"),
    ("My path so far: joined Acme in 2019\n- 2021: led the platform team",
     "My path so far: joined Acme in 2019, 2021: led the platform team"),
    ("Highlights:\n  – Managed a team of 8\n  – 15 engineers hired in one year",
     "Highlights: Managed a team of 8, 15 engineers hired in one year"),
    ("I'd bring three things:\n- Airflow DAGs that run on time\n- dbt models finance trusts",
     "I'd bring three things: Airflow DAGs that run on time, dbt models finance trusts"),
    ("Targets:\r\n- 2019\r\n- 2021", "Targets: 2019, 2021"),
    # a dash after another one is the same dash
    ("I'd bring three things to Northbeam —\n- Airflow DAGs that run on time\n- dbt models finance trusts",
     "I'd bring three things to Northbeam, Airflow DAGs that run on time, dbt models finance trusts"),
    ("I'd bring this –\n- Airflow DAGs", "I'd bring this, Airflow DAGs"),
    ("Led the team — - cut costs by 18%.", "Led the team, cut costs by 18%."),
    ("Led the team — -", "Led the team"),
    # not list items: a figure or a flag at the start of a line, and a range wrapped onto the next line
    ("Churn moved\n-12% in Q1.", "Churn moved -12% in Q1."),
    ("Ran terraform plan\n--refresh-only first.", "Ran terraform plan --refresh-only first."),
    ("Product Analyst, Kestrel Bank, Jan 2020\n– Mar 2023", "Product Analyst, Kestrel Bank, Jan 2020 to Mar 2023"),
    ("Kept the dash —\n-12% churn", "Kept the dash, -12% churn"),
])
def test_a_dash_that_starts_a_list_item_is_punctuation(given, expected):
    assert cleaned(given) == expected


def test_a_list_in_a_letter_paragraph_loses_its_dashes_and_gets_no_note(profile):
    _greeting, paragraphs, _closing = validate.clean_cover_letter("Dear Hiring Manager,", [
        "I'd bring three things to Northbeam —\n- Airflow DAGs that run on time\n- dbt models finance trusts",
        "What I'd bring in my first 90 days:\n- Cut AWS spend by 18%\n- 40% faster deploys with GitHub Actions",
    ], "Sincerely,")
    assert paragraphs == [
        "I'd bring three things to Northbeam, Airflow DAGs that run on time, dbt models finance trusts",
        "What I'd bring in my first 90 days: Cut AWS spend by 18%, 40% faster deploys with GitHub Actions",
    ]
    letter = CoverLetter(greeting="Dear Hiring Manager,", paragraphs=paragraphs, closing="Sincerely,")
    assert validate.dash_problems(validate.build_resume(profile, make_output()), letter) == []


def test_many_list_items_and_line_breaks_take_linear_time():
    started = time.perf_counter()
    assert validate.fix_dashes("Wins:" + "\n- cut AWS spend by 18%" * 20000).count(",") == 19999
    assert validate.fix_dashes("a" + "\u2028" * 50000 + "- b") == "a, b"
    assert validate.fix_dashes("a\n" + " " * 50000 + "x - y") == "a x, y"
    assert time.perf_counter() - started < 1.5


# ------------------------------------------------------------------ counts that look like years
@pytest.mark.parametrize("given,expected", [
    ("Scaled the API to 2000 – 5000 requests per second.", "Scaled the API to 2000-5000 requests per second."),
    ("Served 1900 – 2400 concurrent users during launches.", "Served 1900-2400 concurrent users during launches."),
    ("Handled 2000 - 3000 tickets a month.", "Handled 2000-3000 tickets a month."),
    # a year a word puts in time, or one that is not round, is a year: the dash is punctuation
    ("Launched in 2021 – 12,000 users signed up in the first month.",
     "Launched in 2021, 12,000 users signed up in the first month."),
    ("Went live in 2000 – 5000 users signed up.", "Went live in 2000, 5000 users signed up."),
    ("By the end of 2023 – 4500 merchants were live.", "By the end of 2023, 4500 merchants were live."),
    ("Launched 2021 – 3000 users in month one.", "Launched 2021, 3000 users in month one."),
    ("Rolled out in Q4 2021 – 3000 signups.", "Rolled out in Q4 2021, 3000 signups."),
])
def test_a_count_that_looks_like_a_year_can_start_a_range(given, expected):
    assert cleaned(given) == expected


# ------------------------------------------------------------------ more date forms
@pytest.mark.parametrize("given,expected", [
    ("At Northwind (Jan. 2021 – Mar. 2024), I rebuilt the pipeline.",
     "At Northwind (Jan. 2021 to Mar. 2024), I rebuilt the pipeline."),
    ("Led support (Sept. 2019 – present).", "Led support (Sept. 2019 to present)."),
    ("Owned the H1 2023 – H2 2024 data roadmap with finance.",
     "Owned the H1 2023 to H2 2024 data roadmap with finance."),
    ("Led the pilot from Spring 2023 – Fall 2024 with 3 hospitals.",
     "Led the pilot from Spring 2023 to Fall 2024 with 3 hospitals."),
    ("Taught statistics (Fall 2022 – Spring 2024).", "Taught statistics (Fall 2022 to Spring 2024)."),
    ("Ran planning from H1 2023 – Mar 2024.", "Ran planning from H1 2023 to Mar 2024."),
    # a full stop that ends a sentence still ends it; a framework is no season
    ("Shipped it in May. 2024 – a quiet year.", "Shipped it in May. 2024, a quiet year."),
    ("Built the APIs in Spring Boot 3 – the core of the platform.",
     "Built the APIs in Spring Boot 3, the core of the platform."),
    ("Ran Fall 2024 – a hard quarter – on half the budget.", "Ran Fall 2024, a hard quarter, on half the budget."),
])
def test_abbreviated_months_halves_and_seasons_are_dates(given, expected):
    assert cleaned(given) == expected


# ------------------------------------------------------------------ abbreviations
@pytest.mark.parametrize("given,expected", [
    ("I joined Contoso Ltd. — a 40-person fintech — as its first data engineer.",
     "I joined Contoso Ltd., a 40-person fintech, as its first data engineer."),
    ("At Fabrikam Inc. — Microsoft's Azure partner — I rebuilt billing.",
     "At Fabrikam Inc., Microsoft's Azure partner, I rebuilt billing."),
    ("Unified logs, metrics, traces, etc. — all in one pipeline.",
     "Unified logs, metrics, traces, etc., all in one pipeline."),
    ("Grew sales in the U.S. — our largest market — by 18%.", "Grew sales in the U.S., our largest market, by 18%."),
    ("Joined Fabrikam Mfg. — a 40-person shop — as its first engineer.",
     "Joined Fabrikam Mfg., a 40-person shop, as its first engineer."),
    # the full stop of a sentence
    ("Shipped on time. — Then cut costs.", "Shipped on time. Then cut costs."),
    ("We moved to Snowflake in 2021. — 2022 brought the dbt rewrite.",
     "We moved to Snowflake in 2021. 2022 brought the dbt rewrite."),
    ("Rebuilt the app in Swift. — iOS installs rose 20%.", "Rebuilt the app in Swift. iOS installs rose 20%."),
])
def test_a_dash_after_an_abbreviation_is_still_a_comma(given, expected):
    assert cleaned(given) == expected


# ------------------------------------------------------------------ a bullet that starts with a figure; the ± sign
@pytest.mark.parametrize("from_model", [True, False])
def test_a_bullet_keeps_the_minus_of_a_figure_it_starts_with(profile, from_model):
    out = make_output()
    out.sections.experience[0].bullets = [
        "-12% churn after the onboarding redesign (from 5.0% to 4.4% a month).",
        "-$1.2M in annual cloud spend after rightsizing 140 EC2 instances.",
        "–12% churn after the onboarding redesign.",
        "- -12% churn after the onboarding redesign.",
        "- Cut churn by 12% with a new onboarding flow.",
        "-Cut churn by 12% with a new onboarding flow.",
        "• - Cut churn by 12% with a new onboarding flow.",
    ]
    bullets = validate.build_resume(profile, out, from_model=from_model).experience[0].bullets
    assert bullets == [
        "-12% churn after the onboarding redesign (from 5.0% to 4.4% a month).",
        "-$1.2M in annual cloud spend after rightsizing 140 EC2 instances.",
        "-12% churn after the onboarding redesign.",
        "-12% churn after the onboarding redesign.",
        "Cut churn by 12% with a new onboarding flow.",
        "Cut churn by 12% with a new onboarding flow.",
        "Cut churn by 12% with a new onboarding flow.",
    ]
    assert validate.clean_bullet("-12% churn") == "-12% churn" and validate.clean_bullet("• Led it") == "Led it"


def test_the_plus_minus_sign_survives_cleaning(profile):
    text = "Kept the monthly cash forecast within ±2% of actuals by reconciling 140 accounts every week."
    assert validate.clean_text(text) == text and cleaned(text) == text
    out = make_output()
    out.sections.experience[0].bullets = [text]
    assert validate.build_resume(profile, out).experience[0].bullets == [text]
