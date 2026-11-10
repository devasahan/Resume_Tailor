"""Percentages and their bases, and which numbers are figures: cases from an independent review of the checks."""
import pytest

from app.pipeline import validate
from app.schemas import CoverLetter

from .test_validate import make_output

CASES = [  # (case, text, the percentages without a base, the bases that disagree)
 ("1a", "Maintained 99.9% uptime during growth from 1.2M to 3.4M monthly users by moving checkout to Kubernetes.", [], []),
 ("1b", "Kept 99.95% uptime during lift-and-shift of 40 services to AWS.", [], []),
 ("1c", "Held 99.95% uptime through fall peak season for 4.6K merchants.", [], []),
 ("2a", "Cut costs while maintaining 99.9% uptime across 14 regions.", [], []),
 ("2b", "Increased throughput while maintaining 99.2% accuracy on 4.1M daily events.", [], []),
 ("2c", "Grew users while holding 99.9% uptime through Black Friday.", [], []),
 ("2d", "Cut p95 latency within 99.9% SLA targets by adding Redis caching.", [], []),
 ("2e", "Cut costs yet kept 99.9% uptime by moving batch jobs to Spot.", [], []),
 ("2f", "Lowered costs for top 5% accounts with dedicated support.", [], []),
 ("3a", "Grew enterprise ARR 24.3% year-over-year to $12.4M by shipping usage-based pricing for 140 accounts.", [], []),
 ("3b", "Grew revenue 31.7% YoY, from $3.2M to $4.2M, by launching annual plans.", [], []),
 ("3c", "Grew MAU 4.3% month over month to 1.2M users through referral loops.", [], []),
 ("3d", "Increased bookings 12.6% QoQ to $8.4M by reworking partner pricing.", [], []),
 ("3e", "Cut AWS costs 17.8% annually, saving $41,300.", [], []),
 ("4a", "Grew monthly active users by 22.4% (from 980,000 to 1.2M).", [], []),
 ("4b", "Cut AWS spend by 17.8% (from $1.2M to $986,400 a year).", [], []),
 ("4c", "Grew ARR by 22.4% (from $850,000 to $1.04M) through usage-based pricing.", [], []),
 ("4d", "Cut monthly alert volume by 18.3% (from 12K to 9,800) with deduplication rules.", [], []),
 ("4e", "Cut AWS spend by 17.8% (from $1.2 million to $986,400) by rightsizing EC2.", [], []),
 ("5a", "Consolidated Kafka clusters from 14 to 6, a 17.8% cost reduction, by moving 38 topics to tiered storage.", ["17.8% cost reduction"], []),
 ("5b", "Grew the platform team from 4 to 11 engineers, with a 17.8% cost reduction.", ["17.8% cost reduction"], []),
 ("9a", "Reduced manual QA effort by 37.5% (12 hours a week) with Playwright test suites.", [], []),
 ("9b", "Reduced false positives by 37% (1,200 fewer alerts a month) with tuned thresholds.", [], []),
 ("9c", "Cut on-call pages by 41.7% (60 fewer a month) with alert deduplication in PagerDuty.", [], []),
 ("9d", "Cut manual reporting time by 37.5%, saving 12 hours a week for the finance team.", [], []),
 ("10a", "Saved $41,300 a year (EC2 and RDS rightsizing), a 17.8% cut in AWS spend.", [], []),
 ("10b", "Saved $41,300 a year on AWS (EC2 and RDS), a 17.8% reduction.", [], []),
 ("11a", "Lifted daily orders by 4.6% (from 1,950 to 2,040) after the checkout redesign.", [], []),
 ("11b", "Cut weekly support tickets by 2.5% (from 2,000 to 1,950) with a self-serve help center.", [], []),
 ("12a", "Cut checkout latency by 21%, protecting $2.3M in annual revenue.", ["by 21%"], []),
 ("12b", "Cut p95 latency by 21%, which helped close $1.2M in enterprise deals.", ["by 21%"], []),
 ("12c", "Reduced fraud false positives by 34%, recovering $1.2M in blocked sales.", ["by 34%"], []),
 ("13a", "Kept churn flat while pushing activation up 12.4% with onboarding emails.", ["up 12.4%"], []),
 ("13b", "Kept costs steady and drove signups up 18.2% through referral credits.", ["up 18.2%"], []),
 ("14a", "Cut queue time by 22%, down from 41 to 32 seconds, for 83K players.", [], []),
 ("14b", "Cut API latency by 21% (p95: 820 ms to 648 ms) with Redis caching.", [], []),
 ("14c", "Cut p95 latency by 21%, down to 648 ms, with Redis caching.", [], []),
 ("15a", "Sped up nightly builds by 140% (from 12 to 5 minutes) with remote caching.", [], []),
 ("15b", "Accelerated report generation by 140% (from 12 to 5 minutes).", [], []),
 # must stay as they were
 ("keep1", "Cut latency by 21% and saved $41,300.", ["by 21%"], []),
 ("keep2", "Cut AWS spend by 17.8%, saving the company $41,300 a year.", [], []),
 ("keep3", "Cut queue time by 22% (from 41 to 38 seconds).", [], ["from 41 to 38 seconds"]),
 ("keep4", "Made report generation 50% faster (from 900ms to 600ms) by batching.", [], []),
 ("keep5", "Rightsized EC2 fleets, delivering a 17.8% cost reduction.", ["17.8% cost reduction"], []),
 ("keep6", "Raised trial-to-paid conversion from 3.1% to 3.5%, a 12.9% lift.", [], []),
 ("keep7", "Cut p95 latency from 820 to 648 ms, a 21% drop.", [], []),
 ("keep8", "Kept checkout up 99.95% through Black Friday traffic.", [], []),
]


@pytest.mark.parametrize("case,text,unbased,mismatched", CASES, ids=[case[0] for case in CASES])
def test_each_percentage_is_judged_by_its_own_base(case, text, unbased, mismatched):
    assert validate.unbased_percentages(text) == unbased
    assert validate.base_mismatches(text) == mismatched


def test_names_and_versions_are_not_figures_but_counts_after_a_metric_are():
    text = ("Lifted CSAT 9 points by routing 3,400 weekly tickets. Saved Finance 20 hours a week. Raised NPS 12 points. "
            "Migrated Postgres 40 TB of order history. Ran Microsoft 365 for Fortune 500 retailers on Postgres 15 and "
            "Node.js 20, iOS 17, Python 3.12, GPT-4 and C++17. Built 40 pipelines.")
    assert validate._numbers_in(text) == ["9", "3,400", "20", "12", "40", "40"]
    assert not validate.has_figure("Built a support assistant on GPT-4 that drafts refund replies.")
    assert not validate.has_figure("Migrated checkout from Node.js 16 to Node.js 20.")


def test_a_letter_may_quote_the_companys_own_growth_but_not_hide_a_claim_behind_a_benefit(profile):
    jd = ("Northbeam has grown 40% annually since 2021, and we have grown over 200% in the past two years. Our customer "
          "base expanded 85% in 2024. Up to 20% travel, a 4% 401(k) match.")
    assert validate.posting_figures(jd) == frozenset({"40%", "200%", "85%"})
    letter = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
        "I was drawn to Northbeam's 40% annual growth, and your 85% customer growth.",
        "At Northwind Labs I cut nightly ETL runtime by 20% by moving the largest jobs to Spark.",
    ])
    resume = validate.build_resume(profile, make_output())
    assert validate.unbased_percentage_problems(resume, letter, jd) == ["'by 20%' (cover letter paragraph 2)"]
