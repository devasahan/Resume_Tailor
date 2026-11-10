"""The vague-result check, a third review: a false alarm costs the user a paid fix round, so every rule is tested on
the specific side as well as the vague one. One phrase is one problem; an Oxford comma, an abbreviation or a long
bullet does not hide a figure; 'of' + a specific thing is fine; and years, names and versions are no figures."""
import asyncio
import time

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER
from app.schemas import CoverLetter


def problems(text: str) -> list[str]:
    return validate._plain_problems([("b", text)])


def letter(*paragraphs: str) -> CoverLetter:
    return CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=list(paragraphs))


# ------------------------------------------------------------------ one phrase, one problem
@pytest.mark.parametrize("text,problem", [
    (("Drove operational excellence across the data platform by introducing weekly incident reviews and runbooks "
      "for every Airflow DAG."), "Drove operational excellence"),
    ("Improved operational excellence for the finance team by automating bank reconciliations in Python.",
     "Improved operational excellence"),
    ("Drove data-driven decision making across marketing by building a dbt and Looker reporting layer.",
     "Drove data-driven decision making"),
    ("Drove the Operational Excellence program for the payments platform.", "Drove the Operational Excellence"),
    (("Senior Data Engineer with 8 years of Python and SQL pipeline work on Snowflake, driving data-driven "
      "decision-making for finance and marketing teams."), "driving data-driven decision-making"),
    ("Drove operational excellence across 14 services.", "operational excellence"),  # measured: the cliche alone
    ("Brought operational excellence to on-call.", "operational excellence"),
])
def test_one_vague_phrase_is_one_problem(text, problem):
    assert problems(text) == [f"'{problem}' (b)"]


def test_one_cliche_in_a_cover_letter_is_one_note():
    assert validate.cover_plain_problems(letter("I also drove operational excellence across the on-call rota.")) == [
        "'drove operational excellence' (paragraph 1)"]


# ------------------------------------------------------------------ the clause a figure must be in
@pytest.mark.parametrize("text,vague", [
    # an Oxford comma closes a list, not the clause
    ("Improved operational efficiency across finance, procurement, and payroll by 23% (from 52 to 40 hours a week).",
     False),
    (("Improved team productivity by moving sprint planning, standups, and retros into Linear, cutting meeting time "
      "from 6 to 4 hours a week."), False),
    (("Enhanced cross-functional collaboration between product, design, and engineering, cutting handoff time from "
      "5 days to 2."), False),
    (("Boosted team productivity by automating test data setup, code review checks, and release notes, saving 9 "
      "hours a sprint."), False),
    ("Improved performance across the search, checkout, and payments APIs by 25.6% (p95 from 820 to 610 ms).", False),
    ("Improved team productivity, cutting reviews across design, product, and engineering by 40%.", False),
    # ', and' that starts a new clause still ends it
    ("Improved operational efficiency for the finance team, and migrated 14 Airflow DAGs to Dagster.", True),
    ("Boosted team productivity for the finance, sales, and support teams, and migrated 14 DAGs to Dagster.", True),
    ("Drove alignment between sales, product, and finance on pricing, and cut quote time from 5 days to 2.", True),
    ("Drove alignment with sales, then built the pricing model, and cut quote time from 5 days to 2.", True),
    # a list the phrase itself starts: the figure after its last item is that item's
    ("Improved team productivity, release cadence, and on-call health across 3 platform squads", True),
    # a whole bullet is read, not only its first 100 characters
    (("Improved operational efficiency in the claims team by replacing manual Excel exports with scheduled dbt "
      "models and Looker alerts, cutting weekly reporting from 8 hours to 2."), False),
    (("Boosted team productivity on the payments squad by replacing hand-written Terraform with reusable modules "
      "and pre-commit checks in GitHub Actions, cutting new-service setup from 3 days to 4 hours."), False),
    (("Enhanced cross-functional collaboration between the data platform team and product analytics by setting up "
      "shared dbt docs and a weekly metrics review, cutting ad hoc data requests from 40 to 15 a month."), False),
    (("Improved operational efficiency in the claims team by replacing manual Excel exports with scheduled dbt "
      "models and Looker alerts so analysts stopped rebuilding the weekly pack by hand."), True),
    (("Improved operational efficiency in the claims team by replacing manual Excel exports with scheduled dbt "
      "models and Looker alerts, and moved 14 reports to Power BI."), True),
    # an abbreviation's full stop ends no clause when a lowercase word or a figure follows
    ("Improved operational efficiency across U.S. warehouses by 18% (from 50 to 41 seconds a pick).", False),
    ("Drove alignment between U.S. and U.K. support teams, cutting release delays from 9 days to 2.", False),
    ("Improved team productivity, e.g. cutting code review time from 3 days to 1.", False),
    ("Improved system performance vs. the old Java service by 40% (from 500 to 300 ms).", False),
    ("Improved operational efficiency for Contoso Inc. warehouses by 18% (from 50 to 41 seconds a pick).", False),
    ("Improved operational efficiency in the U.S. Cut pick time from 50 to 41 seconds.", True),
    ("Improved team productivity under Plan B. Cut cycle time from 9 days to 4.", True),
])
def test_a_figure_counts_anywhere_in_its_own_clause(text, vague):
    assert bool(problems(text)) is vague


def test_a_cover_letter_sentence_runs_past_an_abbreviation():
    assert validate.cover_plain_problems(letter(
        "At Northwind I improved operational efficiency across U.S. warehouses by 18% (from 50 to 41 seconds a pick).",
        ("In my last role I improved team productivity across design, product, and engineering by 18% (from 11 to 13 "
         "tickets a sprint)."))) == []
    assert validate.cover_plain_problems(letter(
        "I improved operational efficiency in the U.S. Pick time fell from 50 to 41 seconds.")) == [
        "'improved operational efficiency' (paragraph 1)"]


# ------------------------------------------------------------------ 'of' + a generic owner, not a specific thing
@pytest.mark.parametrize("text", [
    "Improved the performance of database operations in the orders service by batching writes.",
    ("Improved the performance of CRUD operations in the orders service by batching writes and adding composite "
     "indexes."),
    "Improved the performance of read and write operations on the orders table by batching inserts.",
    "Improved the performance of the business logic layer by replacing ORM loops with set-based SQL.",
    "Improved the performance of the Kafka consumer groups by tuning fetch sizes and partition counts.",
    "Improved the efficiency of the EKS node groups by moving them to Graviton and Karpenter.",
    "Improved the efficiency of the GitHub Actions runner groups by adding ARM runners for Go builds.",
    "Improved the performance of AWS Lambda functions by moving them to Graviton and trimming cold starts.",
    "Optimized the performance of the AWS Lambda edge functions that resize product images.",
    "Improved the performance of the SQL window functions in the revenue model by pre-aggregating in dbt.",
    "Improved the performance of the Microsoft Teams bot by caching Graph API calls.",
    "Improved the performance of the company's checkout API with Redis caching and connection pooling.",
    "Improved the efficiency of the finance team's month-end close by automating bank reconciliations in Python.",
    "Improved the performance of the team's CI pipeline with remote caching.",
    "Improved the efficiency of Northwind's operations dashboard with materialized views.",
    "Improved the performance of the operations dashboard by moving heavy queries to materialized views.",
    "Improved the efficiency of staff scheduling by moving rotas from spreadsheets into Deputy.",
    "Improved the performance of unit tests by running them in parallel.",
    "Improved the performance of the people search API with Elasticsearch.",
    # end-to-end-2: the owner word does not end the noun phrase
    ("Improved the performance of the company website by compressing images, lazy-loading third-party scripts and "
     "caching API responses at the edge for returning visitors."),
    ("Improved the performance of business-critical SQL reports by adding indexes, materialized views and a nightly "
     "refresh that finance could rely on before stand-up."),
    "Improved the performance of the company's mobile app by deferring analytics calls.",
    "Improved the performance of the payments team's Kafka consumers by batching offset commits.",
    "Improved the performance of our business intelligence dashboards with extracts.",
    "Increased the efficiency of operations reporting with scheduled dbt models.",
])
def test_of_a_specific_thing_is_not_vague(text):
    assert problems(text) == []


@pytest.mark.parametrize("text", [
    "Improved the overall efficiency of our operations.",
    "Boosted the overall performance of the business.",
    "Enhanced the productivity of the engineering organization.",
    "Improved the efficiency of the operations team across three regions.",
    "Improved the efficiency of business operations with weekly reviews.",
    "Improved the efficiency of our day-to-day operations.",
    "Boosted the productivity of the QA team by adding test data fixtures.",
    "Improved the efficiency of the HR function.",
    "Improved the productivity of all staff.",
    "Improved the efficiency of your operations.",
    # three words before the owner when one is an intensifier, and a company's possessive
    "Boosted the productivity of the entire data engineering team.",
    "Boosted the productivity of the whole customer support team.",
    "Enhanced the effectiveness of the wider product engineering organization.",
    "Improved the efficiency of Northwind's operations.",
    "Improved the efficiency of Northwind’s operations.",
    "Improved the efficiency of the client's finance operations.",
])
def test_of_a_generic_owner_is_vague(text):
    assert problems(text)


def test_a_cover_letter_names_the_companys_operations_as_vague():
    assert validate.cover_plain_problems(letter(
        "I am confident I can improve the efficiency of Northwind's operations.")) == [
        "'improve the efficiency' (paragraph 1)"]


@pytest.mark.parametrize("text,vague", [  # a rejected finding: only 'of' names what got better
    ("Improved performance for the checkout API by adding Redis caching.", True),
    ("Improved performance on the search page by lazy-loading product images.", True),
    ("Improved overall performance for the v2 API.", True),
    ("Improved the performance of the checkout API with Redis caching.", False),
    ("Improved overall performance of the v2 API.", False),
])
def test_for_or_on_a_specific_thing_is_still_vague(text, vague):
    assert bool(problems(text)) is vague


# ------------------------------------------------------------------ what counts as a figure
@pytest.mark.parametrize("text,vague", [
    # a figure after sub-, top- or a minus sign; not one joined to a name
    ("Improved system performance by moving session state to Redis, achieving sub-50ms p99 latency.", False),
    ("Improved system performance with sub-100ms p95 responses after moving reads to replicas.", False),
    ("Improved overall performance (-38% p95 latency) by moving reads to replicas.", False),
    ("Improved overall performance (−38% p95 latency) by moving reads to replicas.", False),
    ("Drove engagement to a top-10 App Store ranking in the Finance category.", False),
    ("Improved team productivity with GPT-4 coding assistants.", True),
    ("Improved operational efficiency during the COVID-19 response.", True),
    ("Improved operational efficiency for the ISO-27001 audit.", True),
    ("Improved team productivity for tier-1 support.", True),
    # a count is not a year
    ("Boosted overall performance to 2047 requests per second by moving the gateway to async I/O.", False),
    ("Improved system performance for 1938 concurrent users by adding read replicas.", False),
    ("Improved team productivity across 2017 support tickets a month by adding Zendesk macros.", False),
    ("Improved system performance for 2000+ concurrent users.", False),
    ("Improved team productivity in a 1900-person support org.", False),
    ("Improved system performance, cutting latency to 1947 ms.", False),
    ("Improved system performance for 2048 concurrent connections by pooling.", False),
    ("Drove alignment on the 2024 roadmap with sales and finance.", True),
    ("Improved operational efficiency in 2024.", True),
    ("Improved operational efficiency during the 2021 migration.", True),
    ("Improved operational efficiency from 2021 to 2023.", True),
    ("Improved operational efficiency in the 2010s.", True),
    ("Improved operational efficiency for the 2024/25 financial year.", True),
    ("Improved operational efficiency for the 2024/2025 budget cycle.", True),
    ("Maximized business outcomes for enterprise accounts (Q3 2024 expansion push)", True),
    # a version, a standard or a product's name is not a figure, unless a measure follows
    ("Boosted team productivity by rolling out Microsoft 365 Copilot to the finance group.", True),
    ("Improved team productivity by upgrading the monorepo to Java 17.", True),
    ("Drove alignment on SOC 2 controls with security and legal.", True),
    ("Drove alignment with Fortune 500 clients on data contracts.", True),
    ("Improved operational efficiency for 5G network rollouts with Ansible playbooks.", True),
    ("Enhanced cross-functional collaboration on the 2FA rollout with security and support.", True),
    ("Boosted team engagement through weekly 1:1s and career check-ins.", True),
    ("Improved operational efficiency for 24/7 support by adding PagerDuty escalation rules.", True),
    ("Improved operational efficiency for Tier 2 support with new runbooks.", True),
    ("Improved system performance by moving the API to Node 20 and Postgres 16.", True),
    ("Improved operational efficiency by automating ISO 27001 evidence collection.", True),
    ("Improved team productivity by moving the codebase to Python 3.12 and uv.", True),
    ("Improved system performance 3x by adding Redis caching.", False),
    ("Boosted team productivity 2x with shared CI templates.", False),
    ("Improved overall performance to 40ms p95 with Redis.", False),
    ("Improved team productivity for 12K support agents with Zendesk macros.", False),
    ("Improved system performance for 2M daily users.", False),
    ("Improved system performance with 16G heaps on the Kafka brokers.", False),
    ("Improved team productivity, saving Finance 120 hours a quarter.", False),
    ("Improved operational efficiency, giving Support 15 hours back each week.", False),
    ("Drove alignment with 14 Fortune 500 clients on data contracts.", False),
    ("Boosted team productivity by rolling out Microsoft 365 Copilot to 1,200 staff.", False),
    ("Boosted overall engagement, reaching NPS 52 after the onboarding redesign.", False),
    ("Improved team engagement with CSAT 4.7 across support channels.", False),
    ("Drove engagement into the Top 10 of the App Store finance chart.", False),
])
def test_only_a_real_figure_makes_a_vague_result_specific(text, vague):
    assert bool(problems(text)) is vague


# ------------------------------------------------------------------ technical terms and literal results
@pytest.mark.parametrize("text,vague", [
    ("Improved performance alerting for the checkout service by adding SLO burn-rate alerts in Datadog.", False),
    ("Improved performance observability by adding OpenTelemetry traces to the order flow.", False),
    ("Enhanced collaboration tools for remote designers by building a Figma-to-Jira sync.", False),
    ("Enhanced collaboration features in the editor by adding live cursors and threaded comments.", False),
    ("Improved engagement tracking by instrumenting Segment events in the mobile app.", False),
    ("Drove performance management and promotions for the data team.", False),
    ("Improved performance reporting for the sales team with Looker.", False),
    ("Enhanced performance attribution models for the fixed income desk.", False),
    ("Improved engagement analytics in Amplitude.", False),
    ("Drove efficiency gains across the operations organization.", True),
    ("Drove performance improvements across the platform.", True),
    ("Enhanced performance, monitoring and alerting for the payments API.", True),
    # 'value based' with or without the hyphen, but not 'based on'
    ("Drove value based care adoption across clinics.", False),
    ("Drove value based pricing for the SaaS tier.", False),
    ("Added value based pricing tiers for enterprise customers.", False),
    ("Drove alignment based on shared OKRs with sales.", True),
    ("Improved efficiency based on the audit findings.", True),
    ("Improved performance based upon customer feedback.", True),
    ("Boosted engagement based around community events.", True),
    ("Added value based on customer feedback.", True),
    # results a system returns, with their speed, are not the 'delivered results' cliche
    ("Rebuilt the product search API in Go so it delivers results in under 120 ms at p95 for 2M daily queries.",
     False),
    ("Tuned Elasticsearch so the catalog search delivers results in 90 ms at p95.", False),
    ("Built a Redis cache that delivers results to the pricing dashboard in 1.2 seconds instead of 14.", False),
    ("Built a streaming job that delivers results in real time to the fraud team.", False),
    ("Delivered results within 90 days of joining the platform team.", True),
    ("Delivered results for 3 enterprise clients in the first quarter.", True),
    ("Delivered measurable results in 30 minutes.", True),
    ("Delivered strong results, growing revenue 12.6% year on year.", True),
    ("Delivered results in the 2020s.", True),
])
def test_technical_terms_and_literal_results_are_not_vague(text, vague):
    assert bool(problems(text)) is vague


# ------------------------------------------------------------------ end to end
class Injects(LLMClient):
    """Puts one bullet into each job's first draft; the fix round's answer (the demo's own text) comes back as is."""

    def __init__(self, inner, bullets):
        self.inner, self.bullets, self.prompts = inner, bullets, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            for job, bullet in zip(result.data["sections"]["experience"], self.bullets):
                job["bullets"][1] = bullet
        return result


def tailor_with(router, db, registry, profile, jd, bullets):
    client = Injects(FakeClient(registry.get("demo-concise")), bullets)
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok"
    return client, result


@pytest.mark.parametrize("bullets,calls", [
    ([("Improved the performance of the company website by compressing images, lazy-loading third-party scripts "
       "and caching API responses at the edge for returning visitors."),
      ("Improved the performance of business-critical SQL reports by adding indexes, materialized views and a "
       "nightly refresh that finance could rely on before stand-up.")], 1),
    (["Improved the efficiency of our day-to-day operations by moving weekly reports into shared dashboards.",
      "Boosted the productivity of the entire data engineering team by moving weekly reports into dashboards."], 2),
])
def test_only_vague_owners_cost_the_fix_round(router, db, registry, profile, jd, bullets, calls):
    client, result = tailor_with(router, db, registry, profile, jd, bullets)
    assert len(client.prompts) == calls
    assert not any("machine-written" in warning for warning in result.ats_warnings)


def test_one_cliche_is_only_noted(router, db, registry, profile, jd):
    client, result = tailor_with(router, db, registry, profile, jd, [
        ("Drove operational excellence across the data platform by introducing weekly incident reviews and runbooks "
         "for every Airflow DAG.")])
    assert len(client.prompts) == 1  # one phrase, one problem: below the fix round's two
    assert any("'Drove operational excellence' (Northwind Labs)" in w and "'operational excellence'" not in w
               for w in result.ats_warnings)


def test_odd_text_never_slows_the_vague_check():
    started = time.perf_counter()
    validate._plain_problems([("b", "Improved team productivity across " + ", x, and y" * 20000)])
    validate._plain_problems([("b", "Improved team productivity" + ", x y z, and w" * 20000)])
    validate._plain_problems([("b", "Drove operational excellence. " * 5000)])
    validate._plain_problems([("b", "Improved the efficiency of " + "Northwind's " * 20000 + "operations")])
    validate.cover_plain_problems(letter("U.S. a " * 20000, "e.g. 1 " * 20000))
    assert time.perf_counter() - started < 2.0
