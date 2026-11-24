"""Each bullet in its job's tense (a reviewer's rule, app/prompts/system.md): the present for the current job ("Build",
"Lead"), the past for every other ("Built", "Led"). The prompt asks for it, and an opening the model writes in the
wrong tense is put right by the app (app/pipeline/tense.py), on every draft, for free."""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import run_tailor
from app.pipeline.tense import (
    base_of,
    past,
    past_opening,
    present_base,
    present_opening,
)
from app.prompts import RETRY_MARKER, tailor_system
from app.schemas import Company, Profile

from .test_validate import make_output


@pytest.mark.parametrize("given,expected", [
    ("Lead the claims API migration to Kubernetes, cutting deploy time from 40 to 12 minutes.",
     "Led the claims API migration to Kubernetes, cutting deploy time from 40 to 12 minutes."),
    ("Design and run the Go order-sync service behind the public REST API.",
     "Designed and ran the Go order-sync service behind the public REST API."),
    ("Design, build and operate the billing pipeline on AWS so finance closed books faster.",
     "Designed, built and operated the billing pipeline on AWS so finance closed books faster."),
    ("Leads a team of five engineers building the onboarding flow in React.",
     "Led a team of five engineers building the onboarding flow in React."),
    ("Optimize slow database queries in PostgreSQL with Redis caching, so the API handled spikes without errors.",
     "Optimized slow database queries in PostgreSQL with Redis caching, so the API handled spikes without errors."),
    ("Co-lead the platform guild that set Terraform standards for every service.",
     "Co-led the platform guild that set Terraform standards for every service."),
    ("Plan and ship weekly releases of the mobile app with Fastlane, so customers got fixes sooner.",
     "Planned and shipped weekly releases of the mobile app with Fastlane, so customers got fixes sooner."),
    ("Mentor four junior engineers on Python testing through daily pairing sessions.",
     "Mentored four junior engineers on Python testing through daily pairing sessions."),
    ("Optimise the nightly ETL jobs in Airflow so analysts had data by 7 am.",
     "Optimized the nightly ETL jobs in Airflow so analysts had data by 7 am."),  # EU spelling follows, in build()
    ("Speed up the checkout API with Redis caching, cutting p95 latency from 810 to 640 ms.",
     "Sped up the checkout API with Redis caching, cutting p95 latency from 810 to 640 ms."),
    ("Debug and fix flaky payment tests in pytest, so releases stopped waiting on reruns.",
     "Debugged and fixed flaky payment tests in pytest, so releases stopped waiting on reruns."),
    ("Apply rate limits to the partner API with Kong, so one client could not slow the rest.",
     "Applied rate limits to the partner API with Kong, so one client could not slow the rest."),
    ("Build the browse and add-to-cart flows in React for 83K customers.",  # a later 'and' word is never changed
     "Built the browse and add-to-cart flows in React for 83K customers."),
    ("Lead the claims API migration to Kubernetes, cut deploy time from 40 to 12 minutes and set up alerts.",
     "Led the claims API migration to Kubernetes, cut deploy time from 40 to 12 minutes and set up alerts."),
    # a word that is a noun as often as a verb is changed when its object follows: a number, a name, a plural noun
    ("Lead 5 data engineers through the Snowflake migration for 2M customer records, cutting query costs by 22%.",
     "Led 5 data engineers through the Snowflake migration for 2M customer records, cutting query costs by 22%."),
    ("Build CI pipelines in GitHub Actions for 12 services, catching 40 regressions a quarter.",
     "Built CI pipelines in GitHub Actions for 12 services, catching 40 regressions a quarter."),
    ("Build dashboards in Looker for 38 analysts, replacing weekly Excel exports.",
     "Built dashboards in Looker for 38 analysts, replacing weekly Excel exports."),
    ("Process 2M insurance claims a day in Spark, cutting the nightly backlog from 9 to 2 hours.",
     "Processed 2M insurance claims a day in Spark, cutting the nightly backlog from 9 to 2 hours."),
    ("Track and reduce cloud costs across 40 AWS accounts with Cost Explorer.",  # 'reduce' is only a verb
     "Tracked and reduced cloud costs across 40 AWS accounts with Cost Explorer."),
])
def test_a_present_tense_opening_is_put_in_the_past(given, expected):
    assert past_opening(given) == expected
    assert past_opening(expected) == expected  # once in the past, nothing more changes


@pytest.mark.parametrize("text", [
    "Led the claims API migration to Kubernetes.",
    "Cut AWS spend 17.8% by moving batch jobs to spot instances.",  # the past is the same word
    "Set up Grafana dashboards for the payments team so on-call engineers spotted failures early.",
    "Lead time dropped from 9 to 4 days after the team moved to trunk-based development.",  # a noun opening
    "Test suites now run in four minutes after parallelizing them in GitHub Actions.",
    "Support tickets fell by a third once the self-serve refund flow went live.",
    "Lead engineer for the payments platform rebuild on AWS.",
    "Build times stayed under ten minutes after caching Docker layers in CI.",
    "API gateway rollout for 40 partner integrations, built in Kong.",  # a name, not a verb
    "Designed the test and build stages of the CI pipeline in GitHub Actions.",  # later nouns stay nouns
    "Built and maintained Go services and REST APIs powering customer-facing product features.",
    # a noun opening: a job title or another noun follows it, not an object
    "Lead data engineer on the Snowflake migration for 2M customer records, cutting query costs by 22%.",
    "Lead backend engineer for the payments platform rebuild on AWS.",
    "Lead Python developer for the claims API, owning 40 endpoints.",
    "Process improvement initiatives across 4 plants, saving $1.2M a year.",
    "Programme manager for the claims platform, owning a budget of $2M.",  # never the verb 'program'
    "Programme management for 6 product lines, cutting slippage by 30%.",
    "Self-service analytics for 40 clinics in Looker, cutting report requests by 60%.",
    "Model serving platform for 12 teams, built on SageMaker.",
    "Reports for 40 clinics, generated nightly in Airflow.",
    "Model output reviews for 3 LLM features, catching 40 regressions.",
    "Load test results for the checkout API in k6.",
])
def test_past_tense_and_noun_openings_are_left_as_they_are(text):
    assert past_opening(text) == text


@pytest.mark.parametrize("change,text", [
    # a later 'and <word>' is never changed on its own: an adjective, or a verb after 'to', 'help' or 'which'
    (present_opening, ("Built CI pipelines in GitHub Actions for manual and automated test suites across 12 services, "
                       "catching 40 regressions a quarter.")),
    (present_opening, ("Added an Airflow DAG over 3K nightly loads with Python checks, which cut failed runs by 18% and "
                       "improved data freshness.")),
    (present_opening, ("Rebuilt the onboarding flow in React with simplified and standardized forms, lifting "
                       "completion by 9%.")),
    (present_opening, "Wrote a migration guide for legacy and modernized services, cutting onboarding from 3 weeks to 1."),
    (past_opening, ("Lead a team of 5 engineers to design and implement the billing pipeline in Python, cutting failed "
                    "charges by 31%.")),
    (past_opening, ("Create runbooks in Confluence that help on-call engineers identify and resolve Airflow incidents "
                    "in under 30 minutes.")),
    # a second verb phrase in the old tense: the bullet stays whole, for the rewrite pass, never in two tenses
    (present_opening, ("Rebuilt the Airflow scheduler in Python, wrote 300 pytest cases and caught 17 regressions "
                       "before each release.")),
    (present_opening, ("Monitored Kafka health in Datadog, triaged incidents and wrote post-mortems that cut repeat "
                       "outages by 30%.")),
    (present_opening, "Modeled the claims data in dbt and optimized the nightly loads in Airflow before 7 am."),
    (present_opening, "Leads the platform team of 6 engineers, owns the roadmap and ships weekly releases."),
    (past_opening, ("Design the claims API in Python, write the OpenAPI spec and set up contract tests in Pact for 12 "
                    "partner teams.")),
    (past_opening, ("Optimize slow database queries in PostgreSQL and add Redis caching layers to handle spikes "
                    "without errors.")),
])
def test_a_bullet_is_never_left_in_two_tenses(change, text):
    assert change(text) == text


def test_the_verb_tables_give_the_right_past():
    assert [past(v) for v in ("lead", "build", "plan", "apply", "deploy", "model", "commit", "cut", "audit")] == [
        "led", "built", "planned", "applied", "deployed", "modeled", "committed", "cut", "audited"]
    assert base_of("Leads") == "lead" and base_of("optimises") == "optimize" and base_of("processes") == "process"
    assert base_of("built") is None and base_of("Kubernetes") is None
    assert base_of("Programme") is None and base_of("cheques") is None  # British nouns, never the verbs


def profile_two_jobs() -> Profile:
    return Profile(name="Taylor Quinn", companies=[
        Company(name="Northwind Labs", start="Jan 2026", end="Present"),
        Company(name="Contoso", start="Jul 2024", end="Dec 2025")])


def test_the_posting_voice_and_a_posting_term_keep_their_words():
    out = make_output(n=2)
    out.jd_analysis.must_have_skills = ["deploy and monitor"]
    out.sections.experience[0].bullets = [  # the current job: the present tense
        "Led the platform team of 6 engineers through the move to Kubernetes, cutting deploy time from 40 to 12 minutes.",
        "Shared dashboards replaced weekly exports for 38 analysts once the Airflow loads ran before 7 am.",
    ]
    out.sections.experience[1].bullets = [  # an older job: the past tense
        "Refine your craft with code reviews and pairing on the payments API in Go every week.",  # the rewrite pass's
        "Deploy and monitor the claims API on GKE with Prometheus alerts, so on-call engineers caught failures early.",
        "Lead the move of 60 services to Kubernetes with Helm and Argo CD, cutting deploy time from 40 to 12 minutes.",
    ]
    resume = validate.build_resume(profile_two_jobs(), out)
    notes = validate.tense_bullets(resume, out.jd_analysis)
    current, older = resume.experience[0].bullets, resume.experience[1].bullets
    assert current[0].startswith("Lead the platform team") and current[1].startswith("Shared dashboards replaced")
    assert older[0].startswith("Refine your craft") and older[1].startswith("Deploy and monitor")
    assert older[2].startswith("Led the move") and notes == [
        "Northwind Labs: wrote 1 bullet(s) in the present tense, as the current job is written.",
        "Contoso: wrote 1 bullet(s) in the past tense, as every earlier job is written."]


def test_a_bullet_left_whole_is_not_counted_in_the_notes():
    out = make_output(n=2)
    current = [  # a second verb in the past, an adjective opening, a one-time event; then a slip that is fixed
        ("Rebuilt the Airflow scheduler in Python, wrote 300 pytest cases and caught 17 regressions before each "
         "release."),
        "Automated Spark tests catch 40 regressions a quarter before each release of the claims pipeline in Snowflake.",
        "Promoted to Senior Data Engineer within 18 months for leading the Airflow migration across 40 pipelines.",
        "Added RAG over past tickets with a PostgreSQL vector index, so suggested answers cite the right help article.",
    ]
    older = [  # a noun opening, a second verb in the present; then a slip that is fixed
        "Lead data engineer on the Snowflake migration for 2M customer records, cutting query costs by 22%.",
        "Design the claims API in Python, write the OpenAPI spec and set up contract tests in Pact for 12 partner teams.",
        "Lead the move of 60 services to Kubernetes with Helm and Argo CD, cutting deploy time from 40 to 12 minutes.",
    ]
    out.sections.experience[0].bullets, out.sections.experience[1].bullets = list(current), list(older)
    resume = validate.build_resume(profile_two_jobs(), out)
    notes = validate.tense_bullets(resume, out.jd_analysis)
    assert resume.experience[0].bullets == [*current[:3], "Add" + current[3].removeprefix("Added")]
    assert resume.experience[1].bullets == [*older[:2], "Led" + older[2].removeprefix("Lead")]
    assert notes == ["Northwind Labs: wrote 1 bullet(s) in the present tense, as the current job is written.",
                     "Contoso: wrote 1 bullet(s) in the past tense, as every earlier job is written."]


@pytest.mark.parametrize("given,expected", [
    ("Added RAG over past tickets with a PostgreSQL vector index, so suggested answers cite the right help article.",
     "Add RAG over past tickets with a PostgreSQL vector index, so suggested answers cite the right help article."),
    ("Designed and shipped the billing API in Go with idempotent retries, cutting failed charges by 31%.",
     "Design and ship the billing API in Go with idempotent retries, cutting failed charges by 31%."),
    ("Leads the platform team of 6 engineers through the Kubernetes move.",
     "Lead the platform team of 6 engineers through the Kubernetes move."),
    ("Co-led the migration to PostgreSQL, cutting nightly load time from 47 to 36 minutes.",
     "Co-lead the migration to PostgreSQL, cutting nightly load time from 47 to 36 minutes."),
    ("Modeled and optimized the nightly claims loads in dbt and Airflow before 7 am.",  # the verbs joined to it
     "Model and optimize the nightly claims loads in dbt and Airflow before 7 am."),
    ("Built CI pipelines in GitHub Actions for 12 services, catching 40 regressions a quarter.",  # a name follows
     "Build CI pipelines in GitHub Actions for 12 services, catching 40 regressions a quarter."),
    ("Reduced cloud spend by 22% by moving batch jobs to spot instances.",  # 'spot' after 'to' is no subject's verb
     "Reduce cloud spend by 22% by moving batch jobs to spot instances."),
    ("Built the claims API in Go, used by 40 partner teams for daily settlement.",  # 'used by': no second verb
     "Build the claims API in Go, used by 40 partner teams for daily settlement."),
    ("Rebuilt the Airflow scheduler in Python and cut failed runs from 40 to 12 a month.",  # 'cut' fits both tenses
     "Rebuild the Airflow scheduler in Python and cut failed runs from 40 to 12 a month."),
    ("Certified 40 engineers on SOC 2 controls in Vanta, cutting audit prep from 6 to 2 weeks.",
     "Certify 40 engineers on SOC 2 controls in Vanta, cutting audit prep from 6 to 2 weeks."),
])
def test_a_past_opening_in_the_current_job_becomes_present(given, expected):
    assert present_opening(given) == expected


@pytest.mark.parametrize("text", [
    "Shared dashboards replaced weekly exports for 38 analysts.",  # an adjective, then the sentence's own verb
    "Automated tests now run in 4 minutes on every pull request.",
    "Improved onboarding cut churn by 12% in the first quarter.",
    "Lead time fell from 9 to 4 days after the release train.",  # a noun
    "Ran evaluation suites that replay real scenarios.",  # would become the weak 'Run': rewritten instead
    "Used Claude with tool calling for 500K+ users.",
    "Build the agentic AI behind the assistant.",  # already present
    "Set up AI observability for the agents.",  # the same in both tenses
    # a result first: the opening is an adjective, the verb after its subject is the sentence's own (a name between)
    "Automated Spark tests catch 40 regressions a quarter before each release of the claims pipeline in Snowflake.",
    "Automated tests catch 40 regressions a quarter before each release of the claims app.",
    "Shared Looker dashboards give 38 analysts current numbers each morning instead of weekly exports.",
    "Shared Looker dashboards replaced weekly Excel exports for 38 analysts once the Airflow loads ran before 7 am.",
    "Automated tests in CI now run in 4 minutes on every pull request.",
    "Automated tests in CI catch 40 regressions a quarter before each release of the claims app.",
    "Automated tests in CI run in 4 minutes on every pull request.",
    "Standardized Terraform modules let 12 teams ship infrastructure changes without platform tickets.",
    "Automated triage in Zendesk saves the support team 10 hours a week.",
    # true only in the past: a passive opening or a one-time event, never a command
    "Promoted to Senior Data Engineer within 18 months for leading the Airflow migration across 40 pipelines.",
    ("Selected as 1 of 4 engineers to rebuild the Spark ingestion layer, cutting nightly load time from 47 to 36 "
     "minutes."),
    "Hired as the first data engineer to build the Snowflake warehouse for 38 analysts.",
    "Recognized by the CTO for leading the Kafka migration across 40 services.",
    "Assigned to the claims team to rebuild its Spark jobs, cutting nightly load time from 47 to 36 minutes.",
    "Certified as a Kubernetes administrator in 2025 while moving 40 services to EKS.",
    "Won the 2025 company hackathon with a Kafka fraud detector adopted by 3 teams.",
    "Earned the AWS Solutions Architect certification in 2025 while leading the move to EKS.",
    "Promoted twice in 3 years for owning the claims API rebuild in Go.",
])
def test_a_present_opening_is_left_alone_when_unsure_or_weak(text):
    assert present_opening(text) == text


def test_the_reverse_tables_give_the_base():
    assert [present_base(w) for w in ("built", "Led", "shipped", "applied", "ran", "Designs")] == [
        "build", "lead", "ship", "apply", "run", "design"]
    assert present_base("build") is None and present_base("Kubernetes") is None


class WrongTense(LLMClient):
    """The demo's draft with a past-tense opening in the current job and a present-tense one in an older job."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            jobs = result.data["sections"]["experience"]
            jobs[0]["bullets"][1] = ("Modeled and optimised the nightly claims loads in dbt and Airflow, so analysts "
                                     "get fresh numbers by 7 am every weekday.")
            jobs[1]["bullets"][1] = ("Design and run the order-sync service in Go with idempotent retries, so partner "
                                     "integrations stayed stable through traffic spikes.")
        return result


@pytest.mark.parametrize("spelling,first,second", [
    ("us", "Model and optimize the nightly claims loads", "Designed and ran the order-sync service"),
    ("eu", "Model and optimise the nightly claims loads", "Designed and ran the order-sync service"),
])
def test_a_slip_in_any_job_is_fixed_on_the_draft(router, db, registry, profile, jd, spelling, first, second):
    client = WrongTense(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    spelling=spelling))
    assert result.status == "ok" and len(client.prompts) == 1  # fixed for free, no fix round
    assert result.resume.experience[0].bullets[1].startswith(first)
    assert result.resume.experience[1].bullets[1].startswith(second)
    assert any(n.endswith("in the present tense, as the current job is written.") for n in result.notes)
    assert any(n.endswith("in the past tense, as every earlier job is written.") for n in result.notes)


def test_the_prompt_asks_for_the_present_tense_in_the_current_job_only():
    text = tailor_system("estimate")
    assert 'in the PRESENT tense for the current job' in text and 'never "Builds" or "Built"' in text
    assert 'in the past tense for every other job ("Built", "Led")' in text
    assert "is not in its job's tense (present for the current job, past for every other)" in text  # the self-check
    assert "Design and run" not in text and "Strong (the current job): \"Plan and launch" in text
    assert '"Build and maintain"' in text and '"Used", "Supported", "Ran" or "Run"' in text  # banned openers
