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
    ("Optimize slow database queries in PostgreSQL and add Redis caching layers to handle spikes without errors.",
     "Optimized slow database queries in PostgreSQL and added Redis caching layers to handle spikes without errors."),
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
])
def test_past_tense_and_noun_openings_are_left_as_they_are(text):
    assert past_opening(text) == text


def test_the_verb_tables_give_the_right_past():
    assert [past(v) for v in ("lead", "build", "plan", "apply", "deploy", "model", "commit", "cut", "audit")] == [
        "led", "built", "planned", "applied", "deployed", "modeled", "committed", "cut", "audited"]
    assert base_of("Leads") == "lead" and base_of("optimises") == "optimize" and base_of("processes") == "process"
    assert base_of("built") is None and base_of("Kubernetes") is None


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


@pytest.mark.parametrize("given,expected", [
    ("Added RAG over past tickets with a PostgreSQL vector index, so suggested answers cite the right help article.",
     "Add RAG over past tickets with a PostgreSQL vector index, so suggested answers cite the right help article."),
    ("Designed and shipped the billing API in Go with idempotent retries, cutting failed charges by 31%.",
     "Design and ship the billing API in Go with idempotent retries, cutting failed charges by 31%."),
    ("Leads the platform team of 6 engineers through the Kubernetes move.",
     "Lead the platform team of 6 engineers through the Kubernetes move."),
    ("Co-led the migration to PostgreSQL, cutting nightly load time from 47 to 36 minutes.",
     "Co-lead the migration to PostgreSQL, cutting nightly load time from 47 to 36 minutes."),
    ("Modeled the claims data in dbt and optimized the nightly loads in Airflow before 7 am.",
     "Model the claims data in dbt and optimize the nightly loads in Airflow before 7 am."),
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
            jobs[0]["bullets"][1] = ("Modeled the claims data in dbt and optimised the nightly loads in Airflow, so "
                                     "analysts get fresh numbers by 7 am every weekday.")
            jobs[1]["bullets"][1] = ("Design and run the order-sync service in Go with idempotent retries, so partner "
                                     "integrations stayed stable through traffic spikes.")
        return result


@pytest.mark.parametrize("spelling,first,second", [
    ("us", "Model the claims data in dbt and optimize", "Designed and ran the order-sync service"),
    ("eu", "Model the claims data in dbt and optimise", "Designed and ran the order-sync service"),
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
