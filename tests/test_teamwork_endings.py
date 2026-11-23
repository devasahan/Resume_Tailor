"""No bullet ends by saying who the work was done with ('..., in close partnership with security and product
teams.'): the prompt forbids it, the app cuts such an ending when the bullet still stands on its own, and a bullet
that says neither how nor what changed is flagged for the rewrite pass and the admin (app/pipeline/validate.py)."""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.tailor import coverage_feedback, run_tailor
from app.prompts import RETRY_MARKER, tailor_system
from app.schemas import Company, Profile

from .test_bullet_tense import profile_two_jobs
from .test_validate import make_output

HIPAA = ("Configured HIPAA compliance controls, audit logs and SSO for enterprise clients in close partnership with "
         "security and product teams.")
LONG = ("Built the claims intake service in Python on GCP for 40 hospital clients, adding SAML single sign-on with Okta, "
        "in close partnership with security and product teams.")


def trimmed(bullets: list[str], must_haves: list[str] | None = None) -> tuple[list[str], list[str]]:
    out = make_output(n=2)
    if must_haves is not None:
        out.jd_analysis.must_have_skills = must_haves
    out.sections.experience[0].bullets = bullets
    resume = validate.build_resume(profile_two_jobs(), out)
    notes = validate.trim_teamwork_tails(resume, out.jd_analysis)
    return resume.experience[0].bullets, notes


def test_a_closing_teamwork_clause_is_cut():
    bullets, notes = trimmed([LONG])
    assert bullets == [("Built the claims intake service in Python on GCP for 40 hospital clients, adding SAML "
                        "single sign-on with Okta.")]
    assert notes == [("Northwind Labs: removed the closing 'in close partnership with security and product teams' "
                      "from a bullet: it said who the work was done with, not how it was done or what changed.")]


@pytest.mark.parametrize("ending", [
    ", working closely with product managers.",
    " in collaboration with the data platform team.",
    ", partnering with QA and design.",
    " alongside the QA and design teams.",
])
def test_other_teamwork_endings_are_cut(ending):
    start = "Moved the nightly billing exports from cron scripts to Airflow DAGs with retries and Slack alerts on failure"
    bullets, _notes = trimmed([start + ending])
    assert bullets == [start + "."]


@pytest.mark.parametrize("bullet", [
    HIPAA,  # 11 words would be left: too thin, so it waits for the rewrite pass
    ("Built the payments service in Go for 40 merchants with Stripe webhooks and idempotent retries, in partnership "
     "with Stripe."),  # a name, not a team
    ("Partnered with the security team to add audit logging to the claims API, working closely with compliance "
     "teams."),  # the partnership is the work
    ("Moved the nightly billing exports from cron scripts to Airflow DAGs with retries and alerts, working closely "
     "with 12 product teams."),  # a figure
    ("Migrated the order jobs, working closely with the owning teams until every output matched, so the cut-over "
     "caused no downtime."),  # not at the end
])
def test_these_bullets_keep_their_words(bullet):
    bullets, notes = trimmed([bullet])
    assert bullets == [bullet] and notes == []


def test_a_must_have_in_the_clause_keeps_it():
    bullet = ("Moved the nightly billing exports from cron scripts to Airflow DAGs with retries and Slack alerts, "
              "working closely with cross-functional teams.")
    bullets, notes = trimmed([bullet], must_haves=["cross-functional teams"])
    assert bullets == [bullet] and notes == []  # its only mention: the rewrite pass places it instead


def test_the_summary_is_left_alone():
    out = make_output(n=2)
    out.sections.summary += " Brings close partnership with product and design teams."
    resume = validate.build_resume(profile_two_jobs(), out)
    summary = resume.summary
    validate.trim_teamwork_tails(resume, out.jd_analysis)
    assert resume.summary == summary


def test_weak_bullets_are_named():
    out = make_output(n=2)
    out.sections.experience[0].bullets = [
        HIPAA,
        "Mentored junior engineers on Python best practices and code review standards during daily pairing sessions.",
        "Optimized slow database queries in PostgreSQL and added Redis caching to handle traffic spikes without errors.",
        "Built the claims API on GCP for 53,000 daily users, cutting p95 latency from 810 to 640 ms.",
    ]
    resume = validate.build_resume(profile_two_jobs(), out)
    weak = [(why, bullet.split()[0]) for company, bullet, why in validate.weak_bullets(resume)
            if company == "Northwind Labs"]  # the current job's (the sample's other job has stub bullets)
    assert ("ends on who the work was done with", "Configured") in weak
    assert ("names no result", "Mentored") in weak
    assert not any(first in ("Optimized", "Built") for _why, first in weak)
    assert validate.weak_bullet_list(validate.weak_bullets(resume)[:1]) == (
        '"Configured HIPAA compliance controls, audit logs and SSO ..." (Northwind Labs: ends on who the work '
        "was done with)")


class Injects(LLMClient):
    """Puts bullets into the current job's first draft; the fix round's answer comes back as the demo writes it."""

    def __init__(self, inner, bullets):
        self.inner, self.bullets, self.prompts = inner, bullets, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            for index, bullet in enumerate(self.bullets, start=1):
                result.data["sections"]["experience"][0]["bullets"][index] = bullet
        return result


def tailored(router, db, registry, profile, jd, bullets):
    client = Injects(FakeClient(registry.get("demo-concise")), bullets)
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    time_limit=0))
    assert result.status == "ok"
    return client, result


def test_one_weak_bullet_is_only_noted(router, db, registry, profile, jd):
    client, result = tailored(router, db, registry, profile, jd, [HIPAA])
    assert len(client.prompts) == 1
    assert any(w.startswith("Bullets that do not say how the work was done or what changed:") and "Configure HIPAA" in w
               for w in result.ats_warnings)  # in the present tense: it is the current job's


def test_two_weak_bullets_ask_for_the_rewrite(router, db, registry, profile, jd):
    client, result = tailored(router, db, registry, profile, jd, [
        HIPAA, "Mentored junior engineers on Python best practices and code review standards in daily pairing."])
    assert len(client.prompts) == 2
    asked = client.prompts[1].split(RETRY_MARKER, 1)[1]
    assert "These bullets do not say how the work was done or what changed:" in asked
    assert "drop an ending that only says who the work was done with" in asked
    assert not validate.weak_bullets(result.resume)  # the demo's own text came back for both


def five_jobs() -> Profile:
    return Profile(name="Taylor Quinn", companies=[
        Company(name=f"Harbor Works {i}", start=f"Jan {2024 - 3 * i}", end="Present" if i == 0 else f"Dec {2026 - 3 * i}")
        for i in range(5)])


@pytest.mark.parametrize("people", ["conftest", "five", "one"])
@pytest.mark.parametrize("metrics", ["estimate", "hints_only"])
@pytest.mark.parametrize("spelling", ["us", "eu"])
def test_the_demo_writes_no_weak_bullet_and_needs_no_fix(router, db, profile, jd, people, metrics, spelling):
    person = {"conftest": profile, "five": five_jobs(),
              "one": Profile(name="Taylor Quinn", companies=[Company(name="Northwind Labs", start="Jan 2022",
                                                                     end="Present")])}[people]
    db.set_settings({"metrics_mode": metrics})
    result = asyncio.run(run_tailor(router, db, profile=person, profile_id=None, jd=jd, model_id="demo-concise",
                                    spelling=spelling, time_limit=0))
    assert result.status == "ok"
    if people != "five":  # five jobs ask for the fix round anyway: the demo repeats one title (title_growth_problems)
        assert result.retries == 0
    assert validate.weak_bullets(result.resume) == []
    assert not any("in the past tense" in n or "removed the closing" in n for n in result.notes)


def test_the_prompt_rules_out_teamwork_endings():
    text = tailor_system("estimate")
    assert "never end a bullet by saying who the work was done with" in text
    assert "close partnership with product and design teams" not in text  # the example the model copied
    assert "Brings end-to-end ownership of payment features" not in text  # self-praise: the summary has no such line
    assert "so finance stopped fixing freight charges by hand" in text
    out = make_output(n=2)
    out.jd_analysis.must_have_skills = ["stakeholder management"]  # shown nowhere: the fix round places it
    fix = coverage_feedback(validate.build_resume(profile_two_jobs(), out), out.jd_analysis)
    assert '"stakeholder management"' in fix and 'never as a closing "with stakeholders" phrase' in fix
