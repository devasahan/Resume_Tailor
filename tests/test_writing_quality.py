"""Fuller summaries and bullets: a bullet budget that fits two pages up front, stronger rules, a thin-text check."""
import asyncio
from datetime import date

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.facts import bullet_caps
from app.pipeline.fit import (
    MAX_PAGES,
    _plan_resume,
    bullet_budget,
    smaller_budget,
    try_styles,
)
from app.pipeline.tailor import run_tailor
from app.prompts import RETRY_MARKER, SHORTEN_NOTE, tailor_system, tailor_user
from app.schemas import Company, Contact, Education, Profile

from .test_validate import make_output

TODAY = date(2026, 10, 12)


def five_jobs(n=5) -> Profile:
    companies = [
        Company(name="Lindy", start="Feb 2025", end="Present", mode="Remote"),
        Company(name="Acquire.com", start="Jul 2024", end="Feb 2025", mode="Remote"),
        Company(name="Orbital Education", start="Jun 2021", end="Jun 2024", mode="Remote"),
        Company(name="Q2 Holdings", start="May 2020", end="May 2021", mode="Onsite", location="Austin, TX"),
        Company(name="Earlier Co", start="Jun 2018", end="Apr 2020", mode="Onsite"),
    ][:n]
    return Profile(
        name="JUAN DANIEL RAMIREZ",
        contact=Contact(email="juan@example.com", phone="(555) 010-2030", linkedin="linkedin.com/in/juan-ramirez",
                        location="Texas, United States"),
        companies=companies,
        education=[Education(school="University of Texas at Austin", degree="B.S.", field="Computer Science", end="2018")],
    )


# ------------------------------------------------------------------ bullet budget
def test_caps_follow_recency_and_length_of_stay():
    profile = five_jobs()
    # Lindy 1 yr 9 mo -> 5; Acquire.com 8 months -> 3; Orbital 3 years -> its recency cap 5; Q2 13 months -> 4
    assert bullet_caps(profile.companies, TODAY) == [5, 3, 5, 4, 3]
    short = [Company(name="A", start="Jun 2026", end="Present"), Company(name="B", start="", end="")]
    assert bullet_caps(short, TODAY) == [2, 6]  # 5 months -> 2; unreadable dates -> no reduction
    long = [Company(name=f"C{i}", start=f"Jan {2010 + 3 * i}", end=f"Dec {2012 + 3 * i}") for i in reversed(range(6))]
    assert bullet_caps(long, TODAY) == [8, 6, 5, 4, 3, 3]  # 3-year stays: the recency caps


def test_budget_gives_recent_roles_the_most_and_fits_two_pages_at_full_length():
    for n in range(1, 6):
        budget = bullet_budget(five_jobs(n), today=TODAY)
        assert len(budget) == n and all(b >= 2 for b in budget)
        assert budget[0] == max(budget)
        # a draft with every bullet (and tech line) at full length fits without asking the model to shorten anything
        found = try_styles(_plan_resume(five_jobs(n), budget))
        assert found is not None and found[1] <= MAX_PAGES
    assert bullet_budget(five_jobs(1), today=TODAY) == [5]  # a short career is not padded out
    assert try_styles(_plan_resume(five_jobs(1), [5]))[1] == 1
    assert bullet_budget(five_jobs(4), today=TODAY) == [5, 3, 5, 4]
    assert bullet_budget(five_jobs(4), today=TODAY, max_pages=1) == [4, 3, 3, 2]  # one page holds far less


def test_budget_holds_for_the_longest_text_the_rules_allow():
    """32-word bullets, an 89-word summary, five skill groups and long tech lines still fit on two pages."""
    from app.pipeline.fit import _PLAN_SUMMARY
    from app.schemas import Resume, ResumeExperience, SkillGroup

    bullet = ("Architected the webhook ingestion and retry pipeline for third-party marketplace integrations on AWS "
              "Lambda and SQS with idempotency keys, so partner order updates arrived reliably and support stopped "
              "re-running failed syncs")
    tech = ["Go", "TypeScript", "PostgreSQL", "Redis", "AWS (ECS, Lambda, SQS)", "Terraform", "GitHub Actions",
            "Datadog", "Docker", "Algolia"]
    summary = _PLAN_SUMMARY + " Brings end-to-end ownership and close partnership with product teams."
    skills = [SkillGroup(category=c, skills=s.split(", ")) for c, s in [
        ("Languages and Frameworks", "Go, TypeScript, React, Next.js, Tailwind CSS, SQL"),
        ("Backend and Integrations", "REST APIs, Webhooks, Third-Party API Integrations, Algolia, Caching, Async Jobs"),
        ("AWS and Infrastructure", "AWS, ECS, Lambda, CloudWatch, S3, CloudFront, RDS, IAM"),
        ("Data", "PostgreSQL, Redis"),
        ("Practices", "CI/CD, Observability, Code Review, Jest, Playwright, Storybook, Feature Flags, TDD"),
    ]]
    for n in range(1, 6):
        profile = five_jobs(n)
        budget = bullet_budget(profile, today=TODAY)
        resume = Resume(profile=profile, headline="Senior Full-Stack Engineer | Go | React | TypeScript | AWS",
                        summary=summary, skills=skills,
                        experience=[ResumeExperience(company_index=i, job_title="Senior Full-Stack Engineer",
                                                     bullets=[bullet] * k, tech=tech) for i, k in enumerate(budget)])
        assert try_styles(resume) is not None, (n, budget)


def test_smaller_budget_takes_from_every_long_role_and_at_least_two():
    assert smaller_budget([8, 6, 5, 3]) == [7, 5, 4, 3]  # every employer above 3
    assert smaller_budget([5, 3, 3, 3]) == [4, 3, 3, 2]  # only one above 3: also the oldest of the rest
    assert smaller_budget([2, 2]) == [2, 2]


# ------------------------------------------------------------------ prompts
def test_prompt_states_the_bullet_count_per_employer():
    profile = five_jobs(3)
    text = tailor_user("A job description", profile.companies, "", bullets=[5, 3, 2], today=TODAY)
    assert ("0: Lindy | Feb 2025 - Present (current role) | Remote | 1 yr 9 mo stay, 3 yrs 8 mo of experience "
            "before it | write exactly 5 bullets") in text
    assert "2: Orbital Education | Jun 2021 - Jun 2024 | Remote | 3 yrs 1 mo stay, first job | write exactly 2 bullets" in text


def test_rules_ask_for_specific_bullets_and_a_full_summary():
    text = tailor_system("estimate")
    assert "write exactly the number of bullets given for each employer" in text
    assert "20-32 words" in text and "WHAT, specifically" in text and "HOW" in text and "RESULT" in text
    assert "at most two pages" in text and "Name the stack in detail" in text and "TECH STACK:" in text
    assert "a must-have listed only in SKILLS or in a tech line does not count" in text
    assert "24-32 items" in text and "Do NOT invent client or product names, certifications" in text
    assert '"Build and maintain"' in text and "Weak:" in text and "Strong:" in text
    assert "Never write a string of short fragments" in text
    assert "Check before answering" in text


def test_shorten_note_no_longer_asks_for_thin_text():
    assert "18 words" not in SHORTEN_NOTE and "under 60 words" not in SHORTEN_NOTE
    assert "60-75 words" in SHORTEN_NOTE and "same quality" in SHORTEN_NOTE


# ------------------------------------------------------------------ thin-text check
def test_thin_text_problems(profile):
    out = make_output()
    out.sections.summary = "Senior Full-Stack Engineer with 7+ years of experience. Builds Go services. Improves reliability."
    out.sections.experience[0].bullets = [
        "Build and maintain Go services and REST APIs powering customer-facing product features.",
        "Deliver React and TypeScript features from backend implementation through frontend release.",
    ]
    problems = validate.thin_text_problems(validate.build_resume(profile, out))
    assert any("summary has only" in p for p in problems)
    assert any("bullets are under 15 words" in p and "Deliver React" in p for p in problems)
    assert any("duty list" in p and "Build and maintain" in p for p in problems)
    warnings = validate.ats_lint(validate.build_resume(profile, out))
    assert any("Summary is short" in w for w in warnings) and any("only 12 words" in w for w in warnings)


class Thin(LLMClient):
    """Writes a thin draft first, a full one once told what was wrong."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if RETRY_MARKER not in kwargs["user"]:
            result.data["sections"]["summary"] = "Data engineer. Builds pipelines."
            for exp in result.data["sections"]["experience"]:
                exp["bullets"] = ["Worked on " + b.split(" ", 1)[1] for b in exp["bullets"]]
        return result


def test_a_thin_draft_is_rewritten_once(router, db, registry, profile, jd):
    client = Thin(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    assert "too thin" in client.prompts[1] and "summary has only" in client.prompts[1]
    assert validate.thin_text_problems(result.resume) == []
    assert validate.word_count(result.resume.summary) >= 60


def test_demo_run_follows_the_budget_without_shortening(router, db, profile, jd):
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.fits and result.retries == 0
    assert [len(e.bullets) for e in result.resume.experience] == bullet_budget(profile, "Letter")
    assert not any("shortened" in n or "Removed" in n for n in result.notes)
