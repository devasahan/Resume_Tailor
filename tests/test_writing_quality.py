"""Fuller summaries and bullets: a bullet budget that fits two pages up front, stronger rules, a thin-text check."""
import asyncio
from datetime import date

import pytest

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
from app.pipeline.tailor import run_checks, run_tailor
from app.prompts import (
    RETRY_MARKER,
    SHORTEN_NOTE,
    cover_system,
    tailor_system,
    tailor_user,
)
from app.schemas import Company, Contact, CoverLetter, Education, Profile, TailorResult

from .test_validate import make_output

TODAY = date(2026, 10, 12)


def five_jobs(n=5) -> Profile:
    companies = [
        Company(name="Northwind Labs", start="Feb 2025", end="Present", mode="Remote"),
        Company(name="Bluefin.com", start="Jul 2024", end="Feb 2025", mode="Remote"),
        Company(name="Contoso Education", start="Jun 2021", end="Jun 2024", mode="Remote"),
        Company(name="Fabrikam Holdings", start="May 2020", end="May 2021", mode="Onsite", location="Austin, TX"),
        Company(name="Earlier Co", start="Jun 2018", end="Apr 2020", mode="Onsite"),
    ][:n]
    return Profile(
        name="TAYLOR QUINN BROOKS",
        contact=Contact(email="taylor@example.com", phone="(555) 010-2030", linkedin="linkedin.com/in/taylor-quinn",
                        location="Texas, United States"),
        companies=companies,
        education=[Education(school="University of Texas at Austin", degree="B.S.", field="Computer Science", end="2018")],
    )


# ------------------------------------------------------------------ bullet budget
def test_caps_follow_recency_and_length_of_stay():
    profile = five_jobs()
    # Northwind Labs 1 yr 9 mo -> 5; Bluefin.com 8 months -> 3; Contoso Education 3 years -> its recency cap 5;
    # Fabrikam Holdings 13 months -> 4
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
        # a draft with every bullet at full length fits without asking the model to shorten anything
        found = try_styles(_plan_resume(five_jobs(n), budget))
        assert found is not None and found[1] <= MAX_PAGES
    assert bullet_budget(five_jobs(1), today=TODAY) == [5]  # a short career is not padded out
    assert try_styles(_plan_resume(five_jobs(1), [5]))[1] == 1
    assert bullet_budget(five_jobs(4), today=TODAY) == [5, 3, 5, 4]
    assert bullet_budget(five_jobs(4), today=TODAY, max_pages=1) == [5, 3, 4, 2]  # one page holds less


def test_budget_holds_for_the_longest_text_the_rules_allow():
    """32-word bullets, an 89-word summary and five skill groups still fit on two pages."""
    from app.pipeline.fit import _PLAN_SUMMARY
    from app.schemas import Resume, ResumeExperience, SkillGroup

    bullet = ("Architected the webhook ingestion and retry pipeline for third-party marketplace integrations on AWS "
              "Lambda and SQS with idempotency keys, so partner order updates arrived reliably and support stopped "
              "re-running failed syncs")
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
                                                     bullets=[bullet] * k) for i, k in enumerate(budget)])
        assert try_styles(resume) is not None, (n, budget)


def test_smaller_budget_takes_from_every_long_role_and_at_least_two():
    assert smaller_budget([8, 6, 5, 3]) == [7, 5, 4, 3]  # every employer above 3
    assert smaller_budget([5, 3, 3, 3]) == [4, 3, 3, 2]  # only one above 3: also the oldest of the rest
    assert smaller_budget([2, 2]) == [2, 2]


# ------------------------------------------------------------------ prompts
def test_prompt_states_the_bullet_count_per_employer():
    profile = five_jobs(3)
    text = tailor_user("A job description", profile.companies, "", bullets=[5, 3, 2], today=TODAY)
    assert ("0: Northwind Labs | Feb 2025 - Present (current role) | Remote | 1 yr 9 mo stay, 3 yrs 8 mo of experience "
            "before it | write exactly 5 bullets") in text
    assert "2: Contoso Education | Jun 2021 - Jun 2024 | Remote | 3 yrs 1 mo stay, first job | write exactly 2 bullets" in text


def test_rules_ask_for_specific_bullets_and_a_full_summary():
    text = tailor_system("estimate")
    assert "write exactly the number of bullets given for each employer" in text
    assert "20-32 words" in text and "WHAT, specifically" in text and "HOW" in text and "RESULT" in text
    assert "at most two pages" in text and "Name the stack in detail" in text and "TECH STACK" not in text
    assert "a must-have listed only in SKILLS does not count" in text and "tech line" not in text
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


# ------------------------------------------------------------------ plain words, not machine-written ones
def test_machine_sounding_words_and_ing_chains_are_found(profile):
    out = make_output()
    out.sections.summary = "Data Engineer who leveraged Spark to deliver robust pipelines for analysts."
    out.sections.experience[0].bullets = [
        "Spearheaded a seamless move of nightly jobs to Airflow in order to cut manual reruns",
        "Rebuilt the loaders, ensuring fresh data, enabling faster reports, and reducing support tickets",
    ]
    out.sections.skills[0].skills.append("Robust Toolkit")  # skills are tool names and are never read
    problems = validate.plain_word_problems(validate.build_resume(profile, out))
    assert problems == ["'leveraged' (summary)", "'robust' (summary)", "'Spearheaded' (Northwind Labs)",
                        "'seamless' (Northwind Labs)", "'in order to' (Northwind Labs)",
                        "a chain of '-ing' clauses (Northwind Labs: 'Rebuilt the loaders, ensuring fresh data, enabling faster ...')"]


def test_everyday_technical_words_are_plain(profile):
    out = make_output()
    out.sections.summary = "Data Engineer who runs Kubernetes orchestration and dynamic pricing jobs."
    out.sections.experience[0].bullets = [
        "Moved the billing jobs to Airflow during the migration, keeping reports on time, so finance closed the month early"]
    assert validate.plain_word_problems(validate.build_resume(profile, out)) == []


class Buzzy(LLMClient):
    """Writes buzzwords into the first draft; the fix comes back in plain words."""

    def __init__(self, inner, words: int):
        self.inner, self.calls, self.words = inner, 0, words

    async def generate_structured(self, **kwargs):
        self.calls += 1
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output":
            bullets = result.data["sections"]["experience"][0]["bullets"]  # not the first: it carries the figures
            bullets[1] = ("Leveraged Python and SQL to build a seamless reporting flow for the product teams, so "
                          "managers saw current numbers every morning")
            if self.words == 1:
                bullets[1] = bullets[1].replace("a seamless", "a simple")
        return result


@pytest.mark.parametrize("words,calls", [(2, 2), (1, 1)])
def test_machine_sounding_drafts_are_rewritten_in_plain_words(router, db, registry, profile, jd, words, calls):
    client = Buzzy(FakeClient(registry.get("demo-concise")), words)
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and client.calls == calls  # one stray word is not worth a model call
    remaining = validate.plain_word_problems(result.resume)
    assert remaining == ([] if words == 2 else ["'Leveraged' (Northwind Labs)"])
    assert any("machine-written" in w for w in result.ats_warnings) == (words == 1)  # still shown to the user


def test_the_prompt_asks_for_plain_words():
    text = tailor_system("estimate")
    assert "PLAIN WORDS" in text and "the way a person would explain the work to a colleague" in text
    assert "leverage, utilize, spearhead, seamless, robust" in text and '"in order to"' in text
    assert "Do not chain \"-ing\" clauses" in text and "Architected" not in text


# ------------------------------------------------------------------ the cover letter
WORDY_LETTER = CoverLetter(greeting="Dear Hiring Manager,", closing="Sincerely,", paragraphs=[
    "Shipping AI systems that people rely on takes robust execution. I like that kind of work.",
    "I built evaluation tools by leveraging automated metrics, ensuring quality, enabling teams, driving adoption.",
    "The services I build fit seamlessly into existing business workflows."])


def test_the_cover_prompt_asks_for_plain_words_and_starts_at_the_greeting():
    text = cover_system()
    assert "PLAIN WORDS" in text and "leverage, utilize, spearhead, seamless, robust" in text
    assert "the letter starts with the greeting" in text and "a date, an address or a subject line" in text
    assert "The system adds them" not in text


def test_cover_letter_wording_is_checked_sentence_by_sentence():
    assert validate.cover_plain_problems(WORDY_LETTER) == [
        "'robust' (paragraph 1)", "'leveraging' (paragraph 2)",
        "a chain of '-ing' clauses (paragraph 2: 'I built evaluation tools by leveraging automated metrics, ...')",
        "'seamlessly' (paragraph 3)"]
    plain = WORDY_LETTER.model_copy(update={"paragraphs": [
        "I built the evaluation service the support team uses every day, and wrong answers dropped as a result.",
        "Before that, I ran the release checks, wrote the test plans, and kept the team's runbooks current."]})
    assert validate.cover_plain_problems(plain) == []  # three '-ing'-free list items in one sentence are fine


def test_machine_sounding_cover_letters_get_a_format_note(profile):
    result = TailorResult(requested_model="m", resume=validate.build_resume(profile, make_output()),
                          analysis=make_output().jd_analysis, cover_letter=WORDY_LETTER)
    run_checks(result, "a job description")
    notes = [w for w in result.ats_warnings if w.startswith("Cover letter wording that can sound machine-written")]
    assert len(notes) == 1 and "'robust' (paragraph 1)" in notes[0] and "Plain words read as your own." in notes[0]
    result.cover_letter = None
    run_checks(result, "a job description")
    assert not any(w.startswith("Cover letter wording") for w in result.ats_warnings)
