"""Must-have JD terms covered automatically, flagged for review, and a career path that shows growth."""
import asyncio
from datetime import date

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import validate
from app.pipeline.facts import (
    bullet_caps,
    experience_before,
    months_phrase,
    stay_months,
)
from app.pipeline.fit import bullet_budget, extra_order
from app.pipeline.tailor import (
    MIN_UNCOVERED_FOR_RETRY,
    coverage_feedback,
    run_checks,
    run_tailor,
)
from app.prompts import RETRY_MARKER, tailor_system
from app.schemas import Company, TailorResult

from .test_validate import make_output
from .test_writing_quality import five_jobs

TODAY = date(2026, 10, 12)


# ------------------------------------------------------------------ career path
def test_long_stays_get_bullets_before_short_recent_ones():
    profile = five_jobs(5)
    order = extra_order(profile, bullet_caps(profile.companies, TODAY), TODAY)
    assert order.index((2, 3)) < order.index((1, 3))  # Orbital Education (3 yrs 1 mo) before Acquire.com (8 mo)
    assert bullet_budget(profile, today=TODAY) == [5, 3, 5, 4, 3]  # two pages hold every cap
    older = [Company(name=f"Old {i}", start=f"Jun {2016 - 2 * i}", end=f"May {2018 - 2 * i}") for i in range(5)]
    long = profile.model_copy(update={"companies": profile.companies + older})
    assert bullet_budget(long, today=TODAY) == [5, 3, 5, 4, 3, 3, 3, 2, 2, 2]  # full pages: the oldest give way


def test_stay_and_experience_before_are_computed_with_overlaps_counted_once():
    jobs = [
        Company(name="Now", start="Jan 2024", end="Present"),
        Company(name="Side gig", start="Jun 2022", end="Mar 2024"),  # overlaps both neighbours
        Company(name="First", start="Jan 2021", end="Dec 2022"),
    ]
    assert months_phrase(stay_months(jobs[0], TODAY)) == "2 yrs 10 mo"
    assert experience_before(jobs, 0, TODAY) == 36  # Jan 2021 - Dec 2023, the overlap counted once
    assert experience_before(jobs, 2, TODAY) == 0
    assert months_phrase(0) == "0 mo" and months_phrase(12) == "1 yr" and months_phrase(None) == ""


def test_title_growth_warning(profile):
    # conftest profile: Northwind (Mar 2022 - Present), Contoso (Jun 2019 - Feb 2022, 33 months), Fabrikam
    titles = {0: "Senior Data Engineer", 1: "Data Engineer", 2: "Data Engineer"}
    out = make_output()
    for exp in out.sections.experience:
        exp.job_title = titles[exp.company_index]
    warnings = validate.title_growth_problems(validate.build_resume(profile, out))
    assert len(warnings) == 1 and warnings[0].startswith("Contoso Systems: 2 yrs 9 mo with the same title")
    for exp in out.sections.experience:
        exp.job_title = "Data Engineer"
    warnings = validate.ats_lint(validate.build_resume(profile, out))
    assert sum("career progression" in w or "same title" in w for w in warnings) == 1  # one warning, not two
    titles = {0: "Senior Data Engineer", 1: "Data Engineer II", 2: "Data Engineer"}
    for exp in out.sections.experience:
        exp.job_title = titles[exp.company_index]
    assert validate.title_growth_problems(validate.build_resume(profile, out)) == []


# ------------------------------------------------------------------ matching and coverage
def test_mentions_tolerates_plural_and_hyphen_but_keeps_boundaries():
    assert validate.mentions("Built REST APIs", "REST API") and validate.mentions("a REST API", "REST APIs")
    assert validate.mentions("event driven design", "event-driven")
    assert validate.mentions("one data pipeline", "Data Pipelines")
    assert not validate.mentions("PostgreSQL", "SQL") and not validate.mentions("Google", "Go")
    assert not validate.mentions("C++", "C") and not validate.mentions("aws", "AWS")
    assert not validate.mentions("schema changes", "schema design")


def with_must_haves(out, terms):
    out.jd_analysis.must_have_skills = terms
    return out


def test_must_haves_absent_or_only_listed(profile):
    out = with_must_haves(make_output(), ["Python", "SQL", "Kafka", "agentic development"])
    out.sections.skills[0].skills.append("Kafka")
    resume = validate.build_resume(profile, out)
    # Python and SQL are in the summary; Kafka only in Skills; agentic development nowhere
    assert validate.unshown_must_haves(resume, out.jd_analysis) == (["agentic development"], ["Kafka"])
    coverage = validate.keyword_coverage(resume, out.jd_analysis)
    assert coverage.skills_only == ["Kafka"] and "agentic development" in coverage.missing


def test_posting_flags_are_info_grouped_and_skip_what_the_hints_back_up(profile):
    out = with_must_haves(make_output(), ["Python", "SQL", "schema design", "incident response"])
    out.sections.experience[0].bullets = [
        ("Led schema design and incident response for the billing data platform, so on-call engineers resolved "
         "outages faster and with less guesswork."),
        "Rebuilt reporting jobs in Python with clear ownership, so analysts got dependable numbers every morning.",
    ]
    resume = validate.build_resume(profile, out)
    flags = validate.posting_match_flags(resume, out.jd_analysis, hints="I used Python and SQL daily.")
    assert len(flags) == 1 and flags[0].level == "info" and flags[0].section == "experience:0"
    assert "schema design, incident response" in flags[0].reason and "Python" not in flags[0].reason


def test_run_checks_warns_and_orders_flags(profile):
    out = with_must_haves(make_output(), ["Python", "Kafka"])
    out.sections.skills[0].skills.append("Kafka")
    out.sections.experience[0].bullets = [
        ("Cut nightly runtime by 23.4% by rewriting Python jobs on Spark with incremental loads and clear alerts "
         "for the data team."),
        "Improved things",
    ]
    result = TailorResult(requested_model="m", resume=validate.build_resume(profile, out), analysis=out.jd_analysis)
    run_checks(result, "a job description")
    assert any(w.startswith("Required by the posting but not shown") and "Kafka" in w for w in result.ats_warnings)
    levels = [f.level for f in result.flags]
    assert levels[0] == "verify" and "Added to match the posting" in result.flags[1].reason


def test_old_runs_without_skills_only_still_load():
    old = TailorResult(requested_model="m").model_dump()
    del old["coverage"]["skills_only"]
    assert TailorResult.model_validate(old).coverage.skills_only == []


# ------------------------------------------------------------------ the one rewrite pass
def test_coverage_feedback_threshold(profile):
    out = with_must_haves(make_output(), ["Python", "SQL"])
    resume = validate.build_resume(profile, out)
    assert coverage_feedback(resume, out.jd_analysis) is None
    out = with_must_haves(make_output(), ["Python", "SQL", "agentic development"])
    text = coverage_feedback(validate.build_resume(profile, out), out.jd_analysis)
    assert MIN_UNCOVERED_FOR_RETRY == 1 and '"agentic development"' in text and "word for word" in text


class Uncovered(LLMClient):
    """Leaves a must-have out of the first draft (or every draft), then covers it once told."""

    def __init__(self, inner, always=False):
        self.inner, self.always, self.prompts = inner, always, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_revision":  # the fix (the analysis is final, so nothing to add there)
            if not self.always:
                draft = kwargs["context"]["draft"]["summary"]
                result.data["summary"] = draft + " Applies agentic development with AI coding agents in a team."
            return result
        result.data["jd_analysis"]["must_have_skills"].append("agentic development")
        return result


def test_a_missing_must_have_is_covered_in_the_one_rewrite(router, db, registry, profile, jd):
    client = Uncovered(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    retry = client.prompts[1]
    assert '"agentic development"' in retry and "<previous_draft>" in retry and "company_index 0" in retry
    assert "agentic development" not in result.coverage.missing
    assert any("agentic development" in f.reason for f in result.flags if f.level == "info")


def test_never_more_than_one_extra_call_for_coverage(router, db, registry, profile, jd):
    # demo-detailed overflows the page, so the shorten call runs too; it must not ask for coverage again
    client = Uncovered(FakeClient(registry.get("demo-detailed")), always=True)
    router.overrides["demo-detailed"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-detailed"))
    assert result.status == "ok" and len(client.prompts) <= 3
    assert sum(RETRY_MARKER in p for p in client.prompts) == 1
    assert any("agentic development" in w for w in result.ats_warnings)  # still visible to the user


# ------------------------------------------------------------------ titles never step down
def _with_titles(titles):
    from app.schemas import ExperienceOut

    out = make_output(n=1)
    out.sections.experience = [
        ExperienceOut(company_index=i, job_title=title, bullets=["Built things", "Improved things"])
        for i, title in enumerate(titles)
    ]
    return validate.build_resume(five_jobs(len(titles)), out)


def test_a_title_that_steps_down_is_raised_to_the_level_before_it():
    # the reported case: Orbital 'Software Engineer II', then Acquire.com plain 'Software Engineer'
    resume = _with_titles(["Backend Engineer", "Software Engineer", "Software Engineer II", "Software Engineer",
                           "Junior Software Engineer"])
    notes = validate.fix_title_regressions(resume)
    assert [e.job_title for e in resume.experience] == [
        "Backend Engineer", "Software Engineer II", "Software Engineer II", "Software Engineer", "Junior Software Engineer"]
    assert notes == [("Acquire.com: title changed from 'Software Engineer' to 'Software Engineer II' so the career "
                      "path never steps down.")]
    assert validate.title_growth_problems(resume) == [] or "step down" not in validate.title_growth_problems(resume)[0]


def test_the_most_recent_title_is_kept_and_the_older_one_lowered():
    resume = _with_titles(["Software Engineer", "Software Engineer II", "Associate Software Engineer"])
    validate.fix_title_regressions(resume)
    assert [e.job_title for e in resume.experience] == [
        "Software Engineer", "Software Engineer", "Associate Software Engineer"]  # the target title stays


def test_manual_step_down_is_warned_and_good_paths_are_left_alone():
    good = ["Backend Engineer", "Software Engineer II", "Software Engineer", "Associate Software Engineer",
            "Junior Software Engineer"]
    resume = _with_titles(good)
    assert validate.fix_title_regressions(resume) == [] and [e.job_title for e in resume.experience] == good
    warned = validate.title_growth_problems(_with_titles(["Data Analyst", "Senior Data Engineer", "Data Engineer"]))
    assert len(warned) == 1 and "step down" in warned[0]


class TitlesOff(LLMClient):
    """Writes titles that step down (first draft only, or every draft with flat=True: the same title everywhere)."""

    def __init__(self, inner, flat=False):
        self.inner, self.flat, self.prompts = inner, flat, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] != "tailor_output":  # a fix only changes what it lists
            return result
        experience = result.data["sections"]["experience"]
        if self.flat and RETRY_MARKER not in kwargs["user"]:
            for exp in experience:
                exp["job_title"] = "Data Engineer"
        elif not self.flat:
            titles = ["Backend Engineer", "Software Engineer", "Software Engineer II"]  # II -> plain: a step down
            for exp, title in zip(experience, titles, strict=False):
                exp["job_title"] = title
        return result


def test_step_down_titles_are_fixed_without_an_extra_call(router, db, registry, jd):
    # the reported case: Orbital Education 'Software Engineer II', then Acquire.com (8 months) 'Software Engineer'
    client = TitlesOff(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=five_jobs(3), profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 0 and len(client.prompts) == 1
    assert [e.job_title for e in result.resume.experience] == ["Backend Engineer", "Software Engineer II", "Software Engineer II"]
    assert any("never steps down" in n for n in result.notes)


def test_flat_titles_are_rewritten_in_the_one_rewrite_pass(router, db, registry, profile, jd):
    client = TitlesOff(FakeClient(registry.get("demo-concise")), flat=True)
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    assert "do not read as a believable career path" in client.prompts[1]
    assert len({e.job_title for e in result.resume.experience}) > 1


# ------------------------------------------------------------------ career flow from the job description
def test_the_career_plan_is_written_before_the_sections():
    from app.llm.schema import strict_json_schema
    from app.schemas import TailorOutput

    schema = strict_json_schema(TailorOutput)
    assert list(schema["properties"]) == ["jd_analysis", "career_plan", "sections"]
    assert "career_plan" in schema["required"]
    step = schema["properties"]["career_plan"]["items"]
    assert step["required"] == ["company_index", "focus", "scope"]


def test_runs_keep_the_plan_and_old_runs_still_load(router, db, profile, jd):
    from app.schemas import TailorResult

    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert [s.company_index for s in result.career_plan] == [0, 1, 2]
    assert [s.scope for s in result.career_plan] == ["owned", "designed", "built"]
    old = result.model_dump()
    del old["career_plan"]
    assert TailorResult.model_validate(old).career_plan == []


def test_tools_that_did_not_exist_yet_are_flagged():
    resume = _with_titles(["Backend Engineer", "Software Engineer II", "Software Engineer", "Software Engineer",
                           "Junior Software Engineer"])
    # five_jobs: Lindy (current), Acquire.com (ends Feb 2025), Orbital (ends Jun 2024), Q2 (ends May 2021), Earlier Co (ends Apr 2020)
    resume.experience[0].bullets = ["Use agentic development and OpenTofu for every release of the platform"]
    resume.experience[2].bullets = ["Moved the infrastructure to OpenTofu and Helm charts on Amazon EKS"]
    resume.experience[3].bullets = ["Managed AWS with OpenTofu and took the helm of the release process"]
    resume.experience[4].bullets = ["Explored agentic development and GitHub Actions for the mobile team"]
    problems = validate.era_problems(resume)
    assert problems == [
        "Q2 Holdings (May 2020 - May 2021) mentions OpenTofu, which only became common in 2024.",
        "Earlier Co (Jun 2018 - Apr 2020) mentions agentic: AI tools belong only to the most recent one or two jobs.",
    ]  # fine: the current job, OpenTofu at Orbital (ends 2024), 'took the helm', GitHub Actions at a job ending 2020


def test_ai_tools_only_at_the_latest_one_or_two_jobs_and_once_they_were_popular():
    titles = ["Backend Engineer", "Software Engineer II", "Software Engineer", "Software Engineer",
              "Junior Software Engineer"]
    # five_jobs: Lindy (current), Acquire.com (ends Feb 2025), Orbital (ends Jun 2024), Q2 (ends May 2021), Earlier Co
    resume = _with_titles(titles)
    resume.experience[0].bullets = ["Use agentic development and LLM evaluations every day"]
    resume.experience[1].bullets = ["Built LLM features and coding agent workflows for the listings team"]
    resume.experience[2].bullets = ["Added an LLM summary to the reporting service for teachers"]
    problems = validate.era_problems(resume)
    assert problems == [
        "Orbital Education (Jun 2021 - Jun 2024) mentions LLM: AI tools belong only to the most recent one or two jobs."
    ]  # LLM (2023) and coding agents (2025) are fine at a job ending 2025; anything goes at the current job
    resume.profile.companies[1].end = "Dec 2024"  # the 2nd job now ends before coding agents were popular
    assert validate.era_problems(resume)[0] == (
        "Acquire.com (Jul 2024 - Dec 2024) mentions coding agent, which only became common in 2025.")


def test_other_tools_count_from_when_they_became_common():
    resume = _with_titles(["Backend Engineer", "Software Engineer II", "Software Engineer"])
    resume.profile.companies[2].end = "Dec 2019"
    resume.experience[2].bullets = ["Set up GitHub Actions pipelines for every service"]
    assert validate.era_problems(resume) == [
        "Orbital Education (Jun 2021 - Dec 2019) mentions GitHub Actions, which only became common in 2020."]
    resume.profile.companies[2].end = "Dec 2020"
    assert validate.era_problems(resume) == []


class Anachronism(LLMClient):
    """Puts OpenTofu in the oldest job on the first draft only."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if RETRY_MARKER not in kwargs["user"]:
            result.data["sections"]["experience"][-1]["bullets"][0] = (
                "Managed every environment with OpenTofu modules and reviewed changes with the team before each release")
        return result


def test_a_tool_from_the_future_is_rewritten_in_the_one_rewrite_pass(router, db, registry, profile, jd):
    client = Anachronism(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise"))
    assert result.status == "ok" and result.retries == 1 and len(client.prompts) == 2
    assert "could not really have used" in client.prompts[1] and "OpenTofu" in client.prompts[1]
    assert validate.era_problems(result.resume) == []


def test_prompt_states_the_career_flow_rules():
    text = tailor_system("estimate")
    assert "CAREER FLOW" in text and "career_plan" in text and "stepping stones" in text
    assert "Only what was really in common use" in text and "never decreases toward the present" in text
    assert "one or two most recent employers" in text and "agentic development from 2025" in text
