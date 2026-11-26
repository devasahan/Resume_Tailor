"""A career for an AI post (prompt v42): titles of their time, and middle jobs whose titles fit their work. Titles
named for AI became common only in 2023, with LLMs, so a job that ended before 2023 is never an 'AI Engineer': the app
writes 'Machine Learning Engineer' there ('ML Platform Engineer' for 'AI Platform Engineer') itself, at no cost and with
a note, unless the person's notes for that job name the title; the current job always keeps the post's title. The
demo's five-job career for a 'Senior AI Engineer' post reads Senior AI Engineer, AI Engineer, Data Engineer, Software
Engineer and Junior Software Engineer. Every person, employer and product here is invented."""
import asyncio
import time

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient, _title_ladder
from app.pipeline import validate
from app.pipeline.tailor import first_draft_notes, run_tailor
from app.prompts import PROMPT_VERSION, RETRY_MARKER, tailor_system
from app.schemas import (
    MAX_PROJECT_CHARS,
    Company,
    Contact,
    Education,
    ExperienceOut,
    Profile,
    TailorRevision,
)

from .test_title_and_language import drop_language, posting
from .test_usual_title import AI_DEV_POST, LONG_TEXTS, built_twice, drafted
from .test_validate import make_output
from .test_writing_quality import five_jobs

AI_ENGINEER_POST = (
    "Senior AI Engineer at Contoso Learning. You will build our AI features: agents, retrieval-augmented generation "
    "(RAG) and LLM evaluation. Required: Python, LLM evaluation, RAG, PostgreSQL, CI/CD, AWS. Nice to have: "
    "Kubernetes, pytest."
)
ERA_REASON = "titles named for AI became common only in 2023, after this job ended."


def career(*jobs: tuple[str, str, str], notes: dict[int, str] | None = None) -> Profile:
    """Taylor Quinn's career, the latest job first: (employer, start, end) for each job; `notes`: the project notes
    typed for a job, by its number."""
    companies = [Company(name=name, start=start, end=end, mode="Remote", projects=(notes or {}).get(index, ""))
                 for index, (name, start, end) in enumerate(jobs)]
    return Profile(name="Taylor Quinn", contact=Contact(email="taylor@example.com", location="Austin, TX"),
                   companies=companies, education=[Education(school="Tailspin University", degree="B.S.",
                                                             field="Computer Science", end="2016")])


def five_ai_jobs() -> Profile:
    """The first preview career: five jobs, the one before the current one ended in December 2023."""
    return career(("Northwind Labs", "Jan 2024", "Present"), ("Bluefin Commerce", "Mar 2022", "Dec 2023"),
                  ("Contoso", "Jun 2020", "Feb 2022"), ("Fabrikam", "Jul 2018", "May 2020"),
                  ("Tailspin Toys", "Jun 2016", "Jun 2018"))


def four_jobs_ending_2022() -> Profile:
    """The second preview career: four jobs, the one before the current one ended in December 2022."""
    return career(("Northwind Labs", "Feb 2023", "Present"), ("Contoso", "Jun 2020", "Dec 2022"),
                  ("Fabrikam", "Jul 2018", "May 2020"), ("Tailspin Toys", "Jun 2016", "Jun 2018"))


def resume_of(*rows: tuple[str, str, str, str], notes: dict[int, str] | None = None):
    """A resume of these jobs, the latest first: (title, employer, start, end) for each."""
    out, bullets = make_output(n=1), ["Built things", "Fixed things"]
    out.sections.experience = [ExperienceOut(company_index=index, job_title=title, bullets=bullets)
                               for index, (title, *_dates) in enumerate(rows)]
    return validate.build_resume(career(*[tuple(dates) for _title, *dates in rows], notes=notes), out)


def titles_of(resume) -> list[str]:
    return [exp.job_title for exp in resume.experience]


def era_note(company: str, before: str, after: str) -> str:
    return f"{company}: title changed from '{before}' to '{after}': {ERA_REASON}"


# ------------------------------------------------------------------ the title a job had before 2023
@pytest.mark.parametrize("title,earlier", [
    ("AI Engineer", "Machine Learning Engineer"),
    ("Artificial Intelligence Engineer", "Machine Learning Engineer"),
    ("LLM Engineer", "Machine Learning Engineer"),
    ("LLMs Engineer", "Machine Learning Engineer"),
    ("Generative AI Engineer", "Machine Learning Engineer"),
    ("GenAI Engineer", "Machine Learning Engineer"),
    ("Gen AI Engineer", "Machine Learning Engineer"),
    ("AI/ML Engineer", "Machine Learning Engineer"),
    ("ML/AI Engineer", "Machine Learning Engineer"),
    ("AI-ML Engineer", "Machine Learning Engineer"),
    ("AI & ML Engineer", "Machine Learning Engineer"),
    ("AI and Machine Learning Engineer", "Machine Learning Engineer"),
    ("Machine Learning and AI Engineer", "Machine Learning Engineer"),
    ("AI Developer", "Machine Learning Engineer"),  # 'Developer' takes the usual noun of the field
    ("AI Platform Engineer", "ML Platform Engineer"),
    ("AI Infrastructure Engineer", "ML Infrastructure Engineer"),
    ("AI/ML Platform Engineer", "ML Platform Engineer"),
    ("ML/AI Platform Engineer", "ML Platform Engineer"),
    ("AI/LLM Platform Engineer", "ML Platform Engineer"),
    ("Data and AI Engineer", "Data and ML Engineer"),
    ("Platform and AI/ML Engineer", "Platform and ML Engineer"),
    ("Cloud AI Engineer", "Cloud ML Engineer"),
    ("NLP/AI Engineer", "NLP Engineer"),
    ("LLMOps Engineer", "MLOps Engineer"),
    ("LLM Ops Engineer", "ML Ops Engineer"),
    # levels before the role, and what follows it, stay
    ("Senior AI Engineer", "Senior Machine Learning Engineer"),
    ("Sr. AI Engineer", "Sr. Machine Learning Engineer"),
    ("Staff AI Platform Engineer", "Staff ML Platform Engineer"),
    ("AI Engineer II", "Machine Learning Engineer II"),
    ("AI Engineer Lead", "Machine Learning Engineer Lead"),
    ("AI Engineer, Platform", "Machine Learning Engineer, Platform"),
    ("Senior AI Engineer (Remote)", "Senior Machine Learning Engineer (Remote)"),
    ("Senior AI Engineer / Tech Lead", "Senior Machine Learning Engineer / Tech Lead"),
    ("AI Engineer: LLM Apps", "Machine Learning Engineer: LLM Apps"),
    ("AI Engineer | Python | SQL", "Machine Learning Engineer | Python | SQL"),
    ("AI Engineer - Search", "Machine Learning Engineer - Search"),
    ("Senior AI Developer, Platform", "Senior Machine Learning Engineer, Platform"),
    # the words put in follow the title's letter case, the noun keeps its own
    ("ai engineer", "machine learning engineer"),
    ("AI engineer", "machine learning engineer"),
    ("AI ENGINEER", "MACHINE LEARNING ENGINEER"),
    ("senior ai platform engineer", "senior ml platform engineer"),
    ("AI Platform developer", "ML Platform engineer"),
    ("ai/ml engineer", "machine learning engineer"),
])
def test_a_title_named_for_ai_becomes_the_title_that_work_had_before_2023(title, earlier):
    assert validate.earlier_title(title) == earlier
    assert validate.earlier_title(earlier) == earlier  # once: nothing named for AI is left


@pytest.mark.parametrize("title", [
    "Machine Learning Engineer", "ML Engineer", "MLOps Engineer", "NLP Engineer", "Computer Vision Engineer",
    "Deep Learning Engineer", "Data Engineer", "Platform Engineer", "Backend Engineer", "Software Engineer",
    "Senior Software Engineer", "Data Developer", "Conversational AI Engineer", "Voice AI Engineer", "AIOps Engineer",
    "AI Ops Engineer", "AI Programmer", "AI Developer Advocate", "AI Engineering Manager", "Applied AI Engineer",
    "Engineer, AI", "AI Research Scientist", "AI Engineers", "Head of AI", "",
])
def test_a_title_not_named_for_ai_stays(title):
    assert validate.earlier_title(title) == title


# ------------------------------------------------------------------ dates
def test_titles_named_for_ai_date_from_the_year_llms_became_common():
    assert validate.AI_TITLE_YEAR == validate.AI_TOOLS["LLM"] == 2023


@pytest.mark.parametrize("end,year", [
    ("Dec 2022", 2022), ("2022", 2022), ("12/2022", 2022), ("2022-12", 2022), ("Jan 2023", 2023), ("2023", 2023),
    ("Present", None), ("present.", None), ("Now", None), ("Current", None), ("", None), ("someday", None),
])
def test_the_year_a_job_ended(end, year):
    assert validate.end_year(end) == year


@pytest.mark.parametrize("end", ["Dec 2022", "2022", "12/2022", "Sept 2021", "2019"])
def test_a_job_that_ended_before_2023_takes_the_title_of_its_time(end):
    assert validate.dated_title("Senior AI Engineer", end) == "Senior Machine Learning Engineer"


@pytest.mark.parametrize("end", ["Jan 2023", "2023", "Present", "", "someday"])
def test_a_job_that_ended_in_2023_or_later_is_held_now_or_has_no_date_keeps_its_title(end):
    assert validate.dated_title("Senior AI Engineer", end) == "Senior AI Engineer"


# ------------------------------------------------------------------ the fix itself, with its note
def test_an_older_ai_title_is_renamed_with_a_note():
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"),
                       ("Software Engineer", "Fabrikam", "Jul 2018", "May 2020"))
    notes = validate.era_title_fixes(resume)
    assert titles_of(resume) == ["Senior AI Engineer", "Machine Learning Engineer", "Software Engineer"]
    assert notes == [("Contoso: title changed from 'AI Engineer' to 'Machine Learning Engineer': titles named for AI "
                      "became common only in 2023, after this job ended.")]
    assert validate.era_title_fixes(resume) == []  # a second pass finds nothing to change


def test_every_older_job_goes_by_its_own_dates():
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2024", "Present"),
                       ("AI Engineer", "Bluefin Commerce", "Mar 2022", "Jan 2023"),  # ended in 2023: kept
                       ("AI Platform Engineer", "Contoso", "Jun 2020", "Feb 2022"),
                       ("LLM Engineer", "Fabrikam", "Jul 2018", "May 2020"),
                       ("AI Engineer", "Tailspin Toys", "Jun 2016", ""))  # an end that cannot be read: kept
    notes = validate.era_title_fixes(resume)
    assert titles_of(resume) == ["Senior AI Engineer", "AI Engineer", "ML Platform Engineer",
                                 "Machine Learning Engineer", "AI Engineer"]
    assert notes == [era_note("Contoso", "AI Platform Engineer", "ML Platform Engineer"),
                     era_note("Fabrikam", "LLM Engineer", "Machine Learning Engineer")]


def test_the_current_job_keeps_the_posts_title_whatever_its_dates():
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Mar 2019", "Present"),  # held since 2019
                       ("AI Engineer", "Contoso", "Jun 2016", "Feb 2019"))
    validate.era_title_fixes(resume)
    assert titles_of(resume) == ["Senior AI Engineer", "Machine Learning Engineer"]
    # no job is marked 'Present': the most recent one is the current job, also when it ended in December 2022
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2021", "Dec 2022"),
                       ("AI Engineer", "Contoso", "Jun 2018", "Dec 2020"))
    notes = validate.era_title_fixes(resume)
    assert titles_of(resume) == ["Senior AI Engineer", "Machine Learning Engineer"]
    assert notes == [era_note("Contoso", "AI Engineer", "Machine Learning Engineer")]


def test_a_job_listed_above_the_current_one_goes_by_its_own_dates():
    rows = [("AI Engineer", "Northwind Labs", "Jan 2022", "Dec 2022"),  # a contract, listed first: it ended later
            ("Senior AI Engineer", "Bluefin Commerce", "Mar 2019", "Present"),  # the job held now
            ("Software Engineer", "Contoso", "Jul 2016", "Feb 2019")]
    resume = resume_of(*rows)
    notes = validate.era_title_fixes(resume)
    assert titles_of(resume) == ["Machine Learning Engineer", "Senior AI Engineer", "Software Engineer"]
    assert notes == [era_note("Northwind Labs", "AI Engineer", "Machine Learning Engineer")]
    resume = resume_of(("AI Engineer", "Northwind Labs", "Jan 2026", "Jun 2026"), *rows[1:])
    assert validate.era_title_fixes(resume) == [] and resume.experience[0].job_title == "AI Engineer"


@pytest.mark.parametrize("notes", [
    "Joined as an AI Engineer on Search.",
    "Search ranking (2021): joined as an ai engineer, then led the ranking rebuild.",
    "Hired as an AI Developer for the search team.",  # either noun: the same role
    "- Senior AI Engineer on the Search team\n- Built the ranking service in Python",
])
def test_a_title_the_persons_notes_name_stays(notes):
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("Senior AI Engineer", "Contoso", "Jun 2020", "Dec 2022"), notes={1: notes})
    assert validate.era_title_fixes(resume) == []
    assert titles_of(resume) == ["Senior AI Engineer", "Senior AI Engineer"]


@pytest.mark.parametrize("notes", [
    "Built an LLM prototype for search.",  # AI work, not the title
    "Ranking with AI: the search team's models.",
    "Worked with the AI Platform team on Search.",
    "Mentored AI engineering interns.",
    "Source: https://contoso.example.com/ai-engineer",  # a source line is not the person's own words
])
def test_notes_that_do_not_name_the_title_keep_nothing(notes):
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("Senior AI Engineer", "Contoso", "Jun 2020", "Dec 2022"), notes={1: notes})
    assert validate.era_title_fixes(resume) == [
        era_note("Contoso", "Senior AI Engineer", "Senior Machine Learning Engineer")]


@pytest.mark.parametrize("target,general,older,renamed", [
    ("Senior AI Engineer", "AI Engineer", "AI Engineer", "Machine Learning Engineer"),
    ("Senior AI Platform Engineer", "AI Platform Engineer", "AI Platform Engineer", "ML Platform Engineer"),
    ("Senior LLM Engineer", "LLM Engineer", "LLM Engineer", "Machine Learning Engineer"),
])
def test_a_renamed_title_before_the_current_one_is_still_in_the_posts_field(target, general, older, renamed):
    resume = resume_of((target, "Northwind Labs", "Jan 2023", "Present"), (older, "Contoso", "Jun 2020", "Dec 2022"),
                       ("Software Engineer", "Fabrikam", "Jul 2018", "May 2020"))
    validate.era_title_fixes(resume)
    assert titles_of(resume) == [target, renamed, "Software Engineer"]
    assert validate.title_path_problems(resume, posting(target, general)) == ([], False)
    assert validate.title_growth_problems(resume) == [] and validate.era_problems(resume) == []


@pytest.mark.parametrize("post,general,titles,older,before", [
    # the usual noun first ('AI Developer' -> 'AI Engineer'), then the title of its time
    ("Senior AI Developer", "AI Developer", ["Senior AI Developer", "AI Developer", "Software Developer"],
     ["Machine Learning Engineer", "Software Developer"], "AI Engineer"),
    # a post that leaves an older Developer title's noun alone still has it take the title of its time
    ("Senior AI Platform Engineer", "AI Platform Engineer", ["Senior AI Platform Engineer", "AI Developer",
                                                             "Software Developer"],
     ["Machine Learning Engineer", "Software Developer"], "AI Developer"),
    ("Senior AI Developer Advocate", "Developer Relations Engineer", ["Senior AI Developer Advocate", "AI Developer",
                                                                      "Data Developer"],
     ["Machine Learning Engineer", "Data Developer"], "AI Developer"),
])
def test_build_gives_an_older_ai_title_the_name_of_its_time_after_the_usual_noun(profile, post, general, titles,
                                                                                  older, before):
    """build()'s title steps (tests/test_usual_title.py's drafted) on conftest's profile, whose Contoso Systems job
    ended in February 2022: test_usual_title moves that end to 2023 to keep its own point; this is the era's case."""
    resume, notes = drafted(profile, posting(post, general), f"{titles[0]} with 9+ years of Python work.", titles,
                            headline=titles[0])
    assert titles_of(resume)[1:] == older
    assert notes == [era_note("Contoso Systems", before, "Machine Learning Engineer")]


def test_build_notes_the_level_and_the_era_both(profile):
    resume, notes = drafted(profile, posting("AI Developer II", "AI Developer"),
                            "AI Developer II with 7+ years of Python work.",
                            ["Senior AI Developer", "AI Developer", "Software Developer"], headline="AI Developer II")
    assert titles_of(resume) == ["Senior AI Engineer", "Machine Learning Engineer", "Software Developer"]
    assert notes == [era_note("Contoso Systems", "AI Engineer", "Machine Learning Engineer"),
                     ("Summary: wrote 'Senior AI Engineer' instead of the posting's level 'AI Developer II', which is "
                      "not your title.")]


@pytest.mark.parametrize("current", ["Senior AI Developer", "Senior AI Developer / Tech Lead", "Lead, AI Developer"])
def test_the_rewrite_pass_draft_of_a_renamed_title_moves_nothing(profile, current):
    first, again = built_twice(profile, posting("Senior AI Developer", "AI Developer"), current, current)
    assert first.experience[1].job_title == "Machine Learning Engineer"
    assert titles_of(again) == titles_of(first)
    assert (again.headline, again.summary) == (first.headline, first.summary)


def test_the_first_drafts_note_stays_while_the_title_does():
    draft = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                      ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"))
    notes = validate.era_title_fixes(draft)
    fixed = draft.model_copy(deep=True)
    assert first_draft_notes(notes, draft, fixed) == notes
    fixed.experience[1].job_title = "Data Engineer"  # the rewrite pass chose another title
    assert first_draft_notes(notes, draft, fixed) == []


# ------------------------------------------------------------------ whole runs: build()'s order
class Titled(LLMClient):
    """The demo, whose first draft has these job titles; with `language` False its current job names no language, so
    the rewrite pass is asked for; `again`: the titles its revision writes again ({job number: title}); `empty`: its
    revision changes nothing. `draft` keeps the first draft as the rewrite pass was shown it."""

    def __init__(self, inner, titles, language=True, again=None, empty=False):
        self.inner, self.titles, self.language, self.again, self.empty = inner, titles, language, again or {}, empty
        self.draft = None

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            sections = result.data["sections"]
            for exp, title in zip(sections["experience"], self.titles):
                exp["job_title"] = title or exp["job_title"]
            if not self.language:
                drop_language(sections)
        elif kwargs["schema_name"] == "tailor_revision":
            self.draft = kwargs["context"]["draft"]
            if self.empty:
                result.data = TailorRevision().model_dump()
            result.data["titles"] += [{"company_index": index, "job_title": title} for index, title in
                                      self.again.items()]
        return result


def run(router, db, registry, person, jd=AI_ENGINEER_POST, **kwargs):
    client = Titled(FakeClient(registry.get("demo-concise")), **kwargs) if kwargs else None
    if client is not None:
        router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=person, profile_id=None, jd=jd, model_id="demo-concise",
                                    save=False))
    assert result.status == "ok", result.error
    return client, result


def test_the_level_fix_runs_first_so_an_older_title_is_compared_as_the_same_role(router, db, registry):
    person = career(("Northwind Labs", "Feb 2023", "Present"), ("Contoso", "Jun 2020", "Dec 2022"))
    jd = AI_ENGINEER_POST.replace("Senior AI Engineer", "AI Engineer", 1)
    _client, result = run(router, db, registry, person, jd=jd, titles=["AI Engineer", "Senior AI Engineer"])
    assert titles_of(result.resume) == ["AI Engineer", "Machine Learning Engineer"]
    assert result.resume.headline == "AI Engineer" and result.retries == 0
    assert validate.title_growth_problems(result.resume) == []
    # both notes are true: the level came down first, then the title took the name of its time
    assert ("Contoso: title changed from 'Senior AI Engineer' to 'AI Engineer' so the career path never steps down."
            in result.notes)
    assert era_note("Contoso", "AI Engineer", "Machine Learning Engineer") in result.notes


def test_a_renamed_title_that_meets_the_same_role_at_a_higher_level_is_raised(router, db, registry):
    person = career(("Northwind Labs", "Feb 2023", "Present"), ("Contoso", "Jun 2021", "Dec 2022"),
                    ("Fabrikam", "Jul 2018", "May 2021"))
    _client, result = run(router, db, registry, person,
                          titles=["Senior AI Engineer", "AI Engineer", "Machine Learning Engineer II"])
    assert titles_of(result.resume) == ["Senior AI Engineer", "Machine Learning Engineer II",
                                        "Machine Learning Engineer II"]
    assert result.retries == 0 and validate.title_growth_problems(result.resume) == []
    assert era_note("Contoso", "AI Engineer", "Machine Learning Engineer") in result.notes
    assert ("Contoso: title changed from 'Machine Learning Engineer' to 'Machine Learning Engineer II' so the career "
            "path never steps down.") in result.notes


def test_a_model_that_writes_ai_engineer_before_2023_gets_it_renamed_once(router, db, registry, profile):
    _client, result = run(router, db, registry, profile, titles=["", "AI Engineer", ""])
    assert titles_of(result.resume) == ["Senior AI Engineer", "Machine Learning Engineer", "Software Engineer"]
    assert result.retries == 0
    assert result.notes.count(era_note("Contoso Systems", "AI Engineer", "Machine Learning Engineer")) == 1


def test_with_a_rewrite_pass_the_title_holds_and_is_noted_once(router, db, registry, profile):
    # the revision writes 'AI Engineer' again: the app renames it again, and the note is shown once
    client, result = run(router, db, registry, profile, titles=["", "AI Engineer", ""], language=False,
                         again={1: "AI Engineer"})
    assert result.retries == 1 and client.draft is not None
    assert titles_of(result.resume) == ["Senior AI Engineer", "Machine Learning Engineer", "Software Engineer"]
    assert result.notes.count(era_note("Contoso Systems", "AI Engineer", "Machine Learning Engineer")) == 1
    assert validate.missing_language(result.resume, result.analysis, AI_ENGINEER_POST) is None  # the fix went in


def test_a_second_build_of_the_unchanged_draft_changes_nothing(router, db, registry, profile):
    client, result = run(router, db, registry, profile, titles=["", "AI Engineer", ""], language=False, empty=True)
    assert result.retries == 1
    draft = client.draft  # the first draft, after build(), as the rewrite pass was shown it
    assert titles_of(result.resume) == [exp["job_title"] for exp in draft["experience"]] == [
        "Senior AI Engineer", "Machine Learning Engineer", "Software Engineer"]
    assert result.resume.headline == draft["headline"] and result.resume.summary == draft["summary"]
    assert result.notes.count(era_note("Contoso Systems", "AI Engineer", "Machine Learning Engineer")) == 1


# ------------------------------------------------------------------ the demo's career for an AI post
FIVE_AI_TITLES = ["Senior AI Engineer", "AI Engineer", "Data Engineer", "Software Engineer", "Junior Software Engineer"]


@pytest.mark.parametrize("person,jd,titles", [
    ("five_jobs", AI_ENGINEER_POST, FIVE_AI_TITLES),
    ("five_ai_jobs", AI_ENGINEER_POST, FIVE_AI_TITLES),
    ("four_jobs_ending_2022", AI_ENGINEER_POST, ["Senior AI Engineer", "Machine Learning Engineer",
                                                 "Software Engineer", "Junior Software Engineer"]),
    ("conftest", AI_ENGINEER_POST, ["Senior AI Engineer", "Machine Learning Engineer", "Software Engineer"]),
    ("five_jobs", AI_DEV_POST, FIVE_AI_TITLES),  # 'Senior AI Developer': the usual noun, the same career
])
def test_the_demo_writes_an_ai_career_without_a_rewrite_pass(router, db, registry, profile, person, jd, titles):
    people = {"five_jobs": five_jobs, "five_ai_jobs": five_ai_jobs, "four_jobs_ending_2022": four_jobs_ending_2022,
              "conftest": lambda: profile}
    _client, result = run(router, db, registry, people[person](), jd=jd)
    resume = result.resume
    assert titles_of(resume) == titles and result.retries == 0
    assert resume.headline == "Senior AI Engineer" and resume.summary.startswith("Senior AI Engineer with ")
    assert not any(ERA_REASON in note for note in result.notes)  # its titles are already those of their time
    assert validate.era_problems(resume) == [] and validate.title_growth_problems(resume) == []
    assert validate.title_path_problems(resume, result.analysis) == ([], False)


@pytest.mark.parametrize("prefix,base,count,current,ladder", [
    ("Senior", "AI Engineer", 5, 0, FIVE_AI_TITLES),
    ("Senior", "AI Engineer", 6, 0, ["Senior AI Engineer", "AI Engineer", "Data Engineer", "Software Engineer",
                                     "Associate Software Engineer", "Junior Software Engineer"]),
    ("Staff", "AI Engineer", 6, 0, ["Staff AI Engineer", "Senior AI Engineer", "AI Engineer", "Data Engineer",
                                    "Software Engineer", "Junior Software Engineer"]),
    ("Senior", "Machine Learning Engineer", 5, 0, ["Senior Machine Learning Engineer", "Machine Learning Engineer",
                                                   "Data Engineer", "Software Engineer", "Junior Software Engineer"]),
    # no room for one more step: as before
    ("Senior", "AI Engineer", 4, 0, ["Senior AI Engineer", "AI Engineer", "Software Engineer",
                                     "Junior Software Engineer"]),
    ("Senior", "AI Engineer", 3, 0, ["Senior AI Engineer", "AI Engineer", "Software Engineer"]),
    ("Senior", "AI Engineer", 5, 1, ["Senior AI Engineer", "Senior AI Engineer", "AI Engineer", "Software Engineer",
                                     "Junior Software Engineer"]),
    ("Staff", "AI Engineer", 5, 0, ["Staff AI Engineer", "Senior AI Engineer", "AI Engineer", "Software Engineer",
                                    "Junior Software Engineer"]),
    # not a senior post in AI's field alone, or not an Engineer: as before
    ("", "AI Engineer", 5, 0, ["AI Engineer", "Associate AI Engineer", "Associate Software Engineer",
                               "Junior Software Engineer", "Junior Software Engineer"]),
    ("Senior", "AI Platform Engineer", 5, 0, ["Senior AI Platform Engineer", "AI Platform Engineer",
                                              "Platform Engineer", "Software Engineer", "Junior Software Engineer"]),
    ("Senior", "Data Engineer", 5, 0, ["Senior Data Engineer", "Data Engineer", "Software Engineer",
                                       "Associate Software Engineer", "Junior Software Engineer"]),
    ("", "Senior iOS Developer", 5, 0, ["Senior iOS Developer", "iOS Developer", "Software Developer",
                                        "Associate Software Developer", "Junior Software Developer"]),
    ("Senior", "AI Programmer", 5, 0, ["Senior AI Programmer", "AI Programmer", "Associate AI Programmer",
                                       "Junior AI Programmer", "Junior AI Programmer"]),
])
def test_the_demos_middle_jobs_fit_their_work(prefix, base, count, current, ladder):
    assert _title_ladder(prefix, base, count, current) == ladder


# ------------------------------------------------------------------ the prompt
def test_the_prompt_asks_for_titles_of_their_time_and_middle_jobs_that_fit_their_work():
    text = tailor_system("estimate")
    assert ("For an AI post, for example: services, APIs and tests first, then data pipelines or search, then machine "
            "learning models in production, then the first LLM features, such as retrieval and evaluation (from 2023), "
            "and the post's core, such as agents, most recently.") in text
    assert ('each step fits that job\'s own bullets (for an AI post: "Machine Learning Engineer" for a job that put '
            'models in production, "Data Engineer" for one that built data pipelines, "Backend Engineer" for one that '
            "built services and APIs), and each earlier title is at most one level lower") in text
    assert ('A title fits its dates, as the tools in its bullets do: titles named for AI ("AI Engineer", "LLM '
            'Engineer", "Generative AI Engineer") became common only in 2023, so an older job that ended before 2023 '
            'has the title that work had then ("Machine Learning Engineer", or "ML Platform Engineer" for "AI Platform '
            'Engineer").') in text
    assert ("each job names only tools that were in common use during its dates (or that its notes name) and that its "
            "title fits those dates.") in text
    assert PROMPT_VERSION == "v42"


# ------------------------------------------------------------------ long crafted titles
LONG_TITLES = [
    "ai " * 7000 + "engineer", "ai " * 7000 + "x", "ai/" * 7000 + "ml engineer", "ai and " * 3000 + "data engineer",
    "data " * 4000 + "ai engineer", "ai" + " " * 20000 + "engineer", "ai" + " " * 20000 + "x",
    ("ai" + " " * 50) * 400 + "engineer x", "senior " * 3000 + "ai engineer", "senior " * 3000 + "x",
    "ai & " * 5000 + "ml engineer", "ai - " * 5000 + "engineer", "ai -" * 5000 + " engineer x",
    "AI Engineer" + " (" * 10000, "AI Engineer" + " / " * 7000, "AI Engineer" + " | x" * 5000,
    "AI Engineer" + " [" * 10000, "(" * 20000, "/" * 20000, "[" * 20000, "-" * 20000, " " * 20000,
    "llm ops " * 3000 + "engineer", "gen ai " * 3000 + "developer x", "machine learning " * 1300 + "ai engineer",
    "AI-" * 7000 + "Engineer", "ai " * 7000 + "engineering manager", "AI/ML Platform " * 1500 + "Engineer",
    "a" * 20000, "Senior AI Engineer" + " II" * 7000, "Senior AI Engineer" + " Lead" * 5000,
    "ai platform " * 2000 + "developer advocate", "ai, " * 5000 + "engineer", "ai/ " * 7000 + "/ engineer",
]


@pytest.mark.parametrize("text", LONG_TITLES + LONG_TEXTS)
def test_long_crafted_titles_and_notes_are_read_in_time(text):
    """Every pattern of the era fix is read in linear time: one that backtracked without end would take minutes. The
    whole fix reads notes as long as a person can type for one employer (MAX_PROJECT_CHARS)."""
    started = time.perf_counter()
    validate.earlier_title(text)
    validate.dated_title(text, "Dec 2022")
    validate.end_year(text)
    validate._notes_name_role(text, text)
    validate._notes_name_role(text, "Senior AI Engineer")
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"), notes={1: text[:MAX_PROJECT_CHARS]})
    resume.experience[1].job_title = text
    validate.era_title_fixes(resume)
    resume.experience[1].job_title = "AI Engineer"  # the notes alone
    validate.era_title_fixes(resume)
    assert time.perf_counter() - started < 1.0
