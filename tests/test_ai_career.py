"""A career for an AI post (prompt v42): titles of their time, and middle jobs whose titles fit their work. Titles
named for AI became common only in 2023, with LLMs, so an older job that ended before 2023 is not an 'AI Engineer': the
app writes 'Machine Learning Engineer' there ('ML Platform Engineer' for 'AI Platform Engineer') itself, at no cost and
with a note, unless the person's notes for that job name the title, and the next level number ('Machine Learning
Engineer II') when the rename alone would repeat the title before it after a stay of 2+ years; the current job always
keeps the post's title. The demo's five-job career for a 'Senior AI Engineer' post reads Senior AI Engineer, AI
Engineer, Data Engineer, Software Engineer and Junior Software Engineer. Every person, employer and product here is
invented."""
import asyncio
import itertools
import time

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient, _title_ladder
from app.pipeline import languages, validate
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
    # still in the field of the title it came from, so the path still leads to that post (title_path_problems: 'MLOps
    # Engineer' to a 'Senior LLMOps Engineer' post)
    assert validate._in_field(validate._field_words(earlier), validate._field_words(title))


@pytest.mark.parametrize("title", [
    "Machine Learning Engineer", "ML Engineer", "MLOps Engineer", "NLP Engineer", "Computer Vision Engineer",
    "Deep Learning Engineer", "Data Engineer", "Platform Engineer", "Backend Engineer", "Software Engineer",
    "Senior Software Engineer", "Data Developer", "Conversational AI Engineer", "Voice AI Engineer", "AIOps Engineer",
    "AI Ops Engineer", "AI Programmer", "AI Developer Advocate", "AI Engineering Manager", "Applied AI Engineer",
    "Engineer, AI", "AI Research Scientist", "AI Engineers", "Head of AI", "",
])
def test_a_title_not_named_for_ai_stays(title):
    assert validate.earlier_title(title) == title


@pytest.mark.parametrize("title,earlier", [
    ("MLOps/LLMOps Engineer", "MLOps Engineer"),
    ("LLMOps/MLOps Engineer", "MLOps Engineer"),
    ("LLMOps & MLOps Engineer", "MLOps Engineer"),
    ("MLOps and LLMOps Engineer", "MLOps Engineer"),
    ("Senior MLOps / LLMOps Engineer II", "Senior MLOps Engineer II"),
    ("Senior ML Ops / LLM Ops Engineer", "Senior ML Ops Engineer"),
    ("LLMOps & ML Ops Engineer", "ML Ops Engineer"),  # the MLOps as written stays
    ("mlops/llmops engineer", "mlops engineer"),
    ("MLOPS/LLMOPS ENGINEER", "MLOPS ENGINEER"),
    ("MLOps/LLMOps Developer", "MLOps Engineer"),
    ("MLOps/LLMOps Platform Engineer", "MLOps Platform Engineer"),
    ("GenAI/MLOps/LLMOps Engineer (Remote)", "MLOps Engineer (Remote)"),
    # without MLOps, 'LLMOps' still becomes 'MLOps'
    ("LLMOps Engineer", "MLOps Engineer"),
    ("ML/LLMOps Engineer", "ML/MLOps Engineer"),
    ("LLMOps Platform Engineer", "MLOps Platform Engineer"),
])
def test_llmops_beside_mlops_goes_rather_than_repeating_it(title, earlier):
    assert validate.earlier_title(title) == earlier
    assert validate.earlier_title(earlier) == earlier  # once: a second build() changes nothing
    assert validate._in_field(validate._field_words(earlier), validate._field_words(title))  # the post's field still


def test_the_note_names_an_mlops_and_llmops_title_of_its_time():
    resume = resume_of(("Senior MLOps/LLMOps Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("MLOps/LLMOps Engineer", "Contoso", "Jun 2020", "Dec 2022"))
    assert validate.era_title_fixes(resume) == [era_note("Contoso", "MLOps/LLMOps Engineer", "MLOps Engineer")]
    assert titles_of(resume) == ["Senior MLOps/LLMOps Engineer", "MLOps Engineer"]  # the current job keeps its title
    assert validate.era_title_fixes(resume) == []  # a second pass finds nothing to change


@pytest.mark.parametrize("text", ["llm ops " * 25000 + "engineer", "llmops " * 25000 + "mlops engineer"])
def test_a_title_of_many_llmops_fields_is_read_in_linear_time(text):
    """Whether 'MLOps' is written too is looked up once per title, never once per field: a title of 25,000 LLMOps
    fields (200,000 characters) takes well under a second; looked up once per field, it took about 3.5 s."""
    started = time.perf_counter()
    validate.earlier_title(text)
    assert time.perf_counter() - started < 1.0


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


@pytest.mark.parametrize("title,notes", [
    ("Data and AI Engineer", "Joined as a Data & AI Engineer."),
    ("AI & ML Engineer", "Joined as an AI and ML engineer."),
    ("AI/ML Engineer", "Joined as an ML/AI engineer."),
    ("Machine Learning and AI Engineer", "Joined as an AI/ML engineer."),
    ("Data/AI/ML Engineer", "Joined as a Data, AI and ML engineer."),
    ("Generative AI Engineer", "Joined as a GenAI engineer."),
    ("GenAI Engineer", "Joined as a Generative AI engineer."),
    ("LLMOps Engineer", "Joined as an LLM Ops engineer."),
    ("LLM Ops Engineer", "Joined as an LLMOps engineer."),
    ("AI Engineer", "Joined as an A.I. engineer."),
    ("AI Engineer", "Joined as a GenAI engineer."),  # a GenAI engineer was an AI engineer too
])
def test_notes_that_name_the_title_in_another_form_keep_it(title, notes):
    """The same role in other words: its fields in another order or with another join, 'GenAI' for 'Generative AI',
    'LLM Ops' for 'LLMOps', 'A.I.' for 'AI'."""
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       (title, "Contoso", "Jun 2020", "Dec 2022"), notes={1: notes})
    assert validate.era_title_fixes(resume) == [] and titles_of(resume)[1] == title


@pytest.mark.parametrize("title,notes,renamed", [
    ("GenAI Engineer", "Joined as an AI engineer.", "Machine Learning Engineer"),  # the notes name a plainer title
    ("Generative AI Engineer", "Joined as an AI engineer.", "Machine Learning Engineer"),
    ("AI Engineer", "Built AI and developer tools for the search team.", "Machine Learning Engineer"),
    ("LLM Engineer", "Owned the LLM and developer tooling roadmap.", "Machine Learning Engineer"),
    ("Data and AI Engineer", "Joined as a data engineer and AI engineer.", "Data and ML Engineer"),
])
def test_notes_that_name_another_role_in_those_words_keep_nothing(title, notes, renamed):
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       (title, "Contoso", "Jun 2020", "Dec 2022"), notes={1: notes})
    assert validate.era_title_fixes(resume) == [era_note("Contoso", title, renamed)]
    assert validate.era_title_fixes(resume) == []  # a second pass changes nothing


@pytest.mark.parametrize("target,general,older,renamed", [
    ("Senior AI Engineer", "AI Engineer", "AI Engineer", "Machine Learning Engineer"),
    ("Senior AI Platform Engineer", "AI Platform Engineer", "AI Platform Engineer", "ML Platform Engineer"),
    ("Senior LLM Engineer", "LLM Engineer", "LLM Engineer", "Machine Learning Engineer"),
    ("Senior LLMOps Engineer", "LLMOps Engineer", "LLMOps Engineer", "MLOps Engineer"),
    ("Senior LLMOps Engineer", "LLMOps Engineer", "Machine Learning Engineer", "Machine Learning Engineer"),
    ("Senior LLM Ops Engineer", "LLM Ops Engineer", "LLM Ops Engineer", "ML Ops Engineer"),
])
def test_a_renamed_title_before_the_current_one_is_still_in_the_posts_field(target, general, older, renamed):
    resume = resume_of((target, "Northwind Labs", "Jan 2023", "Present"), (older, "Contoso", "Jun 2020", "Dec 2022"),
                       ("Software Engineer", "Fabrikam", "Jul 2018", "May 2020"))
    validate.era_title_fixes(resume)
    assert titles_of(resume) == [target, renamed, "Software Engineer"]
    assert validate.title_path_problems(resume, posting(target, general)) == ([], False)
    assert validate.title_growth_problems(resume) == [] and validate.era_problems(resume) == []


@pytest.mark.parametrize("post", [
    "AI Engineer", "Artificial Intelligence Engineer", "LLM Engineer", "LLMs Engineer", "Generative AI Engineer",
    "GenAI Engineer", "AI/ML Engineer", "AI & ML Engineer", "AI Platform Engineer", "AI Infrastructure Engineer",
    "AI/LLM Platform Engineer", "Data and AI Engineer", "Cloud AI Engineer", "NLP/AI Engineer", "LLMOps Engineer",
    "LLM Ops Engineer", "LLMOps Platform Engineer",
])
def test_the_title_of_its_time_is_in_the_field_of_the_post_it_came_from(post):
    """The title the app gives an older job (earlier_title) never asks for a rewrite pass of its own: it is in the field
    of a post with the title it came from (title_path_problems), as 'MLOps Engineer' is for 'Senior LLMOps Engineer'."""
    resume = resume_of((f"Senior {post}", "Northwind Labs", "Jan 2023", "Present"),
                       (post, "Contoso", "Jun 2020", "Dec 2022"),
                       ("Software Engineer", "Fabrikam", "Jul 2018", "May 2020"))
    validate.era_title_fixes(resume)
    assert titles_of(resume)[1] == validate.earlier_title(post) != post
    assert validate.title_path_problems(resume, posting(f"Senior {post}", post)) == ([], False)


@pytest.mark.parametrize("target,general,older,flagged", [
    ("Senior LLMOps Engineer", "LLMOps Engineer", "MLOps Engineer", False),
    ("Senior LLMOps Engineer", "LLMOps Engineer", "Machine Learning Engineer", False),  # the prompt's title of its time
    ("Senior MLOps Engineer", "MLOps Engineer", "LLMOps Engineer", False),
    ("Senior Machine Learning Engineer", "Machine Learning Engineer", "LLMOps Engineer", False),
    # outside AI's field: still asked for
    ("Senior LLMOps Engineer", "LLMOps Engineer", "DevOps Engineer", True),
    ("Senior LLMOps Engineer", "LLMOps Engineer", "Backend Engineer", True),
])
def test_llmops_is_in_ais_field_as_mlops_is(target, general, older, flagged):
    resume = resume_of((target, "Northwind Labs", "Jan 2024", "Present"), (older, "Contoso", "Jun 2021", "Dec 2023"),
                       ("Software Engineer", "Fabrikam", "Jul 2018", "May 2021"))
    problems, strong = validate.title_path_problems(resume, posting(target, general))
    assert (bool(problems), strong) == (flagged, flagged)


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


@pytest.mark.parametrize("language,again,empty", [(True, None, False), (False, {1: "AI/ML Engineer"}, False),
                                                  (False, None, True)])
def test_build_keeps_a_title_the_notes_name_in_another_form(router, db, registry, language, again, empty):
    """The first draft, a rewrite pass whose revision writes the title again, and a second build of the unchanged
    draft all keep the person's own title, with no note."""
    person = career(("Northwind Labs", "Feb 2023", "Present"), ("Contoso", "Jun 2020", "Dec 2022"),
                    ("Fabrikam", "Jul 2018", "May 2020"),
                    notes={1: "Search ranking: joined as an ML/AI engineer, then rebuilt the ranking service."})
    _client, result = run(router, db, registry, person, titles=["", "AI/ML Engineer", ""], language=language,
                          again=again, empty=empty)
    assert titles_of(result.resume)[:2] == ["Senior AI Engineer", "AI/ML Engineer"]
    assert result.retries == (0 if language else 1)
    assert not any(ERA_REASON in note for note in result.notes)


LLMOPS_POST = AI_ENGINEER_POST.replace("Senior AI Engineer", "Senior LLMOps Engineer", 1)


def llmops_career() -> Profile:
    return career(("Northwind Labs", "Jan 2023", "Present"), ("Contoso", "Jun 2020", "Dec 2022"),
                  ("Fabrikam", "Jul 2018", "May 2020"))


def test_an_llmops_rewrite_pass_keeps_the_title_of_its_time_and_notes_it_once(router, db, registry):
    # the rewrite pass is asked for the language alone, and its revision writes 'LLMOps Engineer' again
    client, result = run(router, db, registry, llmops_career(), jd=LLMOPS_POST, titles=["", "LLMOps Engineer", ""],
                         language=False, again={1: "LLMOps Engineer"})
    assert result.retries == 1 and client.draft is not None
    assert [exp["job_title"] for exp in client.draft["experience"]] == titles_of(result.resume) == [
        "Senior LLMOps Engineer", "MLOps Engineer", "Software Engineer"]
    assert result.notes.count(era_note("Contoso", "LLMOps Engineer", "MLOps Engineer")) == 1
    assert validate.title_path_problems(result.resume, result.analysis) == ([], False)


MLOPS_LLMOPS_POST = AI_ENGINEER_POST.replace("Senior AI Engineer", "Senior MLOps/LLMOps Engineer", 1)


@pytest.mark.parametrize("kwargs", [
    {},  # the demo: it gives the job before the current one its title of the time itself
    {"titles": ["", "MLOps/LLMOps Engineer"]},  # a model that writes the post's title there
    {"titles": ["", "MLOps/LLMOps Engineer"], "language": False, "again": {1: "MLOps/LLMOps Engineer"}},
    {"titles": ["", "MLOps/LLMOps Engineer"], "language": False, "empty": True},
])
def test_an_mlops_and_llmops_career_names_mlops_once_in_every_draft(router, db, registry, kwargs):
    client, result = run(router, db, registry, four_jobs_ending_2022(), jd=MLOPS_LLMOPS_POST, **kwargs)
    titles = ["Senior MLOps/LLMOps Engineer", "MLOps Engineer", "Software Engineer", "Junior Software Engineer"]
    assert titles_of(result.resume) == titles and result.retries == (0 if kwargs.get("language", True) else 1)
    assert result.resume.headline == "Senior MLOps/LLMOps Engineer"  # the current job keeps the post's title
    assert result.notes.count(era_note("Contoso", "MLOps/LLMOps Engineer", "MLOps Engineer")) == (1 if kwargs else 0)
    assert validate.title_growth_problems(result.resume) == []
    assert validate.title_path_problems(result.resume, result.analysis) == ([], False)
    if client is not None and client.draft is not None:  # the first draft after build(), as the rewrite pass saw it
        assert [exp["job_title"] for exp in client.draft["experience"]] == titles


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


@pytest.mark.parametrize("written,title", [
    ("", "MLOps Engineer"),  # the demo's own, already a title of its time
    ("MLOps Engineer", "MLOps Engineer"),  # a model that follows the prompt
    ("Machine Learning Engineer", "Machine Learning Engineer"),
    ("LLMOps Engineer", "MLOps Engineer"),  # a model that does not: the app renames it, with its note
])
def test_an_llmops_post_reads_mlops_before_2023_as_its_field(router, db, registry, written, title):
    """'LLMOps Engineer' at a job that ended in 2022 is 'MLOps Engineer' (era_title_fixes, and the prompt asks for a
    title of its time): that title is in the post's field, so no rewrite pass is asked for it."""
    _client, result = run(router, db, registry, four_jobs_ending_2022(), jd=LLMOPS_POST, titles=["", written])
    assert titles_of(result.resume)[:2] == ["Senior LLMOps Engineer", title]
    assert result.resume.headline == "Senior LLMOps Engineer" and result.retries == 0
    assert validate.title_path_problems(result.resume, result.analysis) == ([], False)
    renamed = 1 if written == "LLMOps Engineer" else 0
    assert result.notes.count(era_note("Contoso", "LLMOps Engineer", "MLOps Engineer")) == renamed


@pytest.mark.parametrize("notes,title", [
    ("Search ranking (2021): joined as an AI Engineer on Search, then led the ranking rebuild.", "AI Engineer"),
    ("Hired as an AI developer for the search team.", "AI Engineer"),  # either noun, any letter case
    ("Built an LLM prototype for search.", "Machine Learning Engineer"),  # AI work, not the title
    ("", "Machine Learning Engineer"),
])
def test_the_demo_keeps_an_older_title_the_persons_notes_name(router, db, registry, notes, title):
    """The prompt's exception, as the demo follows it: a job that ended before 2023 keeps the AI title its notes name,
    with no note, since the app changed nothing; other notes give it the title of its time."""
    person = four_jobs_ending_2022()
    person.companies[1].projects = notes
    _client, result = run(router, db, registry, person)
    assert titles_of(result.resume) == ["Senior AI Engineer", title, "Software Engineer", "Junior Software Engineer"]
    assert result.retries == 0 and not any(ERA_REASON in note for note in result.notes)
    assert result.resume.headline == "Senior AI Engineer"


@pytest.mark.parametrize("prefix,base,count,current,ladder", [
    ("Senior", "AI Engineer", 5, 0, FIVE_AI_TITLES),
    ("Senior", "AI Engineer", 6, 0, ["Senior AI Engineer", "AI Engineer", "Data Engineer", "Software Engineer",
                                     "Associate Software Engineer", "Junior Software Engineer"]),
    ("Staff", "AI Engineer", 6, 0, ["Staff AI Engineer", "Senior AI Engineer", "AI Engineer", "Data Engineer",
                                    "Software Engineer", "Junior Software Engineer"]),
    ("Senior", "Machine Learning Engineer", 5, 0, ["Senior Machine Learning Engineer", "Machine Learning Engineer",
                                                   "Data Engineer", "Software Engineer", "Junior Software Engineer"]),
    # LLMOps is AI's field, as MLOps is (validate._FIELD_FAMILY), and so is 'MLOps Engineer', its title before 2023
    ("Senior", "LLMOps Engineer", 5, 0, ["Senior LLMOps Engineer", "LLMOps Engineer", "Data Engineer",
                                         "Software Engineer", "Junior Software Engineer"]),
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


@pytest.mark.parametrize("written", ["", "AI Engineer"])  # the demo's own title, or a model that wrote the AI title
@pytest.mark.parametrize("notes,title", [
    ("Joined as an AI Engineer on Search: built the ranking service.", "AI Engineer"),  # the person's own title
    ("Built an LLM prototype for search.", "Machine Learning Engineer"),  # AI work, not the title
])
def test_with_a_rewrite_pass_the_title_the_notes_name_holds(router, db, registry, profile, notes, title, written):
    """The demo's revision agrees with build(): the title its notes name stays through the rewrite pass, the second
    build() of the draft changes nothing, and the era note shows once, only for a title the app renamed."""
    profile.companies[1].projects = notes  # Contoso Systems, which ended in February 2022
    client, result = run(router, db, registry, profile, titles=["", written, ""], language=False)
    assert result.retries == 1 and client.draft is not None
    assert [exp["job_title"] for exp in client.draft["experience"]] == titles_of(result.resume) == [
        "Senior AI Engineer", title, "Software Engineer"]
    assert result.resume.headline == "Senior AI Engineer"
    renamed = bool(written) and written != title  # the app renamed the model's 'AI Engineer'
    assert result.notes.count(era_note("Contoso Systems", "AI Engineer", "Machine Learning Engineer")) == (
        1 if renamed else 0)
    assert sum(ERA_REASON in note for note in result.notes) == (1 if renamed else 0)


GENAI_POST = AI_ENGINEER_POST.replace("Senior AI Engineer", "Senior Generative AI Engineer", 1)


@pytest.mark.parametrize("prefix,base,count,current,ladder", [
    ("Senior", "Generative AI Engineer", 5, 0, ["Senior Generative AI Engineer", "Generative AI Engineer",
                                                "Data Engineer", "Software Engineer", "Junior Software Engineer"]),
    ("Senior", "Generative AI Engineer", 4, 0, ["Senior Generative AI Engineer", "Generative AI Engineer",
                                                "Software Engineer", "Junior Software Engineer"]),
    ("Senior", "Generative AI Engineer", 5, 1, ["Senior Generative AI Engineer", "Senior Generative AI Engineer",
                                                "Generative AI Engineer", "Software Engineer",
                                                "Junior Software Engineer"]),
    ("Senior", "Gen AI Engineer", 3, 0, ["Senior Gen AI Engineer", "Gen AI Engineer", "Software Engineer"]),
    ("Staff", "Generative AI Engineer", 6, 0, ["Staff Generative AI Engineer", "Senior Generative AI Engineer",
                                               "Generative AI Engineer", "Data Engineer", "Software Engineer",
                                               "Junior Software Engineer"]),
    ("Senior", "Generative AI Platform Engineer", 4, 0, ["Senior Generative AI Platform Engineer",
                                                         "Generative AI Platform Engineer", "Platform Engineer",
                                                         "Software Engineer"]),
    # 'ML Engineer' under 'AI & ML Engineer' reads as 'Machine Learning Engineer' does
    ("Senior", "AI & ML Engineer", 5, 0, ["Senior AI & ML Engineer", "AI & ML Engineer", "Data Engineer",
                                          "Software Engineer", "Junior Software Engineer"]),
    ("Senior", "AI and ML Engineer", 4, 0, ["Senior AI and ML Engineer", "AI and ML Engineer", "Software Engineer",
                                            "Junior Software Engineer"]),
    # 'MLOps Engineer' under 'LLMOps & MLOps Engineer': each is 'MLOps Engineer' before 2023 (earlier_title)
    ("Senior", "LLMOps & MLOps Engineer", 4, 0, ["Senior LLMOps & MLOps Engineer", "LLMOps & MLOps Engineer",
                                                 "Software Engineer", "Junior Software Engineer"]),
    ("Senior", "LLMOps & MLOps Engineer", 5, 0, ["Senior LLMOps & MLOps Engineer", "LLMOps & MLOps Engineer",
                                                 "Data Engineer", "Software Engineer", "Junior Software Engineer"]),
    # a mid-level post's own title is the current job's, which is never dated: its one step named for AI stays
    ("", "Generative AI Engineer", 3, 0, ["Generative AI Engineer", "AI Engineer", "Junior Software Engineer"]),
    ("", "AI & ML Engineer", 3, 0, ["AI & ML Engineer", "ML Engineer", "Junior Software Engineer"]),
])
def test_a_nearer_title_of_the_same_time_is_no_step_in_the_demo(prefix, base, count, current, ladder):
    """'AI Engineer' under 'Generative AI Engineer' reads 'Machine Learning Engineer' at a job that ended before 2023,
    as the post's own title one level lower does there: two steps of one title, so it is no step of its own."""
    assert _title_ladder(prefix, base, count, current) == ladder


@pytest.mark.parametrize("prefix", ["Senior", "Staff", "Lead", "Principal", ""])
@pytest.mark.parametrize("base", ["AI Engineer", "Generative AI Engineer", "Gen AI Engineer", "GenAI Engineer",
                                  "Generative AI Platform Engineer", "Generative AI Data Engineer",
                                  "AI Platform Engineer", "LLM Engineer", "LLMOps Engineer",
                                  "Machine Learning Engineer", "AI & ML Engineer", "AI and ML Engineer",
                                  "AI and Machine Learning Engineer", "LLMOps & MLOps Engineer",
                                  "MLOps/LLMOps Engineer"])
@pytest.mark.parametrize("end", ["Dec 2022", "Dec 2023"])
def test_the_demos_titles_of_their_time_repeat_no_job_before_them(prefix, base, end):
    """Each job after the current one ends on `end` or earlier: dating its title never makes it read the same as the
    next one's ('Machine Learning Engineer' and 'ML Engineer' read the same). A long career's oldest title may repeat,
    as it does on the ladder itself."""
    ends = [end, "Dec 2020", "Dec 2018", "Dec 2016", "Dec 2014", "Dec 2012", "Dec 2010"]
    for count in range(2, 9):
        ladder = _title_ladder(prefix, base, count)
        dated = ladder[:1] + [validate.dated_title(title, ends[index]) for index, title in enumerate(ladder[1:])]
        repeats = [sum(languages.title_words(a) == languages.title_words(b) for a, b in itertools.pairwise(titles))
                   for titles in (ladder, dated)]
        assert repeats[1] <= repeats[0], (count, dated)


@pytest.mark.parametrize("person,titles", [
    ("four_jobs_ending_2022", ["Senior Generative AI Engineer", "Machine Learning Engineer", "Software Engineer",
                               "Junior Software Engineer"]),
    ("five_ai_jobs", ["Senior Generative AI Engineer", "Generative AI Engineer", "Data Engineer", "Software Engineer",
                      "Junior Software Engineer"]),
    ("conftest", ["Senior Generative AI Engineer", "Machine Learning Engineer", "Software Engineer"]),
])
def test_the_demo_writes_a_generative_ai_career_without_a_repeated_title(router, db, registry, profile, person,
                                                                         titles):
    people = {"five_ai_jobs": five_ai_jobs, "four_jobs_ending_2022": four_jobs_ending_2022, "conftest": lambda: profile}
    _client, result = run(router, db, registry, people[person](), jd=GENAI_POST)
    resume = result.resume
    assert titles_of(resume) == titles and result.retries == 0
    assert resume.headline == "Senior Generative AI Engineer"
    assert validate.title_growth_problems(resume) == [] and validate.era_problems(resume) == []
    assert validate.title_path_problems(resume, result.analysis) == ([], False)
    assert not any(ERA_REASON in note for note in result.notes)  # its titles are already those of their time


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
            'Engineer"), unless its project notes name its title (for example "joined as an AI Engineer").') in text
    assert ("each job names only tools that were in common use during its dates (or that its notes name) and that its "
            "title fits those dates (or that its notes name it).") in text
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
    "gen" + " " * 20000 + "ai engineer", "llm" + "-" * 20000 + "ops engineer", "a. " * 7000 + "i. engineer",
    "and " * 5000 + "ai engineer", "ai and developer " * 1500 + "x",
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


# ------------------------------------------------------------------ a rename that meets the title before it
def test_a_rename_that_meets_the_title_before_it_keeps_its_step_up():
    """'AI Engineer' after 'Machine Learning Engineer' was a step up: renamed, it takes the level number above that
    title, so a stay of 2+ years still shows growth (title_growth_problems) at no cost."""
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"),
                       ("Machine Learning Engineer", "Fabrikam", "Jul 2018", "May 2020"),
                       ("Software Engineer", "Tailspin Toys", "Jun 2016", "Jun 2018"))
    notes = validate.era_title_fixes(resume)
    assert titles_of(resume) == ["Senior AI Engineer", "Machine Learning Engineer II", "Machine Learning Engineer",
                                 "Software Engineer"]
    assert notes == [era_note("Contoso", "AI Engineer", "Machine Learning Engineer"),
                     ("Contoso: title changed from 'Machine Learning Engineer' to 'Machine Learning Engineer II' so "
                      "the career path still shows growth.")]
    assert validate.title_growth_problems(resume) == [] and validate.era_title_fixes(resume) == []


def test_steps_build_on_each_other_from_the_oldest_job():
    """Two renamed jobs in a row, each after a stay of 2+ years: each takes the level above the one before it, and
    build()'s next level pass keeps both."""
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"),
                       ("LLM Engineer", "Fabrikam", "Jul 2017", "May 2020"),
                       ("Machine Learning Engineer", "Tailspin Toys", "Jun 2014", "Jun 2017"))
    validate.era_title_fixes(resume)
    assert titles_of(resume) == ["Senior AI Engineer", "Machine Learning Engineer III", "Machine Learning Engineer II",
                                 "Machine Learning Engineer"]
    assert validate.fix_title_regressions(resume) == [] and validate.title_growth_problems(resume) == []


@pytest.mark.parametrize("rows", [
    # a stay under 2 years needs no step (the second level pass gives it the title before it)
    [("AI Engineer", "Contoso", "Jun 2021", "Dec 2022"),
     ("Machine Learning Engineer", "Fabrikam", "Jul 2019", "May 2021")],
    # the model's own repeat is not the rename's
    [("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"), ("AI Engineer", "Fabrikam", "Jul 2018", "May 2020")],
    # a part after the role, or a level word: as it is
    [("AI Engineer, Search", "Contoso", "Jun 2020", "Dec 2022"),
     ("Machine Learning Engineer, Search", "Fabrikam", "Jul 2018", "May 2020")],
    [("Senior AI Engineer", "Contoso", "Jun 2020", "Dec 2022"),
     ("Senior Machine Learning Engineer", "Fabrikam", "Jul 2018", "May 2020")],
])
def test_a_rename_takes_no_step_it_does_not_need_or_cannot_name(rows):
    resume = resume_of(("Staff AI Engineer", "Northwind Labs", "Jan 2023", "Present"), *rows)
    validate.era_title_fixes(resume)
    assert titles_of(resume)[1] == validate.earlier_title(rows[0][0])


@pytest.mark.parametrize("rows,titles", [
    # a current 'AI Engineer' would read as a step down after 'Machine Learning Engineer III'
    ([("AI Engineer", "Northwind Labs", "Jun 2025", "Present"), ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"),
      ("Machine Learning Engineer II", "Fabrikam", "Jul 2017", "May 2020")],
     ["AI Engineer", "Machine Learning Engineer", "Machine Learning Engineer II"]),
    # a later job two levels below 'Machine Learning Engineer III'
    ([("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
      ("Software Engineer", "Bluefin Commerce", "Jan 2021", "Dec 2022"),
      ("AI Engineer", "Contoso", "Jun 2018", "Dec 2020"),
      ("Machine Learning Engineer II", "Fabrikam", "Jul 2015", "May 2018")],
     ["Senior AI Engineer", "Software Engineer", "Machine Learning Engineer", "Machine Learning Engineer II"]),
    # the current job holds the renamed role at a lower level: the next level pass would take the step back
    ([("Machine Learning Engineer", "Northwind Labs", "Jan 2023", "Present"),
      ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"),
      ("Machine Learning Engineer", "Fabrikam", "Jul 2017", "May 2020")],
     ["Machine Learning Engineer", "Machine Learning Engineer", "Machine Learning Engineer"]),
    # a past 'Machine Learning Engineer II' would sit above the current 'Machine Learning Engineer', also two jobs apart
    ([("Machine Learning Engineer", "Northwind Labs", "Jan 2023", "Present"),
      ("ML Platform Engineer", "Bluefin Commerce", "Jan 2021", "Dec 2022"),
      ("AI Engineer", "Contoso", "Jun 2018", "Dec 2020"),
      ("Machine Learning Engineer", "Fabrikam", "Jul 2015", "May 2018")],
     ["Machine Learning Engineer", "ML Platform Engineer", "Machine Learning Engineer", "Machine Learning Engineer"]),
    # a job listed above the current one is never stepped above it
    ([("AI Engineer", "Northwind Labs", "Jan 2020", "Dec 2022"),
      ("Machine Learning Engineer", "Bluefin Commerce", "Mar 2019", "Present"),
      ("Software Engineer", "Contoso", "Jul 2016", "Feb 2019")],
     ["Machine Learning Engineer", "Machine Learning Engineer", "Software Engineer"]),
])
def test_a_step_never_goes_above_a_later_job_or_the_current_one(rows, titles):
    resume = resume_of(*rows)
    notes = validate.era_title_fixes(resume)
    assert titles_of(resume) == titles
    assert not any("still shows growth" in note for note in notes)


@pytest.mark.parametrize("post,general,titles,final", [
    ("Senior AI Engineer", "AI Engineer", ["Senior AI Engineer", "AI Engineer", "Machine Learning Engineer"],
     ["Senior AI Engineer", "Machine Learning Engineer II", "Machine Learning Engineer"]),
    ("Machine Learning Engineer", "Machine Learning Engineer",
     ["Machine Learning Engineer", "LLM Engineer", "Machine Learning Engineer II"],
     ["Machine Learning Engineer", "Machine Learning Engineer", "Machine Learning Engineer"]),
    ("AI Engineer", "AI Engineer", ["AI Engineer", "AI Engineer", "Machine Learning Engineer II"],
     ["AI Engineer", "Machine Learning Engineer II", "Machine Learning Engineer II"]),
])
def test_build_never_takes_a_step_back(profile, post, general, titles, final):
    """build()'s title steps (drafted) on conftest's dates (Contoso Systems Jun 2019 - Feb 2022): a step stays, and
    where it has no room no step is noted, so no later note takes it back."""
    resume, notes = drafted(profile, posting(post, general), f"{post} with 9+ years of Python work.", titles,
                            headline=post)
    assert titles_of(resume) == final
    stepped = {note.split(": ")[0] for note in notes if note.endswith("still shows growth.")}
    lowered = {note.split(": ")[0] for note in notes if note.endswith("never steps down.")}
    assert not stepped & lowered


@pytest.mark.parametrize("older,stepped", [
    ("Machine Learning Engineer", "Machine Learning Engineer II"),
    ("Machine Learning Engineer II", "Machine Learning Engineer III"),
])
def test_build_keeps_the_step_up_without_a_rewrite_pass(router, db, registry, older, stepped):
    person = career(("Northwind Labs", "Feb 2023", "Present"), ("Contoso", "Jun 2020", "Dec 2022"),
                    ("Fabrikam", "Jul 2018", "May 2020"), ("Tailspin Toys", "Jun 2016", "Jun 2018"))
    _client, result = run(router, db, registry, person,
                          titles=["Senior AI Engineer", "AI Engineer", older, "Software Engineer"])
    assert titles_of(result.resume) == ["Senior AI Engineer", stepped, older, "Software Engineer"]
    assert result.retries == 0 and validate.title_growth_problems(result.resume) == []
    assert result.notes.count(f"Contoso: title changed from 'Machine Learning Engineer' to '{stepped}' so the career "
                              "path still shows growth.") == 1


def test_a_rewrite_pass_that_writes_the_ai_title_again_keeps_the_step_and_notes_it_once(router, db, registry):
    person = career(("Northwind Labs", "Feb 2023", "Present"), ("Contoso", "Jun 2020", "Dec 2022"),
                    ("Fabrikam", "Jul 2018", "May 2020"))
    # the revision changes nothing but writes 'AI Engineer' again
    _client, result = run(router, db, registry, person, titles=["", "AI Engineer", "Machine Learning Engineer"],
                          language=False, empty=True, again={1: "AI Engineer"})
    assert result.retries == 1
    assert titles_of(result.resume)[1:] == ["Machine Learning Engineer II", "Machine Learning Engineer"]
    assert result.notes.count(era_note("Contoso", "AI Engineer", "Machine Learning Engineer")) == 1
    assert result.notes.count("Contoso: title changed from 'Machine Learning Engineer' to 'Machine Learning Engineer "
                              "II' so the career path still shows growth.") == 1
    assert validate.title_growth_problems(result.resume) == []


def test_a_rename_that_meets_the_job_before_it_asks_no_rewrite_pass(router, db, registry, profile):
    """The model wrote the AI ladder ('AI Engineer' after 'Machine Learning Engineer') on conftest's dates: the rename
    alone would give Contoso Systems Fabrikam's title after 2 yrs 9 mo, so it takes the next level number, and no
    rewrite pass is paid for and no growth warning is left."""
    _client, result = run(router, db, registry, profile, titles=["", "AI Engineer", "Machine Learning Engineer"])
    assert titles_of(result.resume) == ["Senior AI Engineer", "Machine Learning Engineer II",
                                        "Machine Learning Engineer"]
    assert result.retries == 0 and validate.title_growth_problems(result.resume) == []
    assert not any("same title" in warning for warning in result.ats_warnings)
    assert result.notes.count(era_note("Contoso Systems", "AI Engineer", "Machine Learning Engineer")) == 1
    assert result.notes.count("Contoso Systems: title changed from 'Machine Learning Engineer' to 'Machine Learning "
                              "Engineer II' so the career path still shows growth.") == 1


@pytest.mark.parametrize("jobs,titles", [
    # a second repeat, the model's own, after the one the rename made
    ((("Northwind Labs", "Mar 2022", "Present"), ("Contoso Systems", "Jun 2019", "Feb 2022"),
      ("Fabrikam", "Jul 2017", "May 2019"), ("Tailspin Toys", "Jun 2015", "Jun 2017"),
      ("Bluefin Commerce", "Jun 2013", "Jun 2015")),
     ["", "AI Engineer", "Machine Learning Engineer", "Software Engineer", "Software Engineer"]),
    # the same employer twice: the renamed newer Contoso job is not the older one, whose repeat is the model's own
    ((("Northwind Labs", "Mar 2022", "Present"), ("Contoso", "Jan 2021", "Feb 2022"),
      ("Fabrikam", "Jul 2019", "Dec 2020"), ("Contoso", "Jun 2016", "Jun 2019"),
      ("Tailspin Toys", "Jun 2014", "May 2016")),
     ["", "AI Engineer", "Data Engineer", "Software Engineer", "Software Engineer"]),
])
def test_the_models_own_repeat_beside_a_renamed_title_still_asks_the_rewrite_pass(router, db, registry, jobs, titles):
    _client, result = run(router, db, registry, career(*jobs), titles=titles)
    assert result.retries == 1 and validate.title_growth_problems(result.resume) == []


@pytest.mark.parametrize("text", LONG_TITLES + LONG_TEXTS)
def test_long_crafted_titles_are_stepped_in_time(text):
    """The step reads a crafted title, and one of the same role before it, in linear time."""
    started = time.perf_counter()
    validate._step_above(text, text, "x", "y")
    validate._step_above(validate.earlier_title(text), validate.earlier_title(text), "x", "y")
    resume = resume_of(("Senior AI Engineer", "Northwind Labs", "Jan 2023", "Present"),
                       ("AI Engineer", "Contoso", "Jun 2020", "Dec 2022"),
                       ("Machine Learning Engineer", "Fabrikam", "Jul 2018", "May 2020"))
    resume.experience[1].job_title, resume.experience[2].job_title = text, validate.earlier_title(text)
    validate.era_title_fixes(resume)
    assert time.perf_counter() - started < 1.0
