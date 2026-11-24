"""The owner's rules for the current job and the title path (prompt v39): the line under the name, the summary's first
words and the current job's title read the same ('Senior AI Platform Engineer'); older titles climb into the target's
field (Junior Software Engineer, Software Engineer, Platform Engineer, AI Platform Engineer, Senior AI Platform
Engineer); the current job names the job post's most wanted programming language (or, when the post names none, the
one its kind of work is usually built in), flagged for the person to confirm when their notes do not name it. Every
person, employer and product here is invented."""
import asyncio

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import rules, validate
from app.pipeline.page_fit import design_style
from app.pipeline.tailor import run_checks, run_tailor
from app.prompts import PROMPT_VERSION, RETRY_MARKER, tailor_system
from app.schemas import ExperienceOut, JDAnalysis, Profile, SkillGroup, TailorResult

from .test_validate import make_output
from .test_writing_quality import five_jobs

AI_POST = (
    "Senior AI Platform Engineer at Contoso Learning. You will build the platform behind our AI features: LLM "
    "evaluation, retrieval-augmented generation (RAG), agent tool calling and observability. Required: Python, LLM "
    "evaluation, RAG, Kubernetes, PostgreSQL, CI/CD, AWS. Nice to have: HashiCorp Vault, pytest, Terraform."
)
MARKETING_POST = (
    "Senior Marketing Manager at Fabrikam. You will plan product launches in HubSpot, run SEO and A/B Testing for the "
    "website, own the Product Roadmap messaging and report results to leadership with Analytics dashboards."
)


def posting(target="Senior AI Platform Engineer", general="AI Platform Engineer", must=(), nice=(), keywords=()):
    return JDAnalysis(target_role=target, general_role=general, seniority="Senior", domain="AI",
                      must_have_skills=list(must), nice_to_have_skills=list(nice), keywords=list(keywords))


def titled(titles: list[str]):
    """A resume of len(titles) jobs at five_jobs' employers (Northwind Labs is the current one), with these titles."""
    out = make_output(n=1)
    out.sections.experience = [
        ExperienceOut(company_index=i, job_title=title, bullets=["Built things", "Improved things"])
        for i, title in enumerate(titles)
    ]
    return validate.build_resume(five_jobs(len(titles)), out)


class Changes(LLMClient):
    """The demo model; `change` edits the first draft's sections (the fix round's answer is the demo's own)."""

    def __init__(self, inner, change):
        self.inner, self.change, self.prompts = inner, change, []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["user"])
        result = await self.inner.generate_structured(**kwargs)
        if kwargs["schema_name"] == "tailor_output" and RETRY_MARKER not in kwargs["user"]:
            self.change(result.data["sections"])
        return result

    def fix_round(self) -> str:
        """What the one rewrite pass was asked to fix ("" when it was not asked for)."""
        asked = [user for user in self.prompts if RETRY_MARKER in user]
        return asked[0].split(RETRY_MARKER + ": ", 1)[1] if asked else ""


def tailored(router, db, jd=AI_POST, person: Profile | None = None, change=None, registry=None, **kwargs):
    client = None
    if change is not None:
        client = Changes(FakeClient(registry.get("demo-concise")), change)
        router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=person or five_jobs(), profile_id=None, jd=jd,
                                    model_id="demo-concise", **kwargs))
    assert result.status == "ok", result.error
    return client, result


# ------------------------------------------------------------------ B. one title in three places
@pytest.mark.parametrize("summary,expected", [
    ("Senior AI platform engineer with 7+ years of Python platform work for retail and learning products.",
     "Senior AI Platform Engineer with 7+ years of Python platform work for retail and learning products."),
    ("AI Platform Engineer with 7+ years of Python platform work.",  # the level comes back
     "Senior AI Platform Engineer with 7+ years of Python platform work."),
    ("ai platform engineer with 7+ years of Python platform work.",
     "Senior AI Platform Engineer with 7+ years of Python platform work."),
    ("Staff AI Platform Engineer with 7+ years of Python platform work.",  # one title, one level
     "Senior AI Platform Engineer with 7+ years of Python platform work."),
    ("Builds AI platforms for learning products, with 7+ years of Python work.",  # not the role: left alone
     "Builds AI platforms for learning products, with 7+ years of Python work."),
    ("AI Platform Engineering lead with 7+ years of Python work.",  # a longer word is not the title
     "AI Platform Engineering lead with 7+ years of Python work."),
])
def test_the_headline_and_the_summary_take_the_most_recent_title(profile, summary, expected):
    resume = validate.build_resume(profile, make_output(headline="AI Platform Engineer", summary=summary))
    resume.experience[0].job_title = "Senior AI Platform Engineer"
    validate.same_title(resume)
    assert resume.headline == "Senior AI Platform Engineer"
    assert resume.summary == expected
    assert resume.experience[0].job_title == "Senior AI Platform Engineer"  # the title itself stays as written


@pytest.mark.parametrize("title,headline", [
    ("Senior AI Platform Engineer (Remote)", "Senior AI Platform Engineer"),
    ("Staff Data Engineer, Analytics Infrastructure", "Staff Data Engineer"),
    ("Senior Data Engineer | Python | SQL", "Senior Data Engineer"),
    ("Senior Manager, Customer Success", "Senior Customer Success Manager"),  # a bare role noun takes its function
    ("Software Engineer II", "Software Engineer II"),  # the candidate's own level number stays
    ("Tech Lead", "Tech Lead"),
])
def test_the_headline_is_the_title_without_what_follows_it(profile, title, headline):
    resume = validate.build_resume(profile, make_output(headline="Engineer", summary=f"{title.lower()} with 9+ years."))
    resume.experience[0].job_title = title
    validate.same_title(resume)
    assert resume.headline == headline
    assert resume.summary.startswith(headline + " with 9+ years") or "," in title or "(" in title or "|" in title


def test_with_no_job_the_headline_stays_the_roles_short_name():
    out = make_output(n=0, headline="Senior Data Engineer | Python | SQL")
    resume = validate.build_resume(Profile(name="Taylor Quinn"), out)
    validate.short_role_headline(resume)
    validate.same_title(resume)
    assert resume.headline == "Data Engineer"


def test_the_three_places_keep_one_spelling_in_eu_style(profile):
    analysis = posting(target="Senior Data Modeler", general="Data Modeler", keywords=["dimensional modeling"])
    resume = validate.build_resume(profile, make_output(
        headline="Data Modeler", summary="Senior data modeler with 8 years of dimensional modeling in erwin."))
    resume.experience[0].job_title = "Senior Data Modeler"
    validate.same_title(resume)
    validate.spell_resume(resume, "eu", analysis)
    assert resume.headline == resume.experience[0].job_title == "Senior Data Modeller"
    assert resume.summary.startswith("Senior Data Modeller with 8 years")


class LowerCaseRole(Changes):
    """The first draft's headline is the level-less role and its summary opens with the role in lower case."""

    def __init__(self, inner):
        def change(sections):
            sections["headline"] = "AI Platform Engineer"
            rest = sections["summary"].split(" with ", 1)[1]
            sections["summary"] = "Senior AI platform engineer with " + rest

        super().__init__(inner, change)


def test_a_run_reads_one_title_in_three_places(router, db, registry):
    client = LowerCaseRole(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=five_jobs(), profile_id=None, jd=AI_POST,
                                    model_id="demo-concise"))
    title = result.resume.experience[0].job_title
    assert title == "Senior AI Platform Engineer"
    assert result.resume.headline == title and result.resume.summary.startswith(title + " with ")
    assert "Senior AI platform engineer" not in result.resume.summary


def test_the_prompt_asks_for_one_title_in_three_places():
    text = tailor_system("estimate")
    assert "HEADLINE: the most recent job's title in your plan, word for word" in text
    assert "the line under the name, the summary's first words and the most recent job read the same" in text
    assert '"Senior Data Platform Engineer with 9+ years of software experience' in text  # the summary's example
    assert "Data platform engineer with 9+ years" not in text
    assert "without the hiring company's level number (II, III, L5)" in text


# ------------------------------------------------------------------ E. a title path into the target's field
PLAIN_PATH = ["Senior AI Platform Engineer", "Senior Software Engineer", "Software Engineer II", "Software Engineer",
              "Junior Software Engineer"]


def test_a_path_with_no_older_title_in_the_target_field_is_a_strong_problem():
    problems, strong = validate.title_path_problems(titled(PLAIN_PATH), posting())
    assert strong and len(problems) == 1
    assert "'Senior Software Engineer' at Bluefin.com, the job before the current one" in problems[0]
    assert "'AI Platform Engineer'" in problems[0]


def test_a_path_whose_job_before_the_current_one_is_outside_the_field_is_a_soft_problem():
    path = ["Senior AI Platform Engineer", "Senior Software Engineer", "Platform Engineer", "Software Engineer",
            "Junior Software Engineer"]
    problems, strong = validate.title_path_problems(titled(path), posting())
    assert not strong and len(problems) == 1 and "Bluefin.com" in problems[0]


@pytest.mark.parametrize("titles,target,general", [
    (["Senior AI Platform Engineer", "AI Platform Engineer", "Platform Engineer", "Software Engineer",
      "Junior Software Engineer"], "Senior AI Platform Engineer", "AI Platform Engineer"),  # the owner's path
    (["Senior AI Platform Engineer", "Senior Platform Engineer", "Backend Engineer"], "Senior AI Platform Engineer",
     "AI Engineer"),
    (["Senior AI Platform Engineer", "Machine Learning Engineer", "Software Engineer"], "Senior AI Platform Engineer",
     ""),  # ML is AI's field
    (["Senior AI Platform Engineer", "ML Engineer", "Software Engineer"], "Senior AI Platform Engineer", ""),
    (["Senior Software Engineer", "Software Engineer II", "Software Engineer"], "Senior Software Engineer", ""),  # no
    # field words: nothing to check
    (["Senior AI Platform Engineer"], "Senior AI Platform Engineer", ""),  # one job: nothing before it
    (["Senior Back-End Engineer", "Backend Engineer", "Software Engineer"], "Senior Backend Engineer", ""),
])
def test_a_path_into_the_target_field_is_fine(titles, target, general):
    assert validate.title_path_problems(titled(titles), posting(target, general)) == ([], False)


def test_the_common_title_gives_the_field_a_posting_title_hides_after_a_comma():
    # 'Senior Software Engineer, AI Platform' is a software engineer's title; its common title names the field
    analysis = posting("Senior Software Engineer, AI Platform", "AI Platform Engineer")
    problems, strong = validate.title_path_problems(titled(["Senior AI Platform Engineer", "Senior Software Engineer",
                                                            "Software Engineer"]), analysis)
    assert strong and "is not in the field of 'AI Platform Engineer', and no older title is either." in problems[0]


def test_a_special_posting_title_counts_by_its_common_title():
    # 'Voice' is the specialty: the field is AI ('AI Engineer'), and the specialty itself never makes one
    analysis = posting("Voice AI Engineer", "AI Engineer")
    assert validate.title_path_problems(titled(["AI Engineer", "Backend Engineer", "Software Engineer"]),
                                        analysis)[1]
    assert validate.title_path_problems(titled(["AI Engineer", "Machine Learning Engineer", "Software Engineer"]),
                                        analysis) == ([], False)


def plain_older_titles(sections):
    for exp, title in zip(sections["experience"][1:], PLAIN_PATH[1:]):
        exp["job_title"] = title


def test_older_titles_outside_the_field_are_chosen_again_in_the_fix_round(router, db, registry):
    client, result = tailored(router, db, change=plain_older_titles, registry=registry)
    assert result.retries == 1
    asked = client.fix_round()
    assert "The job titles do not lead into the target's field:" in asked and "JOB TITLES rule" in asked
    assert [e.job_title for e in result.resume.experience] == [
        "Senior AI Platform Engineer", "AI Platform Engineer", "Platform Engineer", "Software Engineer",
        "Junior Software Engineer"]


def one_older_title_outside(sections):
    sections["experience"][1]["job_title"] = "Senior Software Engineer"


def test_a_soft_path_problem_is_asked_only_along_with_another_fix(router, db, registry):
    _client, result = tailored(router, db, change=one_older_title_outside, registry=registry)
    assert result.retries == 0  # on its own it costs no rewrite pass
    assert result.resume.experience[1].job_title == "Senior Software Engineer"

    def also_no_language(sections):
        one_older_title_outside(sections)
        drop_language(sections)

    client, result = tailored(router, db, change=also_no_language, registry=registry)
    assert result.retries == 1 and "The job titles do not lead into the target's field:" in client.fix_round()


def test_the_prompt_asks_for_a_path_into_the_target_field():
    text = tailor_system("estimate")
    assert "The path moves into the target's field as it gets closer to today" in text
    assert '"AI Platform Engineer" or "Senior Platform Engineer" before "Senior AI Platform Engineer"' in text
    assert '("Software Engineer", "Junior Software Engineer")' in text
    assert "each step fits that job's own bullets" in text
    assert "A title never steps down from one job to the next" in text  # the rules kept
    assert "Never give three employers in a row the same title" in text


# ------------------------------------------------------------------ C. the post's language at the current job
@pytest.mark.parametrize("text,found", [
    ("Python, Go and Rust services on Kubernetes", ["Python", "Go", "Rust"]),
    ("Experience with Golang or Rust", ["Go", "Rust"]),
    ("Own the go-to-market plan and the Go-to-market launch", []),
    ("Grow the R&D budget after our Series C round with C-level sponsors", []),
    ("Ship iOS apps in Objective-C and Swift", ["Objective-C", "Swift"]),
    ("Swift delivery and a Ruby anniversary dinner with Julia from finance", []),
    ("Build Ruby on Rails apps and R packages for analysts", ["Ruby", "R"]),
    ("Write C, C++ and C# for the device firmware", ["C", "C++", "C#"]),
    ("Tune PostgreSQL, MySQL and NoSQL stores", []),
    ("Write T-SQL reports and Bash scripts", ["SQL", "Bash"]),
    ("Automate builds with shell scripting", ["Bash"]),
    ("Build JavaScript and TypeScript front ends", ["JavaScript", "TypeScript"]),
    ("Shell, our fuel partner, and a bash for the team", []),
    ("Write the core product services in Go with tests on every change", ["Go"]),
    ("Rebuilt the Go order-sync service behind the partner API", ["Go"]),
    ("Go", ["Go"]),  # a skill keyword of its own, as the language writes it or in lower case
    ("go", ["Go"]),
    ("R", ["R"]),
    ("SWIFT", []),  # the banks' network
    ("Python 3.12", ["Python"]),
])
def test_languages_count_only_where_they_read_as_languages(text, found):
    from app.pipeline import languages

    assert languages.mentioned(text) == found


def test_a_language_is_named_by_any_of_its_names():
    from app.pipeline import languages

    assert languages.names("Write the order service in Golang", "Go")
    assert not languages.names("Plan the Go-to-market launch", "Go")
    assert languages.names("Moved the reports to T-SQL procedures", "SQL")
    assert not languages.names("Write JavaScript widgets", "Java")


def test_the_post_wants_the_first_language_it_asks_for():
    from app.pipeline import languages

    post = posting(must=["LLM evaluation", "Python", "Go"], nice=["Rust"], keywords=["Kubernetes", "TypeScript"])
    assert languages.wanted_language(post, AI_POST) == "Python"
    post.must_have_skills = ["SQL", "dbt", "Python"]  # a language the work is built in before SQL
    assert languages.wanted_language(post) == "Python"
    post.must_have_skills = ["SQL", "dbt"]  # SQL is the post's language here
    assert languages.wanted_language(post) == "SQL"
    post.must_have_skills = ["LLM evaluation"]  # then the nice-to-haves, the keywords and the post's own text
    assert languages.wanted_language(post) == "Rust"
    post.nice_to_have_skills = []
    assert languages.wanted_language(post) == "TypeScript"
    post.keywords = []
    assert languages.wanted_language(post, "You will write the scheduling services in Go.") == "Go"
    assert languages.wanted_language(post, "You will lead go-to-market launches.") is None
    assert languages.wanted_language(None, "") is None


@pytest.mark.parametrize("target,general,usual", [
    ("Senior AI Platform Engineer", "AI Platform Engineer", "Python"),
    ("Machine Learning Engineer", "", "Python"),
    ("Senior Data Engineer", "Data Engineer", "Python"),
    ("Senior iOS Engineer", "iOS Engineer", "Swift"),
    ("Android Developer", "", "Kotlin"),
    ("Senior Front-End Engineer", "Frontend Engineer", "TypeScript"),
    ("Registered Nurse", "", None),
    ("Senior Marketing Manager", "Marketing Manager", None),
    ("Sales Engineer", "", None),
])
def test_work_built_in_code_has_a_usual_language(target, general, usual):
    from app.pipeline import languages

    analysis = posting(target, general)
    assert languages.usual_language(analysis) == usual
    assert languages.codes(analysis) == (usual is not None)


NO_LANGUAGE = [
    "Rebuild the evaluation suite for the AI assistant platform, cutting failed releases from 41 to 28 a quarter.",
    "Design retrieval for the assistant's answers on Kubernetes, letting support teams trust each reply.",
]


def current_job_resume(profile, bullets, older=("Built the reporting service in Python for 40 teams.",)):
    resume = validate.build_resume(profile, make_output())
    resume.experience[0].bullets = list(bullets)
    resume.experience[1].bullets = list(older)
    return resume


def test_the_current_job_names_the_posts_language(profile):
    analysis = posting(must=["Python", "RAG"])
    resume = current_job_resume(profile, NO_LANGUAGE)
    assert validate.missing_language(resume, analysis, AI_POST) == ("Northwind Labs", "Python")  # never older jobs
    resume.experience[0].bullets[1] = NO_LANGUAGE[1].replace("on Kubernetes", "in Go on Kubernetes")
    assert validate.missing_language(resume, analysis, AI_POST) == ("Northwind Labs", "Python")  # not the one wanted
    resume.experience[0].bullets[1] = NO_LANGUAGE[1].replace("on Kubernetes", "in Python on Kubernetes")
    assert validate.missing_language(resume, analysis, AI_POST) is None


def test_with_no_language_in_the_post_any_language_counts_for_work_built_in_code(profile):
    analysis = posting(must=["LLM evaluation", "RAG"])
    resume = current_job_resume(profile, NO_LANGUAGE)
    assert validate.missing_language(resume, analysis, "Build our AI platform.") == ("Northwind Labs", None)
    resume.experience[0].bullets[1] = NO_LANGUAGE[1].replace("on Kubernetes", "in Go on Kubernetes")
    assert validate.missing_language(resume, analysis, "Build our AI platform.") is None
    nurse = posting("Registered Nurse", "", must=["patient care"])
    assert validate.missing_language(current_job_resume(profile, NO_LANGUAGE), nurse, "Care for patients.") is None


def test_the_current_job_is_the_one_marked_present(profile):
    analysis = posting(must=["Python"])
    resume = current_job_resume(profile, NO_LANGUAGE, older=["Built the reporting service in Python for 40 teams."])
    resume.profile.companies[0].end = "Oct 2026"  # no job is current: the most recent one counts
    assert validate.missing_language(resume, analysis, "") == ("Northwind Labs", "Python")


def drop_language(sections):
    """The current job's bullets without the programming languages the demo named."""
    current = sections["experience"][0]
    current["bullets"] = [bullet.replace(" in Python", "").replace("Python", "Kubernetes")
                          for bullet in current["bullets"]]


def test_a_current_job_without_the_posts_language_asks_for_the_fix_round(router, db, registry):
    client, result = tailored(router, db, change=drop_language, registry=registry)
    assert result.retries == 1
    assert ("Name Python in one of Northwind Labs' bullets, where the work used it: the job post asks for it most"
            in client.fix_round())
    assert validate.missing_language(result.resume, result.analysis, AI_POST) is None
    assert not any("programming language" in w for w in result.ats_warnings)


def test_with_no_language_in_the_post_the_fix_round_asks_for_the_usual_one(router, db, registry):
    post = AI_POST.replace("Required: Python, ", "Required: ")
    client, result = tailored(router, db, jd=post, change=drop_language, registry=registry)
    assert result.retries == 1
    assert ("Northwind Labs, the current job, names no programming language: name the one this work is usually built "
            "in (Python, unless its notes name another)") in client.fix_round()
    assert validate.missing_language(result.resume, result.analysis, post) is None


def test_run_checks_warn_while_the_current_job_misses_the_language(profile):
    analysis = posting(must=["Python", "RAG"], keywords=["Python", "RAG"])
    resume = current_job_resume(profile, NO_LANGUAGE)
    result = TailorResult(requested_model="demo-concise", resume=resume, analysis=analysis, jd=AI_POST,
                          page_target=1, style=design_style("Letter"), metrics_mode="estimate", spelling="us")
    run_checks(result, AI_POST)
    assert any(w.startswith("Northwind Labs, the current job, does not name Python") for w in result.ats_warnings)
    old = result.model_copy(update={"page_target": 0, "ats_warnings": []})  # a run saved before: as it was
    run_checks(old, AI_POST)
    assert not any("Python" in w and "current job" in w for w in old.ats_warnings)
    none = result.model_copy(update={"analysis": posting(must=["RAG"], keywords=["RAG"]), "ats_warnings": []})
    run_checks(none, "Build the platform behind our AI features.")  # a post that names no language
    assert any(w.startswith("Northwind Labs, the current job, names no programming language. Name the one you built "
                            "this work in (for example Python)") for w in none.ats_warnings)


def test_the_posts_language_is_flagged_where_the_notes_do_not_name_it(profile):
    analysis = posting(must=["LLM evaluation"], keywords=["Go", "LLM evaluation"])  # Go is no must-have
    bullets = ["Write the evaluation runner in Go for the assistant platform, cutting failed releases from 41 to 28.",
               NO_LANGUAGE[1]]
    resume = current_job_resume(profile, bullets)
    flags = validate.posting_match_flags(resume, analysis, "", language="Go")
    assert [(f.section, f.reason) for f in flags if "(Go)" in f.reason] == [(
        "experience:0", "Added to match the posting (Go). Confirm you can speak to it in an interview, or remove it.")]
    assert not [f for f in validate.posting_match_flags(resume, analysis, "Wrote Golang services at Northwind Labs.",
                                                        language="Go") if "Go" in f.reason]
    assert not [f for f in validate.posting_match_flags(resume, analysis, "") if "(Go)" in f.reason]  # none asked


def test_run_checks_flag_the_posts_language_only_on_new_runs(profile):
    analysis = posting(must=["LLM evaluation"], keywords=["Go", "LLM evaluation"])
    bullets = ["Write the evaluation runner in Go for the assistant platform, cutting failed releases from 41 to 28.",
               NO_LANGUAGE[1]]
    result = TailorResult(requested_model="demo-concise", resume=current_job_resume(profile, bullets),
                          analysis=analysis, jd="Senior AI Platform Engineer. Go, LLM evaluation.", page_target=1,
                          style=design_style("Letter"), metrics_mode="estimate", spelling="us")
    run_checks(result, result.jd)
    assert any(f.section == "experience:0" and "(Go)" in f.reason for f in result.flags)
    old = result.model_copy(update={"page_target": 0, "flags": []})
    run_checks(old, old.jd)
    assert not any("(Go)" in f.reason for f in old.flags)


def test_the_skills_keep_the_language_the_bullets_show(profile):
    elixir = "Write the evaluation runner in Elixir for the assistant platform, cutting failed releases from 41 to 28."
    resume = current_job_resume(profile, [elixir, NO_LANGUAGE[1]])
    resume.skills = [SkillGroup(category="Languages", skills=["Elixir", "Haskell"])]
    rules.shape_skills(resume, posting(must=["RAG"]))
    assert resume.skills[0].skills == ["Elixir"]  # shown in a bullet: kept, though the post does not ask for it


def test_the_prompt_asks_for_the_posts_language_at_the_current_job():
    text = tailor_system("estimate")
    assert "names one programming language in at least one bullet, where the work used it" in text
    assert "Python for AI, LLM and data platform work" in text
    assert "the current job holds about half of all bullets" in text
    assert "cover the job post's must-haves and core responsibilities, one per bullet, strongest first" in text


# ------------------------------------------------------------------ D. the demo model, the prompt version
def test_the_demo_climbs_into_the_target_field_and_names_the_posts_language(router, db):
    _client, result = tailored(router, db)
    resume = result.resume
    assert [e.job_title for e in resume.experience] == [
        "Senior AI Platform Engineer", "AI Platform Engineer", "Platform Engineer", "Software Engineer",
        "Junior Software Engineer"]
    assert resume.headline == "Senior AI Platform Engineer" and resume.summary.startswith(resume.headline + " with ")
    assert validate.missing_language(resume, result.analysis, AI_POST) is None
    assert any("Python" in bullet for bullet in resume.experience[0].bullets)
    assert result.retries == 0
    assert any(f.section == "experience:0" and "Python" in f.reason and f.reason.startswith("Added to match")
               for f in result.flags)  # the person never named it: they confirm it


def test_the_demo_names_python_for_ai_work_when_the_post_names_no_language(router, db):
    _client, result = tailored(router, db, jd=AI_POST.replace("Required: Python, ", "Required: "))
    assert any("Python" in bullet for bullet in result.resume.experience[0].bullets)
    assert result.retries == 0


def test_a_post_for_work_not_built_in_code_asks_for_no_language(router, db):
    _client, result = tailored(router, db, jd=MARKETING_POST)
    assert result.retries == 0
    assert validate.missing_language(result.resume, result.analysis, MARKETING_POST) is None
    assert not any("programming language" in w for w in result.ats_warnings)


def test_the_prompt_version_is_v39():
    assert PROMPT_VERSION == "v39"
