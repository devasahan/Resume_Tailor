"""The owner's rules for the current job and the title path (prompt v39): the line under the name, the summary's first
words and the current job's title read the same ('Senior AI Platform Engineer'); older titles climb into the target's
field (Junior Software Engineer, Software Engineer, Platform Engineer, AI Platform Engineer, Senior AI Platform
Engineer); the current job names the job post's most wanted programming language (or, when the post names none, the
one its kind of work is usually built in), flagged for the person to confirm when their notes do not name it. Every
person, employer and product here is invented."""
import asyncio
import time

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import languages, rules, validate
from app.pipeline.page_fit import design_style
from app.pipeline.tailor import language_fix, language_warning, run_checks, run_tailor
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
    ("Senior AI platform engineer with 9+ years of Python platform work for retail and learning products.",
     "Senior AI Platform Engineer with 9+ years of Python platform work for retail and learning products."),
    ("AI Platform Engineer with 9+ years of Python platform work.",  # the level comes back
     "Senior AI Platform Engineer with 9+ years of Python platform work."),
    ("ai platform engineer with 9+ years of Python platform work.",
     "Senior AI Platform Engineer with 9+ years of Python platform work."),
    ("Staff AI Platform Engineer with 9+ years of Python platform work.",  # one title, one level
     "Senior AI Platform Engineer with 9+ years of Python platform work."),
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
    assert resume.headline == headline == resume.experience[0].job_title  # the job line reads the same
    assert resume.summary.startswith(headline + " with 9+ years")


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
    assert "HEADLINE: the current job's title in your plan (the job marked \"Present (current role)\" in" in text
    assert "the line under the name, the summary's first words and the current job read the same" in text
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
    # what a nurse's, a driver's or a call centre's post says (review LANG-1), and databases (LANG-7)
    ("Coordinate Hepatitis C testing for the clinic", []),
    ("A valid Class C driver's license and reliable transportation", []),
    ("Class C CDL drivers", []),
    ("Build Erlang C staffing models for 3 contact centres", []),
    ("Experience in R & D budgeting", []),
    ("Tune SQL Server queries on Azure SQL and Cloud SQL", []),
    ("Write T-SQL procedures on SQL Server", ["SQL"]),
    ("Write C firmware and R packages", ["C", "R"]),
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
    post.must_have_skills = ["SQL", "dbt"]  # before SQL even when the post only prefers it (review LANG-7)
    assert languages.wanted_language(post) == "Rust"
    post.nice_to_have_skills, post.keywords = [], ["dbt"]  # SQL is the post's language here
    assert languages.wanted_language(post) == "SQL"
    post.nice_to_have_skills, post.keywords = ["Rust"], ["Kubernetes", "TypeScript"]
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
    ("Software Developer", "", "Python"),
    ("Application Developer", "", "Python"),
    # a developer or programmer of work not built in code, and configuration work (review LANG-3)
    ("Business Developer", "Business Development Manager", None),
    ("Instructional Developer", "", None),
    ("Curriculum Developer", "", None),
    ("Real Estate Developer", "", None),
    ("CNC Programmer", "", None),
    ("Epic Applications Analyst", "Applications Analyst", None),
    ("Application Support Engineer", "", None),
    # a title that names its own stack or language
    ("Senior React Developer", "", "TypeScript"),
    ("Node.js Developer", "", "TypeScript"),
    (".NET Developer", "", "C#"),
    ("Ruby on Rails Developer", "", "Ruby"),
    ("Senior Java Developer", "Java Developer", "Java"),
    ("Golang Engineer", "", "Go"),
    ("SQL Server Developer", "", "SQL"),
    ("Go To Market Engineer", "", None),  # a sales title: no Go in it
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
    # the review's fixes (LANG-1, LANG-4): a requirement that offers two, and work not built in code
    assert "a requirement that offers two (\"C/C++\", \"Python or Go\") is met by either" in text
    assert ("Work that is not built in code (nursing, sales, marketing) names no language, unless one of the post's "
            "required skills is a language on its own (then that one)") in text


# ------------------------------------------------------------------ C2. the review's language findings
NURSE_POST = (
    "Registered Nurse, Home Health at Contoso Health. Provide skilled nursing visits, wound care and patient education "
    "for adults recovering at home. Requirements: active RN license, BLS certification, Hepatitis C testing, Epic "
    "charting, and a valid Class C driver's license with reliable transportation."
)
WFM_POST = (
    "Workforce Management Analyst at Fabrikam. Build Erlang C staffing models and intraday forecasts for 3 contact "
    "centres. Required: Erlang C, forecasting, Excel."
)
BIZDEV_POST = (
    "Business Developer at Fabrikam. Find and win new B2B accounts across the Benelux, run discovery calls, manage the "
    "sales pipeline in HubSpot and negotiate annual contracts. Required: B2B sales, HubSpot, negotiation, pipeline "
    "management."
)
GO_POST = (
    "Senior Backend Engineer at Contoso. You will build the payment services behind our checkout: Go microservices, "
    "gRPC APIs, Kubernetes and PostgreSQL. Required: Go, gRPC, Kubernetes, PostgreSQL, CI/CD. Nice to have: Kafka, "
    "Terraform."
)
CARE = "Lead skilled nursing visits for 38 home health patients a week, cutting readmissions from 41 to 28 a quarter."
NOT_CODE = [  # (target, general, must, nice, keywords, post)
    ("Registered Nurse", "Registered Nurse", ["patient education", "wound care", "Epic", "Hepatitis C testing"], [],
     ["home health", "Class C driver's license"], NURSE_POST),
    ("Workforce Management Analyst", "Workforce Analyst", ["Erlang C", "forecasting", "Excel"], [],
     ["Erlang C staffing models"], WFM_POST),
    ("Senior Marketing Manager", "Marketing Manager", ["HubSpot", "SEO", "A/B testing"], ["SQL", "Python"],
     ["HubSpot", "SQL", "Python"], MARKETING_POST + " Nice to have: SQL, Python."),
    ("Business Developer", "Business Development Manager", ["B2B sales", "HubSpot"], [], ["HubSpot"], BIZDEV_POST),
]


def checked(resume, analysis, jd, hints=""):
    """A new run's checks (page_target 1) on this resume."""
    result = TailorResult(requested_model="demo-concise", resume=resume, analysis=analysis, jd=jd, page_target=1,
                          style=design_style("Letter"), metrics_mode="estimate", spelling="us", hints=hints)
    run_checks(result, jd)
    return result


@pytest.mark.parametrize("target,general,must,nice,keywords,jd", NOT_CODE)
def test_work_not_built_in_code_is_asked_for_no_language(profile, target, general, must, nice, keywords, jd):
    """Review LANG-1: a false match in the post ('Class C driver's license', 'Hepatitis C testing', 'Erlang C') or a
    nice-to-have language never makes a nurse, an analyst, a marketer or a salesperson name a language."""
    analysis = posting(target, general, must=must, nice=nice, keywords=keywords)
    assert not languages.codes(analysis)
    assert languages.wanted_language(analysis, jd) is None
    resume = current_job_resume(profile, [CARE], older=[CARE])
    assert validate.missing_language(resume, analysis, jd) is None
    result = checked(resume, analysis, jd)
    assert not [w for w in result.ats_warnings if "programming language" in w]
    assert not [f for f in result.flags if f.reason.startswith("Added to match")]


def test_work_not_built_in_code_names_a_language_its_post_requires_on_its_own(profile):
    analysis = posting("Marketing Analyst", "Marketing Analyst", must=["SQL", "HubSpot", "SEO"])
    assert not languages.codes(analysis)
    resume = current_job_resume(profile, [CARE], older=[CARE])
    assert validate.missing_language(resume, analysis, "") == ("Northwind Labs", "SQL")
    analysis.must_have_skills = ["SQL or Python", "HubSpot"]  # a language among others: no demand
    assert validate.missing_language(resume, analysis, "") is None


def drop_any_language(sections):
    """The current job's bullets without the language the demo wrote ('in C'), as a real model writes a nurse's or a
    salesperson's work."""
    current = sections["experience"][0]
    for language in languages.LANGUAGES:
        current["bullets"] = [bullet.replace(f" in {language.name} ", " ") for bullet in current["bullets"]]


@pytest.mark.parametrize("jd", [post for *_analysis, post in NOT_CODE])
def test_a_run_for_work_not_built_in_code_pays_for_no_language_rewrite(router, db, registry, jd):
    """Review LANG-1 and LANG-3: the demo names no language for a nurse, a call centre's analyst, a marketer or a
    business developer, and no fix round asks for one (three jobs: with five, the demo's title ladder for a post it
    cannot read asks a fix round of its own)."""
    client, result = tailored(router, db, jd=jd, person=five_jobs(3), change=drop_any_language, registry=registry)
    assert "programming language" not in client.fix_round() and "Name C in" not in client.fix_round()
    assert result.retries == 0
    assert validate.missing_language(result.resume, result.analysis, jd) is None
    assert not [w for w in result.ats_warnings if "programming language" in w]
    assert not languages.named("\n".join(result.resume.experience[0].bullets))


COMMON_FORMS = [
    ("Go", "Rebuild the Go payment gateway on Kubernetes, cutting p99 latency from 410 to 180 ms for 2,300 merchants."),
    ("Go", "Port the settlement ledger from Python to Go, cutting CPU cost by 41,000 dollars a year."),
    ("Go", "Rewrite the scheduler with Go's goroutines and channels, cutting p99 latency from 410 to 180 ms."),
    ("Go", "Own the Golang-based ingestion daemon that streams 41 million events a day into Kafka."),
    ("Rust", "Write a Rust parser for partner CSV feeds, cutting malformed imports from 312 to 18 a month."),
    ("Ruby", "Upgrade the Ruby monolith from Rails 5 to 7 for 2,300 merchants, removing 41 deprecated gems."),
    ("C++", "Rewrite the order book in C++17, cutting matching latency from 41 to 12 microseconds."),
    ("Python", "Build Python-based evaluation harnesses for the support assistant, replaying 3,142 conversations."),
    ("Python", "Write Python3 ETL jobs on Airflow for 40 teams, cutting failed runs from 101 to 83 a month."),
    ("Java", "Rebuild the Java-based pricing engine on Spring Boot, cutting quote time from 4.1 to 1.3 seconds."),
    ("TypeScript", "Build TypeScript-first React components for the merchant dashboard, cutting UI bugs by 41."),
    ("C#", "Write C#.NET services for the claims system, cutting average response time from 410 to 180 ms."),
    ("C", "Port the bootloader to C11 on ARM Cortex-M4, freeing 41 KB of flash for new sensor features."),
    ("C", "Write C drivers for the sensor hub, cutting boot time from 2.1 to 0.6 seconds."),
    ("R", "Fit mixed-effects R models for 31 clinical sites, cutting the analysis cycle from 6 to 2 weeks."),
    ("R", "Write R Shiny dashboards for 40 analysts, replacing weekly Excel exports."),
]


@pytest.mark.parametrize("language,bullet", COMMON_FORMS)
def test_a_current_job_naming_the_language_in_a_common_form_is_not_missing_it(profile, language, bullet):
    """Review LANG-2: the way people write a language ('the Go payment gateway', 'from Python to Go', "Go's", 'a Rust
    parser', 'C++17', 'Python-based') names it: no rewrite is asked for, and no warning says it is missing."""
    assert languages.names(bullet, language)
    analysis = posting("Senior Backend Engineer", "Backend Engineer", must=[language, "Kubernetes"],
                       keywords=[language])
    resume = current_job_resume(profile, [bullet, NO_LANGUAGE[1]])
    assert validate.missing_language(resume, analysis, "") is None
    assert not [w for w in checked(resume, analysis, GO_POST).ats_warnings if "programming language" in w]


@pytest.mark.parametrize("language,text", [
    ("Go", "Plan the Go-to-market launch for 3 regions"),
    ("Go", "Drive the go-live of the new checkout"),
    ("Rust", "Apply Rust-proof coating to the bridge"),
    ("Java", "Write JavaScript widgets"),
    ("C", "Ship iOS apps in Objective-C"),
    ("C", "Raise the team to Series C with C-level sponsors"),
    ("R", "Grow the R&D budget and the R & D team"),
    ("Erlang", "Build Erlang C staffing models"),
])
def test_names_still_reads_everyday_words_as_words(language, text):
    assert not languages.names(text, language)


def go_in_common_forms(sections):
    """The current job names Go only as people write it: 'the Go payment gateway', 'from Python to Go'."""
    current = sections["experience"][0]
    current["bullets"] = [
        bullet.replace("Rebuild the Go pipeline", "Rebuild the Go payment gateway")
        .replace("Write the PostgreSQL services in Go", "Port the PostgreSQL ledger from Python to Go")
        .replace("Design Kubernetes components in Go", "Design the Go settlement scheduler on Kubernetes")
        for bullet in current["bullets"]]
    assert not any(" in Go" in bullet for bullet in current["bullets"])  # the change took


def test_a_run_whose_current_job_names_the_language_in_common_forms_pays_for_no_rewrite(router, db, registry):
    client, result = tailored(router, db, jd=GO_POST, change=go_in_common_forms, registry=registry)
    assert any("the Go payment gateway" in bullet for bullet in result.resume.experience[0].bullets)
    assert "Name Go" not in client.fix_round()
    assert result.retries == 0
    assert not [w for w in result.ats_warnings if "programming language" in w]


def test_with_no_language_in_the_post_node_js_names_javascript(profile):
    """Review LANG-3: a Node.js post that names no JavaScript is work built in code, and 'Node.js' at the current job
    names its language."""
    analysis = posting("Node.js Developer", "", must=["Node.js", "Express", "MongoDB"])
    assert languages.wanted_language(analysis) is None and languages.usual_language(analysis) == "TypeScript"
    bullets = ["Build the order APIs on Node.js and Express for 41 partners, cutting p99 latency from 410 to 180 ms.",
               NO_LANGUAGE[1]]
    assert validate.missing_language(current_job_resume(profile, bullets), analysis, "") is None
    assert validate.missing_language(current_job_resume(profile, NO_LANGUAGE), analysis, "") == (
        "Northwind Labs", None)


@pytest.mark.parametrize("must,bullet", [
    (["C/C++", "Linux", "RTOS"], "Rewrite the sensor fusion module in C++ on FreeRTOS, cutting frame drops to 6."),
    (["Python or Go", "Kubernetes"], "Rebuild the ingestion workers in Go on Kubernetes, cutting lag to 6 minutes."),
    (["Java/Kotlin", "Spring Boot"], "Rebuild the payments API in Kotlin on Spring Boot, cutting latency to 180 ms."),
    (["JavaScript/TypeScript", "React"], "Build the React checkout in TypeScript, lifting completed orders to 68%."),
])
def test_a_requirement_that_offers_two_languages_accepts_either(profile, must, bullet):
    """Review LANG-4: 'C/C++' or 'Python or Go' is met by either language; the post's first is not required."""
    analysis = posting("Senior Software Engineer", "Software Engineer", must=must)
    assert len(languages.wanted_languages(analysis)) == 2
    resume = current_job_resume(profile, [bullet, NO_LANGUAGE[1]])
    assert validate.missing_language(resume, analysis, "") is None
    result = checked(resume, analysis, "")
    assert not [w for w in result.ats_warnings if "programming language" in w]
    flags = [f for f in result.flags if f.reason.startswith(f"Added to match the posting ({must[0]}")]
    assert [f.section for f in flags] == ["experience:0"] and flags[0].text == bullet  # the one the job names


def test_a_bullet_is_read_as_the_rule_reads_the_current_job():
    """validate.language_rule and meets_language_rule: what the rule asks of the current job, and whether one bullet
    gives it, read as missing_language reads the job (so trimming can hold the bullet that gives it)."""
    assert validate.language_rule(posting(must=["C/C++", "RTOS"]), "") == ["C", "C++"]
    assert validate.language_rule(posting(must=["RAG"]), "") == []  # AI work whose post names none: any language
    assert validate.language_rule(posting("Registered Nurse", "", must=["wound care"]), NURSE_POST) is None
    assert validate.meets_language_rule("Rewrite the order book in C++17, cutting latency to 12 microseconds.",
                                        ["C", "C++"])
    assert validate.meets_language_rule("Build the order APIs on Node.js for 41 partners.", [])
    assert not validate.meets_language_rule("Plan the Go-to-market launch for 3 regions.", ["Go"])
    assert not validate.meets_language_rule("Plan the launch for 3 regions.", [])


def test_a_job_naming_neither_of_two_languages_is_asked_for_either(profile):
    analysis = posting("Senior Embedded Engineer", "Embedded Engineer", must=["C/C++", "RTOS"])
    resume = current_job_resume(profile, NO_LANGUAGE)
    missing = validate.missing_language(resume, analysis, "")
    assert missing == ("Northwind Labs", "C or C++")
    assert language_fix(*missing, analysis).startswith(
        "Name C or C++ (either one) in one of Northwind Labs' bullets, where the work used it")
    assert language_warning(*missing, analysis).startswith(
        "Northwind Labs, the current job, does not name C or C++ (either one)")


def test_mentioned_is_quick_on_long_crafted_text():
    """Review LANG-5: hints, notes and posts are read in linear time ('C C C ... Python' took minutes)."""
    started = time.perf_counter()
    assert languages.mentioned("C " * 600 + "Python") == ["C", "Python"]
    assert languages.mentioned("Go, " * 600 + "Python") == ["Go", "Python"]
    assert not languages.names("C " * 600 + "and more", "Python")
    assert languages.mentioned("Go" + " " * 1500 + "to market") == []  # a name first, then a long run of spaces
    assert languages.mentioned("Go" + " " * 1500 + "1.22") == ["Go"]
    assert time.perf_counter() - started < 0.5


def test_posting_flags_are_quick_on_the_longest_hints(profile):
    analysis = posting(must=["Python", "RAG"], keywords=["Python"])
    resume = current_job_resume(profile, ["Build the RAG service in Python, replaying 3,142 conversations a release."])
    hints = ("C " * 2000)[:3994] + "Python"  # the 4,000 characters a person may type
    started = time.perf_counter()
    validate.posting_match_flags(resume, analysis, hints, language="Python")
    languages.wanted_language(posting(must=["RAG"]), "R, " * 6000 + "Go")  # a 20,000-character post
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("language,bullet,hints", [
    ("Go", "Rebuild the partner webhook gateway in Go, cutting p99 latency from 410 to 180 ms for 2,300 merchants.",
     "Northwind Labs: I wrote the Go gateway for partner webhooks."),
    ("R", "Build churn forecasts in R with tidymodels, letting retention target 4,100 at-risk accounts each quarter.",
     "Northwind Labs: I used R for the churn forecasts."),
    ("Rust", "Write the CSV feed parser in Rust, cutting malformed partner imports from 312 to 18 a month.",
     "Northwind Labs: wrote a Rust parser for partner feeds."),
    ("Python", "Build the RAG evaluation harness in Python for the support assistant, replaying 3,142 conversations.",
     "At Northwind Labs I built Python-based evaluation tooling and the RAG service."),
])
def test_hints_that_name_the_language_back_it(profile, language, bullet, hints):
    """Review LANG-6: the person's own hints back the language however they write it: no 'Added to match' flag."""
    analysis = posting("Senior Backend Engineer", "Backend Engineer", must=[language, "Kubernetes"],
                       keywords=[language])
    resume = current_job_resume(profile, [bullet])
    flags = validate.posting_match_flags(resume, analysis, hints, language=language)
    assert not [f for f in flags if f"({language}" in f.reason]
    assert [f for f in validate.posting_match_flags(resume, analysis, "", language=language)
            if f"({language}" in f.reason]  # without the hints it is flagged


@pytest.mark.parametrize("target,must,nice,keywords,wanted,bullet", [
    ("Senior DevOps Engineer", ["Bash", "Terraform", "Kubernetes"], ["Python"], [], "Python",
     "Write the Python deploy service for 40 Kubernetes clusters with Terraform, cutting failed rollouts to 6."),
    ("Senior .NET Developer", [".NET", "SQL Server", "Azure"], [], ["C#", ".NET"], "C#",
     "Build the claims API in C# on .NET 8 and Azure, cutting average response time from 410 to 180 ms."),
])
def test_a_language_the_work_is_built_in_beats_sql_or_a_shell(profile, target, must, nice, keywords, wanted, bullet):
    """Review LANG-7: the post's language comes before SQL or a shell across its must-haves, nice-to-haves and
    keywords, as the prompt says; 'SQL Server' is a database, not the SQL language."""
    analysis = posting(target, "", must=must, nice=nice, keywords=keywords)
    assert languages.wanted_language(analysis) == wanted
    resume = current_job_resume(profile, [bullet, NO_LANGUAGE[1]])
    assert validate.missing_language(resume, analysis, "") is None


def test_the_usual_language_added_when_the_post_names_none_is_flagged(profile):
    """Review LANG-8: when the post names no language, the current job names the usual one (Python for AI work), and
    the bullet is flagged for the person to confirm unless their hints or notes name it, as the README says."""
    analysis = posting(must=["LLM evaluation", "RAG", "Kubernetes"], keywords=["RAG", "Kubernetes"])
    bullet = "Build the answer-quality service in Python for the support assistant, replaying 3,142 chats per release."
    resume = current_job_resume(profile, [bullet, NO_LANGUAGE[0]], older=["Built the reporting service for 40 teams."])
    jd = "Senior AI Platform Engineer at Contoso. Required: LLM evaluation, RAG, Kubernetes."
    assert validate.missing_language(resume, analysis, jd) is None
    flags = [f for f in checked(resume, analysis, jd).flags if "Python" in f.reason]
    assert [(f.section, f.text, f.reason) for f in flags] == [("experience:0", bullet, (
        "Added to match the posting (Python). Confirm you can speak to it in an interview, or remove it."))]
    assert not [f for f in checked(resume, analysis, jd, hints="I wrote Python services at Northwind Labs.").flags
                if "Python" in f.reason]


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
    assert any(f.section == "experience:0" and f.reason.startswith("Added to match the posting (Python)")
               for f in result.flags)  # the person never named it: they confirm it (review LANG-8)


def test_a_post_for_work_not_built_in_code_asks_for_no_language(router, db):
    _client, result = tailored(router, db, jd=MARKETING_POST)
    assert result.retries == 0
    assert validate.missing_language(result.resume, result.analysis, MARKETING_POST) is None
    assert not any("programming language" in w for w in result.ats_warnings)


def test_the_prompt_version_is_v39():
    assert PROMPT_VERSION == "v39"
