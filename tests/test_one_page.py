"""One page, cut words not the font (a reviewer's rule, the owner's choice): the plan, the trimming, whole jobs on a page
break, and the run end to end with the demo model. Every person, employer and school here is invented."""
import asyncio
import io
from datetime import date

from docx import Document

from app.pipeline import page_fit
from app.pipeline.fit import LADDER
from app.pipeline.page_fit import (
    LAYOUT,
    design_style,
    extra_order,
    fit_new,
    job_relevance,
    measure,
    plan_pages,
    removal_order,
    trim_to_page,
)
from app.pipeline.tailor import fit_report, refit, run_tailor
from app.render.docx import resume_docx
from app.render.pdf import education_runs
from app.schemas import (
    Company,
    Education,
    FitStyle,
    JDAnalysis,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
)

from .test_render import _lines_by_page

TODAY = date(2026, 11, 20)
BULLET = ("Rebuilt the order sync service behind the partner REST API in Go, adding retries and rate limits for partner "
          "integrations")  # 20 words: two lines


def career(jobs: int, notes: dict[int, str] | None = None) -> Profile:
    notes = notes or {}
    return Profile(name="Taylor Quinn", companies=[
        Company(name=f"Employer {i}", start=f"Jan {2024 - 2 * i}", end="Present" if i == 0 else f"Dec {2025 - 2 * i}",
                projects=notes.get(i, "")) for i in range(jobs)],
        education=[Education(school="Tailspin University", degree="B.S.", field="Computer Science",
                             location="St. Louis, MO", end="2014")])


def resume_with(profile: Profile, counts: list[int], bullet: str = BULLET) -> Resume:
    return Resume(profile=profile, headline="Backend Engineer",
                  summary="Backend engineer with 9+ years of experience building commerce platforms. Focuses on "
                          "partner integrations now, cutting failed syncs from 41 to 6 a week.",
                  skills=[SkillGroup(category="Languages", skills=["Go", "Python", "SQL"])],
                  experience=[ResumeExperience(company_index=i, job_title="Backend Engineer", bullets=[bullet] * n)
                              for i, n in enumerate(counts)])


def test_a_plan_fills_one_page_at_the_design_size_and_the_newest_job_leads():
    for jobs in (2, 3, 4, 5, 6, 7):
        profile = career(jobs)
        pages, counts = plan_pages(profile, today=TODAY)
        assert pages == 1 and len(counts) == jobs and min(counts) >= 1, (jobs, counts)
        assert counts[0] > max(counts[1:]), counts  # the most recent job always gets the most
        plan = page_fit._plan_resume(profile, counts)
        assert page_fit.measure(plan) == 1  # the placeholder text fits at the design's own sizes
        more = [n + 1 if i == 0 else n for i, n in enumerate(counts)]
        if counts[0] < page_fit.bullet_caps(profile.companies, TODAY)[0]:
            assert page_fit.measure(page_fit._plan_resume(profile, more)) == 2  # one more bullet would not fit


def test_two_pages_only_when_one_bullet_per_older_job_does_not_fit():
    pages, counts = plan_pages(career(12), today=TODAY)
    assert pages == 2 and counts[0] > max(counts[1:]) and min(counts) >= 1
    assert page_fit.measure(page_fit._plan_resume(career(12), [2] + [1] * 11)) == 2  # the one-page minimum overflows


def test_the_job_whose_notes_match_the_posting_gets_the_extra_bullet():
    jd = "Senior Backend Engineer: payments, Stripe webhooks, idempotent retries and ledger reconciliation in Go."
    profile = career(4, {2: "Payments ledger: Stripe webhooks with idempotent retries and nightly reconciliation"})
    relevance = job_relevance(profile, jd)
    assert relevance[2] == 1.0 and relevance[1] == relevance[3] == 0.0
    caps, start = [4, 3, 3, 3], [2, 1, 1, 1]
    order = [index for index, _level in extra_order(profile, caps, start, TODAY, relevance) if index]
    plain = [index for index, _level in extra_order(profile, caps, start, TODAY) if index]
    assert order.index(2) < plain.index(2)  # handed its extra sooner than without the notes


def test_trimming_keeps_the_lead_the_first_bullets_and_a_must_haves_only_mention():
    profile = career(3)
    resume = resume_with(profile, [3, 3, 3])
    resume.experience[2].bullets[2] = "Moved billing reports to dbt models with tests, cutting month-end fixes in half"
    analysis = JDAnalysis(target_role="Backend Engineer", seniority="", domain="", must_have_skills=["dbt"],
                          nice_to_have_skills=[], keywords=[])
    order = removal_order(resume, analysis)
    assert (2, 2) == order[-1]  # the only mention of a must-have goes last
    assert all(index > 0 for _position, index in order) and len(order) == 6  # a job's first bullet never goes
    left = [3, 3, 3]
    for position, _index in order[:-1]:
        if position == 0:  # the most recent job gives a bullet only while it still leads, or no other job can give
            assert left[0] - 1 > max(left[1:]) or all(n == 1 for n in left[1:]) or left[2] == 2
        left[position] -= 1
    assert order[0][0] != 0  # from [3, 3, 3] an older job gives first


def test_trim_to_page_removes_the_fewest_bullets_that_fit():
    resume = resume_with(career(3), [9, 8, 8])
    trimmed = trim_to_page(resume, "Letter", 1)
    assert trimmed.pages == 1 and trimmed.removed
    order = removal_order(resume)
    one_fewer = page_fit._without(resume, order, len(trimmed.removed) - 1)
    assert measure(one_fewer) == 2  # one removal fewer would not fit
    assert [len(e.bullets) for e in resume.experience] == [9, 8, 8]  # the draft itself is never changed


def test_one_page_out_of_reach_gives_two_pages_from_the_untrimmed_draft():
    profile = career(12)
    resume = resume_with(profile, [3] * 12)
    fitted = fit_new(resume, "Letter", 1)
    assert fitted.pages == 2 and fitted.style == design_style("Letter") and fitted.style.layout == LAYOUT
    assert any(note.startswith("Kept two pages") for note in fitted.notes)
    assert sum(len(e.bullets) for e in fitted.resume.experience) > 12  # not trimmed down to one bullet a job


def test_the_font_is_reduced_only_when_two_pages_are_out_of_reach():
    resume = resume_with(career(40), [1] * 40)
    fitted = fit_new(resume, "Letter", 1)
    assert fitted.style.font_pt < LADDER[0][0] and fitted.style.layout == LAYOUT
    assert any(note.startswith("Font reduced") for note in fitted.notes)


def test_two_pages_never_split_a_job_in_the_pdf():
    profile = career(4)
    resume = resume_with(profile, [6, 6, 6, 6])
    pages, drawn, _indents = _lines_by_page(resume, design_style("Letter"))
    assert pages == 2
    by_job, job = {}, -1
    for page, _y, _x, text in drawn:
        if text == "EDUCATION":
            break
        if text.startswith("Backend Engineer · "):
            job += 1
        if job >= 0:
            by_job.setdefault(job, set()).add(page)
    assert len(by_job) == 4 and all(len(pages_of_job) == 1 for pages_of_job in by_job.values()), by_job


def test_word_keeps_a_job_together_only_in_the_new_layout():
    resume = resume_with(career(2), [3, 2])
    new = Document(io.BytesIO(resume_docx(resume, design_style("Letter"))))
    bullets = [p for p in new.paragraphs if p.style.name == "List Bullet"]
    assert [bool(p.paragraph_format.keep_with_next) for p in bullets] == [True, True, False, True, False]
    old = Document(io.BytesIO(resume_docx(resume, FitStyle())))
    assert not any(p.paragraph_format.keep_with_next for p in old.paragraphs if p.style.name == "List Bullet")


def test_the_education_row_names_the_degree_and_the_town_in_the_new_layout():
    edu = Education(school="Tailspin University", degree="B.S.", field="Computer Science", location="St. Louis, MO")
    assert "".join(run.text for run in education_runs(edu, 1)) == (
        "B.S. in Computer Science · Tailspin University · St. Louis, MO")
    assert "".join(run.text for run in education_runs(edu, 0)) == "B.S., Computer Science · Tailspin University"
    faculty = Education(school="Tailspin University", degree="Master of Science", field="School of Engineering")
    assert "".join(run.text for run in education_runs(faculty, 1)) == (
        "Master of Science, School of Engineering · Tailspin University")


def test_a_demo_run_fits_one_page_at_the_design_size_without_a_rewrite(router, db, profile, jd):
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    save=False))
    assert result.status == "ok" and result.pages == 1 and result.page_target == 1 and result.fits
    assert result.style == design_style("Letter") and (result.style.font_pt, result.style.margin_in) == LADDER[0]
    assert result.retries == 0 and not any(n.startswith(("Removed", "Font reduced")) for n in result.notes)
    names = [name for name, _seconds in result.timing.steps]
    assert names.index("layout") < names.index("pdf")


def test_a_hand_edit_over_the_page_keeps_the_font_and_says_so(profile):
    resume = resume_with(profile, [8, 7, 7])
    style, pages, fits, notes = asyncio.run(fit_report(resume, "Letter", 1, design_style("Letter")))
    assert style == design_style("Letter") and pages == 2 and fits
    assert notes and notes[0].startswith("Still 2 pages") and "one page" in notes[0]
    old_style, _pages, _fits, _notes = asyncio.run(fit_report(resume, "Letter"))
    assert old_style.layout == 0  # a run saved before keeps the old fitting


def test_a_certification_tick_refits_to_the_runs_own_page(profile):
    resume = resume_with(profile, [8, 7, 7])
    _trimmed, style, pages, fits, notes, removed = asyncio.run(refit(resume, "Letter", 1))
    assert pages == 1 and fits and style.layout == LAYOUT and sum(removed.values()) > 0
    assert notes[-1].startswith("Removed") and "one page" in notes[-1]
