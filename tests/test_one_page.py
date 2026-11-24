"""One page, cut words not the font (a reviewer's rule, the owner's choice): the plan, the trimming, whole jobs on a page
break, and the run end to end with the demo model. Every person, employer and school here is invented."""
import asyncio
import io
import re
from datetime import date

import pytest
from docx import Document

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import page_fit, tailor, validate
from app.pipeline.fit import LADDER, styles
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
from app.render.pdf import education_runs, resume_layout
from app.schemas import (
    Company,
    Education,
    FitStyle,
    JDAnalysis,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
    TailorResult,
)

from .conftest import SAMPLE_JD
from .test_api import tailor as tailor_request
from .test_certifications import ticks, upload
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
    assert [place for place in order if place[0]][-1] == (2, 2)  # the only mention of a must-have goes last ...
    assert order[-2:] == [(0, 2), (0, 1)]  # ... of the older jobs' bullets: the most recent job keeps its lead
    assert all(index > 0 for _position, index in order) and len(order) == 6  # a job's first bullet never goes
    left = [3, 3, 3]
    for position, _index in order:
        if position == 0:  # the most recent job gives a bullet only while it still leads, or no other job can give
            assert left[0] - 1 > max(left[1:]) or all(n == 1 for n in left[1:])
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


# ------------------------------------------------------------------ review fixes (fit flow)
FLOOR = styles("Letter", LAYOUT)[len(LADDER):]  # the size ladder's smallest steps (fit.py COMPACT)
TAIL = ", keeping every partner feed on schedule through the quarterly peak for all 212 downstream report owners"


def floor_pages(resume: Resume) -> int:
    return min(resume_layout(resume, step)[0] for step in FLOOR)


def test_the_last_resort_tries_a_smaller_font_on_the_whole_draft_first():
    """A long career written to its two-page plan, two pages only at a smaller font: no bullet is removed for
    nothing, so the newest job keeps its lead."""
    profile = career(22)
    target, counts = plan_pages(profile, today=TODAY)
    draft = page_fit._plan_resume(profile, counts)
    assert target == 2 and counts[0] > max(counts[1:]) and measure(draft) == 3
    fitted = fit_new(draft, "Letter", target)
    assert fitted.pages == 2 and fitted.style.font_pt < LADDER[0][0] and fitted.style.layout == LAYOUT
    assert fitted.removed == 0 and [len(e.bullets) for e in fitted.resume.experience] == counts
    assert any(n.startswith("Font reduced") for n in fitted.notes)
    assert not any(n.startswith("Removed") for n in fitted.notes)


def test_the_last_resort_removes_only_the_bullets_the_smallest_font_needs():
    resume = resume_with(career(40), [2] * 40)
    order = removal_order(resume)
    assert floor_pages(resume) > 2 and floor_pages(page_fit._without(resume, order, len(order))) <= 2
    fitted = fit_new(resume, "Letter", 1)
    assert fitted.pages == 2 and fitted.style.font_pt < LADDER[0][0]
    assert 0 < fitted.removed < len(order)  # not every removable bullet
    assert sum(len(e.bullets) for e in fitted.resume.experience) == 80 - fitted.removed
    assert floor_pages(page_fit._without(resume, order, fitted.removed - 1)) > 2  # one fewer would not fit
    assert len(fitted.resume.experience[0].bullets) == 2  # the newest job keeps its lead
    [note] = [n for n in fitted.notes if n.startswith("Removed")]
    assert note.startswith(f"Removed {fitted.removed} bullet(s) to fit two pages")


@pytest.mark.parametrize("target", [1, 2])
def test_a_hand_edit_of_a_run_on_a_smaller_font_keeps_its_pages(target):
    """fit_report on the text fit_new laid out at a smaller font (nothing changed): the same pages and size, never 3-4
    pages at the design's own sizes."""
    resume = resume_with(career(40), [1] * 40)
    fitted = fit_new(resume, "Letter", target)
    assert fitted.pages == 2 and fitted.style.font_pt < LADDER[0][0]
    style, pages, fits, notes = asyncio.run(fit_report(fitted.resume, "Letter", target, fitted.style))
    assert (style, pages, fits) == (fitted.style, 2, True)
    assert any(n.startswith("Font reduced") for n in notes) and not any(n.startswith("Still") for n in notes)


def test_a_hand_edit_too_long_even_at_the_smallest_font_says_so():
    resume = resume_with(career(44), [1] * 44)
    style, pages, fits, notes = asyncio.run(fit_report(resume, "Letter", 2, design_style("Letter")))
    assert style in FLOOR and pages == 3 and not fits
    assert notes[-1] == "Still 3 pages at the smallest font (the limit is two). Shorten the text."


def test_the_newest_job_keeps_its_lead_over_older_bullets_that_hold_must_haves():
    resume = resume_with(career(6), [4, 2, 2, 1, 1, 1])
    resume.experience[1].bullets[1] = "Moved event ingestion to Kafka topics with schema checks, cutting lost events"
    resume.experience[2].bullets[1] = "Wrote the Terraform modules for the data stack, cutting new environment setup"
    analysis = JDAnalysis(target_role="Backend Engineer", seniority="", domain="",
                          must_have_skills=["Go", "Kafka", "Terraform"], nice_to_have_skills=[], keywords=[])
    order = removal_order(resume, analysis)
    left = [4, 2, 2, 1, 1, 1]
    for position, _index in order:
        left[position] -= 1
        assert left[0] > max(left[1:]) or all(n == 1 for n in left), (order, left)
    assert order[1:3] == [(2, 1), (1, 1)]  # the older jobs' must-have bullets go before the lead does
    trimmed = trim_to_page(resume, "Letter", 1, analysis)
    assert len(trimmed.resume.experience[0].bullets) > max(len(e.bullets) for e in trimmed.resume.experience[1:])


class ShorterDraft(LLMClient):
    """The demo model; its second draft (the shorter one) is changed by `change`."""

    def __init__(self, inner, change):
        self.inner, self.change, self.kinds = inner, change, []

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        self.kinds.append(kwargs["schema_name"])
        if kwargs["schema_name"] == "tailor_output" and self.kinds.count("tailor_output") == 2:
            self.change(result.data["sections"])
        return result


def longer(sections):  # written to the counts asked, but too long for them
    for exp in sections["experience"]:
        exp["bullets"] = [b if j == 0 else b.rstrip(".") + TAIL + "." for j, b in enumerate(exp["bullets"])]


def without_airflow(sections):  # a must-have the trimmed first draft shows is left out
    sections["summary"] = sections["summary"].replace("Airflow", "the scheduler")
    for exp in sections["experience"]:
        exp["bullets"] = [b.replace("Airflow", "the scheduler") for b in exp["bullets"]]


@pytest.mark.parametrize("change", [longer, without_airflow])
def test_a_shorter_draft_is_kept_only_with_as_many_bullets_and_every_must_have(router, db, registry, profile,
                                                                                 monkeypatch, change):
    """No time limit: a draft longer than its plan is trimmed, and the model is asked once more for the trimmed counts.
    That shorter draft is kept only when it ends with at least as many bullets and still shows every must-have the
    trimmed draft showed."""
    monkeypatch.setattr(tailor, "plan_pages", lambda *args, **kwargs: (1, [8] * len(profile.companies)))
    fits, analyses, real = [], [], tailor.fit_new

    def watch(resume, paper, target, analysis=None, document="Resume"):
        fits.append(real(resume, paper, target, analysis, document))
        analyses.append(analysis)
        return fits[-1]

    monkeypatch.setattr(tailor, "fit_new", watch)
    client = ShorterDraft(FakeClient(registry.get("demo-concise")), change)
    router.overrides["demo-concise"] = client
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", save=False, time_limit=0))
    first, short = fits
    assert result.status == "ok" and client.kinds.count("tailor_output") == 2 and short.removed < first.removed
    assert "Airflow" in analyses[0].must_have_skills  # and the trimmed first draft shows it
    assert validate.mentions(validate.evidence_text(first.resume), "Airflow")
    assert not any("shortened" in note for note in result.notes)
    assert [len(e.bullets) for e in result.resume.experience] == [len(e.bullets) for e in first.resume.experience]
    assert validate.mentions(validate.evidence_text(result.resume), "Airflow") and result.analysis == analyses[0]


class CapReachedMeanwhile(LLMClient):
    """The demo model; once the first draft is back, another run's spend passes the daily cost cap."""

    def __init__(self, inner, db):
        self.inner, self.db, self.kinds = inner, db, []

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        self.kinds.append(kwargs["schema_name"])
        if self.kinds == ["tailor_output"]:
            self.db.save_run(TailorResult(requested_model="demo-detailed", status="ok", cost_usd=5.0), None, "")
        return result


def test_a_shorter_draft_stopped_by_the_cost_cap_keeps_the_trimmed_draft(router, db, registry, profile, monkeypatch):
    monkeypatch.setattr(tailor, "plan_pages", lambda *args, **kwargs: (1, [8] * len(profile.companies)))
    db.set_settings({"daily_cost_cap_usd": 1.0})
    router.overrides["demo-concise"] = CapReachedMeanwhile(FakeClient(registry.get("demo-concise")), db)
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", save=False, time_limit=0))
    assert result.status == "ok" and result.resume is not None and result.pages == 1 and result.fits
    assert any(note.startswith("Removed") and "to fit one page" in note for note in result.notes)
    assert ("A shorter draft was not asked for, so bullets were removed instead: Daily cost cap of $1.00 reached. "
            "Raise it in Settings to continue.") in result.notes
    assert [call.result for call in result.timing.calls if call.kind == "tailor_output"] == ["ok", "failed"]


def fit_notes(notes: list[str]) -> list[str]:
    return [note for note in notes if re.match(r"Removed \d+ bullet|Kept two pages|Font reduced|Still ", note)]


def test_a_tick_and_an_untick_leave_no_stale_or_repeated_fit_notes(client):
    """A tick that removes bullets notes them; the untick that brings them back drops that note, and a tick again
    notes them once. A hand edit drops a tick's notes too. Other notes stay."""
    db = client.app.state.db
    db.set_settings({"certifications_section": True})
    run_id = tailor_request(client, profile_id=upload(client)["id"]).json()["run_id"]
    saved = db.get_run(run_id)
    have = [row.have for row in saved.certifications]
    four = saved.resume.model_copy(deep=True)
    four.certifications = validate.printed_certifications(
        [row.model_copy(update={"have": True}) for row in saved.certifications])
    extra = ("Rebuild the order sync service behind the partner REST API in Go, adding retries and rate limits for "
             "partners")
    while measure(four) == 1:  # fill the page: with all 4 certifications printed, bullets must go
        saved.resume.experience[0].bullets.append(extra)
        four.experience[0].bullets.append(extra)
        assert measure(saved.resume) == 1
    saved.notes = [*saved.notes, "Removed a repeated word: 'the the'."]
    db.update_run(saved)
    for _cycle in range(2):
        ticked = ticks(client, run_id, [True] * len(have)).json()
        assert ticked["message_kind"] == "warn"
        notes = db.get_run(run_id).notes
        [note] = fit_notes(notes)
        assert re.match(r"Removed \d+ bullet\(s\) to fit one page: ", note), notes
        assert len(notes) == len(set(notes)) and "Removed a repeated word: 'the the'." in notes
        back = ticks(client, run_id, have).json()
        assert back["message"] == "The bullets removed earlier are back."
        notes = db.get_run(run_id).notes
        assert fit_notes(notes) == [] and "Removed a repeated word: 'the the'." in notes, notes
    ticks(client, run_id, [True] * len(have))
    run = db.get_run(run_id).resume
    edited = client.put(f"/api/runs/{run_id}/resume", json={
        "headline": run.headline, "summary": run.summary, "skills": [g.model_dump() for g in run.skills],
        "experience": [{"company_index": e.company_index, "job_title": e.job_title, "bullets": e.bullets}
                       for e in run.experience]})
    assert edited.status_code == 200, edited.text
    notes = db.get_run(run_id).notes
    assert fit_notes(notes) == [] and "Removed a repeated word: 'the the'." in notes, notes
