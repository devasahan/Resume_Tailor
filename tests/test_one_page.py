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
    pages_word,
    plan_pages,
    refitted_notes,
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


def unchanged(resume: Resume) -> dict:
    """A hand edit's body (PUT /api/runs/{id}/resume) with the run's text as it is."""
    return {"headline": resume.headline, "summary": resume.summary, "skills": [g.model_dump() for g in resume.skills],
            "experience": [{"company_index": e.company_index, "job_title": e.job_title, "bullets": e.bullets}
                           for e in resume.experience]}


def test_a_tick_and_an_untick_leave_no_stale_or_repeated_fit_notes(client):
    """A tick that removes bullets notes them; the untick that brings them back drops that note, and a tick again
    notes them once. A hand edit keeps the last tick's note: those bullets are still off the resume. Other notes
    stay."""
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
    [ticked_note] = fit_notes(db.get_run(run_id).notes)
    edited = client.put(f"/api/runs/{run_id}/resume", json=unchanged(db.get_run(run_id).resume))
    assert edited.status_code == 200, edited.text
    notes = db.get_run(run_id).notes
    assert fit_notes(notes) == [ticked_note] and "Removed a repeated word: 'the the'." in notes, notes


# ------------------------------------------------------------------ second review fixes (fit)
def edited_past_two_pages(jobs: int) -> tuple[int, Resume]:
    """(page target, text): a run written to its plan, then bullets added by hand until it is over two pages at the
    design's own sizes, where removing bullets would still reach its page."""
    profile = career(jobs)
    target, counts = plan_pages(profile, today=TODAY)
    edited, added = resume_with(profile, counts), 0
    while measure(edited) <= 2:
        edited.experience[added % jobs].bullets.append(BULLET)
        added += 1
    order = removal_order(edited)
    assert measure(edited) == 3 and measure(page_fit._without(edited, order, len(order))) <= target
    return target, edited


@pytest.mark.parametrize("jobs", [6, 18])
def test_a_hand_edit_that_removing_bullets_would_fit_keeps_the_design_size(jobs):
    """A run at the design's own sizes, edited past two pages, where removing bullets would reach its page (6 jobs: one
    page, 18 jobs: two): the font is not reduced, as fit_new and a certification tick on the same text keep the
    design's sizes. A hand edit never removes what was typed, so a note says to shorten the text."""
    target, edited = edited_past_two_pages(jobs)
    assert target == (1 if jobs == 6 else 2)
    style, pages, fits, notes = asyncio.run(fit_report(edited, "Letter", target, design_style("Letter")))
    assert (style, pages, fits) == (design_style("Letter"), 3, False)
    assert notes == [(f"Still 3 pages at the design's own sizes: the resume is meant to fit {pages_word(target)}. "
                      "Shorten the text.")]
    _trimmed, tick_style, tick_pages, tick_fits, _notes, removed = asyncio.run(refit(edited, "Letter", target))
    assert (tick_style, tick_pages, tick_fits) == (style, target, True) and removed  # a tick removes bullets instead


def test_a_hand_edit_of_a_run_on_a_smaller_step_keeps_its_pages_when_removing_bullets_would_fit():
    """A run saved on a smaller step of the size ladder keeps one for text over two pages at the design's own sizes,
    even where removing bullets would reach two pages there (as after run_tailor shaped its skills): an edit never
    turns its two pages into three."""
    target, edited = edited_past_two_pages(18)
    style, pages, fits, notes = asyncio.run(fit_report(edited, "Letter", target, FLOOR[0]))
    assert style.font_pt < LADDER[0][0] and style.layout == LAYOUT and (pages, fits) == (2, True)
    [note] = notes
    assert note.startswith(f"Font reduced to {style.font_pt:g} pt") and note.endswith(" to fit two pages.")


MORE = ("while keeping every partner feed on schedule through the quarterly peak for all downstream report owners and "
        "the finance close team across three regions with clear runbooks and alerts that page the owning team")


def lengthened(bullet: str, words: int) -> str:
    """The bullet with `words` more words at its end."""
    return bullet.rstrip(".") + ", " + " ".join((MORE.split() * 4)[:words]) + "."


class LongDrafts(LLMClient):
    """The demo model with longer bullets: the first draft's by `first` words each (kept through the fix round, as by a
    model that ignores the line rule), the shorter draft's by `second`."""

    def __init__(self, inner, first: int, second: int):
        self.inner, self.first, self.second, self.kinds = inner, first, second, []

    async def generate_structured(self, **kwargs):
        result = await self.inner.generate_structured(**kwargs)
        self.kinds.append(kwargs["schema_name"])
        words = self.first if self.kinds.count("tailor_output") == 1 else self.second
        if kwargs["schema_name"] == "tailor_output" and words:
            for exp in result.data["sections"]["experience"]:
                exp["bullets"] = [lengthened(b, words) for b in exp["bullets"]]
        if kwargs["schema_name"] == "tailor_revision" and self.first:
            result.data["bullets"] = [dict(b, text=lengthened(b["text"], self.first)) for b in result.data["bullets"]]
        return result


def bullets_in(resume: Resume) -> int:
    return sum(len(exp.bullets) for exp in resume.experience)


@pytest.mark.parametrize(("target", "jobs", "counts", "first", "second", "kept"), [
    (1, 7, [5, 4, 4, 3, 3, 3, 3], 45, 0, "shorter"),  # the shorter draft fits one page, the first draft only two
    (2, 16, [6, 4] + [3] * 14, 45, 6, "shorter"),  # the shorter draft fits at the design's sizes, the first at 8 pt
    (2, 18, [4, 3] + [2] * 16, 0, 20, "first"),  # the first draft fits at the design's sizes, the shorter at 8.45 pt
])
def test_a_shorter_draft_is_judged_by_its_pages_and_font_before_its_bullets(router, db, registry, monkeypatch, target,
                                                                            jobs, counts, first, second, kept):
    """No time limit: of the trimmed first draft and the shorter draft, the one on fewer pages, or on as many at a
    larger size, is kept, whichever keeps more bullets: more bullets never make up for an extra page or a smaller
    font."""
    monkeypatch.setattr(tailor, "plan_pages", lambda *args, **kwargs: (target, list(counts)))
    fits, real = [], tailor.fit_new

    def watch(resume, paper, target, analysis=None, document="Resume"):
        fits.append(real(resume, paper, target, analysis, document))
        return fits[-1]

    monkeypatch.setattr(tailor, "fit_new", watch)
    router.overrides["demo-concise"] = LongDrafts(FakeClient(registry.get("demo-concise")), first, second)
    result = asyncio.run(run_tailor(router, db, profile=career(jobs), profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", save=False, time_limit=0))
    trimmed, short = fits
    better, worse = (short, trimmed) if kept == "shorter" else (trimmed, short)
    assert better.style == design_style("Letter") and better.pages == target
    assert worse.pages > target or worse.style.font_pt < better.style.font_pt
    # what bullets alone would choose: the worse layout
    assert bullets_in(better.resume) < bullets_in(worse.resume) or worse.removed < better.removed
    assert result.status == "ok" and (result.style, result.pages) == (better.style, better.pages)
    assert [len(e.bullets) for e in result.resume.experience] == [len(e.bullets) for e in better.resume.experience]
    assert (f"The model shortened the resume to fit {pages_word(target)}." in result.notes) == (kept == "shorter")


def test_a_note_on_a_reduced_font_never_claims_pages_the_text_does_not_fit():
    """Text over two pages even at the smallest step: the notes say the font was reduced and that it is still over,
    never that the smaller font made it fit two pages (a hand edit, and a new run's last resort)."""
    resume = resume_with(career(44), [1] * 44)
    style, pages, fits, notes = asyncio.run(fit_report(resume, "Letter", 2, design_style("Letter")))
    assert style in FLOOR and pages == 3 and not fits
    assert notes == ["Font reduced to 8 pt.",
                     "Still 3 pages at the smallest font (the limit is two). Shorten the text."]
    fitted = fit_new(resume, "Letter", 2)
    assert fitted.pages == 3 and fitted.notes == [
        "Font reduced to 8 pt.", "Still more than two pages after trimming. Shorten the text in the editor."]


def test_a_note_of_removed_bullets_stays_while_they_are_off_the_resume():
    """A new fit's notes replace the earlier fit's notes on the layout (two pages kept, the font reduced, still over).
    A note naming removed bullets stays: only a tick that starts again from the bullets an earlier tick removed drops
    that tick's note (every copy an older version repeated)."""
    made = "Removed 2 bullet(s) to fit one page: Contoso: Rebuilt the ledger export for finance with dbt..."
    ticked = "Removed 1 bullet(s) to fit one page: Fabrikam: Moved reports to SQL views, cutting refresh time..."
    notes = [made, "Kept two pages: even one bullet ...", ticked, "Removed a repeated word: 'the the'.", ticked,
             "Font reduced to 9.1 pt to fit two pages.", "Still 2 pages at the design's own sizes: ..."]
    assert refitted_notes(notes, ["Font reduced to 9.1 pt to fit two pages."]) == [
        made, ticked, "Removed a repeated word: 'the the'.", "Font reduced to 9.1 pt to fit two pages."]
    assert refitted_notes(notes, [], restored=True) == [made, "Removed a repeated word: 'the the'."]
    again = "Removed 2 bullet(s) to fit one page: Fabrikam: Moved reports to SQL views, cutting refresh time..."
    assert refitted_notes(notes, [again], restored=True) == [made, "Removed a repeated word: 'the the'.", again]


def test_a_tick_that_removes_nothing_and_a_hand_edit_keep_the_note_of_removed_bullets(client):
    """The note naming the bullets a run's fit removed when it was made stays while they are off the resume: after a
    certification tick that removes nothing and after a hand edit that changes nothing."""
    db = client.app.state.db
    db.set_settings({"certifications_section": True})
    run_id = tailor_request(client, profile_id=upload(client)["id"]).json()["run_id"]
    saved = db.get_run(run_id)
    made = "Removed 2 bullet(s) to fit one page: Northwind Labs: Rebuilt the partner order sync in Go with retries..."
    saved.notes = [*saved.notes, made, "Removed a repeated word: 'the the'."]  # as the fit wrote it when it was made
    db.update_run(saved)
    have = [row.have for row in saved.certifications]
    counts = [len(e.bullets) for e in saved.resume.experience]
    ticked = ticks(client, run_id, [h or i == have.index(False) for i, h in enumerate(have)])
    assert ticked.status_code == 200, ticked.text
    run = db.get_run(run_id)
    assert [len(e.bullets) for e in run.resume.experience] == counts and not ticked.json().get("message")
    assert fit_notes(run.notes) == [made] and "Removed a repeated word: 'the the'." in run.notes, run.notes
    edited = client.put(f"/api/runs/{run_id}/resume", json=unchanged(run.resume))
    assert edited.status_code == 200, edited.text
    notes = db.get_run(run_id).notes
    assert fit_notes(notes) == [made] and "Removed a repeated word: 'the the'." in notes, notes


def test_a_hand_edit_keeps_an_older_runs_note_of_removed_bullets(client):
    """A run saved before the one-page rule (page_target 0, layout 0) whose fit removed bullets: a hand edit that
    changes nothing keeps the note naming them, as such runs always did."""
    db = client.app.state.db
    profile = Profile(name="Taylor Quinn", companies=[Company(name="Contoso", start="Jan 2020", end="Present"),
                                                      Company(name="Fabrikam", start="Jan 2015", end="Dec 2019")],
                      education=[Education(school="Lakeside University", degree="B.S.", field="Computer Science",
                                           end="2014")])
    resume = resume_with(profile, [1, 1])
    made = "Removed 3 bullet(s) to fit two pages: Contoso: Rebuilt the ledger export for finance with dbt and Airf..."
    analysis = JDAnalysis(target_role="Backend Engineer", seniority="", domain="", must_have_skills=["Go"],
                          nice_to_have_skills=[], keywords=[])
    run_id = db.save_run(TailorResult(requested_model="demo-concise", status="ok", resume=resume, page_target=0,
                                      style=FitStyle(), analysis=analysis,
                                      notes=[made, "Removed a repeated word: 'the the'."]), None, "")
    edited = client.put(f"/api/runs/{run_id}/resume", json=unchanged(resume))
    assert edited.status_code == 200, edited.text
    run = db.get_run(run_id)
    assert run.style.layout == 0 and run.notes == [made, "Removed a repeated word: 'the the'."], run.notes
