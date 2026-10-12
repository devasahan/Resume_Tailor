"""The whole run for one model: prompt -> validate -> merge locked fields -> fit one page -> checks -> save."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import date

from ..llm.base import CostCapError, LLMError
from ..llm.router import Router
from ..llm.schema import strict_json_schema
from ..prompts import (
    PROMPT_VERSION,
    cover_system,
    cover_user,
    tailor_system,
    tailor_user,
)
from ..schemas import (
    CoverLetter,
    CoverLetterOut,
    FitStyle,
    JDAnalysis,
    Profile,
    Resume,
    TailorOutput,
    TailorResult,
)
from ..storage.db import Database
from . import validate
from .facts import candidate_facts
from .fit import bullet_budget, render_smallest, smaller_budget, trim_to_fit, try_styles
from .generate import GenerationFailed, Outcome, generate_validated

log = logging.getLogger(__name__)

# Must-have JD terms missing from the summary and bullets (absent, or only listed) that justify the one rewrite
# pass. 1 means every required term is covered automatically; 2 roughly halves how often the rewrite runs.
MIN_UNCOVERED_FOR_RETRY = 1


def jd_hash(jd: str) -> str:
    return hashlib.sha256(jd.strip().encode("utf-8")).hexdigest()[:16]


def _absorb(total: Outcome, other: Outcome) -> None:
    total.input_tokens += other.input_tokens
    total.output_tokens += other.output_tokens
    total.cost_usd += other.cost_usd
    total.attempts += other.attempts
    total.retries += other.retries


def draft_text(resume: Resume) -> str:
    """The editable sections of a draft as plain text, so a rewrite request can show the model what to revise."""
    lines = [f"Headline: {resume.headline}", f"Summary: {resume.summary}", "Skills:"]
    lines += [f"- {g.category}: {', '.join(g.skills)}" for g in resume.skills]
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index].name
        lines.append(f"company_index {exp.company_index} ({company}): {exp.job_title}")
        lines += [f"- {b}" for b in exp.bullets]
    return "\n".join(lines)


def coverage_feedback(resume: Resume, analysis: JDAnalysis) -> str | None:
    """Rewrite instructions when must-have JD terms are not shown in the summary or a bullet, else None."""
    absent, listed_only = validate.unshown_must_haves(resume, analysis)
    if len(absent) + len(listed_only) < MIN_UNCOVERED_FOR_RETRY:
        return None
    parts = []
    if absent:
        parts.append("These must-have phrases from the job description are not in the summary or any bullet yet: "
                     + ", ".join(f'"{t}"' for t in absent) + ".")
    if listed_only:
        parts.append("These are only listed (in the skills, headline or titles) but not shown in the summary or a "
                     "bullet: " + ", ".join(f'"{t}"' for t in listed_only) + ".")
    return " ".join(parts) + (
        " Put each one in word for word, in the summary or in one existing bullet of the employer where it fits "
        "best (usually the most recent role), replacing weaker words so each bullet stays 18-28 words and the "
        "summary 60-85 words. Do not add or remove bullets: each employer keeps exactly the number given in "
        "<employers>. Use at most two of these phrases per bullet, never as a list, and keep them in "
        "jd_analysis.must_have_skills."
    )


def analysis_terms(analysis: JDAnalysis) -> str:
    """Stand-in for the JD text when it is no longer available (after a manual edit)."""
    return " ".join(
        [analysis.target_role, *analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords,
         *analysis.key_responsibilities]
    )


def run_checks(result: TailorResult, jd_text: str) -> None:
    """(Re)compute coverage, ATS warnings and claim flags from the current resume text."""
    assert result.resume is not None and result.analysis is not None
    result.coverage = validate.keyword_coverage(result.resume, result.analysis)
    result.ats_warnings = validate.ats_lint(result.resume)
    result.ats_warnings += validate.era_problems(result.resume)
    absent, listed_only = validate.unshown_must_haves(result.resume, result.analysis)
    if absent or listed_only:
        result.ats_warnings.append(
            "Required by the posting but not shown in the summary or a bullet: " + ", ".join(absent + listed_only)
            + ". An ATS and a recruiter look for these in your experience, not only in the skills list."
        )
    allowed = f"{result.hints}\n{result.facts}"
    flags = validate.claim_flags(result.resume, jd_text, allowed)
    posting = validate.posting_match_flags(result.resume, result.analysis, result.hints)
    result.flags = [f for f in flags if f.level == "verify"] + posting + [f for f in flags if f.level != "verify"]
    dense = validate.metric_density_problems(result.resume)
    if dense:
        result.ats_warnings.append(
            "Many bullets carry figures (" + "; ".join(dense) + "). Recruiters read a few strong numbers better "
            "than many: consider keeping at most 2 per role."
        )
    rounded = validate.figure_style_problems(result.resume, allowed)
    if rounded:
        result.ats_warnings.append(
            "These figures look rounded or approximate: " + ", ".join(rounded[:8])
            + ". Replace them with your real numbers (specific values such as 23.4% or 21% read as measured)."
        )
    if result.cover_letter:
        result.flags += validate.cover_flags(result.cover_letter, result.hints)


async def fit_report(resume: Resume, paper: str) -> tuple[FitStyle, int, bool, list[str]]:
    """Fit check for a hand-edited resume: never changes the text, only picks the style."""
    found = await asyncio.to_thread(try_styles, resume, paper)
    if found:
        style, pages, _pdf, notes = found
        return style, pages, True, notes
    style, pages, _pdf = await asyncio.to_thread(render_smallest, resume, paper)
    return style, pages, False, [f"Still {pages} pages at the smallest font. Shorten the text."]


async def run_tailor(
    router: Router,
    db: Database,
    *,
    profile: Profile,
    profile_id: int | None,
    jd: str,
    hints: str = "",
    model_id: str,
    allow_fallback: bool = True,
    cover_letter: bool = False,
    cover_model_id: str | None = None,
    company_name: str = "",
    hiring_manager: str = "",
    save: bool = True,
) -> TailorResult:
    started = time.perf_counter()
    settings = router.settings()
    paper = settings["paper_size"]
    metrics_mode = settings.get("metrics_mode", "estimate")
    today = date.today()  # one date for the facts, the bullet budget and the stay lengths in the prompt
    facts = candidate_facts(profile, today)
    result = TailorResult(
        requested_model=model_id, prompt_version=PROMPT_VERSION, hints=hints, facts=facts, company_name=company_name,
        jd=jd,
    )
    outcome: Outcome = Outcome()
    notes: list[str] = []

    try:
        chain = router.chain(model_id, allow_fallback)
        schema = strict_json_schema(TailorOutput)
        budget = await asyncio.to_thread(bullet_budget, profile, paper, today)  # bullets per employer that fit
        context = {"profile": profile.model_dump(), "jd": jd, "hints": hints, "bullets": budget}

        def parse(data: dict):
            """(output, resume, notes). Titles that step down are corrected here, on every draft, for free."""
            out = TailorOutput.model_validate(data)
            resume = validate.build_resume(profile, out)
            return out, resume, validate.fix_title_regressions(resume)

        allowed_figures = f"{hints}\n{facts}"

        def refine(value) -> str | None:
            """Ask once to fix the draft if it uses too many figures, round guesses, thin text, leaves must-have
            JD terms out of the summary and bullets, or has job titles that do not read as a career path. The draft itself is attached: the rewrite prompt
            is the original prompt plus this feedback, so without it the model could not keep what was good."""
            feedback = []
            density = validate.metric_density_problems(value[1])
            if density:
                feedback.append(
                    "Too many bullets carry numbers (" + "; ".join(density) + "). Keep a figure in at most "
                    f"{validate.MAX_FIGURE_BULLETS_PER_JOB} bullets per employer and about 2 in 5 overall; rewrite "
                    "the other bullets to describe the impact in words. Keep the candidate's own figures."
                )
            rounded = validate.figure_style_problems(value[1], allowed_figures)
            if len(rounded) >= 2:
                feedback.append(
                    "Some figures look like round guesses or approximations: " + ", ".join(rounded[:8])
                    + ". Rewrite those figures as specific, measured-looking values (for example 23.4%, 21%, "
                    "17.8%, 1.7TB, 38 analysts): no numbers ending in 0 or 5, no '~', no 'approximately'. Keep any "
                    "figure the candidate gave exactly as it is."
                )
            thin = validate.thin_text_problems(value[1])
            if thin:
                feedback.append(
                    "The draft is too thin: " + "; ".join(thin) + ". Rewrite the summary as 3-4 flowing sentences "
                    "of 60-85 words, and every weak bullet as 18-28 words that say specifically what was done, how, "
                    "and what changed, following the SUMMARY and BULLETS rules."
                )
            coverage = coverage_feedback(value[1], value[0].jd_analysis)
            if coverage:
                feedback.append(coverage)
            era = validate.era_problems(value[1])
            if era:
                feedback.append(
                    "Some bullets mention tools that job could not really have used: " + " ".join(era)
                    + " Move each one to a later job where it was common (AI tools only to the one or two most recent "
                    "jobs), or replace it with a tool of that time; keep the same number of bullets per employer."
                )
            titles = validate.title_growth_problems(value[1])
            if titles:
                feedback.append(
                    "The job titles do not read as a believable career path: " + " ".join(titles) + " Choose the "
                    "titles again following the JOB TITLES rule, using the stay lengths given in <employers>."
                )
            if not feedback:
                return None
            return (
                " ".join(feedback) + " Revise the previous draft below to fix only this, and keep everything else "
                "as it is.\n<previous_draft>\n" + draft_text(value[1]) + "\n</previous_draft>"
            )

        await generate_validated(
            router, chain, system=tailor_system(metrics_mode),
            user=tailor_user(jd, profile.companies, hints, facts, bullets=budget, today=today),
            schema=schema, schema_name="tailor_output", parse=parse, context=context, outcome=outcome,
            refine=refine,
        )
        output, resume, title_notes = outcome.value  # type: ignore[misc]

        # --- fit to one page: shrink style -> ask once more with a smaller bullet budget -> trim bullets
        fit = await asyncio.to_thread(try_styles, resume, paper)
        if fit is None:
            second = Outcome()
            smaller = smaller_budget(budget)
            try:
                await generate_validated(
                    router, [outcome.final_model_id], system=tailor_system(metrics_mode),
                    user=tailor_user(jd, profile.companies, hints, facts, shorten=True, bullets=smaller, today=today),
                    schema=schema, schema_name="tailor_output", parse=parse,
                    context={**context, "shorten": True, "bullets": smaller}, outcome=second,
                    refine=None,  # the shorten call is already the extra call; never let it ask a third time
                )
                _absorb(outcome, second)
                short_output, short_resume, short_title_notes = second.value  # type: ignore[misc]
                fit = await asyncio.to_thread(try_styles, short_resume, paper)
                if fit:
                    output, resume, title_notes = short_output, short_resume, short_title_notes
                    notes.append("The model shortened the bullets to fit one page.")
            except GenerationFailed as exc:
                _absorb(outcome, exc.outcome)
                notes.append("Asking the model for shorter bullets failed, so the text was trimmed instead.")
        if fit:
            style, pages, _pdf, fit_notes = fit
            notes += fit_notes
            fits = True
        else:
            resume, style, pages, _pdf, trim_notes = await asyncio.to_thread(trim_to_fit, resume, paper)
            notes += trim_notes
            fits = pages == 1

        if outcome.input_tokens == 0 and outcome.output_tokens == 0:
            try:
                if router.registry.get(outcome.final_model_id).provider != "fake":
                    notes.append("The provider did not report token usage, so the cost shown ($0) is unknown.")
            except KeyError:
                pass
        notes += title_notes
        result.resume = resume
        result.analysis = output.jd_analysis
        result.career_plan = output.career_plan
        result.style, result.pages, result.fits = style, pages, fits
        result.served_model = outcome.served_model
        result.status = "ok"

        # --- optional cover letter, from the final resume text
        if cover_letter:
            await _cover_letter(
                router, result, outcome, notes, jd=jd, hints=hints,
                company_name=company_name or output.jd_analysis.company, hiring_manager=hiring_manager,
                model_id=cover_model_id or outcome.final_model_id, allow_fallback=allow_fallback,
            )
        run_checks(result, jd)
    except GenerationFailed as exc:
        if exc.outcome is not outcome:
            _absorb(outcome, exc.outcome)
        result.status, result.error = "error", str(exc)[:900]
    except (CostCapError, LLMError) as exc:
        result.status, result.error = "error", str(exc)[:900]
    except Exception as exc:  # never let one model's failure break a compare run
        log.exception("Unexpected error while tailoring with %s", model_id)
        result.status, result.error = "error", f"Unexpected error: {type(exc).__name__}: {str(exc)[:300]}"

    result.notes = notes
    result.attempts = outcome.attempts
    result.retries = outcome.retries
    result.fallback_chain = outcome.chain
    result.input_tokens = outcome.input_tokens
    result.output_tokens = outcome.output_tokens
    result.cost_usd = round(outcome.cost_usd, 6)
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    if save:
        db.save_run(result, profile_id, jd_hash(jd))
    return result


async def _cover_letter(
    router: Router, result: TailorResult, outcome: Outcome, notes: list[str], *, jd: str, hints: str,
    company_name: str, hiring_manager: str, model_id: str, allow_fallback: bool,
) -> str | None:
    """Write result.cover_letter from the final resume. Returns None, or the reason no letter was written (also
    added to notes). A cost-cap stop here leaves the resume intact instead of failing the whole run."""
    assert result.resume is not None and result.analysis is not None
    resume = result.resume

    def parse(data: dict) -> CoverLetter:
        raw = CoverLetterOut.model_validate(data)
        greeting, paragraphs, closing = validate.clean_cover_letter(raw.greeting, raw.paragraphs, raw.closing)
        return CoverLetter(
            greeting=greeting, paragraphs=paragraphs, closing=closing,
            company_name=company_name, hiring_manager=hiring_manager,
        )

    second = Outcome()
    try:
        await generate_validated(
            router, router.chain(model_id, allow_fallback), system=cover_system(),
            user=cover_user(jd, resume, result.analysis.target_role, company_name, hiring_manager, hints),
            schema=strict_json_schema(CoverLetterOut), schema_name="cover_letter", parse=parse,
            context={
                "company_name": company_name, "hiring_manager": hiring_manager,
                "target_role": result.analysis.target_role, "keywords": result.analysis.keywords,
            },
            outcome=second,
        )
        _absorb(outcome, second)
        result.cover_letter = second.value
        return None
    except GenerationFailed as exc:
        _absorb(outcome, exc.outcome)
        message = f"Cover letter failed: {str(exc)[:200]}"
    except CostCapError as exc:
        _absorb(outcome, second)
        message = f"Cover letter not written: {str(exc)[:200]}"
    notes.append(message)
    return message


async def write_cover_letter(
    router: Router, db: Database, run_id: int, *, jd: str, model_id: str
) -> tuple[TailorResult, str | None]:
    """Write (or rewrite) the cover letter of a saved run.

    Returns (updated run, error). The caller has checked that the run exists and has a resume and analysis.
    The spend is saved as its own row dated today (task 'cover_letter'), so the daily cost cap counts it even
    when the run itself is from an earlier day; the run only shows it in cover_cost_usd. The run is read again
    before saving so that edits made on the Review page in the meantime are kept.
    """
    base = db.get_run(run_id)
    assert base is not None and base.resume is not None and base.analysis is not None
    base.cover_letter = None
    outcome, notes, started = Outcome(), [], time.perf_counter()
    error = await _cover_letter(
        router, base, outcome, notes, jd=jd, hints=base.hints,
        company_name=base.company_name or base.analysis.company, hiring_manager="",
        model_id=model_id, allow_fallback=True,
    )
    if outcome.attempts:
        db.save_run(TailorResult(
            requested_model=model_id, served_model=outcome.served_model, status="error" if error else "ok",
            error=error or "", input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens,
            cost_usd=round(outcome.cost_usd, 6), attempts=outcome.attempts, retries=outcome.retries,
            fallback_chain=outcome.chain, latency_ms=int((time.perf_counter() - started) * 1000),
            notes=[f"Cover letter for run {run_id}."],
        ), None, "", task="cover_letter")
    fresh = db.get_run(run_id) or base
    if base.cover_letter is not None:
        fresh.cover_letter = base.cover_letter  # replaces any earlier letter
    fresh.cover_cost_usd = round(fresh.cover_cost_usd + outcome.cost_usd, 6)
    fresh.notes = [n for n in fresh.notes if not n.startswith(("Cover letter failed", "Cover letter not written"))]
    fresh.notes += notes
    if not fresh.jd:
        fresh.jd = jd
    run_checks(fresh, jd)  # recomputes all flags, so cover-letter flags are never duplicated
    db.update_run(fresh)
    return fresh, error
