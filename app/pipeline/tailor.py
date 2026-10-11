"""The whole run for one model: prompt -> validate -> merge locked fields -> fit one page -> checks -> save."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time

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
from .fit import render_smallest, trim_to_fit, try_styles
from .generate import GenerationFailed, Outcome, generate_validated

log = logging.getLogger(__name__)


def jd_hash(jd: str) -> str:
    return hashlib.sha256(jd.strip().encode("utf-8")).hexdigest()[:16]


def _absorb(total: Outcome, other: Outcome) -> None:
    total.input_tokens += other.input_tokens
    total.output_tokens += other.output_tokens
    total.cost_usd += other.cost_usd
    total.attempts += other.attempts
    total.retries += other.retries


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
    allowed = f"{result.hints}\n{result.facts}"
    result.flags = validate.claim_flags(result.resume, jd_text, allowed)
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
    facts = candidate_facts(profile)
    result = TailorResult(requested_model=model_id, prompt_version=PROMPT_VERSION, hints=hints, facts=facts)
    outcome: Outcome = Outcome()
    notes: list[str] = []

    try:
        chain = router.chain(model_id, allow_fallback)
        schema = strict_json_schema(TailorOutput)
        context = {"profile": profile.model_dump(), "jd": jd, "hints": hints}

        def parse(data: dict):
            out = TailorOutput.model_validate(data)
            return out, validate.build_resume(profile, out)

        allowed_figures = f"{hints}\n{facts}"

        def refine(value) -> str | None:
            """Ask once for specific, measured-looking figures if the draft relies on round guesses."""
            problems = validate.figure_style_problems(value[1], allowed_figures)
            if len(problems) < 2:
                return None
            return (
                "Some figures look like round guesses or approximations: " + ", ".join(problems[:8])
                + ". Rewrite only those figures as specific, measured-looking values (for example 23.4%, 21%, "
                "17.8%, 1.7TB, 38 analysts): no numbers ending in 0 or 5, no '~', no 'approximately'. Keep any "
                "figure the candidate gave exactly as it is, and keep everything else unchanged."
            )

        await generate_validated(
            router, chain, system=tailor_system(metrics_mode), user=tailor_user(jd, profile.companies, hints, facts),
            schema=schema, schema_name="tailor_output", parse=parse, context=context, outcome=outcome,
            refine=refine,
        )
        output, resume = outcome.value  # type: ignore[misc]

        # --- fit to one page: shrink style -> ask the model for shorter bullets once -> trim bullets
        fit = await asyncio.to_thread(try_styles, resume, paper)
        if fit is None:
            second = Outcome()
            try:
                await generate_validated(
                    router, [outcome.final_model_id], system=tailor_system(metrics_mode),
                    user=tailor_user(jd, profile.companies, hints, facts, shorten=True), schema=schema,
                    schema_name="tailor_output", parse=parse, context={**context, "shorten": True}, outcome=second,
                    refine=refine,
                )
                _absorb(outcome, second)
                short_output, short_resume = second.value  # type: ignore[misc]
                fit = await asyncio.to_thread(try_styles, short_resume, paper)
                if fit:
                    output, resume = short_output, short_resume
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
        result.resume = resume
        result.analysis = output.jd_analysis
        result.style, result.pages, result.fits = style, pages, fits
        result.served_model = outcome.served_model
        result.status = "ok"

        # --- optional cover letter, from the final resume text
        if cover_letter:
            await _cover_letter(
                router, result, outcome, notes, jd=jd, hints=hints, company_name=company_name,
                hiring_manager=hiring_manager, model_id=cover_model_id or outcome.final_model_id,
                allow_fallback=allow_fallback,
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
) -> None:
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
    except GenerationFailed as exc:
        _absorb(outcome, exc.outcome)
        notes.append(f"Cover letter failed: {str(exc)[:200]}")
