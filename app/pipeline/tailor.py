"""The whole run for one model: prompt -> validate -> merge locked fields -> fit two pages -> checks -> save."""
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
    revise_user,
    tailor_system,
    tailor_user,
)
from ..schemas import (
    CoverLetter,
    CoverLetterOut,
    FitStyle,
    Flag,
    JDAnalysis,
    Profile,
    Resume,
    TailorOutput,
    TailorResult,
    TailorRevision,
)
from ..storage.db import Database
from ..timing import Steps
from . import spelling, validate
from .facts import candidate_facts
from .fit import (
    MAX_PAGES,
    bullet_budget,
    render_smallest,
    smaller_budget,
    trim_to_fit,
    try_styles,
)
from .generate import Ask, GenerationFailed, Outcome, generate_validated, record_sent

log = logging.getLogger(__name__)

# Must-have JD terms missing from the summary and bullets (absent, or only listed) that justify the one rewrite
# pass. 1 means every required term is covered automatically; 2 roughly halves how often the rewrite runs.
MIN_UNCOVERED_FOR_RETRY = 1
# The time limit for a resume (Settings, default 30 s): the AI calls must end this much earlier, leaving time for the
# layout, the checks and saving; the shorter-draft call is not started with less than MIN_SHORTER_SECONDS left.
CHECKS_RESERVE_SECONDS = 4.0
MIN_SHORTER_SECONDS = 10.0
# With the time limit on: the resume model is given (limit + SWITCH_EXTRA_SECONDS) to answer before the next model in
# the chain (Settings > Fallback models) is used instead, and the last model LAST_MODEL_SECONDS, never minutes.
SWITCH_EXTRA_SECONDS = 10.0
LAST_MODEL_SECONDS = 60.0  # the backup usually answers in 11-21 s


def jd_hash(jd: str) -> str:
    return hashlib.sha256(jd.strip().encode("utf-8")).hexdigest()[:16]


def _absorb(total: Outcome, other: Outcome) -> None:
    total.input_tokens += other.input_tokens
    total.output_tokens += other.output_tokens
    total.cost_usd += other.cost_usd
    total.attempts += other.attempts
    total.retries += other.retries


def draft_text(resume: Resume) -> str:
    """The editable sections of a draft as plain text, so a fix request can show the model what to revise. Bullets
    are numbered from 0 within each employer: a fix names the bullets it changes by these numbers."""
    lines = [f"Headline: {resume.headline}", f"Summary: {resume.summary}", "Skills:"]
    lines += [f"- {g.category}: {', '.join(g.skills)}" for g in resume.skills]
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index].name
        lines.append(f"company_index {exp.company_index} ({company}): {exp.job_title}")
        lines += [f"[{index}] {b}" for index, b in enumerate(exp.bullets)]
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
        parts.append("These are only listed (in the skills, headline, titles or certifications) but not shown in the "
                     "summary or a bullet: " + ", ".join(f'"{t}"' for t in listed_only) + ".")
    return " ".join(parts) + (
        " Put each one in word for word, in the summary or in one existing bullet of the employer where it fits "
        "best (usually the most recent role), replacing weaker words so each bullet stays 20-32 words and the "
        "summary 60-85 words. Do not add or remove bullets: each employer keeps exactly the number given in "
        "<employers>. Use at most two of these phrases per bullet, never as a list."
    )


def analysis_terms(analysis: JDAnalysis) -> str:
    """Stand-in for the JD text when it is no longer available (after a manual edit)."""
    return " ".join(
        [analysis.target_role, *analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords,
         *analysis.key_responsibilities]
    )


SPELLING_NAMES = {"us": "American", "eu": "British"}


def run_checks(result: TailorResult, jd_text: str) -> None:
    """(Re)compute coverage, ATS warnings and claim flags from the current resume text."""
    assert result.resume is not None and result.analysis is not None
    result.coverage = validate.keyword_coverage(result.resume, result.analysis)
    result.ats_warnings = validate.ats_lint(result.resume)
    result.ats_warnings += validate.era_problems(result.resume)
    copied = validate.copied_posting_spans(result.resume, jd_text, result.analysis)
    if copied:
        result.ats_warnings.append("Copied word for word from the job description: " + "; ".join(copied[:6])
                                   + ". Keep the keywords, but say the rest in your own words.")
    phrases = validate.skill_phrase_problems(result.resume)
    if phrases:
        result.ats_warnings.append("Skills written as phrases, not keywords: " + ", ".join(phrases[:6])
                                   + ". List the keyword itself (for example SQL, AWS).")
    plain = validate.plain_word_problems(result.resume, result.analysis)
    if plain:
        result.ats_warnings.append("Wording that can sound machine-written: " + "; ".join(plain[:6])
                                   + ". Plain words read as your own.")
    letter_plain = validate.cover_plain_problems(result.cover_letter, result.analysis) if result.cover_letter else []
    if letter_plain:
        result.ats_warnings.append("Cover letter wording that can sound machine-written: " + "; ".join(letter_plain[:6])
                                   + ". Plain words read as your own.")
    unbased = validate.unbased_percentage_problems(result.resume, result.cover_letter, jd_text)
    if unbased:
        result.ats_warnings.append(
            "Percentages without their base: " + "; ".join(unbased[:6]) + ". A recruiter asks 'of what?': add the "
            "amount or the values before and after in words, for example 'by 17.8%, or $41,000 a year', or give the "
            "real numbers alone ('cut queue time from 41 to 32 seconds')."
        )
    mismatched = validate.base_mismatch_problems(result.resume, result.cover_letter)
    if mismatched:
        result.ats_warnings.append(
            "Percentages that do not match their base: " + "; ".join(mismatched[:6]) + ". Check the numbers: from 41 "
            "to 32 is 22%."
        )
    dashes = validate.dash_problems(result.resume, result.cover_letter)
    if dashes:
        result.ats_warnings.append("Dashes used as punctuation: " + "; ".join(dashes[:6]) + ". Recruiters read them "
                                   "as a sign of AI-written text: use a full stop or a comma instead.")
    if result.spelling:  # runs made before the US/EU style buttons get no spelling notes
        style = result.spelling
        names = (result.company_name, result.cover_letter.hiring_manager if result.cover_letter else "")
        other = validate.spelling_problems(result.resume, result.cover_letter, style, result.analysis, names)
        if other:
            result.ats_warnings.append(
                f"Words in {SPELLING_NAMES['eu' if style == 'us' else 'us']} spelling in this {style.upper()} style "
                f"document: {', '.join(other[:8])}. Write them the {SPELLING_NAMES[style]} way, so the spelling is "
                "the same everywhere."
            )
        posting = spelling.detect(jd_text)
        if posting and posting != style:
            result.ats_warnings.append(
                f"This job post uses {SPELLING_NAMES[posting]} spelling, but this resume uses {style.upper()} style. "
                "The post's own keywords keep its spelling, so the two spellings are mixed. If the job is in "
                + ("the UK or the EU, choose EU style." if posting == "eu" else "the US, choose US style.")
            )
    openers = validate.first_bullet_problems(result.resume) if result.metrics_mode == "estimate" else []
    if openers:
        result.ats_warnings.append("These jobs do not open with a measured result: " + ", ".join(openers[:6])
                                   + ". Recruiters read the first bullet of each job first: put that job's biggest "
                                   "win there, with its size and its result.")
    absent, listed_only = validate.unshown_must_haves(result.resume, result.analysis)
    if absent or listed_only:
        result.ats_warnings.append(
            "Required by the posting but not shown in the summary or a bullet: " + ", ".join(absent + listed_only)
            + ". An ATS and a recruiter look for these in your experience, not only in the skills list."
        )
    allowed = f"{result.hints}\n{result.facts}"
    flags = validate.claim_flags(result.resume, jd_text, allowed)
    flags += [  # left in the text after the fix round (or written in by hand): a claim to check
        Flag(section=where, text=term, level="verify",
             reason="Names a certification. Certifications belong only in their own section; remove it from the text.")
        for where, term in validate.certification_mentions(
            result.resume, [row.name for row in result.certifications]
            + [cert.name for cert in result.resume.profile.certifications])
    ]
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
    return style, pages, False, [f"Still {pages} pages at the smallest font (the limit is two). Shorten the text."]


async def refit(resume: Resume, paper: str) -> tuple[Resume, FitStyle, int, bool, list[str], dict[str, int]]:
    """Fit after the printed certifications changed (ticked on the Tailor page): the largest style that keeps two
    pages, or else the fewest bullets removed (the last bullets of the jobs with the most). Returns (resume, style,
    pages, fits, notes, {employer: bullets removed}); the resume given is never changed."""
    found = await asyncio.to_thread(try_styles, resume, paper)
    if found:
        style, pages, _pdf, notes = found
        return resume, style, pages, True, notes, {}
    trimmed, style, pages, _pdf, notes = await asyncio.to_thread(trim_to_fit, resume, paper)
    removed: dict[str, int] = {}
    for before, after in zip(resume.experience, trimmed.experience):
        if len(after.bullets) < len(before.bullets):
            name = resume.profile.companies[before.company_index].name
            removed[name] = removed.get(name, 0) + len(before.bullets) - len(after.bullets)
    return trimmed, style, pages, pages <= MAX_PAGES, notes, removed


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
    user_id: int | None = None,
    spelling: str = "us",
    steps: Steps | None = None,
    time_limit: float | None = None,
) -> TailorResult:
    """`spelling`: "us" (American) or "eu" (British) for every word the model writes, the Tailor page's US style / EU
    style buttons. `steps` (optional) gets a mark at the end of each step, for the job's timing line in the server
    log. `time_limit` (seconds, 0 = none) overrides the Settings time limit (the bake-off compares models without
    one)."""
    started, steps = time.perf_counter(), steps or Steps()
    settings = router.settings()
    paper = settings["paper_size"]
    metrics_mode = settings.get("metrics_mode", "estimate")
    # Settings > Certifications section. Off: no section, no picks, and the model is shown no certifications
    certifications_on = bool(settings.get("certifications_section", False))
    limit = float(settings.get("resume_time_limit_s", 30) or 0) if time_limit is None else time_limit  # 0: none
    deadline = started + limit - CHECKS_RESERVE_SECONDS if limit > 0 else None
    today = date.today()  # one date for the facts, the bullet budget and the stay lengths in the prompt
    facts = candidate_facts(profile, today)
    result = TailorResult(
        requested_model=model_id, prompt_version=PROMPT_VERSION, hints=hints, facts=facts, company_name=company_name,
        jd=jd, metrics_mode=metrics_mode, spelling=spelling,
    )
    outcome: Outcome = Outcome()
    notes: list[str] = []
    steps.mark("setup")

    try:
        chain = router.chain(model_id, allow_fallback)
        schema = strict_json_schema(TailorOutput)
        revision_schema = strict_json_schema(TailorRevision)
        steps.mark("models")

        # the page plan leaves room for the person's certifications only while they can be printed
        planned = profile if certifications_on else profile.model_copy(update={"certifications": []})

        def plan_bullets() -> list[int]:
            steps.mark("thread wait")  # from the request for a worker thread until this line runs in it
            return bullet_budget(planned, paper, today)

        budget = await asyncio.to_thread(plan_bullets)  # bullets per employer that fit
        steps.mark("bullet plan")
        context = {"profile": profile.model_dump(), "jd": jd, "hints": hints, "bullets": budget, "metrics": metrics_mode,
                   "spelling": spelling, "certifications_section": certifications_on}

        def build(out: TailorOutput):
            """(output, resume, notes). The analysis keeps keywords only, the headline the role's short name,
            titles that step down are corrected, and figures and their bases are written in varied shapes (a base in
            brackets at most once), on every draft, for free. Last, every word follows the spelling style (after the
            title fixes, which match the posting's own spelling)."""
            validate.clean_analysis(out.jd_analysis, certification_names(out))
            resume = validate.build_resume(profile, out)
            if certifications_on:  # off: nothing is printed, whatever the model picked
                resume.certifications, _rows = validate.resolve_certifications(profile, out.certifications, today)
            validate.assert_certifications_held(resume)
            # a certification is printed only in its own section, and only when the person has it
            validate.remove_certification_skills(resume, certification_names(out))
            validate.short_role_headline(resume)
            validate.general_role_fixes(resume, out.jd_analysis)  # 'Voice AI Engineer' -> 'AI Engineer'
            notes = validate.fix_title_regressions(resume)
            notes += validate.role_level_fixes(resume, out.jd_analysis)
            validate.vary_figures(resume, f"{hints}\n{facts}")  # the candidate's own figures are never dropped
            validate.spell_resume(resume, spelling, out.jd_analysis, (company_name, hiring_manager))
            return out, resume, notes

        def parse(data: dict):
            return build(TailorOutput.model_validate(data))

        def certification_names(out: TailorOutput) -> list[str]:
            return [pick.name for pick in out.certifications] + [cert.name for cert in profile.certifications]

        def without_certifications(out: TailorOutput) -> JDAnalysis:
            """The analysis the fix round shows, without certification keywords: it is told to use the keywords."""
            names = certification_names(out)
            return out.jd_analysis.model_copy(update={
                "keywords": [k for k in out.jd_analysis.keywords if not validate.names_certification(k, names)]})

        allowed_figures = f"{hints}\n{facts}"

        def refine(value) -> Ask | None:
            """Ask once to fix the draft if it uses too many figures, round guesses, thin text, leaves must-have
            JD terms out of the summary and bullets, copies the posting's voice, repeats its phrases, repeats a word
            by mistake, copies the posting's sentences or lists its phrases as skills, lists qualities as skills,
            sounds machine-written, names tools a job could not have used, opens jobs without a measured result, or
            has job titles that do not read as a career path. The model gets the numbered draft and returns only
            what it changes (a TailorRevision): far less to write than a whole new draft, and everything that was
            fine stays word for word."""
            feedback = []
            density = validate.metric_density_problems(value[1])
            if density:
                feedback.append(
                    "Too many bullets carry numbers (" + "; ".join(density) + "). Keep a figure in at most "
                    f"{validate.MAX_FIGURE_BULLETS_PER_JOB} bullets per employer and about 2 in 5 overall; rewrite "
                    "the other bullets to describe the impact in words. Keep the candidate's own figures."
                )
            openers = validate.first_bullet_problems(value[1]) if metrics_mode == "estimate" else []
            first_job = value[1].profile.companies[value[1].experience[0].company_index].name if openers else ""
            if openers and (len(openers) >= validate.MIN_FIRST_BULLETS_FOR_RETRY or openers[0] == first_job):
                feedback.append(
                    "These jobs open without a measured result: " + ", ".join(openers[:8]) + ". A recruiter reads "
                    "only the first bullet of each job in a six-second scan: rewrite it as that job's biggest win for "
                    "this JD, with its size and its result as figures (for example 'for 83K concurrent players, "
                    "cutting queue-to-lobby time from 41 to 32 seconds'). If a later bullet of the same job already "
                    "has a figure, move that result into the first bullet and say the later one in words, so the job "
                    f"keeps at most {validate.MAX_FIGURE_BULLETS_PER_JOB} bullets with figures."
                )
            unbased = validate.unbased_percentage_problems(value[1]) if metrics_mode == "estimate" else []
            mismatched = validate.base_mismatch_problems(value[1]) if metrics_mode == "estimate" else []
            if len(unbased) >= validate.MIN_UNBASED_FOR_RETRY or mismatched:
                feedback.append(
                    "A percentage means little without its base (20% of 5,000 or of 500,000?). "
                    + (f"These have none: {'; '.join(unbased[:8])}. Give each its base right beside it, in words: the "
                       "amount with its period or the values before and after ('by 17.8%, or $41,000 a year', 'from "
                       "41 to 32 seconds, a 22% drop'), or give the real numbers alone ('cut queue time from 41 to 32 "
                       "seconds'). Write no two figures the same way, and no base in brackets. " if unbased else "")
                    + (f"These bases do not agree with their percentage: {'; '.join(mismatched[:8])}. Make the "
                       "values match the percentage (from 41 to 32 is 22%). " if mismatched else "")
                    + "Keep a figure the candidate gave exactly as it is."
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
                    "of 60-85 words, and every weak bullet as 20-32 words that say specifically what was done, how, "
                    "and what changed, following the SUMMARY and BULLETS rules."
                )
            coverage = coverage_feedback(value[1], value[0].jd_analysis)
            if coverage:
                feedback.append(coverage)
            voice = validate.posting_voice_problems(value[1])
            if voice:
                feedback.append(
                    "These sentences copy the posting's own voice (you, your, we, our): "
                    + "; ".join(f"'{text}'" for text in voice[:3]) + ". Rewrite each one as something the candidate "
                    "did at that employer, in resume voice, keeping the skill it shows."
                )
            repeated = validate.repeated_phrase_problems(value[1], value[0].jd_analysis)
            if repeated:
                feedback.append(
                    "These posting phrases are repeated so often that they read as keyword stuffing: "
                    + ", ".join(repeated[:6]) + ". Keep each one at most twice in the summary and bullets, in the "
                    "places where it fits best, and say the same thing in your own words everywhere else."
                )
            doubled = validate.repeated_word_problems(value[1])
            if doubled:
                feedback.append(
                    "These words are repeated by mistake: " + ", ".join(f"'{pair}'" for pair in doubled[:6])
                    + ". Rewrite each sentence so it reads naturally, keeping its meaning."
                )
            qualities = validate.quality_skill_problems(value[1])
            if qualities:
                feedback.append(
                    "The skills list contains qualities, not skills: " + ", ".join(f"'{s}'" for s in qualities[:6])
                    + ". Replace them with real tools, platforms or methods related to the JD, and show those "
                    "qualities in the bullets instead."
                )
            copied = validate.copied_posting_spans(value[1], jd, value[0].jd_analysis)
            if len(copied) >= validate.MIN_COPIED_FOR_RETRY:
                feedback.append(
                    "These parts copy the job description word for word: " + "; ".join(copied[:8]) + ". Match the "
                    "keyword, not the sentence: keep any must-have keyword they contain, but say the rest in your own "
                    "words (what the candidate did there and what changed), keeping the number of bullets."
                )
            phrases = validate.skill_phrase_problems(value[1])
            if phrases:
                feedback.append(
                    "These skills are phrases from the job description, not keywords: "
                    + ", ".join(f"'{s}'" for s in phrases[:8]) + ". Replace each one through skill_changes with the "
                    "keyword itself, as people list it on a resume (for example 'SQL', 'AWS', 'Copilot')."
                )
            plain = validate.plain_word_problems(value[1], value[0].jd_analysis)
            if len(plain) >= validate.MIN_PLAIN_PROBLEMS_FOR_RETRY:
                feedback.append(
                    "These words or patterns sound machine-written: " + "; ".join(plain[:8]) + ". Rewrite those "
                    "sentences in plain, everyday words, following the PLAIN WORDS rule, and keep what they say, the "
                    "must-have phrases they show and the number of bullets. Replace a vague result ('improved "
                    "operational efficiency') with what specifically changed and for whom."
                )
            era = validate.era_problems(value[1])
            if era:
                feedback.append(
                    "Some bullets mention tools that job could not really have used: " + " ".join(era)
                    + " Move each one to a later job where it was common (AI tools only to the one or two most recent "
                    "jobs), or replace it with a tool of that time; keep the same number of bullets per employer."
                )
            special = validate.specialty_problems(value[1], value[0].jd_analysis)
            if special:
                common = validate.common_title(value[0].jd_analysis)
                feedback.append(
                    f"The {' and the '.join(special)} name{'s' if len(special) == 1 else ''} the posting's specialty "
                    f"'{validate.specialty(value[0].jd_analysis)}', which belongs only in the bullets: the headline, "
                    f"summary and job titles use the common title '{common[1]}' (the COMMON TITLE rule). Rewrite "
                    "them without it, keeping their length and everything else they say."
                )
            named = validate.certification_mentions(value[1], certification_names(value[0]))
            if named:
                feedback.append(
                    "These certification names appear in the headline, summary or bullets: "
                    + ", ".join(f"'{name}'" for name in dict.fromkeys(term for _where, term in named))
                    + (". Certifications are shown only in their own section" if certifications_on
                       else ". Certifications are never named in this resume")
                    + " (the CERTIFICATIONS rule), so take every "
                    "one out of the text. Keep the skill itself and the work it describes (for example 'AWS', not "
                    "'AWS Certified Solutions Architect'), and keep each sentence's length."
                )
            titles = validate.title_growth_problems(value[1])
            if titles:
                feedback.append(
                    "The job titles do not read as a believable career path: " + " ".join(titles) + " Choose the "
                    "titles again following the JOB TITLES rule, using the stay lengths given in <employers>."
                )
            if not feedback:
                return None
            output, resume, first_notes = value

            def revised(data: dict):
                """The fixed draft. A first-draft note about a corrected title, headline or summary stays while
                that text does."""
                out, fixed, notes = build(validate.apply_revision(output, resume, TailorRevision.model_validate(data)))
                same = {resume.profile.companies[a.company_index].name
                        for a, b in zip(resume.experience, fixed.experience) if a.job_title == b.job_title}
                same |= {f.title() for f in ("headline", "summary") if getattr(fixed, f) == getattr(resume, f)}
                return out, fixed, [n for n in first_notes if n.split(": ", 1)[0] in same] + notes

            return Ask(
                required=bool(named),  # a certification claim is removed even past the time limit
                user=revise_user(jd, profile.companies, hints, facts, bullets=budget, today=today,
                                 analysis=without_certifications(output), plan=output.career_plan,
                                 draft=draft_text(resume), problems=" ".join(feedback),
                                 certifications=profile.certifications, certifications_section=certifications_on),
                schema=revision_schema, schema_name="tailor_revision",
                parse=revised,
                context={**context, "draft": resume.model_dump(exclude={"profile"})},  # read only by the demo models
            )

        system_prompt = tailor_system(metrics_mode, spelling, certifications_section=certifications_on)
        user_prompt = tailor_user(jd, profile.companies, hints, facts, bullets=budget, today=today,
                                  certifications=profile.certifications, certifications_section=certifications_on)
        steps.mark("prompts")
        await generate_validated(
            router, chain, system=system_prompt, user=user_prompt,
            schema=schema, schema_name="tailor_output", parse=parse, context=context, outcome=outcome,
            refine=refine, deadline=deadline,
            switch_after=limit + SWITCH_EXTRA_SECONDS if limit > 0 else None,
            last_after=LAST_MODEL_SECONDS if limit > 0 else None,
        )
        steps.mark("AI")
        if outcome.final_model_id and outcome.final_model_id != model_id:
            reason = next((e.split(": ", 1)[1] for e in outcome.errors if e.startswith(f"{model_id}: ")),
                          "it is resting after a recent failure, see the server log")
            notes.append(f"{model_id} could not be used ({reason[:300]}), so {outcome.final_model_id} made this "
                         "resume.")
        if outcome.follow_up:
            notes.append(f"The rewrite pass was {outcome.follow_up} to finish within the time limit of {limit:g} "
                         "seconds (Settings), so this is the first draft with the app's own fixes.")
        output, resume, title_notes = outcome.value  # type: ignore[misc]

        # --- fit two pages at most: pick the style -> ask once more with a smaller bullet budget -> trim bullets
        fit = await asyncio.to_thread(try_styles, resume, paper)
        steps.mark("layout")
        time_left = None if deadline is None else deadline - time.perf_counter()
        # a shorter draft is a whole new draft: it takes about as long as the first one did
        if fit is None and time_left is not None and time_left < max(MIN_SHORTER_SECONDS, outcome.first_call_seconds):
            notes.append("There was no time left (Settings > Time limit) to ask the model for shorter bullets, so the "
                         "text was trimmed to fit two pages.")
        elif fit is None:
            second = Outcome()
            smaller = smaller_budget(budget)
            shorter_system = tailor_system(metrics_mode, spelling, certifications_section=certifications_on)
            shorter_user = tailor_user(jd, profile.companies, hints, facts, shorten=True, bullets=smaller, today=today,
                                       certifications=profile.certifications, certifications_section=certifications_on)
            try:
                shorter = generate_validated(
                    router, [outcome.final_model_id], system=shorter_system, user=shorter_user,
                    schema=schema, schema_name="tailor_output", parse=parse,
                    context={**context, "shorten": True, "bullets": smaller}, outcome=second,
                    refine=None,  # the shorten call is already the extra call; never let it ask a third time
                )
                await (asyncio.wait_for(shorter, time_left) if time_left is not None else shorter)
                _absorb(outcome, second)
                short_output, short_resume, short_title_notes = second.value  # type: ignore[misc]
                fit = await asyncio.to_thread(try_styles, short_resume, paper)
                if fit:
                    output, resume, title_notes = short_output, short_resume, short_title_notes
                    notes.append("The model shortened the bullets to fit two pages.")
            except GenerationFailed as exc:
                _absorb(outcome, exc.outcome)
                notes.append("Asking the model for shorter bullets failed, so the text was trimmed instead.")
            except TimeoutError:
                record_sent(second, router, outcome.final_model_id, shorter_system, shorter_user)  # billed anyway
                _absorb(outcome, second)
                notes.append("Asking the model for shorter bullets ran past the time limit (Settings), so the text was "
                             "trimmed instead.")
            steps.mark("shorter draft")
        if fit:
            style, pages, _pdf, fit_notes = fit
            notes += fit_notes
            fits = True
        else:
            resume, style, pages, _pdf, trim_notes = await asyncio.to_thread(trim_to_fit, resume, paper)
            notes += trim_notes
            fits = pages <= MAX_PAGES
            steps.mark("trimming")

        if outcome.input_tokens == 0 and outcome.output_tokens == 0:
            try:
                if router.registry.get(outcome.final_model_id).provider != "fake":
                    notes.append("The provider did not report token usage, so the cost shown ($0) is unknown.")
            except KeyError:
                pass
        notes += title_notes
        doubled = validate.fix_repeated_words(resume)  # the rewrite pass left some, or there was no pass
        if doubled:
            notes.append("Removed a repeated word: " + ", ".join(f"'{pair}'" for pair in doubled) + ".")
        result.resume = resume
        result.analysis = output.jd_analysis
        result.career_plan = output.career_plan
        result.certifications = (validate.resolve_certifications(profile, output.certifications, today)[1]
                                 if certifications_on else [])  # off: no rows to tick on the Tailor page
        result.style, result.pages, result.fits = style, pages, fits
        result.served_model = outcome.served_model
        result.status = "ok"

        # --- optional cover letter, from the final resume text
        if cover_letter:
            await _cover_letter(
                router, result, outcome, notes, jd=jd, hints=hints,
                company_name=company_name or output.jd_analysis.company, hiring_manager=hiring_manager,
                model_id=cover_model_id or outcome.final_model_id, allow_fallback=allow_fallback, steps=steps,
            )
        run_checks(result, jd)
        steps.mark("checks")
    except GenerationFailed as exc:
        if exc.outcome is not outcome:
            _absorb(outcome, exc.outcome)
        result.status, result.error = "error", str(exc)[:900]
    except (CostCapError, LLMError) as exc:
        result.status, result.error = "error", str(exc)[:900]
    except Exception as exc:  # never let one model's failure break a compare run
        log.exception("Unexpected error while tailoring with %s", model_id)
        result.status, result.error = "error", f"Unexpected error: {type(exc).__name__}: {str(exc)[:300]}"
    if result.status != "ok":
        steps.mark("until it failed")

    result.notes = notes
    result.attempts = outcome.attempts
    result.retries = outcome.retries
    result.fallback_chain = outcome.chain
    result.input_tokens = outcome.input_tokens
    result.output_tokens = outcome.output_tokens
    result.cost_usd = round(outcome.cost_usd, 6)
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    if save:
        db.save_run(result, profile_id, jd_hash(jd), user_id=user_id)
        steps.mark("saving")
    return result


async def _cover_letter(
    router: Router, result: TailorResult, outcome: Outcome, notes: list[str], *, jd: str, hints: str,
    company_name: str, hiring_manager: str, model_id: str, allow_fallback: bool, steps: Steps | None = None,
) -> str | None:
    """Write result.cover_letter from the final resume. Returns None, or the reason no letter was written (also
    added to notes). A cost-cap stop here leaves the resume intact instead of failing the whole run."""
    assert result.resume is not None and result.analysis is not None
    steps = steps or Steps()
    resume, style = result.resume, result.spelling or "us"
    keep = validate.known_names(resume, result.analysis, company_name, hiring_manager)
    terms, roles = validate.skill_terms(resume, result.analysis), validate.role_terms(resume, result.analysis)
    exact = validate.posting_terms(result.analysis)

    def parse(data: dict) -> CoverLetter:
        raw = CoverLetterOut.model_validate(data)
        greeting, paragraphs, closing = validate.clean_cover_letter(raw.greeting, raw.paragraphs, raw.closing)
        letter = CoverLetter(
            greeting=greeting, paragraphs=paragraphs, closing=closing,
            company_name=company_name, hiring_manager=hiring_manager,
        )
        validate.vary_letter_figures(letter, f"{hints}\n{result.facts}")  # a base in brackets at most once
        validate.spell_letter(letter, style, keep, terms, exact, roles)  # the run's style, its sign-off, CV or resume
        return letter

    second = Outcome()
    try:
        chain = router.chain(model_id, allow_fallback)
        schema = strict_json_schema(CoverLetterOut)
        steps.mark("cover models")
        system_prompt = cover_system(style)
        user_prompt = cover_user(jd, resume, result.analysis.target_role, company_name, hiring_manager, hints)
        steps.mark("cover prompts")
        await generate_validated(
            router, chain, system=system_prompt, user=user_prompt,
            schema=schema, schema_name="cover_letter", parse=parse,
            context={
                "company_name": company_name, "hiring_manager": hiring_manager,
                "target_role": result.analysis.target_role, "keywords": result.analysis.keywords,
            },
            outcome=second,
        )
        steps.mark("cover AI")
        _absorb(outcome, second)
        result.cover_letter = second.value
        return None
    except GenerationFailed as exc:
        _absorb(outcome, exc.outcome)
        message = f"Cover letter failed: {str(exc)[:200]}"
    except CostCapError as exc:
        _absorb(outcome, second)
        message = f"Cover letter not written: {str(exc)[:200]}"
    steps.mark("cover AI failed")
    notes.append(message)
    return message


async def write_cover_letter(
    router: Router, db: Database, run_id: int, *, jd: str, model_id: str, user_id: int | None = None,
    steps: Steps | None = None,
) -> tuple[TailorResult, str | None]:
    """Write (or rewrite) the cover letter of a saved run.

    Returns (updated run, error). The caller has checked that the run exists and has a resume and analysis.
    The spend is saved as its own row dated today (task 'cover_letter'), so the daily cost cap counts it even
    when the run itself is from an earlier day; the run only shows it in cover_cost_usd. The run is read again
    before saving so that edits made on the Review page in the meantime are kept. `steps` (optional) gets a mark at
    the end of each step, for the job's timing line in the server log.
    """
    steps = steps or Steps()
    base = db.get_run(run_id)
    assert base is not None and base.resume is not None and base.analysis is not None
    base.cover_letter = None
    outcome, notes, started = Outcome(), [], time.perf_counter()
    steps.mark("reading the run")
    error = await _cover_letter(
        router, base, outcome, notes, jd=jd, hints=base.hints,
        company_name=base.company_name or base.analysis.company, hiring_manager="",
        model_id=model_id, allow_fallback=True, steps=steps,
    )
    if outcome.attempts:
        db.save_run(TailorResult(
            requested_model=model_id, served_model=outcome.served_model, status="error" if error else "ok",
            error=error or "", input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens,
            cost_usd=round(outcome.cost_usd, 6), attempts=outcome.attempts, retries=outcome.retries,
            fallback_chain=outcome.chain, latency_ms=int((time.perf_counter() - started) * 1000),
            notes=[f"Cover letter for run {run_id}."],
        ), None, "", task="cover_letter", user_id=user_id)
    steps.mark("cost record")
    fresh = db.get_run(run_id) or base
    steps.mark("reading the run again")
    if base.cover_letter is not None:
        fresh.cover_letter = base.cover_letter  # replaces any earlier letter
    fresh.cover_cost_usd = round(fresh.cover_cost_usd + outcome.cost_usd, 6)
    fresh.notes = [n for n in fresh.notes if not n.startswith(("Cover letter failed", "Cover letter not written"))]
    fresh.notes += notes
    if not fresh.jd:
        fresh.jd = jd
    run_checks(fresh, jd)  # recomputes all flags, so cover-letter flags are never duplicated
    steps.mark("checks")
    db.update_run(fresh)
    steps.mark("saving")
    return fresh, error
