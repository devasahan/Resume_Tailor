"""Data models.

Three groups:
  * Profile and friends: the FIXED facts from the base resume. The AI never writes these.
  * TailorOutput and friends: what the model returns (all fields required, no defaults, so the
    JSON schema is strict and works across providers; a default, as on CareerStep.product, only serves
    hand-built outputs and old saved runs). TailorRevision is the answer to a fix request: only what changes. Its
    defaults only mean "keep" when a key is left out; the schema still requires every field.
  * Resume / TailorResult: the merged document and the full record of one run.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- fixed facts (from the base resume)
class Contact(BaseModel):
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    location: str = ""
    website: str = ""


class Company(BaseModel):
    name: str
    start: str = ""  # e.g. "Jan 2021"
    end: str = ""  # e.g. "Mar 2023" or "Present"
    mode: str = ""  # "Remote" | "Onsite" | "Hybrid" (blank when the base resume does not say)
    location: str = ""


class Education(BaseModel):
    school: str
    degree: str = ""
    field: str = ""
    start: str = ""
    end: str = ""
    details: str = ""


MAX_CERTIFICATIONS = 4  # the model picks this many per job; a resume prints at most this many


def _as_text(value):
    """A model may send null or a number ("year": 2021) for a text field: read as text instead of failing the answer."""
    if value is None:
        return ""
    return str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


class Certification(BaseModel):
    """A certification or license the person has, copied exactly as written."""
    # From the uploaded resume, or ticked "I have it" on the Tailor page. Only these are ever printed on a resume.
    # (The docstring above is part of the AI resume reader's schema; this comment is not.)
    name: str
    issuer: str = ""
    # 'year' (not 'earned') so resumes saved before the expiry date was kept still load
    year: str = Field("", description='the date it was earned, as written ("2023", "Mar 2023"); "" when not stated')
    expires: str = Field("", description='the date it expires, as written ("Mar 2026"); "" when it does not expire '
                                         'or this is not stated')

    _text = field_validator("name", "issuer", "year", "expires", mode="before")(_as_text)


class Profile(BaseModel):
    name: str
    contact: Contact = Field(default_factory=Contact)
    companies: list[Company] = Field(default_factory=list)  # most recent first
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)  # what the person has ([] in older profiles)


# ---------------------------------------------------------------- what the model returns
class JDAnalysis(BaseModel):
    company: str = ""  # hiring company as the JD names it; "" if it does not (default keeps older runs loadable)
    target_role: str
    # The most similar common title, without the specialty that makes the posting's title a special one ('AI
    # Engineer' for 'Voice AI Engineer'): the headline, the summary and the job titles use it. "" in older runs.
    general_role: str = ""
    seniority: str
    domain: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    keywords: list[str]
    key_responsibilities: list[str]


class SkillGroup(BaseModel):
    category: str
    skills: list[str]  # not called `items`: that is a JSON Schema keyword and confuses some providers


class ExperienceOut(BaseModel):
    company_index: int
    job_title: str
    bullets: list[str]


class TailoredSections(BaseModel):
    headline: str
    summary: str
    skills: list[SkillGroup]
    experience: list[ExperienceOut]


class CareerStep(BaseModel):
    """What one job will show, planned before the bullets are written (see CAREER FLOW in the prompt)."""
    company_index: int
    # Decided first, so the bullets describe work on what that company really offered at the time.
    product: str = Field("", description=(
        "the employer's main product or service during this job's dates, as the company itself names and describes "
        "it (for example \"Acquire.com's marketplace for buying and selling startups\"); \"\" if you do not know "
        "this company"))
    focus: list[str] = Field(description="3-6 of the job description's skills or responsibilities this job shows, in its words")
    scope: str = Field(description="implemented, built, designed, owned or led")


class CertificationPick(BaseModel):
    """One certification for this job, as the CERTIFICATIONS rule in the prompt says. Only one of the candidate's own
    is ever printed; any other is never shown as theirs."""
    # The docstring and descriptions are part of the model's schema, sent in both Settings modes, so they stay true in
    # both. On: a pick that is not the person's own is a suggestion on the Tailor page. Off: picks are never printed
    # or shown; they only keep names out of the text.
    name: str = Field(description=(
        "the certification's official full name, without dash punctuation (for example \"AWS Certified Solutions "
        "Architect Professional\"); for one of the candidate's own, its name exactly as listed"))
    issuer: str = Field("", description="the organization that awards it (for example \"Amazon Web Services\")")
    candidate_index: int = Field(-1, description=(
        "the number of the candidate's own certification when the prompt lists their certifications and this is one "
        "of them, otherwise -1"))

    _text = field_validator("name", "issuer", mode="before")(_as_text)

    @field_validator("candidate_index", mode="before")
    @classmethod
    def _number(cls, value):  # null, "" or "none" from a model: not one of the candidate's
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return -1


class TailorOutput(BaseModel):
    jd_analysis: JDAnalysis
    # Up to 4, best first, decided right after the analysis. Required in the model's schema; the default only serves
    # code that builds an output by hand and older outputs.
    certifications: list[CertificationPick] = Field(default_factory=list, description=(
        "the certifications the CERTIFICATIONS rule asks for, at most 4, best first"))
    # Written before `sections`, so the bullets follow the plan. Required in the model's schema; the default only
    # serves code that builds an output by hand.
    career_plan: list[CareerStep] = Field(default_factory=list)
    sections: TailoredSections

    @field_validator("certifications", mode="before")
    @classmethod
    def _usable_picks(cls, value):
        """A malformed pick is dropped, not the whole draft: the picks are the smallest part of the answer."""
        if not isinstance(value, list):
            return []
        picks = []
        for item in value:
            if isinstance(item, str):  # a bare name
                item = {"name": item}
            try:
                pick = item if isinstance(item, CertificationPick) else CertificationPick.model_validate(item)
            except (ValueError, TypeError):  # pydantic's ValidationError is a ValueError
                continue
            if pick.name.strip() or pick.candidate_index >= 0:
                picks.append(pick)
        return picks


class TitleChange(BaseModel):
    company_index: int
    job_title: str


class BulletChange(BaseModel):
    company_index: int
    bullet_index: int = Field(
        description="the number in [brackets] before the bullet in the draft, counting from 0 within each employer")
    text: str = Field(description="the complete new bullet, without the [number]")


class SkillChange(BaseModel):
    old: str = Field(description="a skill exactly as the draft lists it")
    new: str = Field(description='what replaces it, or "" to remove it')


class TailorRevision(BaseModel):
    """A fix of a draft: only what changes. Everything left out stays as the draft has it, word for word."""
    headline: str = Field("", description='the complete new headline, or "" to keep the draft\'s')
    summary: str = Field("", description='the complete new summary, or "" to keep the draft\'s')
    titles: list[TitleChange] = Field(default_factory=list, description="only the job titles that change")
    bullets: list[BulletChange] = Field(default_factory=list, description="only the bullets that change")
    skill_changes: list[SkillChange] = Field(default_factory=list, description="only the skills that are replaced")


class CoverLetterOut(BaseModel):
    greeting: str
    paragraphs: list[str]
    closing: str


class JobFacts(BaseModel):
    """What a pasted job post says about where the job is done and what it pays, as the AI reads it."""
    work_mode: Literal["Remote", "Hybrid", "On-site", "Not stated"]
    work_mode_detail: str  # "3 days a week in office", "US only"; "" when the post says nothing more
    salary: str  # as written, with currency and period: "$140,000 - $180,000 a year"; "" when not stated


# ---------------------------------------------------------------- merged document
class ResumeExperience(BaseModel):
    company_index: int  # index into Resume.profile.companies
    job_title: str
    bullets: list[str]


class Resume(BaseModel):
    profile: Profile  # fixed facts, copied verbatim from the saved profile
    headline: str
    summary: str
    skills: list[SkillGroup]
    experience: list[ResumeExperience]
    # Printed under CERTIFICATIONS, at most 4, best for the job first: always ones the person has (each is also in
    # profile.certifications). [] in older runs.
    certifications: list[Certification] = Field(default_factory=list)


class CertificationChoice(BaseModel):
    """One of the model's picks for a run, as the Tailor page lists it: `have` means the person has it, so it is
    printed on the resume. The person ticks or unticks it there."""
    name: str
    issuer: str = ""
    year: str = ""  # earned
    expires: str = ""
    have: bool = False


class CoverLetter(BaseModel):
    greeting: str
    paragraphs: list[str]
    closing: str
    company_name: str = ""
    hiring_manager: str = ""


# ---------------------------------------------------------------- checks and run record
class Flag(BaseModel):
    section: str  # headline | summary | skills | experience:<n> | cover_letter
    text: str
    reason: str
    level: Literal["verify", "info"] = "verify"


class Coverage(BaseModel):
    score: float = 0.0  # share of JD keywords found in the resume, 0..1
    found: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    skills_only: list[str] = Field(default_factory=list)  # must-haves listed but not shown in summary or a bullet


class FitStyle(BaseModel):
    font_pt: float = 10.5
    margin_in: float = 0.65
    paper: Literal["Letter", "A4"] = "Letter"
    stacked: bool = False  # each job's title on its own line under the company (used when it costs no bullet)


class TailorResult(BaseModel):
    run_id: int | None = None
    requested_model: str
    served_model: str = ""
    status: Literal["ok", "error"] = "ok"
    error: str = ""
    resume: Resume | None = None
    analysis: JDAnalysis | None = None
    cover_letter: CoverLetter | None = None
    flags: list[Flag] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    ats_warnings: list[str] = Field(default_factory=list)
    pages: int = 0
    fits: bool = False
    style: FitStyle = Field(default_factory=FitStyle)
    notes: list[str] = Field(default_factory=list)  # what the fit step did (shrunk font, trimmed bullets, ...)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cover_cost_usd: float = 0.0  # cover letters written later for this run (each also has its own dated cost row)
    latency_ms: int = 0
    attempts: int = 1  # model calls made in total (resume + cover letter + any retries)
    retries: int = 0  # times a model was asked again because its answer was unusable
    fallback_chain: list[str] = Field(default_factory=list)  # models tried, in order
    prompt_version: str = ""
    hints: str = ""  # the user's own facts, kept so claim checks can be re-run after manual edits
    facts: str = ""  # facts computed from the profile (years of experience) that were given to the model
    company_name: str = ""  # company the user typed on the tailor form (wins over the one read from the JD)
    jd: str = ""  # the job description this run was made for (a cover letter can be written later from it)
    metrics_mode: str = ""  # the "Numbers in bullets" setting the run was made with ("" for older runs)
    spelling: Literal["", "us", "eu"] = ""  # the Tailor page's US/EU style button ("" for older runs: US)
    career_plan: list[CareerStep] = Field(default_factory=list)  # how the model spread the JD across the jobs
    # The model's picks, the rows the Tailor page shows. [] while the Certifications section is off, and in older runs.
    certifications: list[CertificationChoice] = Field(default_factory=list)
    # While the Certifications section is off (Settings): the certifications the job post names, as the model listed
    # them. Never printed and never rows to tick: the checks only keep these names out of the text, as they do the
    # rows' names. [] while it is on (the rows hold the picks) and in older runs.
    posting_certifications: list[str] = Field(default_factory=list)
    # The jobs' bullets as they were before a tick on the Tailor page removed some to keep two pages ([] when none
    # did): each later tick starts from them again, so unticking brings those bullets back.
    untrimmed: list[ResumeExperience] = Field(default_factory=list)
    created_at: str = ""
