"""Data models.

Three groups:
  * Profile and friends: the FIXED facts from the base resume. The AI never writes these.
  * TailorOutput and friends: what the model returns (all fields required, no defaults, so the
    JSON schema is strict and works across providers).
  * Resume / TailorResult: the merged document and the full record of one run.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class Profile(BaseModel):
    name: str
    contact: Contact = Field(default_factory=Contact)
    companies: list[Company] = Field(default_factory=list)  # most recent first
    education: list[Education] = Field(default_factory=list)


# ---------------------------------------------------------------- what the model returns
class JDAnalysis(BaseModel):
    company: str = ""  # hiring company as the JD names it; "" if it does not (default keeps older runs loadable)
    target_role: str
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


class TailorOutput(BaseModel):
    jd_analysis: JDAnalysis
    sections: TailoredSections


class CoverLetterOut(BaseModel):
    greeting: str
    paragraphs: list[str]
    closing: str


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
    latency_ms: int = 0
    attempts: int = 1  # model calls made in total (resume + cover letter + any retries)
    retries: int = 0  # times a model was asked again because its answer was unusable
    fallback_chain: list[str] = Field(default_factory=list)  # models tried, in order
    prompt_version: str = ""
    hints: str = ""  # the user's own facts, kept so claim checks can be re-run after manual edits
    facts: str = ""  # facts computed from the profile (years of experience) that were given to the model
    company_name: str = ""  # company the user typed on the tailor form (wins over the one read from the JD)
    created_at: str = ""
