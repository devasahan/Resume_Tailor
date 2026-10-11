"""Checks that run on every result, whichever model produced it.

  build_resume            merge model output with the locked profile (the AI never touches fixed facts)
  assert_fixed_fields_intact   safety net that the fixed facts are byte-identical to the saved profile
  keyword_coverage        how many JD keywords appear in the resume
  ats_lint                format / length warnings
  claim_flags             numbers and skills the user never supplied, for the review step
"""
from __future__ import annotations

import re
import unicodedata

from ..schemas import (
    CoverLetter,
    Coverage,
    Flag,
    JDAnalysis,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
    TailorOutput,
)

_PUNCT_MAP = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " ", "\t": " "}
)
_BULLET_LEAD = re.compile(r"^[\s\-*•·▪◦‣–—>]+")
_NUMBER = re.compile(r"[$€£]?\d[\d,.]*\s?(?:%|[kKmMbB]\b|\+|x\b)?")
_FIRST_PERSON = re.compile(r"\b(I|me|my|mine|myself|I'm|I've)\b")


def clean_text(text: str) -> str:
    """Normalise punctuation, drop symbols/emoji/control characters, collapse whitespace."""
    text = text.translate(_PUNCT_MAP)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] not in ("S", "C") or ch in "+#$%&<>=|")
    return re.sub(r"\s+", " ", text).strip()


def clean_bullet(text: str) -> str:
    return clean_text(_BULLET_LEAD.sub("", text))


def word_count(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------ merge
def build_resume(profile: Profile, out: TailorOutput) -> Resume:
    """Raise ValueError (retried by the pipeline with this message) if the output cannot be used."""
    sections = out.sections
    n = len(profile.companies)
    indices = [e.company_index for e in sections.experience]
    if sorted(indices) != list(range(n)):
        raise ValueError(
            f"`sections.experience` must have exactly one entry for each company_index 0..{n - 1} "
            f"(got {sorted(indices)})."
        )
    headline = clean_text(sections.headline)
    summary = clean_text(sections.summary)
    if not headline or not summary:
        raise ValueError("`headline` and `summary` must not be empty.")

    skills: list[SkillGroup] = []
    seen: set[str] = set()
    for group in sections.skills:
        items = []
        for item in group.items:
            item = clean_text(item)
            if item and item.lower() not in seen:
                seen.add(item.lower())
                items.append(item)
        if items and clean_text(group.category):
            skills.append(SkillGroup(category=clean_text(group.category), items=items))
    if not skills:
        raise ValueError("`skills` must contain at least one group with items.")

    experience: list[ResumeExperience] = []
    for entry in sorted(sections.experience, key=lambda e: e.company_index):
        title = clean_text(entry.job_title)
        bullets = [b for b in (clean_bullet(x) for x in entry.bullets) if b]
        if not title or len(title) > 80:
            raise ValueError(f"Job title for company_index {entry.company_index} is empty or too long.")
        if not bullets:
            raise ValueError(f"company_index {entry.company_index} has no bullets.")
        experience.append(ResumeExperience(company_index=entry.company_index, job_title=title, bullets=bullets))

    resume = Resume(
        profile=profile.model_copy(deep=True), headline=headline, summary=summary, skills=skills, experience=experience
    )
    assert_fixed_fields_intact(profile, resume)
    return resume


def assert_fixed_fields_intact(original: Profile, resume: Resume) -> None:
    if resume.profile.model_dump() != original.model_dump():
        raise ValueError("Fixed profile fields (name, contact, companies, dates, education) were altered.")
    if [e.company_index for e in resume.experience] != list(range(len(original.companies))):
        raise ValueError("Experience entries no longer map one-to-one onto the profile's companies.")


# ------------------------------------------------------------------ text helpers
def resume_text(resume: Resume) -> str:
    parts = [resume.headline, resume.summary]
    for group in resume.skills:
        parts.append(group.category)
        parts.extend(group.items)
    for exp in resume.experience:
        parts.append(exp.job_title)
        parts.extend(exp.bullets)
    return "\n".join(parts)


def contains_term(text: str, term: str) -> bool:
    term = term.strip()
    if not term:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9+#])"
    flags = 0 if len(term) <= 3 else re.IGNORECASE
    return re.search(pattern, text, flags) is not None


# ------------------------------------------------------------------ checks
def keyword_coverage(resume: Resume, analysis: JDAnalysis) -> Coverage:
    terms: list[str] = []
    for term in [*analysis.must_have_skills, *analysis.keywords]:
        term = term.strip()
        if term and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
    text = resume_text(resume)
    found = [t for t in terms if contains_term(text, t)]
    missing = [t for t in terms if t not in found]
    return Coverage(score=round(len(found) / len(terms), 3) if terms else 0.0, found=found, missing=missing)


def ats_lint(resume: Resume) -> list[str]:
    warnings: list[str] = []
    if word_count(resume.headline) > 14:
        warnings.append(f"Headline is long ({word_count(resume.headline)} words).")
    if word_count(resume.summary) > 70:
        warnings.append(f"Summary is long ({word_count(resume.summary)} words).")
    if _FIRST_PERSON.search(resume.summary):
        warnings.append("Summary uses first-person pronouns.")
    total_skills = sum(len(g.items) for g in resume.skills)
    if not 12 <= total_skills <= 35:
        warnings.append(f"{total_skills} skills listed (aim for roughly 15-28).")
    titles = [e.job_title.lower() for e in resume.experience]
    if len(titles) > 1 and len(set(titles)) == 1:
        warnings.append("The same job title is used at every employer, so there is no visible career progression.")
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index].name
        if not 2 <= len(exp.bullets) <= 7:
            warnings.append(f"{company}: {len(exp.bullets)} bullets (aim for 3-5).")
        openers = [b.split()[0].lower() for b in exp.bullets if b.split()]
        if len(openers) != len(set(openers)):
            warnings.append(f"{company}: two bullets start with the same word.")
        for bullet in exp.bullets:
            if word_count(bullet) > 32:
                warnings.append(f"{company}: a bullet is {word_count(bullet)} words long.")
                break
            if _FIRST_PERSON.search(bullet):
                warnings.append(f"{company}: a bullet uses first-person pronouns.")
                break
    return warnings


def _numbers_in(text: str) -> list[str]:
    return [m.group(0).strip(" ,.") for m in _NUMBER.finditer(text)]


def claim_flags(resume: Resume, jd: str, hints: str) -> list[Flag]:
    """Things the user did not provide and should confirm before sending the resume."""
    flags: list[Flag] = []
    hints_l, jd_l = hints.lower(), jd.lower()

    def check_numbers(section: str, text: str) -> None:
        for token in _numbers_in(text):
            if token and token.lower() not in hints_l:
                flags.append(
                    Flag(section=section, text=text, level="verify",
                         reason=f"Contains '{token}', which is not in your hints. Only keep figures you can back up.")
                )
                return

    check_numbers("headline", resume.headline)
    check_numbers("summary", resume.summary)
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index].name
        flags.append(
            Flag(section=f"experience:{exp.company_index}", text=exp.job_title, level="info",
                 reason=f"Suggested job title at {company}. Use a title you can honestly claim.")
        )
        for bullet in exp.bullets:
            check_numbers(f"experience:{exp.company_index}", bullet)
    for group in resume.skills:
        for item in group.items:
            if item.lower() not in jd_l and item.lower() not in hints_l:
                flags.append(
                    Flag(section="skills", text=item, level="info",
                         reason="Not mentioned in the job description. Inferred for this career path, keep it only if true.")
                )
    return flags


def cover_flags(letter: CoverLetter, hints: str) -> list[Flag]:
    flags: list[Flag] = []
    hints_l = hints.lower()
    for paragraph in letter.paragraphs:
        for token in _numbers_in(paragraph):
            if token.lower() not in hints_l:
                flags.append(
                    Flag(section="cover_letter", text=paragraph, level="verify",
                         reason=f"Contains '{token}', which is not in your hints. Only keep figures you can back up.")
                )
                break
    return flags


def clean_cover_letter(greeting: str, paragraphs: list[str], closing: str) -> tuple[str, list[str], str]:
    paras = [p for p in (clean_text(x) for x in paragraphs) if p]
    if not 2 <= len(paras) <= 6:
        raise ValueError("The cover letter needs between 2 and 6 non-empty paragraphs.")
    greeting, closing = clean_text(greeting), clean_text(closing)
    if not greeting or not closing:
        raise ValueError("`greeting` and `closing` must not be empty.")
    return greeting, paras, closing
