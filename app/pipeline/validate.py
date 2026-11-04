"""Checks that run on every result, whichever model produced it.

  build_resume            merge model output with the locked profile (the AI never touches fixed facts)
  assert_fixed_fields_intact   safety net that the fixed facts are byte-identical to the saved profile
  keyword_coverage        how many JD keywords appear in the resume
  ats_lint                format / length warnings
  claim_flags             numbers and skills the user never supplied, for the review step
  unshown_must_haves      required JD terms missing, or only listed, instead of shown in the summary or a bullet
  posting_match_flags     sentences added to match the JD that the user's hints do not back up
"""
from __future__ import annotations

import re
import unicodedata
from itertools import pairwise

from ..schemas import (
    CoverLetter,
    Coverage,
    ExperienceOut,
    Flag,
    JDAnalysis,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
    TailoredSections,
    TailorOutput,
    TailorRevision,
)
from .facts import months_phrase, parse_point, stay_months

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


# ------------------------------------------------------------------ keywords, not job-description phrases
_SMALL_WORDS = frozenset({
    "a", "an", "the", "and", "or", "nor", "but", "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "into",
    "onto", "over", "under", "via", "per", "across", "like", "than", "then", "our", "your", "their", "its", "his", "her",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "we", "you", "they",
    "using", "through", "within", "while", "when", "where", "which", "who", "how", "all", "each", "both", "not", "so",
    "if", "also", "up", "out", "about", "after", "before", "between", "during", "without", "can", "will", "has", "have",
    "had", "years", "experience",  # '7+ years of experience' opens every summary
})
MAX_TERM_WORDS = 4  # meaningful words: a longer phrase is a piece of a sentence, not what an ATS or a recruiter matches
MAX_NAME_WORDS = 8  # ... but a proper name may be longer: 'Payment Card Industry Data Security Standard (PCI DSS)'

# "AI tools like Copilot, Cursor, or Claude Code": the words after the lead are the tools themselves.
_EXAMPLES_LEAD = re.compile(r"(?<![\w-])(?:like|such as|including|e\.g\.?)(?![\w-])\s*[:,]?\s*", re.IGNORECASE)
_BRACKET_LEAD = re.compile(r"\(\s*(?:like|such as|including|e\.g\.?)(?![\w-])\s*[:,]?\s*", re.IGNORECASE)
_COMMA_SPLIT = re.compile(r"\s*[,;]\s*(?:(?:or|and|&)\s+)?", re.IGNORECASE)  # 'A, B, and C'
_OR_WORD = re.compile(r"\s+or\s+", re.IGNORECASE)  # 'OR-Tools' stays whole
_AND_WORD = re.compile(r"\s+(?:and|&)\s+", re.IGNORECASE)
_LOOSE_JOINER = re.compile(r"^(?:or|and|&)\s+|\s+(?:or|and|&)$", re.IGNORECASE)
# The end of a list, not an item: "Python, Go, etc.", "Copilot, Cursor or similar", "AWS or other cloud platforms"
_LIST_END = re.compile(r"(?:etc|more|others?|similar|the like|so on)\.?$|(?:other|similar) ", re.IGNORECASE)
_BRACKETS = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_SPELLED_OUT = re.compile(r"(.+?)\s*\(([A-Za-z&]{2,})\)")  # 'Extract, Transform, Load (ETL)'
# In the analysis, a bracket of examples or a list is detail, not the keyword: 'AWS (e.g., EC2, S3)', 'AWS (ECS, SQS)'
_DETAIL_BRACKET = re.compile(
    r"\s*\((?:\s*(?:like|such as|including|e\.g\.?)(?![\w-])[^()]*|[^()]*,[^()]*)\)", re.IGNORECASE)

# Prose around a keyword in a posting: "experience with Kafka", "advanced SQL statements", "AWS-hosted systems"
_PROSE_LEAD = re.compile(
    r"^.*?\b(?:experience|knowledge|understanding|familiarity|proficiency|expertise|background)\s+"
    r"(?:of|with|in|using)\s+", re.IGNORECASE)
_ABILITY = re.compile(r"\bability to\b", re.IGNORECASE)
_HOSTED = re.compile(r"^(.+?)-hosted\b.*$", re.IGNORECASE)
_QUALIFIER = re.compile(  # in lower case or capitalized, so that a name in capitals stays: 'SOLID principles'
    r"\b(?:[Aa]dvanced|[Bb]asic|[Ss]trong|[Ss]olid|[Pp]roven|[Ee]xcellent|[Dd]eep|[Gg]ood|[Hh]igh-scale|[Ll]arge-scale"
    r"|[Hh]ands-on|[Ee]xtensive)\s+")
_TOOL_SKILLS = re.compile(r"(\b[A-Z][\w.+#/-]*)\s+skills?$")  # 'Python skills' -> 'Python'; 'people skills' stays
# ... but not where such a word is part of a name ("Deep Learning", "Visual Basic", "self-hosted runners")
_NAMED_TERMS = re.compile(
    r"\b(?:deep (?:learning|neural|reinforcement|links?|linking|packet inspection)|visual basic|basic auth\w*"
    r"|strong (?:consistency|customer authentication|typing)"
    r"|advanced (?:analytics|encryption standard|persistent threats?)"
    r"|good (?:manufacturing|clinical|laboratory|distribution|documentation) practices?"
    r"|(?:self|github|gitlab|microsoft)-hosted"
    r"|solid\.?js|large-scale scrum|(?:basic|advanced(?: cardiac| cardiovascular)?|pediatric advanced) life support)\b",
    re.IGNORECASE,
)
_ABBREVIATION = re.compile(r"\(\s*[A-Z][A-Z0-9&/+.-]+(?:\s+[A-Z][A-Z0-9&/+.-]+)*\s*\)")  # '(BLS)', '(PCI DSS)'


def _meaningful(text: str) -> list[str]:
    """Words that carry meaning: no small words and no lone marks ('-', '&'); a word in capitals counts ('CAN')."""
    words = [word.strip(",.;:!?\"'") for word in text.split()]
    return [w for w in words if re.search(r"[A-Za-z0-9]", w) and (w.isupper() or w.lower() not in _SMALL_WORDS)]


def _is_name(term: str) -> bool:
    """A proper name rather than prose: it has an abbreviation in brackets ('Basic Life Support (BLS)') or is written
    in title case ('Solid Edge', 'Good Documentation Practices', 'SOLID', 'iOS')."""
    if _ABBREVIATION.search(term):
        return True
    words = _meaningful(re.sub(r"\([^)]*\)", "", term))
    return bool(words) and all(not w[0].isalpha() or any(ch.isupper() for ch in w) for w in words)


def _masked(text: str) -> tuple[str, list[str]]:
    """The text with each bracket (nested ones too) replaced by a placeholder, so lists inside brackets stay whole."""
    kept: list[str] = []

    def mask(match: re.Match) -> str:
        kept.append(match.group(0))
        return f"\0{len(kept) - 1}\0"

    text = text.replace("\0", "")
    while (masked := _BRACKETS.sub(mask, text)) != text:
        text = masked
    return text, kept


def _unmasked(text: str, kept: list[str]) -> str:
    while (restored := re.sub(r"\0(\d+)\0", lambda m: kept[int(m.group(1))], text)) != text:
        text = restored
    return text


def _spelled_out(item: str) -> bool:
    """A name with commas in it: 'Extract, Transform, Load (ETL)', 'Water, Sanitation and Hygiene (WASH)'. The
    abbreviation starts with the first word's initial and holds every word's initial, in order."""
    match = _SPELLED_OUT.fullmatch(item)
    if not match:
        return False
    initials, letters = [word[0].lower() for word in _meaningful(match.group(1))], match.group(2).lower()
    rest = iter(letters)
    return bool(initials) and letters[0] == initials[0] and all(initial in rest for initial in initials)


def _list_items(text: str, lead: bool) -> list[str]:
    """Split a list at commas and 'or'; after a lead ('such as AWS and GCP') also at the 'and' before its last item,
    so names that hold an 'and' stay whole: 'Profit and Loss, Balance Sheet and Cash Flow', 'Research and
    Development'."""
    parts = [piece for part in _COMMA_SPLIT.split(text) for piece in _OR_WORD.split(part)]
    if lead and not _OR_WORD.search(text):
        parts[-1:] = _AND_WORD.split(parts[-1])
    return parts


def _tidy(part: str) -> str:
    part = _LOOSE_JOINER.sub("", _BRACKET_LEAD.sub("(", part).strip(" ;:"))  # 'AWS (e.g., EC2, S3)' -> 'AWS (EC2, S3)'
    if not re.search(r"(?:^|[^A-Za-z])[A-Za-z]\.$", part):  # 'Ph.D.' keeps its dot, 'Node.js.' loses it
        part = part.rstrip(". ")  # a leading dot stays: '.NET'
    while part.endswith(")") and part.count(")") > part.count("("):  # "(like Copilot, Cursor)" leaves "Cursor)"
        part = part[:-1].rstrip()
    while part.startswith("(") and part.count("(") > part.count(")"):
        part = part[1:].lstrip()
    return part


def _examples_as_items(item: str) -> list[str]:
    """'AI tools like Copilot, Cursor, or Claude Code' -> ['Copilot', 'Cursor', 'Claude Code']; 'TypeScript or
    JavaScript' and 'Python, Go, and Rust' -> each one. A list in brackets stays with its keyword ('AWS (ECS, SQS)'),
    and so do names with commas in them: 'Extract, Transform, Load (ETL)', 'Microsoft Office: Word, Excel',
    'Java 8, 11, 17'."""
    masked, kept = _masked(item)
    lead = _EXAMPLES_LEAD.search(masked)
    if lead:
        head = masked[:lead.start()].rstrip()
        parts = _list_items(masked[lead.end():], lead=True)
        if head.endswith(","):  # 'SQL, including window functions': SQL is an item too; 'AI tools like ...' is not
            parts.insert(0, head[:-1])
    elif _spelled_out(item) or ":" in masked:
        parts = [masked]
    else:
        parts = _list_items(masked, lead=False)
    parts = [_tidy(_unmasked(part, kept)) for part in parts]
    if not lead and not all(re.search(r"[A-Za-z]", part) for part in parts if part):  # '11' is no item
        parts = [_tidy(item)]
    return [part for part in parts if _meaningful(part) and not _LIST_END.match(part)
            and not _EXAMPLES_LEAD.fullmatch(part)]


def _keyword_of(term: str) -> str:
    """The keyword inside a piece of the posting's prose: 'experience with Kafka' -> 'Kafka', 'advanced SQL' -> 'SQL',
    'AWS-hosted systems' -> 'AWS', 'Strong Python skills' -> 'Python'. A name stays as it is ('Basic Life Support
    (BLS)', 'Solid Edge', 'Deep Learning'); an 'ability to ...' has no keyword ('')."""
    if _ABILITY.search(term):
        return ""
    term = _PROSE_LEAD.sub("", term)
    if not _NAMED_TERMS.search(term):
        term = _HOSTED.sub(r"\1", term)
        if not _is_name(term):
            term = _TOOL_SKILLS.sub(r"\1", _QUALIFIER.sub("", term))
    return term.strip()


def is_keyword(term: str) -> bool:
    """A term as people list it on a resume ('SQL', 'AWS (ECS, SQS)', 'Basic Life Support (BLS)'), not a piece of the
    posting's prose: nothing around the keyword ('advanced', 'AWS-hosted', 'experience with') and at most four
    meaningful words, or eight for a proper name. An abbreviation in brackets does not count."""
    words = _meaningful(re.sub(r"\([^)]*\)", "", term))
    limit = MAX_NAME_WORDS if _is_name(term) else MAX_TERM_WORDS
    return 0 < len(words) <= limit and _keyword_of(term) == term.strip()


def clean_terms(items: list[str]) -> list[str]:
    """Job-description terms as keywords: a list of examples becomes the examples themselves, a piece of the posting's
    prose becomes the keyword in it, or is dropped when it has none, since the coverage rule would otherwise push it
    into a bullet word for word. Duplicates go."""
    terms: list[str] = []
    for item in items:
        for part in _examples_as_items(_DETAIL_BRACKET.sub("", clean_text(item))):
            term = _keyword_of(part)
            if is_keyword(term) and term.lower() not in {t.lower() for t in terms}:
                terms.append(term)
    return terms


def clean_analysis(analysis: JDAnalysis) -> JDAnalysis:
    """The analysis with keyword lists only (changed in place, and returned)."""
    analysis.must_have_skills = clean_terms(analysis.must_have_skills)
    analysis.nice_to_have_skills = clean_terms(analysis.nice_to_have_skills)
    analysis.keywords = clean_terms(analysis.keywords)
    return analysis


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
        cleaned = []
        for raw in group.skills:
            for item in _examples_as_items(clean_text(raw)):  # "AI tools like Copilot, Cursor" -> the tools
                if item and item.lower() not in seen:
                    seen.add(item.lower())
                    cleaned.append(item)
        if cleaned and clean_text(group.category):
            skills.append(SkillGroup(category=clean_text(group.category), skills=cleaned))
    if not skills:
        raise ValueError("`skills` must contain at least one group with skills.")

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


_INDEX_LABEL = re.compile(r"^\s*\[\d+\]\s*")
_FIELD_LABEL = re.compile(r"^\s*(?:headline|summary)\s*:\s*", re.IGNORECASE)
# An edit whose text is clearly another bullet of the same employer rewritten (it shares at least half its words,
# and clearly more than with the bullet it claims to replace) points at the wrong bullet_index.
SHIFT_MIN_OVERLAP, SHIFT_MARGIN = 0.5, 0.2


def _overlap(a: str, b: str) -> float:
    words_a = {w.lower() for w in _WORD.findall(a)}
    words_b = {w.lower() for w in _WORD.findall(b)}
    return len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0.0


def apply_revision(out: TailorOutput, draft: Resume, rev: TailorRevision) -> TailorOutput:
    """The first answer with a fix applied: `draft` is the resume made from `out` (cleaned, titles corrected) and
    exactly what the model was shown; `rev` lists only what changes, so everything else comes from `draft`. The
    analysis, the career plan and the number of bullets stay as they are. Raises ValueError for an edit that
    cannot be applied as meant, so the caller keeps the first draft. Neither input is changed."""
    bullets = {exp.company_index: list(exp.bullets) for exp in draft.experience}
    originals = {index: list(items) for index, items in bullets.items()}
    titles = {exp.company_index: exp.job_title for exp in draft.experience}
    for change in rev.titles:
        if change.company_index not in titles:
            raise ValueError(f"titles: there is no company_index {change.company_index}.")
        if change.job_title.strip():
            titles[change.company_index] = change.job_title
    seen: set[tuple[int, int]] = set()
    for change in rev.bullets:
        where = f"company_index {change.company_index}, bullet_index {change.bullet_index}"
        current = bullets.get(change.company_index)
        if current is None or not 0 <= change.bullet_index < len(current):
            raise ValueError(f"bullets: there is no bullet at {where}.")
        if (change.company_index, change.bullet_index) in seen:
            raise ValueError(f"bullets: {where} is changed twice.")
        seen.add((change.company_index, change.bullet_index))
        text = _INDEX_LABEL.sub("", change.text).strip()
        if not text:
            continue
        own = originals[change.company_index]
        target = _overlap(text, own[change.bullet_index])
        for other_index, other in enumerate(own):
            other_overlap = _overlap(text, other)
            if other_index != change.bullet_index and other_overlap >= SHIFT_MIN_OVERLAP \
                    and other_overlap >= target + SHIFT_MARGIN:
                raise ValueError(f"bullets: the new text for {where} rewrites bullet_index {other_index} instead.")
        current[change.bullet_index] = text
    skills = [SkillGroup(category=group.category, skills=list(group.skills)) for group in draft.skills]
    for change in rev.skill_changes:
        old, new = clean_text(change.old).lower(), clean_text(change.new)
        for group in skills:
            group.skills = [new if item.lower() == old else item for item in group.skills if item.lower() != old or new]
    sections = TailoredSections(
        headline=_FIELD_LABEL.sub("", rev.headline).strip() or draft.headline,
        summary=_FIELD_LABEL.sub("", rev.summary).strip() or draft.summary,
        skills=skills,
        experience=[ExperienceOut(company_index=i, job_title=titles[i], bullets=bullets[i]) for i in sorted(bullets)],
    )
    return out.model_copy(update={"sections": sections})


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
        parts.extend(group.skills)
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


def _word_pattern(word: str, last: bool) -> str:
    """One word of a term; the last one also matches its plural or singular ('API' ~ 'APIs', 'pipelines')."""
    if not last or not word.isalpha():  # 'Node.js', 'CI/CD' stay literal
        return re.escape(word)
    if word.isupper() or len(word) <= 3:  # acronyms: 'API' ~ 'APIs'
        return re.escape(word) + "s?"
    if word.lower().endswith("ss"):
        return re.escape(word) + "(?:es)?"
    if word.lower().endswith("s"):
        return re.escape(word[:-1]) + "(?:s|es)?"
    return re.escape(word) + "(?:s|es)?"


def _term_regex(term: str, case_sensitive: bool | None = None) -> re.Pattern | None:
    words = [w for w in re.split(r"[\s-]+", term.strip()) if w]
    if not words:
        return None
    body = r"[\s-]+".join(_word_pattern(w, i == len(words) - 1) for i, w in enumerate(words))
    if case_sensitive is None:
        case_sensitive = len(term.strip()) <= 3
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9+#])", 0 if case_sensitive else re.IGNORECASE)


def mentions(text: str, term: str, case_sensitive: bool | None = None) -> bool:
    """contains_term, tolerant of the forms an ATS also matches: hyphen or space between words
    ('event-driven' ~ 'event driven') and a plural or singular last word ('REST APIs' ~ 'REST API')."""
    pattern = _term_regex(term, case_sensitive)
    return pattern is not None and pattern.search(text) is not None


def mention_count(text: str, term: str) -> int:
    """How many times `mentions` would find the term."""
    pattern = _term_regex(term)
    return len(pattern.findall(text)) if pattern else 0


def _unique(terms: list[str]) -> list[str]:
    out: list[str] = []
    for term in terms:
        term = term.strip()
        if term and term.lower() not in {t.lower() for t in out}:
            out.append(term)
    return out


def evidence_text(resume: Resume) -> str:
    """Where a skill counts as shown, not just listed: the summary and the bullets."""
    return "\n".join([resume.summary, *(b for exp in resume.experience for b in exp.bullets)])


def unshown_must_haves(resume: Resume, analysis: JDAnalysis) -> tuple[list[str], list[str]]:
    """(absent, listed_only): must-have terms found nowhere in the resume, and ones that appear only in the
    skills list, headline or titles but in no summary sentence or bullet."""
    evidence, everything = evidence_text(resume), resume_text(resume)
    absent, listed_only = [], []
    for term in _unique(analysis.must_have_skills):
        if mentions(evidence, term):
            continue
        (listed_only if mentions(everything, term) else absent).append(term)
    return absent, listed_only


# ------------------------------------------------------------------ checks
def keyword_coverage(resume: Resume, analysis: JDAnalysis) -> Coverage:
    terms = _unique([*analysis.must_have_skills, *analysis.keywords])
    text = resume_text(resume)
    found = [t for t in terms if mentions(text, t)]
    missing = [t for t in terms if t not in found]
    _absent, listed_only = unshown_must_haves(resume, analysis)
    return Coverage(
        score=round(len(found) / len(terms), 3) if terms else 0.0, found=found, missing=missing,
        skills_only=listed_only,
    )


MIN_SUMMARY_WORDS = 55
MIN_BULLET_WORDS = 15
WEAK_OPENERS = (
    "responsible for", "worked on", "helped", "assisted", "participated in", "involved in", "utilized",
    "build and maintain", "built and maintained",
)


def thin_text_problems(resume: Resume) -> list[str]:
    """Signs of a weak draft: a short summary, two or more thin bullets, bullets that open like a duty list."""
    problems: list[str] = []
    words = word_count(resume.summary)
    if words < MIN_SUMMARY_WORDS:
        problems.append(f"the summary has only {words} words")
    bullets = [b for exp in resume.experience for b in exp.bullets]
    thin = [b for b in bullets if word_count(b) < MIN_BULLET_WORDS]
    if len(thin) >= 2:
        problems.append(
            f"{len(thin)} bullets are under {MIN_BULLET_WORDS} words (" + "; ".join(f"'{b}'" for b in thin[:3]) + ")"
        )
    weak = [b for b in bullets if b.lower().startswith(WEAK_OPENERS)]
    if weak:
        problems.append("these bullets open like a duty list: " + "; ".join(f"'{b}'" for b in weak[:3]))
    return problems


# A capitalized pronoun counts only where a sentence starts, and a lowercase one never inside a hyphenated word,
# so standard names stay fine: 'Know Your Customer (KYC)', 'Bring Your Own Device', 'pay-as-you-go'.
_POSTING_VOICE = re.compile(
    r"(?:^|(?<=[.!?]\s))(?:You|Your|Yours|Yourself|We|Our|Ours)\b(?!-)"
    r"|(?<![\w-])(?:you|your|yours|yourself|we|our|ours)(?![\w-])"
)


MIN_COPIED_WORDS = 4  # a run of four words shared with the posting is a copied sentence piece, not a keyword
MIN_COPIED_CONTENT = 3  # ... when at least three of them carry meaning
MIN_COPIED_PROSE = 2  # ... and at least two of those are not keywords: the JD's keywords are meant to be matched
MIN_COPIED_FOR_RETRY = 2  # one copied run is only noted; two or more are worth the single fix round


_WORD_PART = re.compile(r"[A-Za-z][\w+#]*")  # a hyphen splits: 'end-to-end' is matched as 'end to end'
_VOWEL = re.compile(r"[aeiouy]")


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    """(lowercase word, start, end) for each word, and each part of a hyphenated word."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_PART.finditer(text)]


def _stem(word: str) -> str:
    """One form for 'contribute', 'contributes', 'contributed' and 'contributing' (and 'identify', 'identified';
    'focus', 'focused'; 'need', 'needed'), so a copy is found in any tense."""
    if len(word) > 4 and word.endswith(("ies", "ied")):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "e", "s"):
        stem = word[:-len(suffix)]
        if word.endswith(suffix) and len(stem) >= 2 and _VOWEL.search(stem):
            if not (suffix == "s" and stem.endswith("u")) and not (suffix == "ed" and stem.endswith("e")):
                word = stem  # but 'focus' and 'need' are not 'focu' + s and 'ne' + ed
            break
    return word[:-1] if len(word) > 3 and word[-1] == word[-2] else word  # 'debugged' -> 'debugg' -> 'debug'


def copied_posting_spans(resume: Resume, jd: str, analysis: JDAnalysis) -> list[str]:
    """Pieces of the summary and the bullets copied word for word (in any tense) from the job description: runs of
    4+ words found in it, holding at least 3 meaningful words of which at least 2 are not keywords. Keywords (the
    JD's, the target role and the resume's own skills) may be used as they are, so a run made of them ('TypeScript,
    React and PostgreSQL', 'Senior Backend Engineer with') is fine. Overlapping runs are reported once."""
    posting = [_stem(word) for word, _start, _end in _word_spans(clean_text(jd))]
    n = MIN_COPIED_WORDS
    shared = {tuple(posting[i:i + n]) for i in range(len(posting) - n + 1)}
    terms = [*analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords,
             *(item for group in resume.skills for item in group.skills)]
    known = {_stem(word) for term in [*clean_terms(terms), analysis.target_role]
             for word, _start, _end in _word_spans(term)}
    places = [("summary", resume.summary)] + [
        (resume.profile.companies[exp.company_index].name, bullet) for exp in resume.experience for bullet in exp.bullets
    ]
    found: list[str] = []
    for where, text in places:
        words = _word_spans(text)
        stems = [_stem(word) for word, _start, _end in words]
        runs: list[list[int]] = []
        for i in range(len(words) - n + 1):
            if tuple(stems[i:i + n]) in shared:
                if runs and i <= runs[-1][1] + 1:  # windows that overlap or touch become one run
                    runs[-1][1] = i + n - 1
                else:
                    runs.append([i, i + n - 1])
        for first, last in runs:  # judged as a whole: 'Improve the patient experience across all touchpoints'
            content = {stems[j] for j in range(first, last + 1) if words[j][0] not in _SMALL_WORDS}
            if len(content) >= MIN_COPIED_CONTENT and len(content - known) >= MIN_COPIED_PROSE:
                found.append(f"'{text[words[first][1]:words[last][2]]}' ({where})")
    return found


def skill_phrase_problems(resume: Resume) -> list[str]:
    """Skills-list items written as a piece of the posting's prose ('advanced SQL statements', 'AWS-hosted
    systems') rather than as the keyword itself ('SQL', 'AWS'). Lists are already split into their items."""
    return [item for group in resume.skills for item in group.skills if not is_keyword(item)]


def posting_voice_problems(resume: Resume) -> list[str]:
    """Summary sentences and bullets that talk like the posting ("Refine your craft by..."): a resume never
    says you, your, we or our, so these were copied from the JD."""
    texts = [resume.summary, *(b for exp in resume.experience for b in exp.bullets)]
    return [text for text in texts if _POSTING_VOICE.search(text)]


MAX_PHRASE_REPEATS = 2
MIN_PHRASE_WORDS = 3  # 'modern software development', 'test-driven development'; not 'Spring Boot' or 'REST APIs'
_NAME_JOINERS = {"on", "of", "and", "for", "the", "&", "to", "in", "with"}


def _looks_like_name(term: str) -> bool:
    """'Azure Data Factory', 'Ruby on Rails', 'Microsoft Power BI': every main word capitalized, so a product or
    tool, which recurs wherever it was really used. The JD's spelling is kept, and descriptive phrases are lowercase."""
    words = [w for w in re.split(r"[\s-]+", term.strip()) if w and w.lower() not in _NAME_JOINERS]
    return bool(words) and all(w[:1].isupper() or w[:1].isdigit() for w in words)


def repeated_phrase_problems(resume: Resume, analysis: JDAnalysis) -> list[str]:
    """Descriptive JD phrases of three or more words used more than twice in the summary and bullets: keyword
    stuffing. Hyphen-joined parts count as words, as in matching ('test-driven development' is three)."""
    evidence = evidence_text(resume)
    terms = [t for t in _unique([*analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords])
             if len(re.split(r"[\s-]+", t.strip())) >= MIN_PHRASE_WORDS and not _looks_like_name(t)]
    problems = []
    for term in terms:
        count = mention_count(evidence, term)
        if count > MAX_PHRASE_REPEATS:
            problems.append(f"'{term}' ({count} times)")
    return problems


_QUALITY_WORDS = re.compile(
    r"\b(scalable(?!\s+vector\s+graphics)|maintainable|low[\s-]+maintenance|high[\s-]+quality"
    r"|robust(?!\s+(?:control|optimization|statistics|regression)\b)|reliable)\b",
    re.IGNORECASE,
)


def quality_skill_problems(resume: Resume) -> list[str]:
    """Skills-list items that are qualities or outcomes ('High-Quality Code', 'Scalable and Maintainable
    Applications') rather than something a person uses or does."""
    return [skill for group in resume.skills for skill in group.skills if _QUALITY_WORDS.search(skill)]


# Words that make a resume read as machine-written or as marketing copy (the prompt's PLAIN WORDS rule). Words
# with an everyday technical meaning stay allowed: orchestration (Kubernetes), dynamic (pricing), enhance.
_BUZZ = re.compile(
    r"\b(?:leverag(?:e|ed|es|ing)|utiliz(?:e|ed|es|ing|ation)|spearhead(?:ed|s|ing)?|seamless(?:ly)?|robust(?:ly|ness)?"
    r"|cutting[\s-]edge|state[\s-]of[\s-]the[\s-]art|synerg(?:y|ies|istic)|holistic(?:ally)?|innovative|transformative"
    r"|pivotal|paradigm|best[\s-]in[\s-]class|world[\s-]class|game[\s-]chang(?:er|ing)|delv(?:e|ed|es|ing)"
    r"|foster(?:ed|s|ing)?|empower(?:ed|s|ing|ment)?|meticulous(?:ly)?|plethora|myriad|in order to|a wide range of)\b",
    re.IGNORECASE,
)
_ING_CLAUSE = re.compile(r",\s+(?:and\s+)?(?!during\b)[a-z]{4,}ing\b")  # ", ensuring ...", ", and enabling ..."
MAX_ING_CLAUSES = 2  # a third in one bullet reads as a template
MIN_PLAIN_PROBLEMS_FOR_RETRY = 2  # one stray word is only noted; it is not worth a model call


def _plain_problems(places: list[tuple[str, str]]) -> list[str]:
    """Buzzwords in each (where, text), and texts that chain three or more '-ing' clauses."""
    problems: list[str] = []
    for where, text in places:
        problems += [f"'{match.group(0)}' ({where})" for match in _BUZZ.finditer(text)]
        if len(_ING_CLAUSE.findall(text)) > MAX_ING_CLAUSES:
            problems.append(f"a chain of '-ing' clauses ({where}: '{' '.join(text.split()[:8])} ...')")
    return problems


def plain_word_problems(resume: Resume) -> list[str]:
    """Machine-sounding wording in the summary and the bullets: buzzwords, and bullets that chain three or more
    '-ing' clauses. Skills are a list of tool names and are not read."""
    return _plain_problems([("summary", resume.summary)] + [
        (resume.profile.companies[exp.company_index].name, bullet) for exp in resume.experience for bullet in exp.bullets
    ])


def cover_plain_problems(letter: CoverLetter) -> list[str]:
    """The same check on the cover letter, sentence by sentence, so a long paragraph is not read as one '-ing'
    chain."""
    return _plain_problems([(f"paragraph {number}", sentence) for number, paragraph in enumerate(letter.paragraphs, 1)
                            for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence])


_WORD = re.compile(r"[A-Za-z][\w+#]*(?:-[\w+#]+)*")
_REAL_DOUBLES = {"had", "that"}  # 'what it had had', 'said that that': correct English
# 'a plug-in in Java', 'an add-on on Shopify', 'a follow-up up front': a hyphenated word ending in one of these
# may be followed by the same small word on purpose
_PARTICLES = {"in", "on", "up", "out", "off", "to", "of", "by", "at", "for", "with", "as", "over", "down", "back"}


def _repeats(text: str) -> list[tuple[int, int, str]]:
    """Each word that repeats the word just before it, with only a space between: 'the the', and a hyphenated
    word followed by its own last part, 'multi-tenant tenant'. (start, end) of the repeat, and the two words."""
    found = []
    tokens = list(re.finditer(r"\S+", text))
    for before, after in pairwise(tokens):
        first, second = before.group(0), after.group(0).rstrip(".,;:!?)")
        if not (_WORD.fullmatch(first) and _WORD.fullmatch(second)) or text[before.end():after.start()] != " ":
            continue  # 'Java, Java' or a line break: not the same slip
        word = second.lower()
        same = word == first.lower() and word not in _REAL_DOUBLES
        tail = "-" in first and word == first.split("-")[-1].lower() and word not in _PARTICLES
        if same or tail:
            found.append((before.end(), after.start() + len(second), f"{first} {second}"))
    return found


def drop_repeated_words(text: str) -> str:
    """'multi-tenant tenant data schemas' -> 'multi-tenant data schemas'."""
    for start, end, _pair in reversed(_repeats(text)):
        text = text[:start] + text[end:]
    return text


def _resume_texts(resume: Resume) -> list[str]:
    return [resume.headline, resume.summary, *(b for exp in resume.experience for b in exp.bullets)]


def repeated_word_problems(resume: Resume) -> list[str]:
    """Words the model repeated by mistake in the headline, summary and bullets ('multi-tenant tenant')."""
    return [pair for text in _resume_texts(resume) for _s, _e, pair in _repeats(text)]


def fix_repeated_words(resume: Resume) -> list[str]:
    """Remove the repeats that are still there after the rewrite pass; returns what was removed."""
    fixed = repeated_word_problems(resume)
    resume.headline, resume.summary = drop_repeated_words(resume.headline), drop_repeated_words(resume.summary)
    for exp in resume.experience:
        exp.bullets = [drop_repeated_words(b) for b in exp.bullets]
    return fixed


_LEVEL_WORDS = {
    "intern": -1, "trainee": -1, "junior": 0, "jr": 0, "associate": 0, "graduate": 0,
    "senior": 3, "sr": 3, "staff": 4, "lead": 4, "principal": 5,
}
_LEVEL_NUMBERS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}


def title_level(title: str) -> tuple[str, int]:
    """('software engineer', 2) for 'Software Engineer II': the role without its level words, and the level
    (intern -1, junior/associate 0, plain 1, II 2, III or senior 3, staff or lead 4, principal 5)."""
    words = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", title.lower().replace(".", ""))
    level, base = None, []
    for index, word in enumerate(words):
        if word in _LEVEL_WORDS and level is None:
            level = _LEVEL_WORDS[word]
        elif index == len(words) - 1 and index > 0 and word in _LEVEL_NUMBERS:
            level = max(level or 0, _LEVEL_NUMBERS[word])
        else:
            base.append(word)
    return " ".join(base), (1 if level is None else level)


_ROLE_LEVEL = re.compile(r"^(?P<role>.*?\S)\s+(?P<level>I{1,3}|IV|V|[1-5]|L[1-9]|Level\s+[1-9])$", re.IGNORECASE)


def role_level_fixes(resume: Resume, analysis: JDAnalysis) -> list[str]:
    """The headline and summary name the posting's role without the hiring company's level number (II, L5) when
    the candidate's latest title does not have that level: "Senior Software Engineer II with 7+ years" claims a
    level the candidate never held. Changes the resume in place and returns a note per changed field."""
    match = _ROLE_LEVEL.match(analysis.target_role.strip())
    if not match or not resume.experience:
        return []
    role, level = match.group("role"), match.group("level")
    level_pattern = r"\s+".join(map(re.escape, level.split()))
    if re.search(rf"(?<![\w-]){level_pattern}$", resume.experience[0].job_title.strip(), re.IGNORECASE):
        return []  # the candidate's latest title really has this level
    role_pattern = r"\s+".join(map(re.escape, role.split()))
    phrase = re.compile(rf"(?<![\w-])({role_pattern})\s+{level_pattern}(?![\w-])", re.IGNORECASE)
    notes = []
    for field in ("headline", "summary"):
        text = getattr(resume, field)
        fixed = phrase.sub(lambda m: m.group(1), text)
        if fixed != text:
            setattr(resume, field, fixed)
            notes.append(f"{field.title()}: wrote '{role}' instead of the posting's level '{analysis.target_role}', "
                         "which is not your title.")
    return notes


def _steps_down(newer: str, older: str) -> bool:
    """A newer title below the older one: same role at a lower level, or a different role 2+ levels lower."""
    (new_base, new_level), (old_base, old_level) = title_level(newer), title_level(older)
    return new_level < old_level and (new_base == old_base or old_level - new_level >= 2)


def fix_title_regressions(resume: Resume) -> list[str]:
    """Make sure no job title steps down from one employer to the next (for example 'Software Engineer II'
    then 'Software Engineer'). Changes the resume in place and returns a note per change.

    Same role at a lower level: the newer title is raised to the older one, unless the newer job is the most
    recent (its title matches the target role) or raising it would step above the job after it; then the older
    title is lowered instead. Different roles are only reported by title_growth_problems."""
    exps, notes = resume.experience, []  # most recent first
    for i in range(1, len(exps)):
        newer, older = exps[i - 1], exps[i]
        (new_base, new_level), (old_base, old_level) = title_level(newer.job_title), title_level(older.job_title)
        if new_base != old_base or new_level >= old_level:
            continue
        above = title_level(exps[i - 2].job_title) if i >= 2 else None
        can_raise = i - 1 > 0 and above is not None and (above[0] != old_base or above[1] >= old_level)
        changed, before = (newer, newer.job_title) if can_raise else (older, older.job_title)
        changed.job_title = older.job_title if can_raise else newer.job_title
        company = resume.profile.companies[changed.company_index].name
        notes.append(f"{company}: title changed from '{before}' to '{changed.job_title}' so the career path never "
                     "steps down.")
    return notes


# When each tool became COMMON at work (not its first release): a job that ended before that year should not
# mention it. Lenient on purpose: compared with the job's END year. Only tools a recent career can get wrong.
COMMON_YEAR: dict[str, int] = {
    "Kubernetes": 2017, "Helm": 2018, "Amazon EKS": 2019, "EKS": 2019, "Istio": 2019, "Next.js": 2019,
    "GitHub Actions": 2020, "dbt": 2020, "Argo CD": 2021, "Tailwind CSS": 2021, "FastAPI": 2021, "Delta Lake": 2021,
    "Azure Synapse": 2021, "OpenTelemetry": 2022, "Vite": 2022, "Astro": 2023, "Polars": 2023, "OpenTofu": 2024,
    "Bun": 2024, "Microsoft Fabric": 2024,
}
# AI tools, with the year they became popular at work. They also belong only to the most recent AI_RECENT_JOBS jobs.
AI_TOOLS: dict[str, int] = {
    "ChatGPT": 2023, "LLM": 2023, "large language model": 2023, "GPT-4": 2023, "LangChain": 2023,
    "prompt engineering": 2023, "vector database": 2023, "retrieval-augmented generation": 2023,
    "AI coding assistant": 2023, "Copilot": 2022, "agentic": 2025, "coding agent": 2025,
}
AI_RECENT_JOBS = 2
_EVERYDAY_WORDS = {"Helm", "Astro", "Vite", "Copilot", "Bun", "Polars"}  # matched only as written: "at the helm" is fine
_CURRENT = ("", "present", "current", "now", "today")


def _mentions_tool(text: str, term: str) -> bool:
    return mentions(text, term, case_sensitive=term in _EVERYDAY_WORDS or None)


def era_problems(resume: Resume) -> list[str]:
    """Tools a job could not really have used: an AI tool outside the most recent one or two jobs, or any tool in a
    job that ended before it became common (e.g. OpenTofu at a job that ended in 2021). The current job, and jobs
    whose end date cannot be read, are never flagged for dates."""
    problems: list[str] = []
    for index, exp in enumerate(resume.experience):  # most recent first
        company = resume.profile.companies[exp.company_index]
        text = "\n".join([exp.job_title, *exp.bullets])
        when = f"{company.name} ({company.start or '?'} - {company.end or 'Present'})"
        if index >= AI_RECENT_JOBS:
            for term in AI_TOOLS:
                if _mentions_tool(text, term):
                    problems.append(f"{when} mentions {term}: AI tools belong only to the most recent one or two jobs.")
        end = parse_point(company.end, is_end=True)
        if end is None or company.end.strip().lower() in _CURRENT:
            continue
        for term, year in {**COMMON_YEAR, **(AI_TOOLS if index < AI_RECENT_JOBS else {})}.items():
            if year > end[0] and _mentions_tool(text, term):
                problems.append(f"{when} mentions {term}, which only became common in {year}.")
    return problems


def title_growth_problems(resume: Resume) -> list[str]:
    """At most one warning about the career path: a title that steps down from one job to the next, the same
    title everywhere, the same title at 3+ employers in a row, or a stay of 2+ years with the same title as the
    job before it."""
    exps = resume.experience  # most recent first
    titles = [e.job_title.strip().lower() for e in exps]
    for i in range(len(exps) - 1):
        if _steps_down(exps[i].job_title, exps[i + 1].job_title):
            company = resume.profile.companies[exps[i].company_index].name
            return [(f"{company}: '{exps[i].job_title}' comes after '{exps[i + 1].job_title}', which reads as a step "
                     "down. Use the same level or higher than the job before it.")]
    if len(titles) > 1 and len(set(titles)) == 1:
        return ["The same job title is used at every employer, so there is no visible career progression."]
    for i in range(2, len(titles)):
        if titles[i] == titles[i - 1] == titles[i - 2]:
            return [(f"The title '{exps[i].job_title}' is used at 3 employers in a row, so there is no visible "
                     "career progression.")]
    problems = []
    for i in range(len(exps) - 1):  # the oldest job has nothing before it to grow from
        if titles[i] != titles[i + 1]:
            continue
        company = resume.profile.companies[exps[i].company_index]
        months = stay_months(company)
        if months is not None and months >= 24:
            problems.append(
                f"{company.name}: {months_phrase(months)} with the same title as the job before it. Show growth, "
                "for example a level step ('Software Engineer II') or the specialty title."
            )
    return problems[:1]


def ats_lint(resume: Resume) -> list[str]:
    warnings: list[str] = []
    if word_count(resume.headline) > 14:
        warnings.append(f"Headline is long ({word_count(resume.headline)} words).")
    words = word_count(resume.summary)
    if words > 95:
        warnings.append(f"Summary is long ({words} words; aim for 60-85).")
    elif words < MIN_SUMMARY_WORDS:
        warnings.append(f"Summary is short ({words} words; aim for 60-85).")
    if _FIRST_PERSON.search(resume.summary):
        warnings.append("Summary uses first-person pronouns.")
    total_skills = sum(len(g.skills) for g in resume.skills)
    if not 12 <= total_skills <= 40:
        warnings.append(f"{total_skills} skills listed (aim for roughly 24-32).")
    warnings += title_growth_problems(resume)
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index].name
        if not 2 <= len(exp.bullets) <= 8:
            warnings.append(f"{company}: {len(exp.bullets)} bullets (aim for 3-8, the most in recent roles).")
        openers = [b.split()[0].lower() for b in exp.bullets if b.split()]
        if len(openers) != len(set(openers)):
            warnings.append(f"{company}: two bullets start with the same word.")
        for bullet in exp.bullets:
            if word_count(bullet) > 36:
                warnings.append(f"{company}: a bullet is {word_count(bullet)} words long.")
                break
            if word_count(bullet) < MIN_BULLET_WORDS:
                warnings.append(
                    f"{company}: a bullet has only {word_count(bullet)} words; say how it was done and what changed."
                )
                break
            if _FIRST_PERSON.search(bullet):
                warnings.append(f"{company}: a bullet uses first-person pronouns.")
                break
    return warnings


def _numbers_in(text: str) -> list[str]:
    return [m.group(0).strip(" ,.") for m in _NUMBER.finditer(text)]


_FIGURE_CORE = re.compile(r"[$€£]?(\d[\d,]*)(\.\d+)?")
_APPROX = re.compile(r"~\s?[$€£]?\d[\d,.]*\s?%?|\b(?:approximately|approx\.)\s+[$€£]?\d[\d,.]*\s?%?", re.IGNORECASE)


def is_round_figure(token: str) -> bool:
    """25, 30%, 40+, $120K: whole numbers of 10 or more ending in 0 or 5. Years (1900-2100) are not figures."""
    match = _FIGURE_CORE.match(token)
    if not match or match.group(2):
        return False
    value = int(match.group(1).replace(",", "") or 0)
    if 1900 <= value <= 2100 and token.isdigit():
        return False
    return value >= 10 and value % 5 == 0


def figure_style_problems(resume: Resume, allowed_text: str) -> list[str]:
    """Estimated figures that look like round guesses or approximations ("35%", "40+", "~30%").

    Figures that appear in `allowed_text` (the user's hints and the computed facts) are never reported.
    """
    allowed = allowed_text.lower()
    problems: list[str] = []
    texts = [resume.summary] + [bullet for exp in resume.experience for bullet in exp.bullets]
    for text in texts:
        for match in _APPROX.finditer(text):
            problems.append(match.group(0).strip())
        for token in _numbers_in(text):
            if token and token.lower() not in allowed and is_round_figure(token):
                problems.append(token)
    return list(dict.fromkeys(problems))


MAX_FIGURE_BULLETS_PER_JOB = 2
MAX_FIGURE_SHARE = 0.5


def has_figure(text: str) -> bool:
    """True if the text carries a number other than a plain year."""
    for token in _numbers_in(text):
        if token.isdigit() and 1900 <= int(token) <= 2100:
            continue
        if token:
            return True
    return False


def metric_density_problems(resume: Resume) -> list[str]:
    """Employers with more than 2 bullets carrying figures, and an overall share above half."""
    problems: list[str] = []
    total = with_figure = 0
    for exp in resume.experience:
        count = sum(has_figure(b) for b in exp.bullets)
        total += len(exp.bullets)
        with_figure += count
        if count > MAX_FIGURE_BULLETS_PER_JOB:
            company = resume.profile.companies[exp.company_index].name
            problems.append(f"{company}: {count} of {len(exp.bullets)} bullets have figures")
    if total and with_figure / total > MAX_FIGURE_SHARE:
        problems.append(f"{with_figure} of {total} bullets overall have figures")
    return problems


MIN_FIRST_BULLETS_FOR_RETRY = 2  # two jobs that open without a figure (or the most recent one) are worth the fix round


def first_bullet_problems(resume: Resume) -> list[str]:
    """Jobs whose first bullet carries no figure, most recent first. A recruiter's six-second scan reads only that
    bullet, so it should show the job's biggest win with its size and its result."""
    return [resume.profile.companies[exp.company_index].name for exp in resume.experience
            if exp.bullets and not has_figure(exp.bullets[0])]


# What a draft adds after the role: a domain, areas of work or skills ("AI Engineer | AI Platforms", "Full-Stack
# Engineer - Payments")
_AFTER_ROLE = re.compile(r"\s*\|.*$|\s+-\s+.*$")


# Level words before a role ("Senior Data Engineer"); "Tech Lead" and a bare "Lead" stay
_LEADING_LEVEL = re.compile(
    r"^(?:(?:senior|sr|staff|lead|principal|junior|jr|associate|graduate|mid-level|entry-level)\.?\s+)+(?=\S)",
    re.IGNORECASE,
)


def _without_level(role: str) -> str:
    """'Senior Data Engineer' -> 'Data Engineer', 'AI Engineer II' -> 'AI Engineer'."""
    role = _LEADING_LEVEL.sub("", role.strip())
    match = _ROLE_LEVEL.match(role)
    return match.group("role") if match else role


def short_role_headline(resume: Resume) -> None:
    """The headline is the role's short, common name, never the posting's full title: what a draft adds after a
    '|' or a spaced dash goes, and so do brackets, the part after a comma and the level ('Staff Data Engineer,
    Analytics Infrastructure' -> 'Data Engineer'). A bare role noun takes its function from after the comma
    ('Manager, Customer Success' -> 'Customer Success Manager'). Changes the resume in place."""
    text = re.sub(r"\s*\([^)]*\)", "", _AFTER_ROLE.sub("", resume.headline))
    role, _comma, rest = text.partition(",")
    role, rest = _without_level(role), _without_level(rest)
    if len(role.split()) == 1 and rest:
        role = f"{rest} {role}"
    if role.strip():
        resume.headline = role.strip()


def _quoted(items: list[str]) -> str:
    """'a'; 'a' and 'b'; 'a', 'b' and 'c'."""
    quoted = [f"'{item}'" for item in items]
    return " and ".join(quoted) if len(quoted) <= 2 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def claim_flags(resume: Resume, jd: str, hints: str) -> list[Flag]:
    """Things the user did not provide and should confirm before sending the resume."""
    flags: list[Flag] = []
    hints_l, jd_l = hints.lower(), jd.lower()

    def check_numbers(section: str, text: str) -> None:
        """One flag naming every estimate in the text (a job's first bullet gives its size and its result)."""
        tokens = list(dict.fromkeys(t for t in _numbers_in(text) if t and t.lower() not in hints_l))
        if not tokens:
            return
        if len(tokens) == 1:
            reason = f"'{tokens[0]}' is an estimate, not a number you gave. Replace it with your real figure or remove it."
        else:
            reason = (f"{_quoted(tokens)} are estimates, not numbers you gave. Replace them with your real figures or "
                      "remove them.")
        flags.append(Flag(section=section, text=text, level="verify", reason=reason))

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
        for item in group.skills:
            if item.lower() not in jd_l and item.lower() not in hints_l:
                flags.append(
                    Flag(section="skills", text=item, level="info",
                         reason="Related skill added (not in the job description). Keep it only if you have it.")
                )
    return flags


MAX_POSTING_FLAGS = 10


def posting_match_flags(resume: Resume, analysis: JDAnalysis, hints: str) -> list[Flag]:
    """One 'info' flag per bullet or summary sentence that carries must-have terms the user's hints do not back
    up: the app writes those in to match the posting, so the user should confirm them. The first place each term
    appears is flagged, most recent employer first, then the summary."""
    places: list[tuple[str, str]] = [
        (f"experience:{exp.company_index}", bullet) for exp in resume.experience for bullet in exp.bullets
    ]
    places += [("summary", s) for s in re.split(r"(?<=[.!?])\s+", resume.summary) if s.strip()]
    grouped: dict[tuple[str, str], list[str]] = {}
    for term in _unique(analysis.must_have_skills):
        if mentions(hints, term):
            continue
        for place in places:
            if mentions(place[1], term):
                grouped.setdefault(place, []).append(term)
                break
    flags = [
        Flag(section=section, text=text, level="info",
             reason="Added to match the posting (" + ", ".join(terms) + "). Confirm you can speak to it in an "
                    "interview, or remove it.")
        for (section, text), terms in grouped.items()
    ]
    return flags[:MAX_POSTING_FLAGS]


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
    paras = [p for p in (drop_repeated_words(clean_text(x)) for x in paragraphs) if p]
    if not 2 <= len(paras) <= 6:
        raise ValueError("The cover letter needs between 2 and 6 non-empty paragraphs.")
    greeting, closing = clean_text(greeting), clean_text(closing)
    if not greeting or not closing:
        raise ValueError("`greeting` and `closing` must not be empty.")
    return greeting, paras, closing
