"""The reviewer's rules for a one-page resume (prompt v38) that code keeps, on every draft, for free:

  * official_names: a tool written as its maker writes it, one way everywhere (PostgreSQL, pytest, GitHub, Node.js);
    the form the posting's analysis lists wins, else the posting's own form when it uses only another (an ATS matches
    it letter for letter, in any case: letter case alone never wins)
  * fix_numbers: numbers as numerals and every figure written one way (numerals.py)
  * summary_trim: a summary sentence of self-praise ("Brings sound tradeoff judgment ..."), or a 4th sentence, goes
    unless it holds a must-have's only mention
  * shape_skills: at most 3 rows, the posting's must-haves first; an item no summary or bullet shows goes, unless the
    posting asks for it (job sites scan the skills list)

and the checks that ask for the rewrite pass, or warn the admin, when the model broke one: weak opening verbs ("Used",
"Supported", "Run"), bullets of 3 lines, a summary of 4 lines, self-praise, a summary that lists the skills, more than
one "so ..." ending in a job, a current job with too few figures.
"""
from __future__ import annotations

import re

from ..schemas import FitStyle, JDAnalysis, Profile, Resume
from . import tense
from .facts import is_current
from .numerals import normal
from .page_fit import design_style
from .validate import (
    _sentence_pieces,
    evidence_text,
    has_figure,
    mentions,
    note_figures,
    word_count,
)

# ---------------------------------------------------------------------- official names
# Each tool's official name and the forms models write instead. Only forms that can mean nothing else: 'Vault' or
# 'calendar' could be anything, so those are left to the prompt.
OFFICIAL_NAMES: dict[str, tuple[str, ...]] = {
    "PostgreSQL": ("Postgres", "Postgresql", "PostGres", "PostgreSql", "postgres", "postgresql"),
    "pytest": ("Pytest", "PyTest", "py.test"),
    "GitHub": ("Github", "github"),
    "GitHub Actions": ("Github Actions", "Github actions", "GitHub actions", "github Actions", "github actions"),
    "GitLab": ("Gitlab", "gitlab"),
    "JavaScript": ("Javascript", "javascript"),
    "TypeScript": ("Typescript", "typescript"),
    "Node.js": ("NodeJS", "NodeJs", "Nodejs", "nodejs", "Node.JS", "node.js"),
    "Next.js": ("NextJS", "NextJs", "Nextjs", "nextjs"),
    "HashiCorp": ("Hashicorp", "hashicorp"),
    "MongoDB": ("Mongodb", "MongoDb", "mongodb"),
    "MySQL": ("Mysql", "MySql", "mysql"),
    "GraphQL": ("Graphql", "GraphQl", "graphql"),
    "OpenAI": ("Openai", "OpenAi", "openai"),
    "LangChain": ("Langchain", "langchain"),
    "FastAPI": ("FastApi", "Fastapi", "fastapi"),
    "PyTorch": ("Pytorch", "pytorch"),
    "TensorFlow": ("Tensorflow", "tensorflow"),
    "DynamoDB": ("DynamoDb", "Dynamodb", "dynamodb"),
    "OpenTelemetry": ("Opentelemetry", "opentelemetry"),
    "OAuth": ("Oauth", "oauth"),
    "Elasticsearch": ("ElasticSearch", "elasticsearch"),
    "Datadog": ("DataDog", "datadog"),
    "BigQuery": ("Bigquery", "bigquery"),
    "NumPy": ("Numpy", "numpy"),
    "scikit-learn": ("Scikit-learn", "Scikit-Learn", "sklearn"),
    "macOS": ("MacOS", "MacOs", "macos"),
    "Salesforce": ("SalesForce", "salesforce"),
    "WordPress": ("Wordpress", "wordpress"),
    "Power BI": ("PowerBI", "PowerBi", "Powerbi"),
}


_TOOL = {form: canonical for canonical, aliases in OFFICIAL_NAMES.items() for form in (canonical, *aliases)}
# a form as its tool writes it: the first of the tool's forms that differs from it only by letter case, the official
# name first ('github' is GitHub, 'postgres' is Postgres), so letter case alone never makes a form win
_SPELLING = {form: next(f for f in (canonical, *OFFICIAL_NAMES[canonical]) if f.lower() == form.lower())
             for form, canonical in _TOOL.items()}
# any form, on its own (never inside a word, a path, a link or an address: 'github.com', 'postgres://'), the longest
# first, so 'GitHub Actions' is read as one name and never as 'GitHub'
_ANY_FORM = re.compile(r"(?<![\w./@#-])(?:" + "|".join(map(re.escape, sorted(_TOOL, key=len, reverse=True)))
                       + r")(?![\w/@#-]|\.\w|://)")


def _terms(analysis: JDAnalysis | None) -> list[str]:
    if analysis is None:
        return []
    return [*analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords]


def _spellings_in(texts: list[str]) -> dict[str, list[str]]:
    """For each official name the texts use, the spellings of its forms they use, in order."""
    used: dict[str, list[str]] = {}
    for text in texts:
        for found in _ANY_FORM.finditer(text):
            spellings = used.setdefault(_TOOL[found.group(0)], [])
            if _SPELLING[found.group(0)] not in spellings:
                spellings.append(_SPELLING[found.group(0)])
    return used


def name_targets(analysis: JDAnalysis | None, posting: str = "") -> dict[str, str]:
    """For each official name, the form to write: the form the analysis lists (a must-have's first: coverage looks for
    it word for word), else the official one, unless the posting's text uses only another form, which then wins.
    Letter case never decides (an ATS matches a name in any case), and a form is written in its own case."""
    listed, used = _spellings_in(_terms(analysis)), _spellings_in([posting])
    targets = {}
    for canonical in OFFICIAL_NAMES:
        if canonical in listed:
            targets[canonical] = listed[canonical][0]
        elif canonical in used and canonical not in used[canonical]:
            targets[canonical] = used[canonical][0]
        else:
            targets[canonical] = canonical
    return targets


def _writing(analysis: JDAnalysis | None, posting: str) -> dict[str, str]:
    """How each form is written (name_targets); a form the analysis lists is never written as another one."""
    targets = name_targets(analysis, posting)
    listed = {spelling for spellings in _spellings_in(_terms(analysis)).values() for spelling in spellings}
    return {form: _SPELLING[form] if _SPELLING[form] in listed else targets[canonical]
            for form, canonical in _TOOL.items()}


def _named(text: str, writing: dict[str, str]) -> str:
    return _ANY_FORM.sub(lambda found: writing[found.group(0)], text)


def official_names(resume: Resume, analysis: JDAnalysis | None = None, posting: str = "") -> list[str]:
    """Every tool name in the headline, summary, skills, titles and bullets as name_targets says (PostgreSQL, not
    Postgres; pytest, not Pytest), one way everywhere; the skills are de-duplicated after. Never the fixed facts.
    Returns no notes: it changes how a name is written, never what is said."""
    writing = _writing(analysis, posting)
    resume.headline, resume.summary = _named(resume.headline, writing), _named(resume.summary, writing)
    for exp in resume.experience:
        exp.job_title = _named(exp.job_title, writing)
        exp.bullets = [_named(bullet, writing) for bullet in exp.bullets]
    seen: set[str] = set()
    for group in resume.skills:
        items = []
        for item in (_named(item, writing) for item in group.skills):
            if item.lower() not in seen:
                seen.add(item.lower())
                items.append(item)
        group.skills = items
    resume.skills = [group for group in resume.skills if group.skills] or resume.skills
    return []


# ---------------------------------------------------------------------- numbers
def fix_numbers(resume: Resume, analysis: JDAnalysis | None = None) -> list[str]:
    """Numbers as numerals and figures one way (numerals.normal) in the headline, summary and bullets; a posting term
    or a skills item that holds a number word keeps it ('five whys'; 'two-phase commit' is a compound anyway). Returns
    no notes: how a number is written, never what it says."""
    keep = (_terms(analysis) + [company.name for company in resume.profile.companies]
            + [item for group in resume.skills for item in group.skills])
    resume.headline, resume.summary = normal(resume.headline, keep), normal(resume.summary, keep)
    for exp in resume.experience:
        exp.bullets = [normal(bullet, keep) for bullet in exp.bullets]
    return []


def letter_fixes(paragraphs: list[str], analysis: JDAnalysis | None = None, posting: str = "",
                 companies: list[str] | None = None, skills: list[str] | None = None) -> list[str]:
    """A cover letter's paragraphs with the resume's official names and numbers (official_names, fix_numbers), so the
    two documents write them the same way; `skills`: the resume's skills items."""
    writing = _writing(analysis, posting)
    keep = _terms(analysis) + list(companies or []) + list(skills or [])
    return [normal(_named(paragraph, writing), keep) for paragraph in paragraphs]


def number_words(resume: Resume, analysis: JDAnalysis | None = None) -> list[str]:
    """Texts that still write a number as a word or a figure another way (a hand edit): for the admin's warnings. What
    fix_numbers keeps on purpose is never listed: a posting term ('three lines of defense', from `analysis`), an
    employer's name or a skills item ('five whys')."""
    keep = (_terms(analysis) + [company.name for company in resume.profile.companies]
            + [item for group in resume.skills for item in group.skills])
    texts = [resume.summary, *(bullet for exp in resume.experience for bullet in exp.bullets)]
    return [text for text in texts if normal(text, keep) != text]


# ---------------------------------------------------------------------- the summary
SELF_PRAISE = re.compile(  # 'Brings 8 years of ... experience', 'Recognized with the 2025 ... Award': facts, no praise
    r"^\s*(?:brings|bringing)\b(?!\s+(?:[\w-]+\s+){0,2}(?:\d|experience\b|decades?\b))"
    r"|^\s*(?:recognized|recognised)\b(?!\s+(?:with|by)\b|\s+(?:[\w-]+\s+){0,2}\d)"
    r"|^\s*(?:known|passionate|dedicated|proven)\b"
    r"|\b(?:sound|strong|excellent|exceptional|outstanding|proven|solid|keen|superb)\s+(?:[\w-]+\s+){0,2}"
    r"(?:judgment|judgement|communication|communicator|leadership|track\s+record|abilit(?:y|ies)|skills|work\s+ethic"
    r"|eye\s+for|instincts?)\b"
    r"|\b(?:results[- ]driven|detail[- ]oriented|team\s+player|self[- ]starter|go[- ]getter|highly\s+motivated"
    r"|hard[- ]working|passionate\s+about|track\s+record\s+of)\b",
    re.IGNORECASE)
MAX_SUMMARY_SENTENCES = 3
# 'U.S.', 'e.g.', 'vs.', 'etc.', 'Mt.', 'Dr.', 'Inc.', 'Jr.', 'No.', the 'T.' of a name: a full stop that ends no
# summary sentence, whatever follows it
_ABBREVIATION = re.compile(
    r"(?<![\w.])(?:(?:[A-Za-z]\.){2,}|[A-Z]\.|(?:vs|etc|Mt|St|Dr|Mr|Mrs|Ms|Inc|Ltd|Co|Corp|Jr|Sr|No)\.)$")


def _sentence_parts(text: str) -> list[list[str]]:
    """The text's sentences, each as its pieces and the spaces between them (validate._sentence_pieces): a piece after
    an abbreviation or an initial ('U.S. Army', 'e.g. React', 'T. Quinn') belongs to the same sentence."""
    pieces = _sentence_pieces(text)
    parts: list[list[str]] = []
    for index in range(0, len(pieces), 2):
        if parts and _ABBREVIATION.search(parts[-1][-1]):
            parts[-1] += [pieces[index - 1], pieces[index]]
        elif pieces[index].strip():
            parts.append([pieces[index]])
    return parts


def _sentences(text: str) -> list[str]:
    return ["".join(parts) for parts in _sentence_parts(text)]


def self_praise(summary: str) -> list[str]:
    """The summary's sentences that praise the candidate or name qualities instead of showing them."""
    return [sentence for sentence in _sentences(summary) if SELF_PRAISE.search(sentence)]


def summary_trim(resume: Resume, analysis: JDAnalysis | None = None) -> list[str]:
    """A summary of at most 3 sentences without self-praise: a praising sentence ('Brings strong judgment to every
    project.'), then any sentence after the 3rd, is removed whole unless it holds a must-have's only mention in the
    summary and the bullets; at least 2 sentences always stay. An abbreviation or an initial ends no sentence ('U.S.
    Army'), and praise after one stays for the rewrite pass: whether a new sentence began there is unknown. Returns a
    note when one went."""
    parts = _sentence_parts(resume.summary)
    sentences = ["".join(pieces) for pieces in parts]
    must = list(analysis.must_have_skills) if analysis else []
    bullets = "\n".join(bullet for exp in resume.experience for bullet in exp.bullets)

    def needed(index: int, kept: list[int]) -> bool:
        rest = " ".join(sentences[i] for i in kept if i != index) + "\n" + bullets
        return any(mentions(sentences[index], term) and not mentions(rest, term) for term in must)

    kept = list(range(len(sentences)))
    removed = []
    for index in [i for i in kept if SELF_PRAISE.search(parts[i][0])]:  # in its first piece: a sentence for sure
        if len(kept) > 2 and not needed(index, kept):
            kept.remove(index)
            removed.append(sentences[index])
    for index in reversed(kept[MAX_SUMMARY_SENTENCES:]):
        if len(kept) > MAX_SUMMARY_SENTENCES and not needed(index, kept):
            kept.remove(index)
            removed.append(sentences[index])
    if not removed:
        return []
    resume.summary = " ".join(sentences[i].strip() for i in kept)
    return ["Summary: removed " + "; ".join(f"'{s.strip()[:80]}'" for s in removed)
            + " (self-praise, or more than 3 sentences)."]


def summary_skill_items(resume: Resume) -> list[str]:
    """The skills items the summary names: more than 3 reads as the skills list again."""
    return [item for group in resume.skills for item in group.skills if mentions(resume.summary, item)]


MAX_SUMMARY_SKILLS = 3


# ---------------------------------------------------------------------- skills
MAX_SKILL_ROWS = 3


def _rank(item: str, analysis: JDAnalysis | None) -> int:
    """0 a must-have, 1 a nice-to-have, 2 one of the posting's keywords, 3 anything else."""
    if analysis is None:
        return 3
    for rank, terms in enumerate((analysis.must_have_skills, analysis.nice_to_have_skills, analysis.keywords)):
        if any(item.lower() == term.lower() or mentions(item, term) or mentions(term, item) for term in terms):
            return rank
    return 3


MAX_SKILL_ITEMS = 18


def _shown(evidence: str, item: str) -> bool:
    """The summary or a bullet shows the skills item: as `mentions` finds it, or with a version right after it ('C++17',
    'C#11', 'Python3'), after a dotted prefix ('ASP.NET' shows '.NET'), or as the noun of an '-ing' method, plural
    too ('unit tests' shows 'Unit Testing', 'A/B tests' 'A/B Testing')."""
    name = item.strip()
    if not name:
        return False
    if mentions(evidence, name):
        return True
    flags = 0 if len(name) <= 3 else re.IGNORECASE  # as mentions: 'Go' is not 'go'
    after = r"(?![A-Za-z0-9+#])"
    if (not name[-1].isalnum() or len(name) >= 3) and re.search(  # never 'R2' for R
            rf"(?<![A-Za-z0-9]){re.escape(name)}\d+(?:\.\d+)*{after}", evidence, flags):
        return True
    if name.startswith(".") and re.search(rf"(?<=[A-Za-z]){re.escape(name)}{after}", evidence, flags):
        return True
    head, _space, last = name.rpartition(" ")
    if not last.lower().endswith("ing") or len(last) < 6:  # never 'Swing' or 'Ping'
        return False
    stem = last[:-3]
    nouns = {stem, stem + "e"} | ({stem[:-1]} if stem[-1] == stem[-2] else set())  # test, cache, log (logging)
    return any(mentions(evidence, f"{head} {noun}".strip()) for noun in nouns)


def shape_skills(resume: Resume, analysis: JDAnalysis | None = None) -> list[str]:
    """The skills as the reviewer asked: an item no summary sentence or bullet shows is left out, unless the posting
    asks for it (its must-haves, nice-to-haves and keywords stay for the job sites' scans); the posting's must-haves
    come first, then its nice-to-haves, keywords and the rest, rows in the same order; at most 3 rows: a 4th row's
    items join the 3rd, the posting's terms first, while the list holds at most 18 items. Changes the resume; returns
    a note naming what was left out (when no item is shown, the list stays as it is)."""
    evidence = evidence_text(resume)
    unshown: list[str] = []
    rows = []
    for group in resume.skills:
        ranked = []
        for item in group.skills:
            rank = _rank(item, analysis)
            if rank < 3 or _shown(evidence, item):
                ranked.append((rank, item))
            else:
                unshown.append(item)
        if ranked:
            ranked.sort(key=lambda pair: pair[0])  # stable: the model's order within a rank
            rows.append((min(rank for rank, _item in ranked), group, [item for _rank_, item in ranked]))
    rows.sort(key=lambda row: row[0])
    for _best, group, items in rows:
        group.skills = items
    groups = [group for _best, group, _items in rows]
    crowded: list[str] = []
    if len(groups) > MAX_SKILL_ROWS:
        last, extra = groups[MAX_SKILL_ROWS - 1], [item for group in groups[MAX_SKILL_ROWS:] for item in group.skills]
        groups = groups[:MAX_SKILL_ROWS]
        total = sum(len(group.skills) for group in groups)
        for item in sorted(extra, key=lambda item: _rank(item, analysis)):  # stable: the posting's terms first
            if item.lower() in {i.lower() for group in groups for i in group.skills}:
                continue
            if _rank(item, analysis) < 3 or total < MAX_SKILL_ITEMS:
                last.skills.append(item)
                total += 1
            else:
                crowded.append(item)
    if groups:
        resume.skills = groups
    else:
        unshown = []  # nothing was left out
    notes = []
    if unshown:
        notes.append("Skills: left out what no bullet shows: " + ", ".join(unshown[:8])
                     + (f" and {len(unshown) - 8} more" if len(unshown) > 8 else "") + ".")
    if crowded:
        notes.append(f"Skills: kept {MAX_SKILL_ROWS} rows, so left out " + ", ".join(crowded[:8])
                     + (f" and {len(crowded) - 8} more" if len(crowded) > 8 else "") + ".")
    return notes


# ---------------------------------------------------------------------- checks
_WEAK_OPENING = re.compile(
    r"^\s*(?P<word>used|use|uses|using|supported|support|supports|supporting|ran|run|runs|running|helped|help|helps"
    r"|handled|handle|handles)\b(?:\s+(?P<next>[A-Za-z][\w-]*))?",
    re.IGNORECASE)
_WEAK_BASE = {"used": "use", "uses": "use", "using": "use", "supported": "support", "supports": "support",
              "supporting": "support", "ran": "run", "runs": "run", "running": "run", "helped": "help",
              "helps": "help", "handled": "handle", "handles": "handle"}


def weak_openers(resume: Resume) -> list[str]:
    """Bullets that open with a weak verb ('Used ...', 'Supported ...', 'Run ...'): it says nothing about what was
    built, changed or decided. An opening that is a noun is fine ('Support tickets fell ...', 'Use cases ...')."""
    found = []
    for exp in resume.experience:
        for bullet in exp.bullets:
            match = _WEAK_OPENING.match(bullet)
            if not match:
                continue
            word = match.group("word").lower()
            base = _WEAK_BASE.get(word, word)
            if (match.group("next") or "").lower() in tense.NOUN_AFTER.get(base, ()):
                continue
            found.append(bullet)
    return found


MIN_WEAK_OPENERS_FOR_RETRY = 2


def text_lines(resume: Resume, paper: str = "Letter", style: FitStyle | None = None) -> tuple[int, list[list[int]]]:
    """(the summary's lines, each job's bullets' lines) at the design's own sizes, or at `style` (a saved run's own)."""
    from ..render.pdf import Run, resume_sheet

    sheet = resume_sheet(resume, style or design_style(paper))
    line = sheet.size * sheet.line
    summary = len(sheet.wrap([Run(resume.summary)], sheet.width, sheet.size))
    return summary, [[round(sheet.bullet_height(bullet) / line) for bullet in exp.bullets]
                     for exp in resume.experience]


MAX_BULLET_LINES, MAX_SUMMARY_LINES = 2, 3
MIN_LONG_BULLETS_FOR_RETRY = 2


def long_text(resume: Resume, paper: str = "Letter", style: FitStyle | None = None) -> tuple[list[str], int]:
    """(bullets of 3 lines or more, the summary's lines when over 3, else 0) at the design's own sizes, or at `style`
    (a saved run's own)."""
    summary, lines = text_lines(resume, paper, style)
    long = [bullet for exp, counts in zip(resume.experience, lines) for bullet, n in zip(exp.bullets, counts)
            if n > MAX_BULLET_LINES]
    return long, summary if summary > MAX_SUMMARY_LINES else 0


_SO_ENDING = re.compile(r",?\s+so\s+(?:that\s+)?(?!(?:far|much|many|few|long|well)\b)[a-z]", re.IGNORECASE)
MAX_SO_ENDINGS_PER_JOB = 1


def so_endings(resume: Resume) -> list[tuple[str, list[str]]]:
    """(employer, its bullets) for each job where more than one bullet ends with a "so ..." clause: the reviewer asked
    for varied endings."""
    found = []
    for exp in resume.experience:
        ending = [bullet for bullet in exp.bullets
                  if any(m.start() > len(bullet) * 0.3 for m in _SO_ENDING.finditer(bullet))]
        if len(ending) > MAX_SO_ENDINGS_PER_JOB:
            found.append((resume.profile.companies[exp.company_index].name, ending))
    return found


MIN_CURRENT_FIGURE_BULLETS = 2


def current_job_figures(resume: Resume) -> tuple[str, int] | None:
    """(employer, bullets with a figure) when the current job has fewer than 2 bullets with figures, else None."""
    for exp in resume.experience:
        company = resume.profile.companies[exp.company_index]
        if is_current(company, exp.company_index):
            count = sum(has_figure(bullet) for bullet in exp.bullets)
            if count < min(MIN_CURRENT_FIGURE_BULLETS, len(exp.bullets)):
                return company.name, count
            return None
    return None


def notes_figure_tip(profile: Profile) -> str | None:
    """A tip when the current job's project notes give fewer than 2 figures: the AI uses the person's own numbers
    first, and the reviewer wants that job's scale and results."""
    from .notes import clean_notes

    for index, company in enumerate(profile.companies):
        if is_current(company, index):
            if len(note_figures(clean_notes(company.projects))) >= MIN_CURRENT_FIGURE_BULLETS:
                return None
            return (f"Tip: add numbers for {company.name} to its project notes on the Profile page (users or runs per "
                    "day, tests run and regressions caught, a cost saved): the AI uses your own numbers first.")
    return None


def summary_words(resume: Resume) -> int:
    return word_count(resume.summary)
