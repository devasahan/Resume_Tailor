"""Programming languages: which ones a job post wants most, and whether a text names one (prompt v39: the current job
names the post's most wanted language in at least one bullet, where the work used it; when the post names none, the
language its kind of work is usually built in, for example Python for AI work).

Some names are also everyday words, people's names or a single letter ('Go', 'R', 'C', 'Swift', 'Rust', 'Ruby',
'Dart', 'Julia', 'Shell', 'Bash'). Reading what a post asks for (mentioned), they count only where they read as the
language, written that way ('Go', never 'go'): as a skill keyword of their own ('Go', 'Go 1.22'), with a word of code
after them ('Go services', 'the Go order-sync service', 'Ruby on Rails', 'R packages'), after 'in' or 'using' ('written
in Go'), or in a list with another language ('Python, Go or Rust'). 'Go-to-market', 'R&D', 'R & D', 'Series C',
'C-level', 'Objective-C', 'Hepatitis C testing', "a Class C driver's license" and 'Erlang C' (a call centre's staffing
formula) never name Go, R, C or Erlang, a database ('SQL Server', 'Azure SQL') is not the SQL language, and 'dinner
with Julia' names no language.

Checking that a text names a language the post wants (names, named) reads more loosely, the way people write it: 'the
Go payment gateway', 'from Python to Go', "Go's goroutines", 'a Rust parser', 'C++17', 'Python3', 'Python-based';
C and R before any word of code ('C drivers', 'R models'), and C by its standards ('C11'). Each reading takes time in
proportion to the text's length, however it is written ('C C C ... Python').
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from ..schemas import JDAnalysis


@dataclass(frozen=True)
class Language:
    name: str  # as a resume writes it: 'Go', 'C#'
    forms: tuple[str, ...] = ()  # other names that mean only it, in any letter case: 'Golang', 'T-SQL'
    words: tuple[str, ...] = ()  # names that are everyday words too: counted in this case only, where they read as code
    kind: str = "code"  # 'query' (SQL) or 'shell' (Bash): a post's language for building the work comes first


LANGUAGES: tuple[Language, ...] = (
    Language("Python"), Language("TypeScript"), Language("JavaScript"), Language("Java"),
    Language("Go", forms=("Golang",), words=("Go",)), Language("Rust", words=("Rust",)), Language("C#"),
    Language("C++"), Language("C", words=("C",)), Language("Kotlin"), Language("Swift", words=("Swift",)),
    Language("Ruby", words=("Ruby",)), Language("PHP"), Language("Scala"), Language("R", words=("R",)),
    Language("SQL", forms=("T-SQL", "Transact-SQL", "PL/SQL"), kind="query"), Language("Elixir", words=("Elixir",)),
    Language("Dart", words=("Dart",)), Language("Objective-C"), Language("Perl"), Language("MATLAB"),
    Language("Haskell"), Language("Lua"),
    Language("Bash", forms=("shell scripting", "shell scripts", "shell script", "Bash scripting"),
             words=("Bash", "Shell"), kind="shell"),
    Language("PowerShell", kind="shell"), Language("Apex", words=("Apex",)), Language("Clojure"), Language("F#"),
    Language("Julia", words=("Julia",)), Language("Groovy", words=("Groovy",)),
    Language("Solidity", words=("Solidity",)),
    Language("Erlang"), Language("OCaml"), Language("Fortran"), Language("COBOL"), Language("VBA"),
    Language("Visual Basic", forms=("VB.NET",)), Language("ABAP"), Language("Verilog"), Language("SystemVerilog"),
    Language("VHDL"), Language("Zig"),
)
_BY_NAME = {language.name.lower(): language for language in LANGUAGES}
# every way to write a language that means only it ('Python', 'golang'), and the names that are words too ('Go')
_SURE = {form.lower(): language for language in LANGUAGES
         for form in ((() if language.name in language.words else (language.name,)) + language.forms)}
_WORDS = {word: language for language in LANGUAGES for word in language.words}


def _alternation(forms) -> str:
    return "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))  # 'C++' before 'C'


# Never inside a word or a name: not after a letter, '#', '+', '.' or '-' ('PostgreSQL', 'Objective-C'), and not
# before a letter, '#', '+' or '&', nor a hyphen or a dot that joins more ('C++', 'R&D', 'Go-to-market'); a name that is
# a word too never before an apostrophe either ("Julia's team")
_BEFORE = r"(?<![\w#+.\-])"
_SURE_FORMS = re.compile(rf"{_BEFORE}(?:{_alternation(_SURE)})(?![\w#+&]|[\-.]\w)", re.IGNORECASE)
_WORD_FORMS = re.compile(rf"{_BEFORE}(?:{_alternation(_WORDS)})(?![\w#+&'’]|[\-.]\w)")
# what reads as code after a name that is a word too, right after it or one or two words later ('Go services', 'the Go
# order-sync service', 'Ruby on Rails', 'R packages'); the words between are never small words ('Go to the services
# page', 'Julia from the apps team')
_SMALL = (r"(?:to|the|a|an|and|or|nor|for|with|of|in|on|at|by|from|as|is|are|was|be|it|its|our|your|their|this|that"
          r"|live|ahead|back|out|over|through|up|down|into|get|got)")
_CODE_NOUNS = (
    "code", "coding", "codebases?", "programming", "programmers?", "languages?", "developers?", "development",
    "engineers?", "engineering", "services?", "microservices?", "backends?", "back-end", "servers?", "scripts?",
    "scripting", "librar(?:y|ies)", "packages?", "modules?", "apps?", "applications?", "SDKs?", "APIs?", "runtimes?",
    "toolchains?", "compilers?", "tests?", "testing", "stack", "workers?", "CLIs?", "tools?", "tooling", "functions?",
    "lambdas?", "notebooks?", "pipelines?", r"on[ \t]+Rails", "classes", "triggers", "crates?", "gems?", "frameworks?",
    "ecosystem", "concurrency", "goroutines", "generics", "automation", "firmware", "drivers?", "models?", "queries",
    "contracts?",
)
# words of code that follow a single letter outside code too ('Hepatitis C testing', "a Class C driver's license",
# 'Erlang C staffing models'): reading a post, they do not make C or R a language
_LETTER_NOUNS = frozenset({"tests?", "testing", "drivers?", "models?"})


def _code_after(nouns) -> re.Pattern:
    return re.compile(rf"(?:[ \t]+(?!{_SMALL}\b)[\w/.+#-]+){{0,2}}[ \t]+(?:{'|'.join(nouns)})\b", re.IGNORECASE)


_CODE_AFTER = _code_after(_CODE_NOUNS)
_LETTER_AFTER = _code_after([noun for noun in _CODE_NOUNS if noun not in _LETTER_NOUNS])
# what reads as code right before one ('written in Go', 'using R', 'experience with Rust')
_CODE_BEFORE = re.compile(r"\b(?:in|using|(?:proficien(?:t|cy)|fluen(?:t|cy)|experience|expertise|skilled)[ \t]+with"
                          r"|knowledge[ \t]+of)[ \t]+$", re.IGNORECASE)
# between two languages of a list: commas, slashes, brackets, 'and', 'or', versions ('Python 3.12, Go or Rust'); a
# version is read whole, so a long run of digits never makes the match slow
_LIST_GAP = re.compile(r"(?:[\s,/&()]|\d+(?:\.\d+)*(?![\d.])|\band\b|\bor\b|\bplus\b)*", re.IGNORECASE)
# 'Go 1.22': a skill keyword of its own; each run of spaces can be read only one way, so a long one is never slow
_VERSION = re.compile(r"[ \t]*(?:\d+(?:\.\d+)*[ \t]*)?(?:[.,;:][ \t]*)?")
# a keyword of its own, as the language writes it or in lower case ('Go', 'go', 'r'); never in capitals ('SWIFT', the
# banks' network)
_ALONE = re.compile(r"\s*([A-Za-z]+)(?:[ \t]+\d+(?:\.\d+)*)?\s*")
_LOWER_WORDS = {word.lower(): language for word, language in _WORDS.items()}
# never a language, however it is read: two capitals joined by '&' ('R & D', 'P&L'; but 'R & C' are two languages,
# and so are 'C & T-SQL'), and 'Erlang C' (and its A and B)
_NEVER = re.compile(rf"{_BEFORE}(?:(?!R[ \t]*&[ \t]*C\b|C[ \t]*&[ \t]*R\b)[A-Z][ \t]*&[ \t]*[A-Z]|Erlang[ \t-]?[A-C])"
                    r"(?![\w#+&]|[\-.]\w)")
# a database, not the SQL language, when a post is read: 'SQL Server', 'Azure SQL', 'Cloud SQL'
_DATABASES = re.compile(r"\bSQL[ \t]+Server\b|\b(?:Azure|Cloud)[ \t]+SQL\b", re.IGNORECASE)


def _outside(spans: list[tuple[int, int]]):
    """A test of whether [start, end) overlaps none of the spans (in order, apart)."""
    starts = [start for start, _end in spans]

    def outside(start: int, end: int) -> bool:
        k = bisect.bisect_left(starts, end) - 1  # the last span that starts before `end`
        return k < 0 or spans[k][1] <= start
    return outside


def _blocked(text: str, databases: bool) -> list[tuple[int, int]]:
    """The spans of the text that never name a language (_NEVER), and its databases ('SQL Server') when `databases`."""
    spans = sorted([m.span() for m in _NEVER.finditer(text)]
                   + ([m.span() for m in _DATABASES.finditer(text)] if databases else []))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _spans(text: str, letters: re.Pattern, databases: bool) -> list[tuple[int, int, str]]:
    """(start, end, language) of each place the text names a language (see mentioned), in order. `letters`: what reads
    as code after C or R; `databases`: whether a database ('SQL Server') is kept out. Linear in the text's length: each
    name is tested once, a list only between neighbours."""
    outside = _outside(_blocked(text, databases))
    every = [(m.start(), m.end(), _SURE[m.group(0).lower()].name) for m in _SURE_FORMS.finditer(text)]
    longer = _outside([(start, end) for start, end, _name in every])
    items = [(start, end, name, True) for start, end, name in every if outside(start, end)]
    lead = len(text) - len(text.lstrip())  # where the text's first word starts
    for match in _WORD_FORMS.finditer(text):
        start, end = match.span()
        if not longer(start, end) or not outside(start, end):  # part of a longer name ('Shell scripting'), or 'R & D'
            continue
        name = _WORDS[match.group(0)].name
        reads = ((start == lead and _VERSION.fullmatch(text, end))
                 or (letters if len(name) == 1 else _CODE_AFTER).match(text, end)
                 or _CODE_BEFORE.search(text[max(0, start - 60):start]))
        items.append((start, end, name, bool(reads)))
    items.sort()
    # a name in a list with a language is one too ('Python, Go or Rust'): along each run of neighbours that only a list
    # gap parts, forwards and then backwards
    sure = [item[3] for item in items]
    joined = [_LIST_GAP.fullmatch(text, items[i][1], items[i + 1][0]) is not None for i in range(len(items) - 1)]
    for i in range(1, len(items)):
        sure[i] = sure[i] or (sure[i - 1] and joined[i - 1])
    for i in range(len(items) - 2, -1, -1):
        sure[i] = sure[i] or (sure[i + 1] and joined[i])
    return [(start, end, name) for (start, end, name, _read), good in zip(items, sure) if good]


def _alone(text: str) -> str | None:
    """The language a name that is a word too is when it is the whole text, maybe with a version ('Go', 'go', 'R 4.3');
    None otherwise."""
    alone = _ALONE.fullmatch(text)
    word = alone.group(1) if alone else ""
    if word in _WORDS or (word.islower() and word in _LOWER_WORDS):
        return (_WORDS.get(word) or _LOWER_WORDS[word]).name
    return None


def mentioned(text: str) -> list[str]:
    """The languages the text names, each once, in the order they first appear ('Python', 'Go'); a name that is a word
    too only where it reads as code (see the module docstring)."""
    text = text or ""
    word = _alone(text)
    if word:
        return [word]
    return list(dict.fromkeys(name for _start, _end, name in _spans(text, _LETTER_AFTER, databases=True)))


def canonical(name: str) -> str:
    """A language's name as a resume writes it ('golang' -> 'Go'); the name itself when it is no language listed."""
    found = mentioned(name)
    return found[0] if len(found) == 1 else name


# The looser reading of names(): any of a language's names, also with a version, "'s" or a compound after it ('Python3',
# 'C++17', "Go's", 'Python-based', 'C#.NET'), and a name that is a word too without a word of code around it ('the Go
# payment gateway', 'from Python to Go'), though still only as the language writes it ('Go', not 'go'). C and R keep
# the careful reading, before any word of code ('C drivers'); C also by its standards ('C11'), R by its own tools ('R
# Shiny', 'RStudio').
_SUFFIX = r"(?:\d+(?:\.\d+)*)?(?i:['’]s\b|-(?:based|first|only|heavy|native|powered|driven)\b|\.net\b)?"
_END = r"(?![\w#+&]|[\-.]\w)"
_LETTER_FORMS = {"C": r"C(?:89|90|99|11|17|18|23)", "R": r"R(?=[ \t]+(?:Shiny|Markdown)\b)|RStudio"}


def _loose_pattern(language: Language) -> re.Pattern | None:
    if len(language.name) == 1:
        return re.compile(rf"{_BEFORE}(?:{_LETTER_FORMS[language.name]}){_END}")
    if language.name in language.words:  # as the language writes it ('Go'); its other names in any case ('golang')
        body = (f"(?i:{_alternation(language.forms)})|" if language.forms else "") + re.escape(language.name)
    else:
        body = f"(?i:{_alternation((language.name, *language.forms))})"
    return re.compile(rf"{_BEFORE}(?:{body}){_SUFFIX}{_END}")


_LOOSE = {language.name: pattern for language in LANGUAGES if (pattern := _loose_pattern(language)) is not None}
# Node.js: JavaScript's runtime, which names a language when the post names none (names_a_language)
_NODE = re.compile(rf"{_BEFORE}Node(?:\.js|JS){_END}", re.IGNORECASE)


def _loose(text: str, only: str | None = None) -> list[tuple[int, str]]:
    """(position, language) where the text names a language in the looser reading of names(), each language at its
    first place, in order; `only`: that language alone."""
    found = [(start, name) for start, _end, name in _spans(text, _CODE_AFTER, databases=False)]
    word = _alone(text)
    if word:
        found.append((len(text) - len(text.lstrip()), word))
    outside = _outside(_blocked(text, databases=False))
    for name, pattern in _LOOSE.items():
        if only in (None, name):
            first = next((m.start() for m in pattern.finditer(text) if outside(*m.span())), None)
            if first is not None:
                found.append((first, name))
    return sorted(place for place in found if only in (None, place[1]))


def names(text: str, language: str) -> bool:
    """Whether the text names the language, by any of its names ('Golang' names Go), in the looser reading people
    write it in ('the Go payment gateway', "Go's", 'C++17', 'Python-based'; see the module docstring)."""
    return bool(_loose(text or "", canonical(language)))


def named(text: str) -> list[str]:
    """The languages the text names in the looser reading of names(), each once, in the order they first appear."""
    return list(dict.fromkeys(name for _start, name in _loose(text or "")))


def names_a_language(text: str) -> bool:
    """Whether the text names any programming language, as names() reads it, 'Node.js' counting as JavaScript: what a
    current job's bullets need when the post names none."""
    return bool(named(text)) or _NODE.search(text or "") is not None


def either(found: list[str]) -> str | None:
    """Languages any one of which will do, as a sentence names them: 'C or C++', 'Python, Go or Rust'; None for none."""
    if not found:
        return None
    return found[0] if len(found) == 1 else f"{', '.join(found[:-1])} or {found[-1]}"


def _kind(name: str) -> str:
    language = _BY_NAME.get(name.lower())
    return language.kind if language else "code"


def _most_wanted(found: list[str]) -> str | None:
    """The first language the work is built in, before SQL or a shell; else the first one; None for none."""
    return next((name for name in found if _kind(name) == "code"), found[0] if found else None)


# a skill that is one language on its own, maybe with a version ('SQL', 'Python 3.12', 'T-SQL', 'golang')
_EVERY_NAME = [form for language in LANGUAGES for form in (language.name, *language.forms)]
_ONLY = re.compile(rf"\s*(?:{_alternation(_EVERY_NAME)})(?:[ \t]*v?\d+(?:\.\d+)*)?\s*", re.IGNORECASE)


def _on_its_own(term: str) -> bool:
    """Whether a skill is one language and nothing else ('SQL', 'Python 3'), not one among other words ('SQL or
    Python', 'Python scripting', 'Hepatitis C testing')."""
    return _ONLY.fullmatch(term) is not None and len(mentioned(term)) == 1


def wanted_languages(analysis: JDAnalysis | None, jd: str = "") -> list[str]:
    """The languages the job post wants most, any one of which the current job names: those of its first skill that
    names a language the work is built in, among its must-haves, then its nice-to-haves, then its keywords (so 'SQL,
    dbt' required and Python preferred wants Python), else those of its first skill that names SQL or a shell; for work
    built in code, then the first language its own text names, a language the work is built in first. A skill that
    offers two ('C/C++', 'Python or Go') is met by either: both are given, a language the work is built in first. Work
    not built in code (codes) wants only a language one of its must-haves is on its own ('SQL' for a marketing
    analyst), never one the post only prefers or one its text seems to name ("a valid Class C driver's license").
    [] when it names none."""
    if analysis is None:
        return []
    built_in_code = codes(analysis)
    terms = ([*analysis.must_have_skills, *analysis.nice_to_have_skills, *analysis.keywords] if built_in_code
             else [term for term in analysis.must_have_skills if _on_its_own(term)])
    found = [names_ for names_ in map(mentioned, terms) if names_]
    chosen = next((names_ for names_ in found if any(_kind(name) == "code" for name in names_)),
                  found[0] if found else [])
    if chosen:
        return sorted(chosen, key=lambda name: _kind(name) != "code")
    first = _most_wanted(mentioned(jd)) if built_in_code else None
    return [first] if first else []


def wanted_language(analysis: JDAnalysis | None, jd: str = "") -> str | None:
    """The language the job post wants most: the first of wanted_languages ('SQL, Python' wants Python; 'SQL, dbt'
    wants SQL; 'C/C++' wants C, or C++ just as well). None when it names none."""
    wanted = wanted_languages(analysis, jd)
    return wanted[0] if wanted else None


# A title's fields as one word each ('Machine Learning' is ML, 'Back-End' is Backend), in lower case
_FIELD_FORMS = (
    (re.compile(r"\bmachine[\s-]+learning\b"), "ml"), (re.compile(r"\bartificial[\s-]+intelligence\b"), "ai"),
    (re.compile(r"\bback[\s-]?end\b"), "backend"), (re.compile(r"\bfront[\s-]?end\b"), "frontend"),
    (re.compile(r"\bfull[\s-]?stack\b"), "fullstack"), (re.compile(r"\bsite[\s-]+reliability\b"), "sre"),
    (re.compile(r"\bdev[\s-]?ops\b"), "devops"), (re.compile(r"\bml[\s-]?ops\b"), "mlops"),
    (re.compile(r"\binfra\b"), "infrastructure"), (re.compile(r"\bgen[\s-]?ai\b"), "ai"),
)


def title_words(title: str) -> list[str]:
    """A title's words in lower case, each field as one word: 'Senior Machine Learning Engineer' -> ['senior', 'ml',
    'engineer']; 'Back-End Developer' -> ['backend', 'developer']."""
    text = (title or "").lower()
    for pattern, form in _FIELD_FORMS:
        text = pattern.sub(form, text)
    return re.findall(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]", text)


# Work built in code: a developer or a programmer, or an engineer, scientist, architect or analyst of one of these
# fields ('Senior AI Platform Engineer', 'Data Scientist'), with the language that work is usually built in
_USUAL: dict[str, str] = {
    "ios": "Swift", "android": "Kotlin", "mobile": "Kotlin", "frontend": "TypeScript", "web": "TypeScript",
    "fullstack": "TypeScript", "embedded": "C", "firmware": "C", "game": "C++", "games": "C++", "robotics": "C++",
    "blockchain": "Solidity", "salesforce": "Apex", "ai": "Python", "ml": "Python", "mlops": "Python",
    "llm": "Python", "data": "Python", "platform": "Python", "backend": "Python", "software": "Python",
    "devops": "Python", "devsecops": "Python", "dataops": "Python", "sre": "Python", "infrastructure": "Python",
    "cloud": "Python", "automation": "Python", "vision": "Python", "nlp": "Python", "api": "Python",
    "integration": "Python", "integrations": "Python", "application": "Python", "applications": "Python",
}
# a title's own stack, before its field ('Senior React Developer' builds in TypeScript, '.NET Developer' in C#)
_STACK: dict[str, str] = {
    "react": "TypeScript", "angular": "TypeScript", "vue": "TypeScript", "vue.js": "TypeScript",
    "next.js": "TypeScript", "node": "TypeScript", "node.js": "TypeScript", "nodejs": "TypeScript", "net": "C#",
    "asp.net": "C#", "dotnet": "C#", "unity": "C#", "rails": "Ruby", "django": "Python", "flask": "Python",
    "fastapi": "Python", "spring": "Java", "laravel": "PHP", "wordpress": "PHP", "drupal": "PHP", "flutter": "Dart",
}
_CODE_ROLES = frozenset({"developer", "developers", "programmer", "programmers", "sdet"})
_FIELD_ROLES = frozenset({"engineer", "engineers", "scientist", "scientists", "architect", "architects", "analyst",
                          "analysts"})
# an application's analysts and support configure it, not build it in code ('Epic Applications Analyst', 'Application
# Support Engineer')
_CONFIGURES = frozenset({"analyst", "analysts", "support"})
# developers and programmers of work not built in code ('Business Developer', 'Instructional Developer', 'Real Estate
# Developer', 'CNC Programmer', 'Radio Programmer')
_NOT_CODE = re.compile(
    r"\b(?:business|sales|commercial|accounts?|partners?|partnerships?|channel|market|real[\s-]+estate|property"
    r"|land|housing|curriculum|instructional|courses?|courseware|training|e-?learning|learning[\s-]+(?:experience"
    r"|content|design|and[\s-]+development)|content|recipes?|menus?|talent|leadership|organi[sz]ational|cnc|cam|plc"
    r"|robot|broadcast|radio|tv|television|film|festival|music|product|programs?|fund|fundraising|donor)\b",
    re.IGNORECASE)


def usual_language(analysis: JDAnalysis | None) -> str | None:
    """The language this kind of work is usually built in, from the posting's title and its common title: the language
    or the stack the title names ('Senior Java Developer', 'Senior React Developer' -> TypeScript, '.NET Developer' ->
    C#), else Python for AI, LLM, data and platform work, Swift for iOS, Kotlin for Android, TypeScript for front ends
    ...; None for work not built in code ('Registered Nurse', 'Marketing Manager', 'Sales Engineer', 'Business
    Developer', 'CNC Programmer', 'Epic Applications Analyst')."""
    if analysis is None:
        return None
    title = f"{analysis.target_role} {analysis.general_role}"
    words = title_words(title)
    coder = any(word in _CODE_ROLES for word in words)
    if not coder and not any(word in _FIELD_ROLES for word in words):
        return None
    # the title's language by the careful reading ('Go Developer', not 'Go To Market Engineer'), a database's too ('SQL
    # Server Developer'), else its stack
    own = ([name for _start, _end, name in _spans(title, _LETTER_AFTER, databases=False)]
           or [_STACK[word] for word in words if word in _STACK])
    if own:
        return own[0]
    configures = any(word in _CONFIGURES for word in words)
    fields = [_USUAL[word] for word in words
              if word in _USUAL and not (configures and word in ("application", "applications"))]
    if fields:
        return fields[0]
    return "Python" if coder and not _NOT_CODE.search(title) else None


def codes(analysis: JDAnalysis | None) -> bool:
    """Whether the posting is for work built in code (usual_language): its current job names a language even when the
    post names none."""
    return usual_language(analysis) is not None
