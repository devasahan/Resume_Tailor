"""Programming languages: which one a job post wants most, and whether a text names it (prompt v39: the current job
names the post's most wanted language in at least one bullet, where the work used it; when the post names none, the
language its kind of work is usually built in, for example Python for AI work).

Some names are also everyday words, people's names or a single letter ('Go', 'R', 'C', 'Swift', 'Rust', 'Ruby',
'Dart', 'Julia', 'Shell', 'Bash'). Written that way ('Go', never 'go'), they count only where they read as the language:
as a skill keyword of their own ('Go', 'Go 1.22'), with a word of code after them ('Go services', 'the Go order-sync
service', 'Ruby on Rails', 'R packages'), after 'in' or 'using' ('written in Go'), or in a list with another language
('Python, Go or Rust'). 'Go-to-market', 'R&D', 'Series C', 'C-level' and 'Objective-C' never name Go, R or C, and
neither does 'dinner with Julia'.
"""
from __future__ import annotations

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
_CODE_NOUN = (r"(?:code|coding|codebases?|programming|programmers?|languages?|developers?|development|engineers?"
              r"|engineering|services?|microservices?|backends?|back-end|servers?|scripts?|scripting|librar(?:y|ies)"
              r"|packages?|modules?|apps?|applications?|SDKs?|APIs?|runtimes?|toolchains?|compilers?|tests?|testing"
              r"|stack|workers?|CLIs?|tools?|tooling|functions?|lambdas?|notebooks?|pipelines?|on[ \t]+Rails|classes"
              r"|triggers|crates?|gems?|frameworks?|ecosystem|concurrency|goroutines|generics|automation|firmware"
              r"|drivers?|models?|queries|contracts?)")
_CODE_AFTER = re.compile(rf"(?:[ \t]+(?!{_SMALL}\b)[\w/.+#-]+){{0,2}}[ \t]+{_CODE_NOUN}\b", re.IGNORECASE)
# what reads as code right before one ('written in Go', 'using R', 'experience with Rust')
_CODE_BEFORE = re.compile(r"\b(?:in|using|(?:proficien(?:t|cy)|fluen(?:t|cy)|experience|expertise|skilled)[ \t]+with"
                          r"|knowledge[ \t]+of)[ \t]+$", re.IGNORECASE)
# between two languages of a list: commas, slashes, brackets, 'and', 'or', versions ('Python 3.12, Go or Rust'); a
# version is read whole, so a long run of digits never makes the match slow
_LIST_GAP = re.compile(r"(?:[\s,/&()]|\d+(?:\.\d+)*(?![\d.])|\band\b|\bor\b|\bplus\b)*", re.IGNORECASE)
_VERSION = re.compile(r"[ \t]*(?:\d+(?:\.\d+)*)?[ \t]*[.,;:]?[ \t]*")  # 'Go 1.22': a skill keyword of its own
# a keyword of its own, as the language writes it or in lower case ('Go', 'go', 'r'); never in capitals ('SWIFT', the
# banks' network)
_ALONE = re.compile(r"\s*([A-Za-z]+)(?:[ \t]+\d+(?:\.\d+)*)?\s*")
_LOWER_WORDS = {word.lower(): language for word, language in _WORDS.items()}


def mentioned(text: str) -> list[str]:
    """The languages the text names, each once, in the order they first appear ('Python', 'Go'); a name that is a word
    too only where it reads as code (see the module docstring)."""
    text = text or ""
    alone = _ALONE.fullmatch(text)
    word = alone.group(1) if alone else ""
    if word in _WORDS or (word.islower() and word in _LOWER_WORDS):
        return [(_WORDS.get(word) or _LOWER_WORDS[word]).name]
    spans = [(m.start(), m.end(), _SURE[m.group(0).lower()].name) for m in _SURE_FORMS.finditer(text)]
    unsure = []
    for match in _WORD_FORMS.finditer(text):
        start, end = match.span()
        if any(s < end and start < e for s, e, _name in spans):  # part of a longer name ('Shell scripting')
            continue
        found = (start, end, _WORDS[match.group(0)].name)
        alone_here = not text[:start].strip() and _VERSION.fullmatch(text[end:])
        if alone_here or _CODE_AFTER.match(text, end) or _CODE_BEFORE.search(text[max(0, start - 60):start]):
            spans.append(found)
        else:
            unsure.append(found)
    grown = True
    while unsure and grown:  # a name in a list with a language is one too ('Python, Go or Rust')
        grown = False
        for found in list(unsure):
            start, end, _name = found
            if any(_LIST_GAP.fullmatch(text[e:start] if e <= start else text[end:s]) for s, e, _n in spans):
                spans.append(found)
                unsure.remove(found)
                grown = True
    return list(dict.fromkeys(name for _start, _end, name in sorted(spans)))


def canonical(name: str) -> str:
    """A language's name as a resume writes it ('golang' -> 'Go'); the name itself when it is no language listed."""
    found = mentioned(name)
    return found[0] if len(found) == 1 else name


def names(text: str, language: str) -> bool:
    """Whether the text names the language, by any of its names ('Golang' names Go), where it reads as code."""
    return canonical(language) in mentioned(text)


def _kind(name: str) -> str:
    language = _BY_NAME.get(name.lower())
    return language.kind if language else "code"


def _most_wanted(found: list[str]) -> str | None:
    """The first language the work is built in, before SQL or a shell; else the first one; None for none."""
    return next((name for name in found if _kind(name) == "code"), found[0] if found else None)


def wanted_language(analysis: JDAnalysis | None, jd: str = "") -> str | None:
    """The language the job post wants most: the first among its must-haves, then its nice-to-haves, then its keywords,
    then its own text, a language the work is built in before SQL or a shell in each ('SQL, Python' wants Python;
    'SQL, dbt' wants SQL). None when it names none."""
    tiers = [analysis.must_have_skills, analysis.nice_to_have_skills, analysis.keywords] if analysis else []
    for terms in tiers:
        found = list(dict.fromkeys(name for term in terms for name in mentioned(term)))
        if found:
            return _most_wanted(found)
    return _most_wanted(mentioned(jd))


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
_CODE_ROLES = frozenset({"developer", "developers", "programmer", "programmers", "sdet"})
_FIELD_ROLES = frozenset({"engineer", "engineers", "scientist", "scientists", "architect", "architects", "analyst",
                          "analysts"})


def usual_language(analysis: JDAnalysis | None) -> str | None:
    """The language this kind of work is usually built in, from the posting's title and its common title: Python for
    AI, LLM, data and platform work, Swift for iOS, Kotlin for Android, TypeScript for front ends ...; None for work not
    built in code ('Registered Nurse', 'Marketing Manager', 'Sales Engineer')."""
    if analysis is None:
        return None
    words = title_words(f"{analysis.target_role} {analysis.general_role}")
    coder = any(word in _CODE_ROLES for word in words)
    fields = [_USUAL[word] for word in words if word in _USUAL]
    if fields and (coder or any(word in _FIELD_ROLES for word in words)):
        return fields[0]
    return "Python" if coder else None


def codes(analysis: JDAnalysis | None) -> bool:
    """Whether the posting is for work built in code (usual_language): its current job names a language even when the
    post names none."""
    return usual_language(analysis) is not None
