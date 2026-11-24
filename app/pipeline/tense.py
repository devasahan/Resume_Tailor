"""The tense of resume bullets.

The current job is written in the present tense ("Build", "Lead") and every other job in the past ("Built", "Led")
(app/prompts/system.md, BULLETS). A model sometimes slips ("Added RAG over ..." in the current job, "Lead the claims API
migration ..." in an old one), so a bullet's opening verb, and the verbs joined to it ("Design and run ..."), are put
in its job's tense here (past_opening, present_opening). As in spelling.py, only words in the lists below are changed:
an unknown word stays as written, and so does an opening that reads as a noun ("Lead time fell from ...", "Lead data
engineer on ...", "Shared dashboards replaced ...", "Automated Spark tests catch ..."). No other word is ever changed:
a bullet with a second verb phrase in its opening's own form ("Rebuilt ..., wrote 300 pytest cases and caught ...",
"Leads ..., owns ... and ships ...") is left whole, as changing its opening alone would leave its verbs in two forms.
Only that form counts: "Leads the data team and automated ..." becomes "Lead the data team and automated ...", as its
later verb was in the other tense already. A past opening is never turned into a weak present one ("Ran" into "Run",
"Used" into "Use"): such a bullet is rewritten instead (validate.weak_openers); nor into a command when it is true only
in the past ("Promoted to ...", "Won the 2025 hackathon"). A duty is, also before "to" or "as" ("Reported to the CTO"
-> "Report to the CTO").
"""
from __future__ import annotations

import re

from .spelling import EU_ONLY, _with_case, us_word

# base -> past, where the past is not the base plus -ed
IRREGULAR = {
    "become": "became", "begin": "began", "break": "broke", "bring": "brought", "build": "built", "buy": "bought",
    "catch": "caught", "choose": "chose", "deal": "dealt", "do": "did", "draw": "drew", "drive": "drove",
    "fall": "fell", "fight": "fought", "find": "found", "fly": "flew", "get": "got", "give": "gave", "go": "went",
    "grow": "grew", "hold": "held", "keep": "kept", "lay": "laid", "lead": "led", "lend": "lent", "make": "made",
    "meet": "met", "outgrow": "outgrew", "overcome": "overcame", "oversee": "oversaw", "pay": "paid",
    "rebuild": "rebuilt", "redo": "redid", "remake": "remade", "rerun": "reran", "rethink": "rethought",
    "rewrite": "rewrote", "rise": "rose", "run": "ran", "see": "saw", "seek": "sought", "sell": "sold", "send": "sent",
    "shrink": "shrank", "speak": "spoke", "speed": "sped", "spend": "spent", "spin": "spun", "stand": "stood",
    "take": "took", "teach": "taught", "tell": "told", "think": "thought", "understand": "understood",
    "undertake": "undertook", "uphold": "upheld", "win": "won", "withdraw": "withdrew", "write": "wrote",
}
# an opening that is always the verb, never an adjective before its noun ('Made nightly builds 21% faster', 'Built
# ranking models that make ...'): a simple past whose participle differs, and 'built', 'made', 'led'. Not 'Paid search
# campaigns drove ...', 'Thought leadership posts drove ...', 'Rebuilt onboarding flows lift ...' or 'Custom-built ...'
_VERB_ONLY_PASTS = frozenset({"built", "made", "led", "wrote", "rewrote", "grew", "outgrew", "drove", "took",
                              "undertook", "gave", "began", "broke", "chose", "drew", "flew", "rose", "saw",
                              "oversaw", "shrank", "spoke", "withdrew", "overcame", "did", "redid"})
# the past is the base itself: nothing to change ("Cut", "Set up")
SAME = frozenset({
    "bid", "broadcast", "cast", "cut", "forecast", "hit", "let", "offset", "put", "quit", "read", "reset", "rid",
    "set", "shed", "shut", "split", "spread", "upset",
})
# the last consonant doubles: plan -> planned
DOUBLED = frozenset({
    "admit", "chat", "commit", "compel", "control", "debug", "dub", "emit", "equip", "excel", "flag", "format", "log",
    "map", "omit", "patrol", "permit", "pin", "plan", "plot", "plug", "prep", "program", "propel", "refer", "remap",
    "replan", "rig", "scan", "scrap", "scrub", "ship", "slot", "spot", "step", "stop", "submit", "swap", "tag",
    "transfer", "trim", "unpin", "unplug", "unwrap", "vet", "wrap",
})
# the past adds -ed, -d (after a final e) or -ied (after a consonant and y)
REGULAR = frozenset({
    "accelerate", "achieve", "acquire", "adapt", "add", "address", "adjust", "administer", "adopt", "advise",
    "advocate", "align", "allocate", "analyze", "answer", "anticipate", "apply", "approve", "architect", "archive",
    "arrange", "assemble", "assess", "assign", "assist", "attain", "audit", "augment", "author", "automate", "balance",
    "benchmark", "boost", "brief", "budget", "calculate", "calibrate", "capture", "catalog", "centralize", "certify",
    "chair", "champion", "change", "chart", "check", "clarify", "classify", "clean", "close", "coach", "codify",
    "collaborate", "collect", "combine", "communicate", "compare", "compile", "complete", "compose", "compress",
    "compute", "conduct", "configure", "connect", "consolidate", "construct", "consult", "containerize", "contribute",
    "convert", "coordinate", "correct", "craft", "create", "cultivate", "curate", "customize", "decommission",
    "decouple", "decrease", "deduplicate", "define", "delegate", "deliver", "demonstrate", "deploy", "deprecate",
    "derive", "design", "detect", "determine", "develop", "devise", "diagnose", "digitize", "direct", "discover",
    "dispatch", "distribute", "document", "double", "draft", "earn", "edit", "educate", "eliminate", "embed", "enable",
    "encrypt", "enforce", "engage", "engineer", "enhance", "enrich", "ensure", "establish", "estimate", "evaluate",
    "examine", "execute", "expand", "expedite", "experiment", "explain", "explore", "export", "expose", "extend",
    "extract", "facilitate", "finalize", "fix", "follow", "formalize", "formulate", "gather", "generate", "govern",
    "guide", "halve", "handle", "harden", "harmonize", "help", "hire", "host", "identify", "implement", "import",
    "improve", "incorporate", "increase", "index", "influence", "inform", "ingest", "initiate", "inspect", "install",
    "institute", "instrument", "integrate", "interpret", "interview", "introduce", "invent", "invest", "investigate",
    "isolate", "iterate", "join", "justify", "label", "launch", "lift", "load", "localize", "locate", "lower",
    "maintain", "manage", "market", "maximize", "measure", "mediate", "mentor", "merge", "migrate", "minimize",
    "mobilize", "model", "moderate", "modernize", "modify", "modularize", "monetize", "monitor", "motivate", "move",
    "navigate", "negotiate", "normalize", "notify", "obtain", "onboard", "open", "operate", "optimize", "orchestrate",
    "organize", "outline", "outperform", "overhaul", "own", "package", "pair", "parallelize", "participate", "partner",
    "patch", "perform", "pilot", "pinpoint", "pioneer", "port", "position", "predict", "prepare", "present",
    "preserve", "prevent", "price", "prioritize", "process", "procure", "produce", "profile", "promote", "propose",
    "protect", "prototype", "prove", "provide", "provision", "publish", "purchase", "pursue", "qualify", "quantify",
    "query", "raise", "rank", "reach", "realign", "rearchitect", "rebalance", "recommend", "reconcile", "record",
    "recover", "recruit", "redesign", "reduce", "refactor", "refine", "reform", "reinforce", "relaunch", "release",
    "relocate", "remediate", "remove", "render", "renegotiate", "reorganize", "repair", "replace", "replatform",
    "report", "represent", "research", "reshape", "resolve", "respond", "restore", "restructure", "retain", "retire",
    "retrain", "return", "reuse", "revamp", "review", "revise", "revitalize", "rework", "roll", "route", "safeguard",
    "save", "scale", "schedule", "scope", "score", "screen", "script", "secure", "segment", "select", "serve",
    "service", "shape", "share", "shift", "shorten", "showcase", "simplify", "simulate", "solve", "source", "specify",
    "sponsor", "stabilize", "standardize", "start", "steer", "store", "streamline", "strengthen", "structure",
    "study", "summarize", "supervise", "supply", "support", "survey", "sustain", "switch", "synchronize",
    "synthesize", "tackle", "target", "test", "track", "train", "transcribe", "transform", "transition", "translate",
    "triage", "triple", "troubleshoot", "tune", "uncover", "unify", "unlock", "update", "upgrade", "use", "utilize",
    "validate", "verify", "visualize", "volunteer", "widen", "wire", "work",
})
VERBS = frozenset(IRREGULAR) | SAME | DOUBLED | REGULAR
# words after an opening verb that make it a noun ('Lead time fell', 'Test suites now run in four minutes')
NOUN_AFTER = {
    "build": {"time", "times", "failure", "failures", "minutes", "cache", "pipeline", "pipelines", "server", "servers"},
    "design": {"review", "reviews", "doc", "docs", "token", "tokens", "sprint", "sprints", "team", "teams", "debt"},
    "lead": {"time", "times", "engineer", "engineers", "developer", "developers", "designer", "architect", "analyst",
             "generation", "scoring", "nurturing", "list", "lists", "role", "roles", "conversion", "conversions"},
    "load": {"time", "times", "balancer", "balancers", "balancing", "test", "tests", "testing"},
    "model": {"accuracy", "performance", "latency", "drift", "training", "output", "outputs", "quality"},
    "process": {"improvement", "improvements", "change", "changes", "map", "maps", "documentation", "owner", "owners"},
    "query": {"latency", "time", "times", "volume", "cost", "costs", "plan", "plans"},
    "release": {"cadence", "frequency", "note", "notes", "cycle", "cycles", "time", "times", "train", "trains"},
    "report": {"generation", "accuracy", "request", "requests"},
    "run": {"time", "times", "rate", "rates", "cost", "costs", "books", "book"},
    "score": {"accuracy", "drift"},
    "support": {"tickets", "ticket", "volume", "queue", "queues", "cost", "costs", "request", "requests", "staff",
                "engineer", "engineers", "agent", "agents", "teams", "team", "call", "calls", "escalation",
                "escalations"},
    "test": {"suite", "suites", "coverage", "case", "cases", "result", "results", "run", "runs", "failure", "failures",
             "time", "times", "environment", "environments", "data", "plan", "plans", "automation"},
    "track": {"record", "records"},
    "use": {"case", "cases"},
    "work": {"order", "orders", "item", "items", "streams", "stream", "flow", "flows", "load", "loads"},
}
# nouns as often as verbs: such an opening is changed only when its object follows ('Lead the move', 'Lead 5 engineers',
# 'Build CI pipelines', 'Build dashboards'), never in a noun opening ('Lead data engineer on ...', 'Model serving
# platform for ...', 'Self-service analytics for ...')
NOUN_VERBS = frozenset(NOUN_AFTER) | {"lead", "model", "process", "program", "report", "service"}
# a job title after such an opening: it is a noun ('Lead Python developer for ...')
ROLE_NOUNS = frozenset({
    "engineer", "engineers", "developer", "developers", "scientist", "scientists", "designer", "designers",
    "architect", "architects", "analyst", "analysts", "manager", "managers", "contributor", "contributors",
})
# true only once, so never a command for the current job ('Won the 2025 company hackathon', 'Promoted twice')
ONE_TIME = frozenset({"win", "earn", "become", "award", "name", "elect", "promote", "hire", "select"})
# after an opening past participle, a passive ('Promoted to ...', 'Selected as ...', 'Recognized by ...')
_PASSIVE_NEXT = frozenset({"to", "as", "by"})
# verbs that are a duty, never a passive, before their word: the current job's own work ('Reported to the CTO', 'Served
# as tech lead'). Only these, each with its own word: a change made once stays as written ('Moved to ...', 'Migrated to
# ...'), and so do 'Reported by ...' and 'Reported as ...'
_ACTIVE_BEFORE = {"contribute": "to", "respond": "to", "report": "to", "present": "to", "serve": "as", "partner": "as"}
# a word after the opening verb group that shows it is a verb: an object or a particle comes next
_VERB_NEXT = frozenset({
    "a", "an", "the", "its", "their", "his", "her", "our", "each", "every", "all", "both", "this", "that", "these",
    "those", "several", "multiple", "new", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "dozens", "hundreds", "thousands", "up", "out", "down", "off", "over", "into", "through",
    "across", "away", "back",
})
# an object's first word after a verb ('saves the team', 'gives 38 analysts'), never a particle
_OBJECT_START = _VERB_NEXT - {"up", "out", "down", "off", "over", "into", "through", "across", "away", "back", "that",
                              "new"}
# words after which a verb form is not a subject's own verb ('for automated tests', 'to spot instances', 'that cut')
_NOT_SUBJECT = _VERB_NEXT | frozenset({
    "for", "with", "without", "to", "on", "in", "at", "by", "from", "onto", "using", "via", "of", "under", "per",
    "after", "before", "within", "between", "around", "about", "behind", "during", "toward", "towards", "against",
    "like", "than", "and", "or", "but", "nor", "so", "which", "who", "where", "when", "while", "as", "once", "until",
    "because", "if", "many", "more", "most", "some", "any", "no", "other",
})
# words between a subject and its verb ('Automated tests in CI run ...', 'Shared dashboards across 6 teams give ...')
_SUBJECT_PREP = frozenset({"in", "on", "at", "across", "from", "of"})
# a subject's verb that is not in the lists: an auxiliary or a modal, or 'now' / 'still' before the verb
_FINITE_WORDS = frozenset({"is", "are", "was", "were", "has", "have", "had", "now", "still", "can", "could", "will",
                           "would", "should", "may", "might", "must"})
# a finite verb in the opening clause: the opening word is part of its subject ('Lead time dropped from 9 to 4 days')
_FINITE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|now|still|" + "|".join(sorted(set(IRREGULAR.values()))) + r")\b"
    r"|\b[a-z]{2,}ed\s+(?:from|to|by|under|below|above|over|after|within|across|in|for|\d|[$€£])"
)
_FIRST_WORD = re.compile(r"\s*([A-Za-z][A-Za-z'-]*)")
_JOINED = re.compile(r"(\s*,\s*(?:and\s+|&\s+)?|\s+(?:and|&)\s+)([A-Za-z][A-Za-z-]*)")
_NEXT_WORD = re.compile(r"\s+([A-Za-z0-9][\w'-]*)")
_CLAUSE_END = re.compile(r"[,;:.!?]")
_SUBJECT_END = re.compile(r"[,;:](?:\s|$)|[.!?](?:\s|$)")  # the first comma, not the one in '1,200'
_TOKEN = re.compile(r"[A-Za-z0-9][\w'-]*")
# the first word of another phrase: '..., wrote 300 pytest cases and caught ...', also after '(and' or a comma with no
# space ('... in Go (and automated its tests)', '... in Go,and automated ...'); with 'by' after it: a participle
_PHRASE_START = re.compile(r"(?:[,;]\s*(?:(?:and|&)\s+)?|(?:\s+|\()(?:and|&)\s+)(?:(?:then|also)\s+)?"
                           r"([A-Za-z][A-Za-z-]*)(\s+by\b)?")
# the last part of a hyphenated adjective ('role-based access', 'short-lived tokens'), no second verb
_MODIFIER_TAILS = frozenset({"based", "lived", "focused", "oriented", "related", "powered", "backed", "minded",
                             "grained", "centered", "centred"})
# a first part that makes the word a verb ('re-based 40 branches', 'co-founded the guild')
_VERB_PREFIXES = frozenset({"re", "co", "de", "pre", "un", "up", "over", "out", "under", "mis", "dis"})
MAX_JOINED = 3  # 'Design, build, test and ship' at most
MAX_PREFIX = 6  # 'Co-lead', 'Re-architect', 'Fine-tune', 'Self-host'


def past(base: str) -> str:
    """The past tense of a listed verb's base form (lowercase)."""
    if base in IRREGULAR:
        return IRREGULAR[base]
    if base in SAME:
        return base
    if base in DOUBLED:
        return base + base[-1] + "ed"
    if base.endswith("e"):
        return base + "d"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return base[:-1] + "ied"
    return base + "ed"


def base_of(word: str) -> str | None:
    """The listed verb a present-tense word is ('Leads' -> 'lead', 'optimises' -> 'optimize', 'applies' -> 'apply'),
    or None for any other word (a past form, a noun, an unknown verb). A British noun is never the American verb it is
    spelled like ('programme' is not 'program', 'cheques' not 'checks')."""
    if word.lower() in EU_ONLY:
        return None
    lower = us_word(word)
    if lower in VERBS:
        return lower
    if lower.endswith("ies") and lower[:-3] + "y" in VERBS:
        return lower[:-3] + "y"
    if lower.endswith("s") and lower[:-1] in VERBS:
        return lower[:-1]
    if lower.endswith("es") and lower[:-2] in VERBS:
        return lower[:-2]
    return None


def _converted(word: str) -> str | None:
    """The word in the past tense, keeping its case and a short prefix ('Co-lead' -> 'Co-led'); None when it is not a
    listed present-tense verb or its past is the same word."""
    if any(ch.isupper() for ch in word[1:]):  # 'API', 'GraphQL': a name, never a verb
        return None
    prefix, rest = "", word
    if "-" in word:
        head, tail = word.rsplit("-", 1)
        if not head.isalpha() or len(head) > MAX_PREFIX:
            return None
        prefix, rest = head + "-", tail
    base = base_of(rest)
    if base is None:
        return None
    new = past(base)
    if new == rest.lower():
        return None
    return prefix + _with_case(new, rest if not prefix else rest.lower())


def _plural(word: str) -> bool:
    """A plural noun's shape ('tests', 'dashboards', 'APIs'); not a name ('Kubernetes', 'Redis') nor a word such as
    'process', 'status', 'analysis' or 'analytics'."""
    if re.fullmatch(r"[A-Z]{2,}s", word):
        return True
    return word.islower() and len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ics"))


def _form(word: str) -> str | None:
    """Which form of a listed verb the word is: 'past' ('wrote', 'automated', and any word ending in -ed, also after a
    short first part: 're-based', and a modifier such as 'role-based', which _second_verb tells apart), 'third' ('owns')
    or 'base' ('write'). None for a name, a word with a long or non-letter first part ('add-to-cart',
    'customer-focused'), any other word, and a verb whose past is the same word ('cut', 'set'), which fits either
    tense."""
    if not word.islower():
        return None
    if "-" in word:
        head, word = word.rsplit("-", 1)
        if not head.isalpha() or len(head) > MAX_PREFIX:
            return None
    lower = us_word(word)
    if lower in PAST_TO_BASE:
        return "past"
    base = base_of(word)
    if base is not None and base != lower:
        return "third"
    if base is not None:
        return "base" if past(base) != base else None
    return "past" if re.fullmatch(r"[a-z]{3,}ed", lower) else None


def _second_verb(text: str, at: int, old: set[str]) -> bool:
    """Whether a verb in one of the old forms starts another phrase after the opening verbs ('..., wrote 300 pytest
    cases and caught ...', '... to design and implement ...'): changing the opening alone would leave the bullet in
    two tenses, so it is left whole. A past participle before 'by' is no verb of its own ('..., used by 40 teams'), nor
    is a hyphenated adjective ('... and role-based access', see _modifier)."""
    for match in _PHRASE_START.finditer(text, at):
        form = _form(match.group(1))
        if (form in old and not (form == "past" and match.group(2))
                and not _modifier(match.group(1), text, match.end(1))):
            return True
    return False


def _modifier(word: str, text: str, end: int) -> bool:
    """Whether the word (ending at `end` in the text) is a hyphenated adjective ('role-based access', 'short-lived
    tokens'): its last part in _MODIFIER_TAILS, its first part no verb prefix ('re-based' is a verb), and no object or
    particle after it ('laser-focused the team', 'piggy-backed on the stream' are verbs)."""
    head, _, tail = word.lower().rpartition("-")
    if not head or head in _VERB_PREFIXES or tail not in _MODIFIER_TAILS:
        return False
    nxt = _NEXT_WORD.match(text, end)
    after = nxt.group(1).lower() if nxt else ""
    return not (after in _VERB_NEXT or after in {"on", "onto", "it", "them"} or after[:1].isdigit())


def _object_follows(words_after: str) -> bool:
    """Whether the words after an opening that is as often a noun start its object: a name or a plural noun ('Build
    CI pipelines', 'Build dashboards'), and no job title follows within three words ('Lead Python developer for')."""
    words = _TOKEN.findall(words_after)[:3]
    if not words or any(word.lower() in ROLE_NOUNS for word in words):
        return False
    return words[0][:1].isupper() or _plural(words[0])


def past_opening(text: str) -> str:
    """The bullet with its opening verb, and the verbs joined to it, in the past tense: 'Lead the move ...' ->
    'Led the move ...', 'Design and run ...' -> 'Designed and ran ...'. No later word is changed. Unchanged when the
    opening word is not a listed present-tense verb, when it reads as a noun ('Lead time fell from 9 to 4 days', 'Test
    suites now run in four minutes'; a word that is as often a noun needs its object after it, see NOUN_VERBS), and
    when another verb phrase follows in the opening's own form ('Design the claims API, write the OpenAPI spec and
    ...')."""
    first = _FIRST_WORD.match(text)
    if not first or _converted(first.group(1)) is None:
        return text
    words = [(first.start(1), first.end(1))]
    at = first.end(1)
    for _ in range(MAX_JOINED):
        joined = _JOINED.match(text, at)
        if not joined or joined.group(2)[:1].isupper() or base_of(joined.group(2).rsplit("-", 1)[-1]) is None:
            break  # a capital is a name ('Plan and Build Co.'); an unlisted word ends the verbs
        words.append((joined.start(2), joined.end(2)))
        at = joined.end(2)
    bases = {base_of(text[start:end].rsplit("-", 1)[-1]) for start, end in words}
    nxt = _NEXT_WORD.match(text, at)
    next_word = nxt.group(1).lower() if nxt else ""
    opening_base = base_of(first.group(1).rsplit("-", 1)[-1])
    if next_word not in _VERB_NEXT and not next_word[:1].isdigit():
        if next_word in NOUN_AFTER.get(opening_base or "", ()):
            return text
        clause = _CLAUSE_END.search(text, at)
        words_after = text[at:clause.start() if clause else len(text)][:200]
        if _FINITE.search(words_after):
            return text
        if bases <= NOUN_VERBS and not _object_follows(words_after):
            return text  # 'Lead data engineer on ...': a noun ('Track and reduce' has a verb that is never one)
    if _second_verb(text, at, {_form(text[start:end].lower()) for start, end in words} & {"base", "third"}):
        return text
    for start, end in sorted(words, reverse=True):
        new = _converted(text[start:end])
        if new:
            text = text[:start] + new + text[end:]
    return text


# past -> base for every listed verb whose past differs from it ('built' -> 'build', 'shipped' -> 'ship')
PAST_TO_BASE = {past(base): base for base in sorted(VERBS) if past(base) != base}
# never made the opening verb: a weak one, which says nothing about what was built or changed (the bullet is rewritten)
WEAK_BASES = frozenset({"run", "use", "support", "help", "assist", "work", "utilize", "handle", "participate"})
# words that end the opening clause before its object's details: a finite verb after them is not the subject's
_CLAUSE_STOP = re.compile(r"[,;:.!?]|\s(?:for|with|to|on|in|at|by|from|into|across|through|using|via|so|that|which|"
                          r"while|and|or)\s")
_PAST_FORM = re.compile(r"\b(?:[a-z]{3,}ed|" + "|".join(sorted(set(IRREGULAR.values()) | SAME)) + r")\b")


def present_base(word: str) -> str | None:
    """The listed verb a past-tense or third-person word is ('Built' -> 'build', 'shipped' -> 'ship', 'Leads' ->
    'lead'), or None for any other word (a present base form, a noun, an unknown verb)."""
    lower = us_word(word)
    if lower in PAST_TO_BASE:
        return PAST_TO_BASE[lower]
    base = base_of(word)
    return base if base is not None and base != lower else None


def _present(word: str) -> str | None:
    """The word in the present tense ('Built' -> 'Build', 'Co-led' -> 'Co-lead', 'Designs' -> 'Design'), keeping its
    case; None when it is not a listed past or third-person verb."""
    if any(ch.isupper() for ch in word[1:]):  # 'API', 'GraphQL': a name, never a verb
        return None
    prefix, rest = "", word
    if "-" in word:
        head, tail = word.rsplit("-", 1)
        if not head.isalpha() or len(head) > MAX_PREFIX:
            return None
        prefix, rest = head + "-", tail
    base = present_base(rest)
    if base is None:
        return None
    return prefix + _with_case(base, rest if not prefix else rest.lower())


def _subject_verb(text: str, at: int) -> bool:
    """Whether the words after the opening, up to the first comma, are a subject with its own verb ('Automated Spark
    tests catch ...', 'Shared Looker dashboards replaced ...', 'Automated tests in CI run ...'): the opening is then an
    adjective. A verb form after a preposition, an article, 'and' or 'that' is not that verb ('for automated test
    suites', 'to spot instances'), nor a present one after a singular noun or a possessive ('a vector index', 'each
    partner's export jobs') unless an object follows ('triage in Zendesk saves the team 10 hours')."""
    end = _SUBJECT_END.search(text, at)
    tokens = _TOKEN.findall(text[at:end.start() if end else len(text)][:200])
    plural = False  # a plural subject so far: 'tests', 'tests in CI' (not 'tests for the', 'jobs to spot')
    for before, word, after in zip(tokens, tokens[1:], [*tokens[2:], ""]):
        if _plural(before) and "'" not in before:  # 'each partner's export jobs': a possessive is no subject
            plural = True
        elif before.lower() in _NOT_SUBJECT and before.lower() not in _SUBJECT_PREP:
            plural = False
        if (not word.islower() or before.lower() in _NOT_SUBJECT or before[:1].isdigit()
                or _PAST_FORM.fullmatch(before.lower())):  # 'a shared automated pipeline': an adjective before
            continue
        if word in _FINITE_WORDS or _PAST_FORM.fullmatch(word):
            return True
        base = base_of(word)
        if base is None or after.lower() in NOUN_AFTER.get(base, ()):
            continue
        if (base == us_word(word) and plural) or after.lower() in _OBJECT_START or after[:1].isdigit():
            return True
    return False


def present_opening(text: str) -> str:
    """The bullet with its opening verb, and the verbs joined to it, in the present tense, for the current job:
    'Added RAG over ...' -> 'Add RAG over ...', 'Designed and shipped the ...' -> 'Design and ship the ...',
    'Leads the ...' -> 'Lead the ...'. No later word is changed. Unchanged when the opening is not a listed past or
    third-person verb; when the words after it, up to the first comma, read as a sentence of their own ('Shared Looker
    dashboards replaced weekly exports', 'Automated Spark tests catch ...': the opening is an adjective, never one of
    _VERB_ONLY_PASTS); when another verb phrase follows in the opening's own form: a past one after a past opening
    ('..., wrote 300 pytest cases and caught ...'), a third-person one after a third-person opening ('Leads ..., owns
    ... and ships ...'); when it is true only in the past ('Promoted to ...', 'Selected as ...', 'Won the 2025
    hackathon', see ONE_TIME; a duty is not: 'Reported to ...', 'Served as ...', see _ACTIVE_BEFORE); and when a verb
    would become a weak one ('Ran ...', 'Used ...': rewritten instead, see WEAK_BASES). A third-person opening is put
    in the base form even when a past verb follows ('Leads the data team and automated ...' -> 'Lead the data team and
    automated ...'): that verb was in the other tense already."""
    first = _FIRST_WORD.match(text)
    if not first or _present(first.group(1)) is None:
        return text
    words = [(first.start(1), first.end(1))]
    at = first.end(1)
    for _ in range(MAX_JOINED):
        joined = _JOINED.match(text, at)
        if not joined or joined.group(2)[:1].isupper():
            break  # a capital is a name ('Plan and Build Co.')
        word = joined.group(2).rsplit("-", 1)[-1]
        if present_base(word) is None and base_of(word) is None:
            break  # an unlisted word ends the verbs
        words.append((joined.start(2), joined.end(2)))
        at = joined.end(2)
    bases = [present_base(text[start:end].rsplit("-", 1)[-1]) or base_of(text[start:end].rsplit("-", 1)[-1])
             for start, end in words]
    if any(base in WEAK_BASES or base in ONE_TIME for base in bases):
        return text
    forms = {_form(text[start:end].lower()) for start, end in words}
    nxt = _NEXT_WORD.match(text, at)
    next_word = nxt.group(1) if nxt else ""
    if (next_word.lower() in _PASSIVE_NEXT and _form(text[words[-1][0]:words[-1][1]].lower()) == "past"
            and _ACTIVE_BEFORE.get(bases[-1]) != next_word.lower()):
        return text  # 'Promoted to ...', 'Selected as ...': a passive, never a command ('Reported to ...' is a duty)
    if next_word.lower() not in _VERB_NEXT and not next_word[:1].isdigit():  # a name next too ('Automated Spark tests')
        if next_word.lower() in NOUN_AFTER.get(bases[0] or "", ()):
            return text
        stop = _CLAUSE_STOP.search(text, at)
        clause = text[at:stop.start() if stop else len(text)][:200]
        if (_FINITE.search(clause) or _PAST_FORM.search(clause.replace("-", " _"))
                or (first.group(1).lower() not in _VERB_ONLY_PASTS and _subject_verb(text, at))):
            return text  # 'Shared dashboards replaced ...': the opening describes the subject of another verb
    if _second_verb(text, at, forms & {"past", "third"}):
        return text
    for start, end in sorted(words, reverse=True):
        new = _present(text[start:end])
        if new:
            text = text[:start] + new + text[end:]
    return text
