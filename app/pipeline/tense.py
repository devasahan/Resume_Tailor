"""The tense of resume bullets.

The current job is written in the present tense ("Build", "Lead") and every other job in the past ("Built", "Led")
(app/prompts/system.md, BULLETS). A model sometimes slips ("Added RAG over ..." in the current job, "Lead the claims API
migration ..." in an old one), so a bullet's opening verb, and the verbs joined to it ("Design and run ..."), are put
in its job's tense here (past_opening, present_opening). As in spelling.py, only words in the lists below are changed:
an unknown word stays as written, and so does an opening that reads as a noun ("Lead time fell from ...", "Test suites
now run ...", "Shared dashboards replaced ..."). A past opening is never turned into a weak present one ("Ran" into
"Run", "Used" into "Use"): such a bullet is rewritten instead (validate.weak_openers).
"""
from __future__ import annotations

import re

from .spelling import _with_case, us_word

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
# never a noun or a noun's modifier: in a bullet whose opening was changed, a later 'and <verb>' is changed too
VERB_ONLY = frozenset({
    "add", "automate", "consolidate", "create", "define", "deliver", "develop", "eliminate", "enable", "establish",
    "expand", "extend", "hire", "identify", "implement", "improve", "integrate", "introduce", "maintain", "migrate",
    "modernize", "negotiate", "optimize", "reduce", "refactor", "remove", "replace", "resolve", "simplify",
    "standardize", "streamline", "teach",
})
# words after an opening verb that make it a noun ('Lead time fell', 'Test suites now run in four minutes')
NOUN_AFTER = {
    "build": {"time", "times", "failures", "minutes", "cache", "pipeline", "pipelines", "server", "servers"},
    "design": {"review", "reviews", "doc", "docs", "tokens", "sprints", "team", "teams", "debt"},
    "lead": {"time", "times", "engineer", "engineers", "developer", "developers", "designer", "architect", "analyst",
             "generation", "scoring", "nurturing", "list", "lists", "role", "roles", "conversion", "conversions"},
    "load": {"time", "times", "balancer", "balancers", "balancing", "tests", "testing"},
    "model": {"accuracy", "performance", "latency", "drift", "training", "outputs", "quality"},
    "process": {"improvements", "changes", "maps", "documentation", "owners"},
    "query": {"latency", "time", "times", "volume", "costs", "plans"},
    "release": {"cadence", "frequency", "notes", "cycle", "cycles", "time", "times", "trains"},
    "report": {"generation", "accuracy", "requests"},
    "run": {"time", "times", "rate", "rates", "cost", "costs", "books", "book"},
    "score": {"accuracy", "drift"},
    "support": {"tickets", "ticket", "volume", "queue", "queues", "costs", "requests", "staff", "engineers", "agents",
                "teams", "team", "calls", "escalations"},
    "test": {"suite", "suites", "coverage", "case", "cases", "results", "runs", "failures", "time", "times",
             "environment", "environments", "data", "plans", "automation"},
    "track": {"record", "records"},
    "use": {"case", "cases"},
    "work": {"orders", "items", "streams", "stream", "flow", "flows", "load", "loads"},
}
# a word after the opening verb group that shows it is a verb: an object or a particle comes next
_VERB_NEXT = frozenset({
    "a", "an", "the", "its", "their", "his", "her", "our", "each", "every", "all", "both", "this", "that", "these",
    "those", "several", "multiple", "new", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "dozens", "hundreds", "thousands", "up", "out", "down", "off", "over", "into", "through",
    "across", "away", "back",
})
# a finite verb in the opening clause: the opening word is part of its subject ('Lead time dropped from 9 to 4 days')
_FINITE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|now|still|" + "|".join(sorted(set(IRREGULAR.values()))) + r")\b"
    r"|\b[a-z]{2,}ed\s+(?:from|to|by|under|below|above|over|after|within|across|in|for|\d|[$€£])"
)
_FIRST_WORD = re.compile(r"\s*([A-Za-z][A-Za-z'-]*)")
_JOINED = re.compile(r"(\s*,\s*(?:and\s+|&\s+)?|\s+(?:and|&)\s+)([A-Za-z][A-Za-z-]*)")
_NEXT_WORD = re.compile(r"\s+([A-Za-z0-9][\w'-]*)")
_CLAUSE_END = re.compile(r"[,;:.!?]")
_LATER_AND = re.compile(r"\b(?:and|&)\s+([a-z]+)\b")
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
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
    or None for any other word (a past form, a noun, an unknown verb)."""
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


def past_opening(text: str) -> str:
    """The bullet with its opening verb, and the verbs joined to it, in the past tense: 'Lead the move ...' ->
    'Led the move ...', 'Design and run ...' -> 'Designed and ran ...'. A later 'and <verb>' that is never a noun is
    changed with them ('... and add Redis caching' -> '... and added Redis caching'). Unchanged when the opening word
    is not a listed present-tense verb, or reads as a noun ('Lead time fell from 9 to 4 days', 'Test suites now run
    in four minutes')."""
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
    nxt = _NEXT_WORD.match(text, at)
    next_word = nxt.group(1).lower() if nxt else ""
    opening_base = base_of(first.group(1).rsplit("-", 1)[-1])
    if next_word not in _VERB_NEXT and not next_word[:1].isdigit():
        if next_word in NOUN_AFTER.get(opening_base or "", ()):
            return text
        clause = _CLAUSE_END.search(text, at)
        if _FINITE.search(text[at:clause.start() if clause else len(text)][:200]):
            return text
    edits = [(start, end, _converted(text[start:end])) for start, end in words]
    sentence = _SENTENCE_END.search(text, at)
    stop = sentence.start() if sentence else len(text)
    for later in _LATER_AND.finditer(text, at, stop):
        if us_word(later.group(1)) in VERB_ONLY:
            edits.append((later.start(1), later.end(1), past(us_word(later.group(1)))))
    for start, end, new in sorted(edits, reverse=True):
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


def present_opening(text: str) -> str:
    """The bullet with its opening verb, and the verbs joined to it, in the present tense, for the current job:
    'Added RAG over ...' -> 'Add RAG over ...', 'Designed and shipped the ...' -> 'Design and ship the ...',
    'Leads the ...' -> 'Lead the ...'. Unchanged when the opening is not a listed past or third-person verb, when the
    words after it read as a sentence of their own ('Shared dashboards replaced weekly exports', 'Automated tests now
    run ...': the opening is an adjective), and when a verb would become a weak one ('Ran ...', 'Used ...': rewritten
    instead, see WEAK_BASES)."""
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
    if any(base in WEAK_BASES for base in bases):
        return text
    nxt = _NEXT_WORD.match(text, at)
    next_word = nxt.group(1) if nxt else ""
    if next_word.lower() not in _VERB_NEXT and not next_word[:1].isdigit() and not next_word[:1].isupper():
        if next_word.lower() in NOUN_AFTER.get(bases[0] or "", ()):
            return text
        stop = _CLAUSE_STOP.search(text, at)
        clause = text[at:stop.start() if stop else len(text)][:200]
        if _FINITE.search(clause) or _PAST_FORM.search(clause.replace("-", " _")):
            return text  # 'Shared dashboards replaced ...': the opening describes the subject of another verb
    edits = [(start, end, _present(text[start:end])) for start, end in words]
    sentence = _SENTENCE_END.search(text, at)
    stop = sentence.start() if sentence else len(text)
    for later in _LATER_AND.finditer(text, at, stop):  # '... and optimized the loads' -> '... and optimize the loads'
        base = PAST_TO_BASE.get(us_word(later.group(1)))
        if base in VERB_ONLY:
            edits.append((later.start(1), later.end(1), base))
    for start, end, new in sorted(edits, reverse=True):
        if new:
            text = text[:start] + new + text[end:]
    return text
