"""US or EU (British) spelling for the text the app writes: the user picks "US style" or "EU style" on the Tailor page.

Recruiters in the UK and the EU read American spelling ("optimization") as a CV nobody localised, and recruiters in
the US read British spelling the same way. The model is told the style; this module makes sure every word follows it.

The word lists are explicit, never a general -ize -> -ise rule: advertise, compromise, enterprise, exercise, expertise,
promise, revise, supervise and surprise are correct in American English too. Words whose British form depends on the
meaning (program/programme, license/licence, practice/practise, meter/metre) are only ever changed from British to
American. Names keep their owner's spelling: employers, schools, the hiring company, and capitalised words inside a
brand's name (Unity Catalog, AWS Organizations, Google Optimize, Azure Security Center).
"""
from __future__ import annotations

import re
from collections.abc import Iterable

STYLES = ("us", "eu")

# -ize (US) / -ise (EU): stem + iz/is + ending, with an optional prefix ("reorganize", "deprioritized")
IZE_STEMS = (
    "amort", "annual", "anonym", "apolog", "binar", "collateral", "commodit", "docker", "featur", "lemmat",
    "production", "revital", "securit", "templat", "author", "canonical", "capital", "categor", "central", "character", "commercial",
    "component", "computer", "conceptual", "container", "contextual", "critic", "crystall", "custom", "democrat",
    "digit", "digital", "discret", "emphas", "empath", "energ", "equal", "evangel", "familiar", "final", "formal",
    "galvan", "general", "global", "harmon", "human", "hypothes", "incentiv", "industrial", "initial",
    "institutional", "internal", "international", "item", "jeopard", "legitim", "linear", "local", "marginal",
    "material", "maxim", "mechan", "memor", "minim", "mobil", "modern", "modular", "monet", "neutral", "normal",
    "operational", "optim", "organ", "parallel", "parameter", "parametr", "penal", "personal", "polar", "popular",
    "pressur", "priorit", "product", "pseudonym", "public", "quant", "random", "raster", "rational", "real", "recogn",
    "regular", "revolution", "sanit", "scrutin", "sensit", "serial", "social", "special", "stabil", "standard",
    "steril", "strateg", "subsid", "summar", "symbol", "synchron", "synthes", "systemat", "theor", "token", "urban",
    "util", "vector", "virtual", "visual", "weapon",
)
# Never "", "m" or "t": emphasis, synthesis, criticism and specialist are the same in both styles
IZE_ENDINGS = ("e", "es", "ed", "ing", "er", "ers", "ation", "ations", "ational", "able", "ability")
PREFIXES = ("", "re", "de", "un", "non", "pre", "over", "under", "co", "mis", "dis")
# -yze (US) / -yse (EU)
YZE_STEMS = ("anal", "paral", "catal", "dial", "hydrol", "electrol")
YZE_ENDINGS = ("e", "es", "ed", "ing", "er", "ers")
# -or (US) / -our (EU); rigorous, humorous, laborious and honorary are the same in both
OUR_WORDS = (
    "behavior", "behaviors", "behavioral", "behaviorally", "color", "colors", "colored", "coloring", "colorful",
    "favor", "favors", "favored", "favoring", "favorable", "favorably", "favorite", "favorites", "favoritism",
    "honor", "honors", "honored", "honoring", "honorable", "labor", "labors", "labored", "laboring", "neighbor",
    "neighbors", "neighboring", "neighborhood", "neighborhoods", "humor", "rumor", "rumors", "endeavor",
    "endeavors", "endeavored", "flavor", "flavors", "harbor", "harbors", "vapor", "rigor", "vigor", "armor",
    "candor", "odor", "parlor", "valor", "demeanor", "misdemeanor", "tumor", "tumors", "unfavorable",
    "unfavorably", "laborer", "laborers", "flavored",
)
RE_WORDS = {
    "center": "centre", "centers": "centres", "centered": "centred", "centering": "centring", "fiber": "fibre",
    "fibers": "fibres", "liter": "litre", "liters": "litres", "theater": "theatre", "theaters": "theatres",
    "caliber": "calibre", "meager": "meagre", "maneuver": "manoeuvre", "maneuvers": "manoeuvres",
    "maneuvered": "manoeuvred", "maneuvering": "manoeuvring", "maneuverability": "manoeuvrability",
    "centerpiece": "centrepiece", "centerpieces": "centrepieces",
}
LL_WORDS = {
    "modeled": "modelled", "modeling": "modelling", "modeler": "modeller", "modelers": "modellers",
    "remodeled": "remodelled", "remodeling": "remodelling", "labeled": "labelled", "labeling": "labelling",
    "relabeled": "relabelled", "mislabeled": "mislabelled", "unlabeled": "unlabelled", "canceled": "cancelled",
    "canceling": "cancelling", "traveled": "travelled", "traveling": "travelling", "traveler": "traveller",
    "travelers": "travellers", "signaled": "signalled", "signaling": "signalling", "leveled": "levelled",
    "leveling": "levelling", "channeled": "channelled", "channeling": "channelling", "tunneled": "tunnelled",
    "tunneling": "tunnelling", "fueled": "fuelled", "fueling": "fuelling", "refueled": "refuelled",
    "totaled": "totalled", "totaling": "totalling", "counseled": "counselled", "counseling": "counselling",
    "counselor": "counsellor", "counselors": "counsellors", "dialed": "dialled", "dialing": "dialling",
    "equaled": "equalled", "funneled": "funnelled", "rivaled": "rivalled", "unrivaled": "unrivalled",
    "jeweler": "jeweller", "marvelous": "marvellous", "fulfill": "fulfil", "fulfills": "fulfils",
    "fulfillment": "fulfilment", "enroll": "enrol", "enrolls": "enrols", "enrollment": "enrolment",
    "enrollments": "enrolments", "installment": "instalment", "installments": "instalments", "skillful": "skilful",
    "skillfully": "skilfully", "willful": "wilful", "instill": "instil", "instills": "instils", "distill": "distil",
    "distills": "distils", "labeler": "labeller", "labelers": "labellers", "funneling": "funnelling",
    "rivaling": "rivalling", "equaling": "equalling", "unraveled": "unravelled", "unraveling": "unravelling",
}
OTHER_WORDS = {
    "catalog": "catalogue", "catalogs": "catalogues", "cataloged": "catalogued", "cataloging": "cataloguing",
    "analog": "analogue", "analogs": "analogues", "defense": "defence", "defenses": "defences", "offense": "offence",
    "offenses": "offences", "pretense": "pretence", "gray": "grey", "grayscale": "greyscale", "aging": "ageing",
    "jewelry": "jewellery", "mold": "mould", "molds": "moulds", "molded": "moulded", "molding": "moulding",
    "judgment": "judgement", "judgments": "judgements", "acknowledgment": "acknowledgement", "aluminum": "aluminium",
    "skeptical": "sceptical", "skepticism": "scepticism", "toward": "towards", "grayed": "greyed",
    "graying": "greying", "acknowledgments": "acknowledgements", "colorize": "colourise", "colorized": "colourised",
    "colorizing": "colourising",
}
# British words changed to American only: the other way depends on the meaning (a software program, the verb to
# license, a medical practice, a gas meter), or both forms are American too (towards)
EU_ONLY = {
    "programme": "program", "programmes": "programs", "licence": "license", "licences": "licenses",
    "practise": "practice", "practised": "practiced", "practises": "practices", "practising": "practicing",
    "metre": "meter", "metres": "meters", "kilometre": "kilometer", "kilometres": "kilometers",
    "centimetre": "centimeter", "millimetre": "millimeter", "cheque": "check", "cheques": "checks",
    "artefact": "artifact", "artefacts": "artifacts", "whilst": "while", "amongst": "among", "learnt": "learned",
    "enquiry": "inquiry", "enquiries": "inquiries", "focussed": "focused", "focussing": "focusing",
    "benefitted": "benefited", "maths": "math",
}
NEVER_TO_US = {"towards"}  # American too
# Capitalised words that are nearly always part of a product's name (Webpack Bundle Analyzer, React Fiber)
NAME_PRONE = {"analyzer", "analyzers", "analyser", "analysers", "fiber", "fibre", "harbor", "harbour", "tokenizers",
              "tokenisers", "catalog", "catalogue", "modeler", "modeller", "modelers", "modellers"}
# A capitalised word after one of these, in the same run of capitalised words, belongs to a product's name
BRANDS = {
    "AWS", "Amazon", "Azure", "Microsoft", "Google", "GCP", "Apache", "IBM", "Oracle", "SAP", "Salesforce", "Adobe",
    "Atlassian", "Jira", "GitHub", "GitLab", "Databricks", "Snowflake", "Unity", "Webpack", "Hugging", "Elastic",
    "Splunk", "Datadog", "Cisco", "VMware", "HashiCorp", "Meta", "Apple", "Android", "Firebase", "React", "Tableau",
    "ServiceNow", "HubSpot", "Zendesk", "Shopify", "Stripe", "Twilio", "Collibra", "Alation", "Informatica",
    "Talend", "Purview", "Dataplex", "Glue", "SPSS", "Confluence", "Bitbucket", "Okta", "Gurobi", "CPLEX",
}
SIGN_OFFS = {
    "eu": {"sincerely": "Kind regards", "sincerely yours": "Yours sincerely", "respectfully": "Kind regards"},
    "us": {"yours sincerely": "Sincerely", "yours faithfully": "Sincerely"},
}
# 'my resume' -> 'my CV' in an EU style letter, only where it names the document: never 'the resume parser'. Never
# the other way: 'CV' also means computer vision or cross-validation, and American readers know the word.
_CV_WORDING = re.compile(
    r"\b(my|the|this|attached|enclosed|your)\s+r[eé]sum[eé](?=\s*(?:[.,;:!?)]|$)|\s+(?:and|or|for|with|as|to|which|that"
    r"|shows?|includes?|lists?|highlights?|outlines?|details?|has|is|was|below|above)\b)", re.IGNORECASE)


def _pairs() -> dict[str, str]:
    """Every American word in the lists, with its British form."""
    pairs: dict[str, str] = {}
    for prefix in PREFIXES:
        for stem in IZE_STEMS:
            for ending in IZE_ENDINGS:
                pairs[f"{prefix}{stem}iz{ending}"] = f"{prefix}{stem}is{ending}"
    for prefix in ("", "re"):
        for stem in YZE_STEMS:
            for ending in YZE_ENDINGS:
                pairs[f"{prefix}{stem}yz{ending}"] = f"{prefix}{stem}ys{ending}"
    for word in OUR_WORDS:
        pairs[word] = word.replace("or", "our", 1)
    pairs.update(RE_WORDS)
    pairs.update(LL_WORDS)
    pairs.update(OTHER_WORDS)
    return pairs


US_TO_EU = _pairs()
# "analyses" is also the plural of "analysis", the same in both styles: never changed to "analyzes"
EU_TO_US = {eu: us for us, eu in US_TO_EU.items() if not eu.endswith("yses") and eu not in NEVER_TO_US} | EU_ONLY
TABLES = {"eu": US_TO_EU, "us": EU_TO_US}
_FORMS: dict[str, frozenset[str]] = {}
for _us, _eu in [*US_TO_EU.items(), *EU_ONLY.items()]:
    _both = frozenset({_us, _eu}) | _FORMS.get(_us, frozenset()) | _FORMS.get(_eu, frozenset())
    for _form in _both:
        _FORMS[_form] = _both
# The word families that tell a text's own spelling (not towards, learnt or whilst, which Americans also write)
_TELLING = {word for word in US_TO_EU if word not in OTHER_WORDS} | {
    eu for us, eu in US_TO_EU.items() if us not in OTHER_WORDS}

_WORD = re.compile(r"[A-Za-z]+")
_CAPITALISED_RUN_GAP = re.compile(r"^[ \t]+$")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+|^\s*)$")  # not after "(": a bracket usually holds a name


def twins(word: str) -> frozenset[str]:
    """The word in both spellings, lowercase ('optimisation' -> {'optimisation', 'optimization'})."""
    lower = word.lower()
    return _FORMS.get(lower, frozenset({lower}))


def us_word(word: str) -> str:
    """The American form of a word, lowercase, for comparisons that must not depend on the spelling style. Both forms
    of an ambiguous word ('analyses') map to one form here: only comparisons use this."""
    lower = word.lower()
    if lower in EU_TO_US:
        return EU_TO_US[lower]
    if lower.endswith("yses") and lower in _FORMS:
        return next((form for form in _FORMS[lower] if form.endswith("yzes")), lower)
    return lower


def us_text(text: str) -> str:
    """The text with every listed word in its American form, lowercase: for comparing two texts."""
    return _WORD.sub(lambda match: us_word(match.group(0)), text.lower())


def _phrase_regex(phrase: str) -> re.Pattern | None:
    words = _WORD.findall(phrase)
    if not words:
        return None
    body = r"[^A-Za-z]+".join("(?:" + "|".join(sorted(map(re.escape, twins(w)), key=len, reverse=True)) + ")"
                              for w in words)
    return re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE)


def _spans(text: str, phrases: Iterable[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for phrase in {p.strip() for p in phrases if p and p.strip()}:
        pattern = _phrase_regex(phrase)
        if pattern is not None:
            spans += [match.span() for match in pattern.finditer(text)]
    return spans


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


_PACKAGE_PREFIXES = ("scikit-", "sklearn-", "py", "react-", "vue-", "ngx-", "node-", "django-", "flask-", "pytest-",
                     "eslint-", "babel-", "webpack-", "tf-", "torch")


def _in_identifier(text: str, match: re.Match, title: bool) -> bool:
    """True when the word is part of code, a package, a path or a CSS name rather than English."""
    start = max(text.rfind(" ", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
    ends = [end for end in (text.find(" ", match.end()), text.find("\n", match.end())) if end != -1]
    token = text[start: min(ends) if ends else len(text)].strip("()[]{}\"'`,;:!?.")
    if "." in token or "_" in token or token.startswith("-") or token.count("/") >= 2:
        return True
    if token.islower() and token.startswith(_PACKAGE_PREFIXES):  # scikit-optimize, react-color
        return True
    return title and token.islower() and token.count("-") >= 2  # a lowercase package name in the skills list


def _with_case(word: str, template: str) -> str:
    return word[:1].upper() + word[1:] if template[:1].isupper() else word


def _after_brand(text: str, words: list[re.Match], index: int) -> bool:
    """True when a brand's name starts the run of capitalised words that this word ends ('Unity Catalog')."""
    position = index - 1
    while position >= 0:
        previous = words[position]
        gap = text[previous.end(): words[position + 1].start()]
        if not _CAPITALISED_RUN_GAP.match(gap) or not previous.group(0)[:1].isupper():
            return False
        if previous.group(0) in BRANDS:
            return True
        position -= 1
    return False


def convert_and_list(text: str, style: str, keep: Iterable[str] = (), terms: Iterable[str] = (),
                     title: bool = False) -> tuple[str, list[tuple[str, str]]]:
    """(text in the style, [(word, new word), ...]); see convert."""
    table = TABLES[style]
    if not text:
        return text, []
    words = list(_WORD.finditer(text))
    if not any(match.group(0).lower() in table for match in words):
        return text, []
    kept, term_spans = _spans(text, keep), _spans(text, terms)
    parts: list[str] = []
    changes: list[tuple[str, str]] = []
    last = 0
    for index, match in enumerate(words):
        word = match.group(0)
        new = table.get(word.lower())
        if new is None or _inside(match.start(), kept) or any(ch.isupper() for ch in word[1:]):
            continue  # not a listed word, a name, ALL CAPS or camelCase
        if _in_identifier(text, match, title):
            continue  # 'scipy.optimize', 'pg_catalog', '--color-primary', 'webpack-bundle-analyzer'

        if word[0].isupper():
            if word.lower() in NAME_PRONE or _after_brand(text, words, index):
                continue
            starts_sentence = _SENTENCE_START.search(text[: match.start()]) is not None
            if not title and not starts_sentence and not _inside(match.start(), term_spans):
                continue  # a capitalised word inside a sentence is a name, unless it is a known skill term
        new = _with_case(new, word)
        parts += [text[last: match.start()], new]
        last = match.end()
        changes.append((word, new))
    parts.append(text[last:])
    return "".join(parts), changes


def convert(text: str, style: str, keep: Iterable[str] = (), terms: Iterable[str] = (), title: bool = False) -> str:
    """The text in the style ("us" or "eu"). `keep`: names never changed. `terms`: skill terms, changed even when
    capitalised inside a sentence. `title`: a heading, skill or job title, where capitals do not mark a name."""
    return convert_and_list(text, style, keep, terms, title)[0]


def other_style_words(text: str, style: str, keep: Iterable[str] = (), terms: Iterable[str] = (),
                      title: bool = False) -> list[str]:
    """Words in the text that are spelled in the other style (what convert would change)."""
    return [word for word, _new in convert_and_list(text, style, keep, terms, title)[1]]


def detect(text: str) -> str:
    """The spelling a text is written in: "us", "eu", or "" when it does not clearly show one."""
    us = eu = 0
    for match in _WORD.finditer(text):
        lower = match.group(0).lower()
        if lower in _TELLING:
            if lower in US_TO_EU:
                us += 1
            elif lower in EU_TO_US:
                eu += 1
    if eu >= 2 and eu > us:
        return "eu"
    if us >= 2 and us > eu:
        return "us"
    return ""


def sign_off(closing: str, style: str) -> str:
    """'Sincerely,' -> 'Kind regards,' in EU style; 'Yours sincerely,' -> 'Sincerely,' in US style."""
    text = closing.strip()
    comma = "," if text.endswith(",") else ""
    return SIGN_OFFS[style].get(text.rstrip(",").strip().lower(), text.rstrip(",")) + comma


def cv_wording(text: str, style: str) -> str:
    """'my resume' -> 'my CV' in EU style (a cover letter's own words); US style leaves the text as it is."""
    return _CV_WORDING.sub(r"\1 CV", text) if style == "eu" else text
