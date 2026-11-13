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
from functools import lru_cache

STYLES = ("us", "eu")

# -ize (US) / -ise (EU): stem + iz/is + ending, with an optional prefix ("reorganize", "deprioritized")
IZE_STEMS = (
    "amort", "annual", "anonym", "apolog", "binar", "collateral", "commodit", "docker", "featur", "lemmat",
    "production", "revital", "securit", "templat", "author", "canonical", "capital", "categor", "central", "character", "commercial",
    "component", "computer", "conceptual", "container", "contextual", "critic", "crystall", "custom", "democrat",
    "digit", "digital", "discret", "emphas", "empath", "energ", "equal", "evangel", "familiar", "final", "formal",
    "galvan", "general", "global", "harmon", "hospital", "human", "hypothes", "immun", "incentiv", "industrial",
    "initial", "external", "privat",
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
    "unfavorably", "laborer", "laborers", "flavored", "endeavoring",
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
    "relabeling": "relabelling", "mislabeling": "mislabelling", "trialed": "trialled", "trialing": "trialling",
    "marshaled": "marshalled", "marshaling": "marshalling", "unmarshaled": "unmarshalled",
    "unmarshaling": "unmarshalling",
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
    "benefitted": "benefited", "benefitting": "benefiting", "maths": "math", "centimetres": "centimeters",
    "millimetres": "millimeters",
}
NEVER_TO_US = {"towards"}  # American too
# Capitalised words that are nearly always part of a product's name (Webpack Bundle Analyzer, React Fiber)
NAME_PRONE = {"analyzer", "analyzers", "analyser", "analysers", "fiber", "fibre", "harbor", "harbour", "tokenizers",
              "tokenisers", "modeler", "modeller", "modelers", "modellers"}
# A capitalised word after one of these, in the same run of capitalised words, belongs to a product's name
BRANDS = {
    "AWS", "Amazon", "Azure", "Microsoft", "Google", "GCP", "Apache", "IBM", "Oracle", "SAP", "Salesforce", "Adobe",
    "Atlassian", "Jira", "GitHub", "GitLab", "Databricks", "Snowflake", "Unity", "Webpack", "Hugging", "Elastic",
    "Splunk", "Datadog", "Cisco", "VMware", "HashiCorp", "Meta", "Apple", "Android", "Firebase", "React", "Tableau",
    "ServiceNow", "HubSpot", "Zendesk", "Shopify", "Stripe", "Twilio", "Collibra", "Alation", "Informatica",
    "Talend", "Purview", "Dataplex", "Glue", "SPSS", "Confluence", "Bitbucket", "Okta", "Gurobi", "CPLEX",
    "Spark", "Kafka", "Confluent", "Erwin", "erwin", "Iceberg", "Polaris", "Toad", "Navicat",
}
# After a brand only a word that names a thing keeps the brand's spelling (AWS Organizations, Compute Optimizer, Google
# Optimize, Azure Security Center, Unity Catalog), and so does a word inside a longer name (CPLEX Optimization Studio,
# AWS Cost Optimization Hub); an activity follows the style (Snowflake Data Modelling, Tableau Visualisation, Spark
# Optimisations)
_PRODUCT_WORD = re.compile(
    r"(?:i[sz]es?|i[sz]ers?|y[sz]ers?|organi[sz]ations|cent(?:er|re)s?|catalog(?:ue)?s?|colou?rs?)$")
# Names of products and bodies without a brand in front, kept as written wherever they appear capitalised
PRODUCT_NAMES = ("System Center", "Security Center", "Compute Optimizer", "Catalyst Optimizer",
                 "Center for Internet Security", "Centers for Disease Control", "Cloud Armor",
                 "Marketing Cloud Personalization", "SQL Developer Data Modeler")
SIGN_OFFS = {
    "eu": {"sincerely": "Kind regards", "sincerely yours": "Yours sincerely", "respectfully": "Kind regards",
           "respectfully yours": "Kind regards", "yours truly": "Kind regards", "very truly yours": "Kind regards",
           "truly yours": "Kind regards"},
    "us": {"yours sincerely": "Sincerely", "yours faithfully": "Sincerely"},
}
# 'my resume' -> 'my CV' in an EU style letter, only where it names the document: never 'the resume parser'. Never
# the other way: 'CV' also means computer vision or cross-validation, and American readers know the word.
_RESUME_ADJECTIVE = r"(?:(?:updated|full|current|latest|attached|enclosed|complete|detailed|one-page|two-page)\s+)?"
_RESUME_COMPOUND = (r"(?:parser|parsing|data|screening|screener|builder|template|templates|tool|tools|matching|matcher"
                    r"|scanner|service|services|database|format|formats|writer|writing|bank|app|api|model|models"
                    r"|pipeline|project|generator|scoring|score|scores|search|ranking|extraction|classifier)")
# ... nor where it shares a software word with another noun: 'my resume to job matching service' (never 'my resume and
# two writing samples', 'my resume and project portfolio')
_RESUME_TECH = (r"(?:parser|parsing|screening|screener|builder|matching|matcher|scanner|database|generator|scoring"
                r"|extraction|classifier)")
# What 'the resume' does when it is the document: 'As the resume shows', 'the resume and portfolio links'
_RESUME_DOES = (r"(?:(?:shows?|includes?|lists?|details?|highlights?|outlines?|reflects?|demonstrates?"
                r"|summari[sz]es?|has|is|was|below|above)\b|(?:and|or)\s+(?:portfolio|cover\s+letter|references"
                rf"|transcripts?|(?:writing|work|code)\s+samples)\b(?!\s+{_RESUME_COMPOUND}\b))")
_NOT_CV_ALREADY = r"(?!\s*(?:[/(]|\b(?:or|and|aka)\b)\s*(?:CV|C\.V\.|curriculum\s+vitae)\b)"  # 'my resume/CV'
_CV_WORDING = re.compile(
    rf"\b(?P<own>my|your|attached|enclosed)\s+(?P<adj>{_RESUME_ADJECTIVE})r[eé]sum[eé]"
    rf"(?![\w-]|\s+{_RESUME_COMPOUND}\b|\s+(?:and|or|to)\s+(?:[\w-]+\s+){{1,2}}{_RESUME_TECH}\b){_NOT_CV_ALREADY}"
    rf"|\b(?P<the>the|this)\s+(?P<adj2>{_RESUME_ADJECTIVE})r[eé]sum[eé](?=\s*(?:[.,;:!?)]|$)|\s+{_RESUME_DOES})",
    re.IGNORECASE)


def _pairs() -> dict[str, str]:
    """Every American word in the lists, with its British form."""
    pairs: dict[str, str] = {}
    for prefix in PREFIXES:
        for stem in IZE_STEMS:
            for ending in IZE_ENDINGS:
                pairs[f"{prefix}{stem}iz{ending}"] = f"{prefix}{stem}is{ending}"
    for prefix in ("", "re", "over"):
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
# A title before a name, as it is written there ('Ms. Gray', 'Dr. Patel'): never 'ms' after a number or the 'st' of
# '1st'
_TITLES = frozenset({"Mr", "Mrs", "Ms", "Mx", "Dr", "Prof", "St", "Sr", "Jr", "Rev", "Hon"})
_TITLE_BEFORE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]+)\.$")
# A list item's marker, when it begins a line: '- ', an en or em dash, 'o ', '> ', '+ ', '1) '
_LIST_MARKER = re.compile(r"(?:[-\u2013\u2014>+o]|\d{1,2}\))$")


def _list_item(text: str, position: int) -> bool:
    """True when the text before `position` ends with a list marker that begins a line ('- Optimise data models')."""
    marker = _LIST_MARKER.search(text, max(0, position - 3), position)
    if marker is None:
        return False
    before = marker.start()
    while before > 0 and text[before - 1] in " \t":
        before -= 1
    return before == 0 or text[before - 1] in "\n\r"


def _starts_sentence(text: str, start: int) -> bool:
    """True when the word at `start` begins the text, a sentence, a line or a list item; never after 'Ms.' or 'Dr.',
    and not after "(": a bracket usually holds a name."""
    position = start
    while position > 0 and text[position - 1] in " \t":
        position -= 1
    if position == 0 or text[position - 1] in "\n\r\u2022\u00b7\u25aa\u25e6\u2023*":
        return True
    if position < start and _list_item(text, position):
        return True  # a job post's '- ' list: '- Optimise data models'
    mark = text[position - 1]
    if position == start or mark not in ".!?:;":
        return False
    title = _TITLE_BEFORE.search(text, max(0, position - 8), position)
    return not (mark == "." and title and title.group(1) in _TITLES)


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


@lru_cache(maxsize=8192)  # a long keep-list (project notes) would otherwise compile every pattern for every text
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


def _name_spans(text: str, names: Iterable[str]) -> list[tuple[int, int]]:
    """Where the names appear: a capitalised name only where it is capitalised ('Color' the employer, never 'lab color
    workflows')."""
    spans: list[tuple[int, int]] = []
    for name in sorted({n.strip() for n in names if n and n.strip()}):
        pattern = _phrase_regex(name)
        if pattern is not None:
            capital = name[:1].isupper()
            spans += [match.span() for match in pattern.finditer(text) if not capital or match.group(0)[:1].isupper()]
    return spans


def _exact_words(text: str, phrases: Iterable[str]) -> dict[int, str]:
    """{where a word starts: that word as the phrase spells it} for every word inside one of the phrases (the job
    post's keywords, which keep the post's spelling)."""
    found: dict[int, str] = {}
    for phrase in sorted({p.strip() for p in phrases if p and p.strip()}):
        pattern = _phrase_regex(phrase)
        if pattern is None:
            continue
        own = _WORD.findall(phrase)
        for match in pattern.finditer(text):
            for word, target in zip(_WORD.finditer(match.group(0)), own):
                found.setdefault(match.start() + word.start(), target)
    return found


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


_PACKAGE_PREFIXES = ("scikit-", "sklearn-", "py", "react-", "vue-", "ngx-", "node-", "django-", "flask-", "pytest-",
                     "eslint-", "babel-", "webpack-", "tf-", "torch")
_CODE_PARTS = frozenset({"html", "css", "js", "ts", "json", "yaml", "cli", "sdk", "api", "plugin", "loader", "lib",
                         "utils", "kit"})
# Folders that start a path: those only code uses (src/normalize), and those that are English words too, which start
# one only with two or more slashes (app/utils/normalize; 'build/optimization work' is a list)
_CODE_FOLDERS = frozenset({"src", "lib", "pkg", "cmd", "bin", "etc", "usr", "var", "dist", "utils", "util",
                           "node_modules", "vendor"})
_FOLDERS = frozenset({"app", "apps", "internal", "tests", "test", "docs", "home", "config", "scripts", "packages",
                      "modules", "build"})
_COMMAND_BEFORE = re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?$|\bnpx\s+$")
_MAKE_BEFORE = re.compile(r"\bmake\s+$")  # 'make optimize' runs a target; 'make personalized offers' is English
_CALL = re.compile(r"[A-Za-z0-9]\((?!(?:s|es|ies)\))")  # normalize(df), never 'organization(s)'
_INLINE_CODE = re.compile(r"`[^`\n]+`")  # `make optimize`
_BARE_VERB = re.compile(r"[a-z]+(?:i[sz]|y[sz])e")  # 'optimize': a script's, target's or package's name after a
# folder, a code word or 'make' ('scripts/optimize', 'json-normalize'), where English has a noun or an adjective
# ('build/optimization', 'json-serialized')


def _in_identifier(text: str, match: re.Match, title: bool) -> bool:
    """True when the word is part of code, a package, a path or a CSS name rather than English."""
    start = max(text.rfind(" ", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
    ends = [end for end in (text.find(" ", match.end()), text.find("\n", match.end())) if end != -1]
    raw = text[start: min(ends) if ends else len(text)]
    token = raw.strip("()[]{}\"'`,;:!?.")
    if "." in token or "_" in token or token.startswith(("-", "/", "@", "~")) or _CALL.search(raw):
        return True  # scipy.optimize, pg_catalog, --color-primary, /authorize, @angular/localize, normalize(df)
    parts = token.split("/")
    if len(parts) > 1 and (any(re.search(r"\d", part) for part in parts) or parts[0] in _CODE_FOLDERS or (
            parts[0] in _FOLDERS and (len(parts) > 2 or _BARE_VERB.fullmatch(parts[-1])))):
        return True  # a path (src/utils/normalize), never a list such as 'analysis/modeling/visualization'
    lower = token.islower()
    if lower and token.startswith(_PACKAGE_PREFIXES):  # scikit-optimize, react-color
        return True
    pieces = token.split("-")
    if lower and len(pieces) > 1 and _CODE_PARTS.intersection(pieces) and (
            pieces[-1] in _CODE_PARTS or len(pieces) > 2 or any(_BARE_VERB.fullmatch(piece) for piece in pieces)):
        return True  # sanitize-html, color-loader, json-normalize; never an English compound such as 'api-centered'
    before = text[max(0, start - 20): start]
    if _COMMAND_BEFORE.search(before) or (_MAKE_BEFORE.search(before) and _BARE_VERB.fullmatch(token)):
        return True  # npm run optimize, make optimize
    return title and lower and token.count("-") >= 2  # a lowercase package name in the skills list


def _with_case(word: str, template: str) -> str:
    return word[:1].upper() + word[1:] if template[:1].isupper() else word


_BRAND_LISTS = re.compile(r"(?<![\w-])(?:" + "|".join(sorted(map(re.escape, BRANDS), key=len, reverse=True))
                          + r")\s*(?:\([^()]*\)|:[^.;]*)")


def names_brand(text: str) -> bool:
    """True when the text (a skill group's name: 'AWS', 'Cloud (Azure)') names a brand."""
    return any(word in BRANDS for word in re.findall(r"[A-Za-z]+", text))


def _brand_before(text: str, words: list[re.Match], index: int) -> bool | None:
    """Whether a brand starts the run of capitalised words that leads to the word ('Unity' for 'Catalog' in 'Unity
    Catalog'); None when that run goes back to the start of the text without one."""
    position = index - 1
    while position >= 0:
        previous = words[position]
        if not _CAPITALISED_RUN_GAP.match(text[previous.end(): words[position + 1].start()]):
            return False
        if previous.group(0) in BRANDS:
            return True
        if not previous.group(0)[:1].isupper():
            return False
        position -= 1
    return None if not text[: words[0].start()].strip() else False


def _inside_name(text: str, words: list[re.Match], index: int) -> bool:
    """True when a capitalised word (or a brand) comes right before the word, in the same run: 'erwin Data Modeler',
    'React Fiber'; never 'Fibre broadband' at the start of a sentence."""
    if index == 0:
        return False
    previous = words[index - 1]
    return (_CAPITALISED_RUN_GAP.match(text[previous.end(): words[index].start()]) is not None
            and (previous.group(0)[:1].isupper() or previous.group(0) in BRANDS))


def _name_goes_on(text: str, words: list[re.Match], index: int) -> bool:
    """True when another capitalised word follows the word in the same run: it is inside a longer name ('CPLEX
    Optimization Studio', 'Cost Optimization Hub', 'User Behavior Analytics')."""
    if index + 1 >= len(words):
        return False
    following = words[index + 1]
    return (_CAPITALISED_RUN_GAP.match(text[words[index].end(): following.start()]) is not None
            and following.group(0)[:1].isupper())


def _after_brand(text: str, words: list[re.Match], index: int, lists: list[tuple[int, int]], brand: bool) -> bool:
    """True when the word names part of a brand's product: in the run of capitalised words a brand starts ('Unity
    Catalog', or inside a longer name: 'CPLEX Optimization Studio'), in a list right after a brand ('AWS
    (Organizations, Control Tower)', 'AWS: Compute Optimizer') or, with `brand`, as the last word of a one- or
    two-word skill in a group named after one ('Organizations', 'Compute Optimizer'; never 'Data Center Migration',
    'Data Centers' or 'Color Grading', which are activities)."""
    lower = words[index].group(0).lower()
    thing = _PRODUCT_WORD.search(lower) is not None
    if thing and _inside(words[index].start(), lists):
        return True
    found = _brand_before(text, words, index)
    if found:
        return thing or _name_goes_on(text, words, index)
    if found is None and brand and thing:
        head = _WORD.findall(text.split("(")[0])
        return len(head) <= 2 and index == len(head) - 1 and (not lower.endswith("s") or lower.startswith("organi"))
    return False


def _starts_longer_title(text: str, match: re.Match, title: bool, role_spans: list[tuple[int, int]]) -> bool:
    """True when the word is followed by more of a title, skill or role: 'Analog' in 'Analog Design Engineer' or
    'Analog Circuit Design' is a word of the job, even where a one-word name is spelled the same ('Analog' for Analog
    Devices); 'Engineer at Color' still ends with the name."""
    if any(start <= match.start() and match.end() < end for start, end in role_spans):
        return True
    return title and re.match(r"[ \t-]+[A-Za-z]", text[match.end():]) is not None


def _convert(text: str, style: str, keep: Iterable[str] = (), terms: Iterable[str] = (), title: bool = False, *,
             exact: Iterable[str] = (), roles: Iterable[str] = (),
             brand: bool = False) -> tuple[str, list[tuple[int, str, str]]]:
    """(text in the style, [(where the word starts, word, new word), ...]); see convert."""
    table = TABLES[style]
    if not text:
        return text, []
    words = list(_WORD.finditer(text))
    exact_words = _exact_words(text, exact) if exact else {}
    if not exact_words and not any(match.group(0).lower() in table for match in words):
        return text, []
    names = [*keep, *PRODUCT_NAMES]
    kept = _name_spans(text, [name for name in names if len(_WORD.findall(name)) > 1])
    one_word = _name_spans(text, [name for name in names if len(_WORD.findall(name)) == 1])
    term_spans = _spans(text, terms)
    starts = {match.start(): index for index, match in enumerate(words)}
    # The role, unless a brand comes right before it: 'Data Modeler' is the job, 'erwin Data Modeler' a product
    role_spans = [span for span in _spans(text, roles)
                  if span[0] not in starts or not _brand_before(text, words, starts[span[0]])]
    lists = [match.span() for match in _BRAND_LISTS.finditer(text)]
    code = [match.span() for match in _INLINE_CODE.finditer(text)]
    parts: list[str] = []
    changes: list[tuple[int, str, str]] = []
    last = 0
    for index, match in enumerate(words):
        word, lower = match.group(0), match.group(0).lower()
        if any(ch.isupper() for ch in word[1:]) or _inside(match.start(), code):
            continue  # ALL CAPS, camelCase or inline code (`make optimize`)
        role = _inside(match.start(), role_spans)  # the job ('Data Modeler'): the style's spelling, before a keyword's
        if _inside(match.start(), kept) or (_inside(match.start(), one_word)
                                            and not _starts_longer_title(text, match, title, role_spans)):
            continue  # a name; a one-word name ('Analog' for Analog Devices) never at the head of a job or a skill
        target = None if role else exact_words.get(match.start())
        if target is not None:  # a word of one of the job post's keywords: the post's own spelling
            new = target.lower()
            if new == lower or new not in twins(word) or _in_identifier(text, match, title):
                continue
            if word[0].isupper() and ((lower in NAME_PRONE and _inside_name(text, words, index))
                                      or _after_brand(text, words, index, lists, brand)):
                continue  # a product's name holding the keyword: 'AWS Glue Data Catalog' for 'data catalogue'
        else:
            new = table.get(lower)
            if new is None or _in_identifier(text, match, title):
                continue  # not a listed word; 'scipy.optimize', 'pg_catalog', 'webpack-bundle-analyzer'
            if word[0].isupper():
                if not role and (lower in NAME_PRONE or _after_brand(text, words, index, lists, brand)):
                    continue
                if not (title or role or _starts_sentence(text, match.start())
                        or _inside(match.start(), term_spans)):
                    continue  # a capitalised word inside a sentence is a name, unless it is a skill or the role
        new = _with_case(new, word)
        parts += [text[last: match.start()], new]
        last = match.end()
        changes.append((match.start(), word, new))
    parts.append(text[last:])
    return "".join(parts), changes


def convert_and_list(text: str, style: str, keep: Iterable[str] = (), terms: Iterable[str] = (),
                     title: bool = False, *, exact: Iterable[str] = (), roles: Iterable[str] = (),
                     brand: bool = False) -> tuple[str, list[tuple[str, str]]]:
    """(text in the style, [(word, new word), ...]); see convert."""
    new_text, changes = _convert(text, style, keep, terms, title, exact=exact, roles=roles, brand=brand)
    return new_text, [(word, new) for _start, word, new in changes]


def convert(text: str, style: str, keep: Iterable[str] = (), terms: Iterable[str] = (), title: bool = False, *,
            exact: Iterable[str] = (), roles: Iterable[str] = (), brand: bool = False) -> str:
    """The text in the style ("us" or "eu"). `keep`: names never changed, where capitalised; a one-word name still
    changes at the head of a job or a skill ('Analog Design Engineer' for Analog Devices). `terms`: skill terms,
    changed even when capitalised inside a sentence. `title`: a heading, skill or job title,
    where capitals do not mark a name. `exact`: the job post's keywords, written as the post spells them, except in
    the role and inside a product's name. `roles`: the role and the job titles, always in the style and never a
    product's name. `brand`: a skill in a group named after a brand ('AWS': 'Organizations')."""
    return convert_and_list(text, style, keep, terms, title, exact=exact, roles=roles, brand=brand)[0]


def other_style_words(text: str, style: str, keep: Iterable[str] = (), terms: Iterable[str] = (),
                      title: bool = False, *, roles: Iterable[str] = (), brand: bool = False,
                      exact: Iterable[str] = ()) -> list[str]:
    """Words in the text that are spelled in the other style (what convert would change to the style). `exact`: the
    job post's keywords, which keep the post's spelling on purpose: a word written as the post spells it is never
    listed, in whatever case."""
    posting = _exact_words(text, exact) if exact else {}
    changes = _convert(text, style, keep, terms, title, roles=roles, brand=brand)[1]
    return [word for start, word, _new in changes if posting.get(start, "").lower() != word.lower()]


def detect(text: str) -> str:
    """The spelling a text is written in: "us", "eu", or "" when it does not clearly show one."""
    us = eu = 0
    for match in _WORD.finditer(text):
        lower = match.group(0).lower()
        if match.group(0)[:1].isupper() and not _starts_sentence(text, match.start()):
            continue  # a name: AWS Organizations, Azure Security Center
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
    if style != "eu":
        return text
    return _CV_WORDING.sub(lambda m: f"{m.group('own') or m.group('the')} {m.group('adj') or m.group('adj2') or ''}CV",
                           text)
