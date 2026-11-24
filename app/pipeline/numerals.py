"""Numbers as numerals, and every figure written one way (the reviewer's rule, prompt v38): "7 schools" and "under 5
days", never "seven" or "five"; "500K+" (never "500k+"), "$1.2M", "$2M" (never "$2 million"), "38%" (never "38 percent"),
"684ms" and "1.7TB" (no space, as the prompt's own examples write them).

Only lowercase number words from two to ninety-nine are changed: never "one" (except in a range such as "one or two"),
"zero", an ordinal ("first"), a capitalised name ("Two Sigma", "Three.js"), a compound ("two-factor", "one-on-one",
"three-tier"), a word that is part of a name, a bigger number or a fraction ("three.js", "two's complement", "two
hundred", "two thirds", "a two million-dollar contract") or a phrase the caller keeps (the posting's own terms). The
trust checks read the person's hints and notes through numerals() too, so a "7" written for their "seven" is still their
figure (app/pipeline/validate.py).
"""
from __future__ import annotations

import re

_ONES = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ALL = {"one": 1, **_ONES, **_TEENS, **_TENS}
_WORD = r"(?:" + "|".join(sorted(_ALL, key=len, reverse=True)) + r")"
# 'one or two', 'two to three', 'two and three': a range of number words, 'one' included; else a number word, or tens
# and ones ('twenty-five', 'twenty five'), on its own: never inside a word, a compound or a name. One pass, the range
# first, so a range left as written is never half changed.
_NUMBER_WORD = re.compile(
    rf"(?<![\w-])(?P<low>{_WORD})(?P<link>\s+(?:or|to|and)\s+)(?P<high>{_WORD})(?![\w-])"
    rf"|(?<![\w-])(?P<tens>{'|'.join(_TENS)})(?:[- ](?P<ones>{'|'.join(_ONES)}|one))?(?![\w-])"
    rf"|(?<![\w-])(?P<word>{_WORD})(?![\w-])"
)
# what makes a number word part of more: a name ('three.js'), a possessive ("two's complement"), a bigger number ('two
# hundred', 'two million-dollar') or a fraction ('two thirds'; but 'two third-party APIs' are two)
_PART_OF_MORE = re.compile(
    r"\.[A-Za-z]|['’]s\b|\s+(?:(?:hundreds?|thousands?|millions?|billions?)\b|(?:dozens?|score|half|halves|thirds?"
    r"|quarter|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?|hundredths?|thousandths?)(?![\w-]))", re.IGNORECASE)
# quarters are a time span ('within two quarters' -> 'within 2 quarters') unless they can be a fraction: 'three
# quarters' (3/4, or 9 months), 'two quarters of the budget'
_QUARTERS = re.compile(r"\s+quarters(?![\w-])(?P<of>\s+of\b)?", re.IGNORECASE)
# right after a bigger number in words: the 'twenty' of 'one hundred twenty', 'a thousand and five' (never the 'five' of
# '$2 million and five weeks')
_AFTER_SCALE = re.compile(rf"\b(?:{_WORD}|a)[\s-]+(?:hundred|thousand|million|billion)(?:\s+and)?[\s-]+$",
                          re.IGNORECASE)


def _value(match: re.Match) -> int | None:
    if match.group("tens"):
        return _TENS[match.group("tens")] + (_ALL[match.group("ones")] if match.group("ones") else 0)
    word = match.group("word")
    return None if word == "one" else _ALL[word]


def _part_of_more(match: re.Match) -> bool:
    """The number word (or range) belongs to a name, a bigger number or a fraction: it stays as written."""
    text, start = match.string, match.start()
    quarters = _QUARTERS.match(text, match.end())
    return (_PART_OF_MORE.match(text, match.end()) is not None
            or (quarters is not None and (quarters.group("of") is not None or match.group(0).endswith("three")))
            or _AFTER_SCALE.search(text, max(0, start - 40), start) is not None)


def _kept_spans(text: str, keep: list[str]) -> list[tuple[int, int]]:
    spans = []
    for phrase in keep:
        if not phrase or not re.search(_WORD, phrase.lower()):
            continue
        for found in re.finditer(re.escape(phrase), text, re.IGNORECASE):
            spans.append(found.span())
    return spans


def numerals(text: str, keep: list[str] | None = None) -> str:
    """The text with its number words as numerals: 'seven international schools' -> '7 international schools',
    'one or two releases' -> '1 or 2 releases'. Never 'one' alone, 'zero', a capitalised word, a compound
    ('two-factor'), a word that is part of more ('three.js', 'two hundred', 'two thirds'), or a word inside a phrase of
    `keep` (the posting's own terms, the skills)."""
    kept = _kept_spans(text, keep or [])

    def written(match: re.Match) -> str:
        if _part_of_more(match) or any(s < match.end() and match.start() < e for s, e in kept):
            return match.group(0)
        if match.group("low"):
            if match.group("low") == match.group("high") == "one":  # 'one to one' stays
                return match.group(0)
            return f"{_ALL[match.group('low')]}{match.group('link')}{_ALL[match.group('high')]}"
        value = _value(match)
        return match.group(0) if value is None else str(value)

    return _NUMBER_WORD.sub(written, text)


_UNIT_WORDS = re.compile(  # never a compound: 'a 2 million-dollar contract', 'a 5 million dollar budget'
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>million|mn|billion|bn|thousand|per\s?cent|percent)\b"
    r"(?![-‐]\w|\s+(?:dollar|euro|pound)s?\b)", re.IGNORECASE)
_SHORT = {"million": "M", "mn": "M", "billion": "B", "bn": "B", "thousand": "K", "percent": "%", "per cent": "%",
          "percent ": "%", "percent.": "%"}
_LOWER_K = re.compile(r"(?<=\d)k\b(?!\s*(?:m|g|b)\b)")  # 500k -> 500K; never 5km, 5kg or 5kb
_MONEY_SUFFIX = re.compile(r"(?<=[$€£])(\d[\d,]*(?:\.\d+)?)([mb])\b")  # $1.2m -> $1.2M (alone, m is minutes or meters)
_PERCENT_SPACE = re.compile(r"(?<=\d)\s+%")
_UNIT_SPACE = re.compile(r"(?<=\d)\s+(ms|TB|GB|MB|KB|PB)\b")  # 684 ms -> 684ms, 1.7 TB -> 1.7TB


def figure_format(text: str) -> str:
    """Every figure written the same way: '500k+' -> '500K+', '$1.2m' -> '$1.2M', '$2 million' -> '$2M', '38 percent'
    -> '38%', '38 %' -> '38%', '684 ms' -> '684ms', '1.7 TB' -> '1.7TB'."""
    def short(match: re.Match) -> str:
        unit = match.group("unit").lower()
        return match.group("num") + ("%" if "cent" in unit else _SHORT[unit])

    text = _UNIT_WORDS.sub(short, text)
    text = _LOWER_K.sub("K", text)
    text = _MONEY_SUFFIX.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    text = _PERCENT_SPACE.sub("%", text)
    return _UNIT_SPACE.sub(r"\1", text)


def normal(text: str, keep: list[str] | None = None) -> str:
    """numerals, then figure_format."""
    return figure_format(numerals(text, keep))
