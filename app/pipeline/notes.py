"""The person's project notes (Profile page, one text per employer) as the app uses them. Notes are often pasted from
research, with links and lists of sources: the model reads only the projects (clean_notes), and the checks trust only
what the model reads (validate.project_notes)."""
from __future__ import annotations

import re

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)|<?https?://\S+|\bwww\.\S+", re.IGNORECASE)
_SOURCE_WORDS = (r"(?:sources?(?: list| used)?|links?|useful links|references?(?: used)?|citations?|works cited"
                 r"|bibliography|further reading|reading list|resources|press coverage)")
# a heading over a list of sources: 'Sources', '**Sources**', '#### **Works cited**', 'Links and references:', 'Sources (2)'
_SOURCES_HEADING = re.compile(rf"^\W*{_SOURCE_WORDS}(?:\s*(?:and|&)\s*{_SOURCE_WORDS})?(?:\s*\(\d+\))?\W*$",
                              re.IGNORECASE)
_SOURCE_LINE = re.compile(rf"^\W*{_SOURCE_WORDS}\s*:\s*\S", re.IGNORECASE)  # 'Source: the company blog'
_CITED = re.compile(r"\b(?:accessed|retrieved)\b", re.IGNORECASE)  # 'Title, accessed March 3, 2026, https://...'
_LIST_ITEM = re.compile(r"^\s*(?:[-*•+▪◦‣]|\(?(?:\d{1,3}|[a-z])[.)])\s")
_MARKS_ONLY = re.compile(r"^\s*(?:[-*•+▪◦‣_=#>|:]+|\(?\d{1,3}[.)]?)\s*$")  # '-', '---', '3.', '(4)', '|---|'


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def _links_only(line: str) -> bool:
    """A line with nothing but links on it (and a list mark or a '[1]'): a source's address."""
    rest = re.sub(r"^\s*(?:\[\d{1,3}\]|\(?(?:\d{1,3}|[a-z])[.)]|[-*•+▪◦‣])?", "", _LINK.sub("", line))
    return bool(_LINK.search(line)) and not re.search(r"\w", rest)


def _sources_end(lines: list[str], start: int) -> int:
    """Where the list of sources under the heading at `start` ends. A heading that is itself a list item ('  - Sources:'
    under a project) covers the lines indented under it and link items at its own level. Any other heading covers
    list items, link lines, a title line over its link line and blank lines, and ends at the next other line, at a
    list item less indented than the first source, or, after a blank line, at a list item without a link when every
    source had one (the person's projects again)."""
    nested, level = bool(_LIST_ITEM.match(lines[start])), _indent(lines[start])
    first: int | None = None  # the first source's indent
    linked = True  # every source so far had a link

    def size(i: int, after_blank: bool) -> int:  # how many lines from i belong to the list (0: it has ended)
        line = lines[i]
        item, link, indent = bool(_LIST_ITEM.match(line)), bool(_LINK.search(line)), _indent(line)
        if nested:
            return 1 if indent > level or (item and link and indent == level) else 0
        if item or link:
            if item and first is not None and indent < first:
                return 0
            return 0 if after_blank and item and not link and linked else 1
        return 2 if i + 1 < len(lines) and _links_only(lines[i + 1]) else 0

    i = start + 1
    while i < len(lines):
        after_blank = not lines[i].strip()
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j == len(lines):
            return j
        taken = size(j, after_blank)
        if not taken:
            return i
        first = _indent(lines[j]) if first is None else first
        linked = linked and any(_LINK.search(line) for line in lines[j:j + taken])
        i = j + taken
    return i


def clean_notes(text: str) -> str:
    """One employer's notes as the model reads them: no links (a Markdown link keeps its words), no list of sources
    (see _sources_end), no 'Source: ...' line or citation ('..., accessed March 3, 2026, https://...'), no line left
    with only a list mark or a number, and at most one blank line in a row."""
    lines = text.strip().splitlines()
    kept: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _SOURCES_HEADING.match(line):
            i = _sources_end(lines, i)
            continue
        i += 1
        if _SOURCE_LINE.match(line) or (_CITED.search(line) and _LINK.search(line)):
            continue
        line = _LINK.sub(lambda match: match.group(1) or "", line).rstrip()
        if line.strip() and _MARKS_ONLY.match(line):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
