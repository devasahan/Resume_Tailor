"""The person's project notes (Profile page, one text per employer) as the app uses them. Notes are often pasted from
research, with links and a list of sources: the model reads only the projects (clean_notes), and the checks trust only
what the model reads (validate.project_notes)."""
from __future__ import annotations

import re

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)|<?https?://\S+|\bwww\.\S+", re.IGNORECASE)
_SOURCE_WORDS = r"(?:sources?|links?|references?|citations?|bibliography|further reading|reading list)"
# a heading over a list of sources: 'Sources', '**Sources**', '## Links and references', 'References:'
_SOURCES_HEADING = re.compile(rf"^\W*{_SOURCE_WORDS}(?:\s*(?:and|&)\s*{_SOURCE_WORDS})?\W*$", re.IGNORECASE)
_SOURCE_LINE = re.compile(rf"^\W*{_SOURCE_WORDS}\s*:\s*\S", re.IGNORECASE)  # 'Source: the company blog'
_LIST_ITEM = re.compile(r"^\s*(?:[-*•+▪◦‣]|\(?\d{1,3}[.)])\s")
_MARKS_ONLY = re.compile(r"^\s*(?:[-*•+▪◦‣_=#>|:]+|\(?\d{1,3}[.)]?)\s*$")  # '-', '---', '3.', '(4)', '|---|'


def clean_notes(text: str) -> str:
    """One employer's notes as the model reads them: no links (a Markdown link keeps its words), no list of sources
    (a 'Sources' heading and the list items, link lines and blank lines after it, up to the next other line), no
    'Source: ...' line, no line left with only a list mark or a number, and at most one blank line in a row."""
    lines: list[str] = []
    in_sources = False
    for line in text.strip().splitlines():
        if _SOURCES_HEADING.match(line):
            in_sources = True
            continue
        if in_sources and (not line.strip() or _LINK.search(line) or _LIST_ITEM.match(line)):
            continue
        in_sources = False
        if _SOURCE_LINE.match(line):
            continue
        line = _LINK.sub(lambda match: match.group(1) or "", line).rstrip()
        if line.strip() and _MARKS_ONLY.match(line):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
