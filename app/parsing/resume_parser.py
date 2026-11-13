"""Uploaded base resume (PDF / DOCX / TXT) -> text -> Profile.

A model does the extraction when one is chosen (the resume text IS sent to that provider). If none is
chosen, or the model fails or the daily cost cap is reached, the rule-based parser runs instead. An
upload is saved as read; the Profile page lists it with warnings, and a bad read is fixed by deleting
it and uploading again (or choosing another reader in Settings).
"""
from __future__ import annotations

import io
import re
import zipfile

import pdfplumber
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from ..llm.base import CostCapError
from ..llm.router import Router
from ..llm.schema import strict_json_schema
from ..pipeline.generate import GenerationFailed, Outcome, generate_validated
from ..prompts import parse_system, parse_user
from ..schemas import Profile, TailorResult
from ..storage.db import Database
from .heuristic import _is_header, parse_resume_text

MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 24_000
MAX_PDF_PAGES = 10
MAX_UNZIPPED_BYTES = 50 * 1024 * 1024  # a DOCX is a zip: refuse ones that expand into something huge


_NUM_ID = qn("w:pPr") + "/" + qn("w:numPr") + "/" + qn("w:numId")


def _list_styles(doc) -> tuple[dict[str, tuple[bool, bool] | None], tuple[bool, bool]]:
    """Each style id's (list style, drawn with a mark), read in one pass over the styles: a list style's name starts
    with 'List' ('List Bullet', 'List Paragraph'); it is drawn with a mark when it or a style it is based on has
    numbering (numId 0 is none). None for a style of another type. And the same for the default paragraph style."""
    found: dict[str, tuple[bool, str | None, str | None, bool]] = {}
    default_id = None
    try:
        elements = doc.styles.element.findall(qn("w:style"))
    except Exception:  # noqa: BLE001  any styles part python-docx cannot read: the text is still read, unmarked
        elements = []
    for element in elements:
        style_id = element.get(qn("w:styleId"))
        if style_id is None or style_id in found:  # the first style with an id counts, as in python-docx
            continue
        name = element.find(qn("w:name"))
        num = element.find(_NUM_ID)
        base = element.find(qn("w:basedOn"))
        paragraph = element.get(qn("w:type")) == "paragraph"
        found[style_id] = (name is not None and (name.get(qn("w:val")) or "").startswith("List"),
                           None if num is None else num.get(qn("w:val")), None if base is None else base.get(qn("w:val")),
                           paragraph)
        if paragraph and element.get(qn("w:default")) in ("1", "true", "on"):
            default_id = style_id

    def drawn(style_id: str | None) -> bool:
        for _ in range(20):  # a chain of styles each based on the next
            entry = found.get(style_id) if style_id else None
            if entry is None:
                return False
            if entry[1] is not None:
                return entry[1] != "0"
            style_id = entry[2]
        return False

    by_id = {sid: (entry[0], drawn(sid)) if entry[3] else None for sid, entry in found.items()}
    return by_id, (by_id.get(default_id) or (False, False)) if default_id else (False, False)


def _paragraph_text(paragraph: Paragraph, styles: tuple[dict, tuple[bool, bool]]) -> str:
    """A Word paragraph's text, with '• ' before it when Word draws a bullet or number before it in a list style ('List
    Bullet', 'List Paragraph' with numbering): the text has no mark, and the reader knows a bullet by its mark. A list
    style without numbering (a bullet taken off) has none on the page, and a numbered section heading stays one."""
    pstyle = paragraph._p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    style_id = pstyle.get(qn("w:val")) if pstyle is not None else None
    listed = styles[0].get(style_id) if style_id else None
    if listed is None:  # no style, one not defined, or one of another type: the default paragraph style
        listed = styles[1]
    text = paragraph.text
    if not (listed[0] and text.strip()):
        return text
    own = paragraph._p.find(_NUM_ID)
    drawn = own.get(qn("w:val")) != "0" if own is not None else listed[1]
    # a section heading Word numbers ("1. Experience"), the whole paragraph: not "Skills: Python, SQL"
    heading = len(text.split()) <= 3 and not re.search(r"[:,;]", text.strip(" :")) and _is_header(text) is not None
    return f"• {text}" if drawn and not heading else text


def extract_text(data: bytes, filename: str) -> str:
    if len(data) > MAX_BYTES:
        raise ValueError("File is larger than 5 MB.")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages[:MAX_PDF_PAGES])
        except Exception as exc:
            raise ValueError(f"Could not read the PDF: {exc}") from exc
    elif ext == "docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if sum(info.file_size for info in archive.infolist()) > MAX_UNZIPPED_BYTES:
                    raise ValueError("The DOCX expands to an unreasonable size.")
            doc = Document(io.BytesIO(data))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read the DOCX: {exc}") from exc
        styles = _list_styles(doc)
        lines = [_paragraph_text(p, styles) for p in doc.paragraphs]
        for table in doc.tables:  # some resumes keep contact info or dates in tables
            for row in table.rows:
                lines.extend(cell.text for cell in row.cells)
        text = "\n".join(lines)
    elif ext in ("txt", "md", ""):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError("Upload a PDF, DOCX or TXT file.")
    text = text.strip()
    if len(text) < 30:
        raise ValueError("No readable text found. Scanned PDFs (images) are not supported; export a text PDF or DOCX.")
    return text[:MAX_CHARS]


def profile_warnings(profile: Profile) -> list[str]:
    warnings: list[str] = []
    if not profile.name.strip():
        warnings.append("Name not found.")
    if not profile.contact.email:
        warnings.append("Email not found.")
    if not profile.contact.phone:
        warnings.append("Phone not found.")
    if not profile.companies:
        warnings.append("No employers found. Add each company with its dates.")
    for company in profile.companies:
        if not company.start or not company.end:
            warnings.append(f"{company.name}: start or end date is missing.")
        if not company.mode:
            warnings.append(f"{company.name}: work mode (Remote / Onsite / Hybrid) is blank, so it won't be shown.")
    if not profile.education:
        warnings.append("No education found.")
    return warnings


async def read_resume(
    router: Router, db: Database, text: str, model_id: str | None, *, user_id: int | None = None
) -> tuple[Profile, str, list[str]]:
    """Return (profile, method, notes). `method` is 'model:<id>' or 'rules'; notes say why a model was not used."""
    method = "rules"
    notes: list[str] = []
    profile: Profile | None = None
    if model_id:
        outcome: Outcome = Outcome()
        try:
            await generate_validated(
                router, [model_id], system=parse_system(), user=parse_user(text),
                schema=strict_json_schema(Profile), schema_name="profile",
                parse=lambda data: Profile.model_validate(data), context={"resume_text": text}, outcome=outcome,
            )
            profile, method = outcome.value, f"model:{model_id}"  # type: ignore[assignment]
            for company in profile.companies:  # type: ignore[union-attr]
                company.projects = ""  # only the person types these, on the Profile page
        except GenerationFailed as exc:
            outcome = exc.outcome
            notes.append(f"The model could not parse the resume ({str(exc)[:160]}). Used the rule-based parser instead.")
        except CostCapError as exc:
            notes.append(f"{str(exc)[:200]} Used the rule-based parser instead.")
        record_cost(db, model_id, outcome, user_id)
    if profile is None:
        profile = parse_resume_text(text)
    return profile, method, notes


async def parse_resume(
    router: Router, db: Database, text: str, model_id: str | None, *, user_id: int | None = None
) -> tuple[Profile, str, list[str]]:
    """Return (profile, method, warnings): read_resume's notes plus profile_warnings."""
    profile, method, notes = await read_resume(router, db, text, model_id, user_id=user_id)
    return profile, method, notes + profile_warnings(profile)


def record_cost(db: Database, model_id: str, outcome: Outcome, user_id: int | None = None, task: str = "parse") -> None:
    """A row with what a model call outside a tailoring run cost, so the daily cost cap counts it."""
    if not outcome.input_tokens and not outcome.output_tokens:
        return
    record = TailorResult(
        requested_model=model_id, served_model=outcome.served_model, status="ok" if outcome.value else "error",
        input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens,
        cost_usd=round(outcome.cost_usd, 6), attempts=outcome.attempts, fallback_chain=outcome.chain,
    )
    db.save_run(record, None, "", task=task, user_id=user_id)
