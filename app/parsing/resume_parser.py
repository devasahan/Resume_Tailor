"""Uploaded base resume (PDF / DOCX / TXT) -> text -> Profile.

A model does the extraction when one is chosen (the resume text IS sent to that provider). If none is
chosen, or the model fails, the rule-based parser runs instead. The user always reviews and corrects
the result on the Profile page before it is saved and locked.
"""
from __future__ import annotations

import io

import pdfplumber
from docx import Document

from ..llm.router import Router
from ..llm.schema import strict_json_schema
from ..pipeline.generate import GenerationFailed, Outcome, generate_validated
from ..prompts import parse_system, parse_user
from ..schemas import Profile, TailorResult
from ..storage.db import Database
from .heuristic import parse_resume_text

MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 24_000


def extract_text(data: bytes, filename: str) -> str:
    if len(data) > MAX_BYTES:
        raise ValueError("File is larger than 5 MB.")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as exc:
            raise ValueError(f"Could not read the PDF: {exc}") from exc
    elif ext == "docx":
        try:
            doc = Document(io.BytesIO(data))
        except Exception as exc:
            raise ValueError(f"Could not read the DOCX: {exc}") from exc
        lines = [p.text for p in doc.paragraphs]
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


async def parse_resume(
    router: Router, db: Database, text: str, model_id: str | None
) -> tuple[Profile, str, list[str]]:
    """Return (profile, method, warnings). `method` is 'model:<id>' or 'rules'."""
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
        except GenerationFailed as exc:
            outcome = exc.outcome
            notes.append(f"The model could not parse the resume ({str(exc)[:160]}). Used the rule-based parser instead.")
        _record_cost(db, model_id, outcome)
    if profile is None:
        profile = parse_resume_text(text)
    return profile, method, notes + profile_warnings(profile)


def _record_cost(db: Database, model_id: str, outcome: Outcome) -> None:
    if not outcome.input_tokens and not outcome.output_tokens:
        return
    record = TailorResult(
        requested_model=model_id, served_model=outcome.served_model, status="ok" if outcome.value else "error",
        input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens,
        cost_usd=round(outcome.cost_usd, 6), attempts=outcome.attempts, fallback_chain=outcome.chain,
    )
    db.save_run(record, None, "", task="parse")
