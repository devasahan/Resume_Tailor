"""HTML (Jinja2) -> PDF (WeasyPrint). Single column, real text, standard fonts: what ATS parsers read best."""
from __future__ import annotations

import threading
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from ..schemas import Contact, CoverLetter, FitStyle, Resume

_render_lock = threading.Lock()  # WeasyPrint/Pango are not guaranteed thread-safe; rendering is quick

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=select_autoescape(["html", "j2"]),
)


def contact_line(contact: Contact) -> str:
    parts = [contact.email, contact.phone, contact.linkedin, contact.location, contact.website]
    return " | ".join(part for part in parts if part)


def fmt_dates(start: str, end: str) -> str:
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


def render_resume_html(resume: Resume, style: FitStyle) -> str:
    return _env.get_template("resume.html.j2").render(
        resume=resume, p=resume.profile, style=style, contact=contact_line(resume.profile.contact), dates=fmt_dates
    )


def render_resume_pdf(resume: Resume, style: FitStyle) -> tuple[bytes, int]:
    """Return (pdf bytes, page count)."""
    html = render_resume_html(resume, style)
    with _render_lock:
        document = HTML(string=html).render()
        return document.write_pdf(), len(document.pages)


def render_cover_pdf(profile_resume: Resume, letter: CoverLetter, paper: str = "Letter") -> tuple[bytes, int]:
    today = date.today()
    html = _env.get_template("cover.html.j2").render(
        p=profile_resume.profile,
        letter=letter,
        paper=paper,
        contact=contact_line(profile_resume.profile.contact),
        today=f"{today:%B} {today.day}, {today.year}",
    )
    with _render_lock:
        document = HTML(string=html).render()
        return document.write_pdf(), len(document.pages)
