"""Make the resume fit on exactly one page, in the least destructive order:

  1. shrink font and margins down a ladder (never below 9.5 pt / 0.5 in)
  2. (done by the caller) ask the model once for shorter bullets
  3. drop the last bullet of the roles with the most bullets, oldest first, keeping 2 per role
"""
from __future__ import annotations

from ..render.pdf import render_resume_pdf
from ..schemas import FitStyle, Resume

LADDER = [(10.5, 0.65), (10.25, 0.6), (10.0, 0.55), (9.75, 0.5), (9.5, 0.5)]


def styles(paper: str) -> list[FitStyle]:
    return [FitStyle(font_pt=f, margin_in=m, paper=paper) for f, m in LADDER]  # type: ignore[arg-type]


def try_styles(resume: Resume, paper: str = "Letter") -> tuple[FitStyle, int, bytes, list[str]] | None:
    """Return the first style that renders on one page, or None."""
    ladder = styles(paper)
    for index, style in enumerate(ladder):
        pdf, pages = render_resume_pdf(resume, style)
        if pages == 1:
            notes = []
            if index:
                notes.append(f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in to fit one page.")
            return style, pages, pdf, notes
    return None


def render_smallest(resume: Resume, paper: str = "Letter") -> tuple[FitStyle, int, bytes]:
    style = styles(paper)[-1]
    pdf, pages = render_resume_pdf(resume, style)
    return style, pages, pdf


def trim_to_fit(resume: Resume, paper: str = "Letter") -> tuple[Resume, FitStyle, int, bytes, list[str]]:
    """Remove bullets until one page (or nothing more can be removed). Works on a copy."""
    work = resume.model_copy(deep=True)
    removed: list[str] = []
    while True:
        style, pages, pdf = render_smallest(work, paper)
        if pages == 1:
            break
        candidates = [e for e in work.experience if len(e.bullets) > 2]
        if not candidates:
            break
        most = max(len(e.bullets) for e in candidates)
        target = [e for e in candidates if len(e.bullets) == most][-1]  # the oldest among the longest
        company = work.profile.companies[target.company_index].name
        removed.append(f"{company}: {target.bullets.pop()}")
    notes = [f"Font reduced to {style.font_pt:g} pt and margins to {style.margin_in:g} in."]
    if removed:
        notes.append(f"Removed {len(removed)} bullet(s) to fit one page: " + " / ".join(r[:60] + "..." for r in removed))
    if pages != 1:
        notes.append("Still more than one page after trimming. Shorten the text in the editor.")
    return work, style, pages, pdf, notes
