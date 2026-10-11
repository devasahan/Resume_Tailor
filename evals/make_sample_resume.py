"""Regenerates the sample base resume (TXT, DOCX, PDF) used to try the app.

    python evals/make_sample_resume.py

The sample holds only the fixed facts, like the base resume the app expects: name, contact details,
employers with dates and work mode, and education. Everything in it is invented.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402
from docx.shared import Pt  # noqa: E402

from app.render.pdf import Run, Sheet  # noqa: E402

HERE = Path(__file__).resolve().parent

NAME = "Jordan Rivera"
CONTACT = "Austin, TX | jordan.rivera@example.com | (555) 010-2030 | linkedin.com/in/jordanrivera"
EMPLOYERS = [
    ("Northwind Labs | Remote", "Mar 2022 - Present"),
    ("Contoso Systems | Dallas, TX | Onsite", "Jun 2019 - Feb 2022"),
    ("Fabrikam Inc. | Hybrid", "Jul 2016 - May 2019"),
]
EDUCATION = [("University of Texas at Austin", "B.S. in Computer Science, 2016")]


def write_txt() -> None:
    lines = [NAME, CONTACT, "", "EXPERIENCE"]
    for employer, dates in EMPLOYERS:
        lines += [employer, dates]
    lines += ["", "EDUCATION"]
    for school, degree in EDUCATION:
        lines += [school, degree]
    (HERE / "sample_base_resume.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_docx() -> None:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    doc.add_paragraph().add_run(NAME).bold = True
    doc.add_paragraph(CONTACT)
    doc.add_paragraph().add_run("EXPERIENCE").bold = True
    for employer, dates in EMPLOYERS:
        doc.add_paragraph().add_run(employer).bold = True
        doc.add_paragraph(dates)
    doc.add_paragraph().add_run("EDUCATION").bold = True
    for school, degree in EDUCATION:
        doc.add_paragraph().add_run(school).bold = True
        doc.add_paragraph(degree)
    doc.save(HERE / "sample_base_resume.docx")


def write_pdf() -> None:
    sheet = Sheet("Letter", 54, 11, 1.4, f"{NAME} - Base resume", NAME)
    sheet.para([Run(NAME, True)], size=16, after=2)
    sheet.para([Run(CONTACT)], after=10)
    sheet.para([Run("EXPERIENCE", True)], after=2)
    for employer, dates in EMPLOYERS:
        sheet.para([Run(employer, True)])
        sheet.para([Run(dates)], after=4)
    sheet.space(6)
    sheet.para([Run("EDUCATION", True)], after=2)
    for school, degree in EDUCATION:
        sheet.para([Run(school, True)])
        sheet.para([Run(degree)])
    (HERE / "sample_base_resume.pdf").write_bytes(sheet.finish()[0])


if __name__ == "__main__":
    write_txt()
    write_docx()
    write_pdf()
    print("Wrote sample_base_resume.txt, .docx and .pdf in", HERE)
