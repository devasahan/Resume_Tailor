"""The sample files shipped for a first try must parse into exactly the sample profile."""
from pathlib import Path

import pytest

from app.parsing.heuristic import parse_resume_text
from app.parsing.resume_parser import extract_text
from app.schemas import Profile

EVALS = Path(__file__).resolve().parent.parent / "evals"
EXPECTED = Profile.model_validate_json((EVALS / "sample_profile.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["sample_base_resume.txt", "sample_base_resume.docx", "sample_base_resume.pdf"])
def test_sample_base_resume_parses_to_the_sample_profile(name):
    text = extract_text((EVALS / name).read_bytes(), name)
    assert parse_resume_text(text).model_dump() == EXPECTED.model_dump()


def test_sample_job_descriptions_are_substantial():
    files = sorted((EVALS / "jds").glob("*.txt"))
    assert len(files) >= 3 and all(len(f.read_text(encoding="utf-8")) > 400 for f in files)
