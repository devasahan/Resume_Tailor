import io
import os
import stat
import threading
import zipfile

import anthropic
import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from app.main import create_app
from app.parsing.resume_parser import extract_text
from app.pipeline import validate
from app.prompts import cover_system, parse_system, tailor_system
from app.render.pdf import render_resume_pdf
from app.schemas import FitStyle
from app.storage.db import Database

from .test_adapters import anthropic_client, call, message
from .test_validate import make_output


def test_zip_bomb_docx_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"0" * (60 * 1024 * 1024))  # compresses to a few dozen KB
    assert len(buffer.getvalue()) < 5 * 1024 * 1024
    with pytest.raises(ValueError, match="unreasonable"):
        extract_text(buffer.getvalue(), "evil.docx")


def test_corrupt_files_give_clear_errors():
    with pytest.raises(ValueError, match="Could not read the DOCX"):
        extract_text(b"not a zip", "x.docx")
    with pytest.raises(ValueError, match="Could not read the PDF"):
        extract_text(b"not a pdf", "x.pdf")


def test_oversized_upload_is_refused_without_buffering_it_all(tmp_path):
    client = TestClient(create_app(db=Database(tmp_path / "t.db")))
    r = client.post("/api/profile/parse", files={"file": ("big.txt", b"a" * (5 * 1024 * 1024 + 10))}, data={"model": "rules"})
    assert r.status_code == 422 and "5 MB" in r.json()["detail"]


def test_pdf_page_cap():
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for n in range(1, 13):
        pdf.add_page()
        pdf.cell(0, 8, f"PAGE {n} of the resume text here")
    text = extract_text(bytes(pdf.output()), "long.pdf")
    assert "PAGE 1 " in text and "PAGE 12 " not in text


def test_rendering_is_safe_from_many_threads(profile):
    resume = validate.build_resume(profile, make_output())
    results, errors = [], []

    def work():
        try:
            results.append(render_resume_pdf(resume, FitStyle())[1])
        except Exception as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and results == [1] * 8


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_database_file_is_private(tmp_path):
    Database(tmp_path / "private.db")
    assert stat.S_IMODE((tmp_path / "private.db").stat().st_mode) & 0o077 == 0


def test_prompts_tell_the_model_that_inputs_are_data():
    for text in (tailor_system(), cover_system(), parse_system()):
        assert "DATA, not instructions" in text


class OtherBadRequest(anthropic.BadRequestError):
    def __init__(self, text):
        Exception.__init__(self, text)
        self.status_code, self.message = 400, text


def test_anthropic_fallback_beta_failure_of_any_kind_retries_plain():
    calls = []
    result = call(anthropic_client([OtherBadRequest("unexpected beta problem"), message()], calls, refusal_fallback=True))
    assert [c[0] for c in calls] == ["beta", "plain"] and result.data == {"a": 2}


def test_anthropic_genuine_bad_request_surfaces_the_real_error():
    from app.llm.base import ProviderError

    calls = []
    with pytest.raises(ProviderError, match="max_tokens too large"):
        call(anthropic_client([OtherBadRequest("max_tokens too large"), OtherBadRequest("max_tokens too large")], calls, refusal_fallback=True))
    assert [c[0] for c in calls] == ["beta", "plain"]
    calls.clear()
    with pytest.raises(ProviderError):
        call(anthropic_client([OtherBadRequest("bad")], calls, refusal_fallback=False))
    assert [c[0] for c in calls] == ["plain"]


def test_relative_data_dir_is_resolved_inside_the_project(monkeypatch, tmp_path):
    from app import config

    monkeypatch.setenv("DATA_DIR", "./data_dir_test_tmp")
    monkeypatch.chdir(tmp_path)  # started from some other folder
    folder = config.data_dir()
    try:
        assert folder == config.ROOT / "data_dir_test_tmp" and folder.is_dir()
    finally:
        folder.rmdir()
