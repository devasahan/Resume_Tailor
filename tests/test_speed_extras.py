"""Two small waits taken out of a run, with the same AI and the same files:

- each model keeps one connection pool, so an AI call reuses the connection (and, on a host, its TLS setup through the
  proxy) that the call before it opened, and a call stopped at the time limit never leaves the pool broken;
- the latest resume PDFs are kept in memory, so the preview and the downloads that follow a run serve the run's own
  file instead of laying the resume out again, while any change to the resume, its style or its title is a new file.
"""
import asyncio
import io
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pypdf import PdfReader

from app.llm import openai_compat_client
from app.llm.base import ProviderError
from app.llm.openai_compat_client import TIMEOUT, OpenAICompatClient
from app.llm.registry import ModelSpec
from app.main import document_word
from app.pipeline.fit import COMPACT
from app.pipeline.tailor import run_tailor
from app.render import pdf as pdf_module
from app.render.pdf import (
    RECENT_PDFS,
    document_name,
    recent_resume_pdf,
    remember_resume_pdf,
    render_resume_pdf,
)
from app.schemas import (
    Certification,
    FitStyle,
    Resume,
    ResumeExperience,
    SkillGroup,
    TailorResult,
)
from app.timing import FONTS

from .conftest import SAMPLE_JD, add_user
from .test_api import save_profile

# ---------------------------------------------------------------------- one connection pool per model
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
ANSWER = {
    "model": "m-1",
    "choices": [{"message": {"role": "assistant", "content": '{"a": 7}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 9},
}
MEDIUM_SECONDS = 1.0  # a call still running while another one is stopped beside it
LIMIT_SECONDS = 0.5  # the time limit that stops a "slow" call


@pytest.fixture
def model_server(monkeypatch):
    """A model's API on a local port, over real TCP (no mock transport). It notes the client port of each request
    (the same port is the same connection) and of each caller that left before its answer. The user message sets the
    pace: "slow" waits until the caller gives up (at most 10 s), "medium" answers after MEDIUM_SECONDS, any other at
    once."""
    monkeypatch.setenv("POOL_TEST_KEY", "pool-test-key")
    ports: list[int] = []
    left: list[int] = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        pace = (await request.json())["messages"][-1]["content"]
        ports.append(request.client.port)
        if pace == "slow":
            for _ in range(200):
                if await request.is_disconnected():
                    left.append(request.client.port)
                    break
                await asyncio.sleep(0.05)
        elif pace == "medium":
            await asyncio.sleep(MEDIUM_SECONDS)
        return JSONResponse(ANSWER)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}/v1", "ports": ports, "left": left}
    server.should_exit = True
    thread.join(timeout=5)


def model_client(server: dict) -> OpenAICompatClient:
    spec = ModelSpec(id="m", label="M", provider="openai_compat", model="m-1", base_url=server["url"],
                     key_env="POOL_TEST_KEY", price_in=1, price_out=2, settings={"json_mode": "json_schema"})
    return OpenAICompatClient(spec)


async def ask(client: OpenAICompatClient, pace: str = "now") -> dict:
    result = await client.generate_structured(system="sys", user=pace, schema=SCHEMA, schema_name="thing",
                                              max_tokens=50)
    return result.data


def test_two_calls_in_one_event_loop_share_one_connection(model_server):
    client = model_client(model_server)

    async def two_calls() -> list[dict]:
        return [await ask(client), await ask(client)]

    assert asyncio.run(two_calls()) == [{"a": 7}, {"a": 7}]
    first, second = model_server["ports"]
    assert first == second  # one TCP connection: on a host, one TLS setup through its proxy for both calls


def test_the_pool_outlives_a_call_and_keeps_the_time_limits(model_server):
    client = model_client(model_server)

    async def pools() -> tuple[httpx.AsyncClient, httpx.AsyncClient, httpx.AsyncClient]:
        before = client._client()
        await ask(client)
        after = client._client()
        await after.aclose()  # a closed pool (nothing closes it today) is replaced by the next call
        await ask(client)
        return before, after, client._client()

    before, after, replaced = asyncio.run(pools())
    assert before is after and replaced is not after and not replaced.is_closed
    assert replaced.timeout == TIMEOUT == httpx.Timeout(240.0, connect=15.0)
    assert model_server["ports"][0] != model_server["ports"][1]


def test_a_call_stopped_at_the_time_limit_is_followed_by_a_working_call(model_server):
    client = model_client(model_server)

    async def stopped_then_asked() -> tuple[dict, httpx.AsyncClient, httpx.AsyncClient]:
        pool = client._client()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ask(client, "slow"), LIMIT_SECONDS)
        return await asyncio.wait_for(ask(client), 10), pool, client._client()

    answer, pool, next_pool = asyncio.run(stopped_then_asked())
    assert answer == {"a": 7}
    assert next_pool is not pool and not pool.is_closed  # a new pool next; the old one is left to calls beside it
    stopped, fresh = model_server["ports"]
    assert stopped != fresh  # the stopped call's connection is never used again
    deadline = time.monotonic() + 5
    while not model_server["left"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert model_server["left"] == [stopped]  # it was closed, not left half-used in a pool


def test_a_stopped_call_never_breaks_the_calls_beside_it(model_server):
    """The pool is shared: stopping one call must not close it under another call that is still running."""
    client = model_client(model_server)

    async def side_by_side() -> tuple[dict, dict, bool]:
        beside = asyncio.ensure_future(ask(client, "medium"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ask(client, "slow"), LIMIT_SECONDS)
        running = not beside.done()
        return await beside, await ask(client), running

    medium, after, running = asyncio.run(side_by_side())
    assert running and medium == after == {"a": 7}
    assert len(model_server["ports"]) == 3


def test_each_event_loop_gets_its_own_connection(model_server):
    """A connection belongs to the event loop that opened it: a call in a new loop (tests run several asyncio.run
    loops) never uses the old loop's pool."""
    client = model_client(model_server)
    assert asyncio.run(ask(client)) == asyncio.run(ask(client)) == {"a": 7}
    first, second = model_server["ports"]
    assert first != second


# ---------------------------------------------------------------------- the latest resume PDFs, kept in memory
def small_resume(profile) -> Resume:
    return Resume(
        profile=profile, headline="Data Engineer", summary="Builds data pipelines for analytics teams.",
        skills=[SkillGroup(category="Languages", skills=["Python", "SQL"])],
        experience=[ResumeExperience(company_index=0, job_title="Data Engineer",
                                     bullets=["Built an ETL pipeline that loads 2 TB a day."])],
    )


def pdf_text(pdf: bytes) -> str:
    return " ".join(" ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages).split())


def test_an_eu_style_run_calls_its_resume_a_cv():
    assert document_name("eu") == "CV"
    assert document_name("us") == document_name("") == document_name(None) == "Resume"
    assert document_word(TailorResult(requested_model="m", spelling="eu")) == "CV"
    assert document_word(TailorResult(requested_model="m")) == "Resume"  # runs made before the style buttons


def test_laying_out_a_resume_keeps_no_file(profile):
    """render_resume_pdf always lays the resume out (the server's warm-up relies on it) and keeps nothing itself."""
    resume = small_resume(profile)
    loads = FONTS.count
    render_resume_pdf(resume, FitStyle())
    render_resume_pdf(resume, FitStyle())
    assert FONTS.count == loads + 2
    assert recent_resume_pdf(resume, FitStyle(), "Resume") is None


def test_the_runs_own_pdf_is_served_without_a_new_layout(client, user_client, profile, monkeypatch):
    """What the job does after its layout: keep the fitted file. The preview and the download then serve it as it
    is, though the run was saved and read back from the database in between (the same text gives the same key)."""
    state = client.app.state
    result = asyncio.run(run_tailor(state.router, state.db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", spelling="eu"))
    document = document_name(result.spelling)
    pdf, _pages = render_resume_pdf(result.resume, result.style, document)
    remember_resume_pdf(result.resume, result.style, document, pdf)

    def no_layout(*_args, **_kwargs):
        raise AssertionError("the resume was laid out again")

    monkeypatch.setattr(pdf_module, "Sheet", no_layout)
    preview = client.get(f"/api/runs/{result.run_id}/resume.pdf")
    download = client.get(f"/api/runs/{result.run_id}/resume.pdf?download=1")
    assert preview.status_code == download.status_code == 200
    assert preview.content == download.content == pdf
    assert PdfReader(io.BytesIO(preview.content)).metadata.title == "Jordan Rivera - CV"
    assert download.headers["content-disposition"] == 'attachment; filename="Jordan_Rivera_CV.pdf"'
    assert user_client.get(f"/api/runs/{result.run_id}/resume.pdf").status_code == 404  # still only for its owner


def test_an_edited_resume_is_laid_out_again_then_kept(client):
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "model": "demo-concise"}).json()
    url = f"/api/runs/{run['run_id']}/resume.pdf"
    saved = client.app.state.db.get_run(run["run_id"])
    remember_resume_pdf(saved.resume, saved.style, document_word(saved), b"%PDF-kept")
    assert client.get(url).content == b"%PDF-kept"

    edit = {"headline": run["resume"]["headline"], "summary": "Builds dependable data platforms for analysts.",
            "skills": run["resume"]["skills"], "experience": run["resume"]["experience"]}
    assert client.put(f"/api/runs/{run['run_id']}/resume", json=edit).status_code == 200
    loads = FONTS.count
    fresh = client.get(url).content
    assert FONTS.count > loads and fresh != b"%PDF-kept"
    assert "Builds dependable data platforms for analysts." in pdf_text(fresh)
    loads = FONTS.count
    assert client.get(url + "?download=1").content == fresh  # the download after the preview: kept, not laid out
    assert FONTS.count == loads


def test_a_changed_resume_style_or_title_is_another_file(profile):
    resume, style = small_resume(profile), FitStyle()
    remember_resume_pdf(resume, style, "Resume", b"%PDF-kept")
    edited = resume.model_copy(deep=True)
    edited.summary = "Builds data platforms for analytics teams."
    ticked = resume.model_copy(deep=True)
    ticked.certifications = [Certification(name="Cloud Data Engineer Associate", issuer="Contoso Academy")]
    assert recent_resume_pdf(resume.model_copy(deep=True), FitStyle(), "Resume") == b"%PDF-kept"  # equal content
    changed = [
        (edited, style, "Resume"),  # an edit
        (ticked, style, "Resume"),  # a ticked certification
        (resume, FitStyle(font_pt=10.0, margin_in=0.55), "Resume"),
        (resume, FitStyle(paper="A4"), "Resume"),
        (resume, FitStyle(spacing=COMPACT), "Resume"),  # the same sizes with the last steps' closer gaps
        (resume, style, "CV"),  # the same text in EU style: another title and language
    ]
    assert [recent_resume_pdf(*key) for key in changed] == [None] * len(changed)
    compact, in_line = FitStyle(spacing=COMPACT), FitStyle(spacing=COMPACT, inline_skills=True)  # the last two steps
    remember_resume_pdf(resume, compact, "Resume", b"%PDF-compact")
    assert recent_resume_pdf(resume, in_line, "Resume") is None  # they differ only in where the skills go
    remember_resume_pdf(resume, in_line, "Resume", b"%PDF-in-line")
    assert recent_resume_pdf(resume, compact, "Resume") == b"%PDF-compact"
    assert recent_resume_pdf(resume, in_line, "Resume") == b"%PDF-in-line"


def test_only_the_latest_files_are_kept(profile):
    resume, style = small_resume(profile), FitStyle()
    for n in range(RECENT_PDFS):
        remember_resume_pdf(resume, style, f"Doc {n}", f"%PDF-{n}".encode())
    assert recent_resume_pdf(resume, style, "Doc 0") == b"%PDF-0"  # read again: now the most recently used
    remember_resume_pdf(resume, style, "Doc 2", b"%PDF-2 again")  # the same file again takes no second place
    remember_resume_pdf(resume, style, "Doc new", b"%PDF-new")  # one too many: the least recently used one goes
    kept = {n: recent_resume_pdf(resume, style, f"Doc {n}") for n in [*range(RECENT_PDFS), "new"]}
    assert kept[1] is None and kept[0] == b"%PDF-0" and kept[2] == b"%PDF-2 again" and kept["new"] == b"%PDF-new"
    assert sum(pdf is not None for pdf in kept.values()) == RECENT_PDFS == len(pdf_module._recent_pdfs)


def test_threads_keeping_and_reading_files_at_once(profile):
    """A job's worker thread keeps its file while request threads read theirs: no error, a file is never served for
    another resume, and at most RECENT_PDFS files are kept."""
    resume, style = small_resume(profile), FitStyle()
    documents = [f"Doc {n}" for n in range(3 * RECENT_PDFS)]  # more than are kept: files go all the time

    def work(seed: int) -> int:
        served = 0
        for step in range(1000):
            document = documents[(seed * 7 + step) % len(documents)]
            if step % 3:
                found = recent_resume_pdf(resume, style, document)
                assert found in (None, document.encode())
                served += found is not None
            else:
                remember_resume_pdf(resume, style, document, document.encode())
        return served

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible, so a missing lock shows here
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            served = sum(pool.map(work, range(8)))  # an error in a thread is raised here
    finally:
        sys.setswitchinterval(previous)
    assert served > 0 and len(pdf_module._recent_pdfs) <= RECENT_PDFS


@pytest.mark.parametrize("delete", ["the account", "all data"])
def test_deleting_an_account_or_all_data_forgets_the_kept_files(client, profile, delete):
    resume, style = small_resume(profile), FitStyle()
    remember_resume_pdf(resume, style, "Resume", b"%PDF-kept")
    url = f"/api/users/{add_user(client)}" if delete == "the account" else "/api/data"
    assert client.delete(url).json() == {"deleted": True}
    assert recent_resume_pdf(resume, style, "Resume") is None


def test_a_model_silent_for_the_whole_read_timeout_is_not_asked_again(model_server, monkeypatch):
    """A model that says nothing for the read timeout (4 minutes on the host) would make every retry another 4 minutes:
    it is reported at once, after one request."""
    monkeypatch.setattr(openai_compat_client, "TIMEOUT", httpx.Timeout(0.5, connect=2.0))
    client = model_client(model_server)
    started = time.perf_counter()
    with pytest.raises(ProviderError, match="no answer within 0.5 seconds"):
        asyncio.run(ask(client, "slow"))
    assert len(model_server["ports"]) == 1 and time.perf_counter() - started < 3


@pytest.mark.parametrize(("spelling", "title"), [("us", "Resume"), ("eu", "CV")])
def test_a_finished_run_serves_the_pdf_its_own_job_made(client, profile, monkeypatch, spelling, title):
    """run_tailor keeps the PDF it laid out to fit, with the preview's title and language: the first preview after a
    run (with its cover letter too) needs no layout at all."""
    state = client.app.state
    result = asyncio.run(run_tailor(state.router, state.db, profile=profile, profile_id=None, jd=SAMPLE_JD,
                                    model_id="demo-concise", spelling=spelling, cover_letter=True))
    assert result.status == "ok" and result.cover_letter is not None

    def no_layout(*_args, **_kwargs):
        raise AssertionError("the resume was laid out again")

    monkeypatch.setattr(pdf_module, "Sheet", no_layout)
    preview = client.get(f"/api/runs/{result.run_id}/resume.pdf")
    assert preview.status_code == 200
    assert PdfReader(io.BytesIO(preview.content)).metadata.title == f"Jordan Rivera - {title}"
