"""The speed-ups that leave the resume and the cover letter exactly as they were: a size that is only tried is laid out
without fonts (only the file made loads them), the letter's request goes out before the layout starts, a run reads
the Settings once, the letter's PDF made for its preview is kept for the download, and a pasted job post is checked at
once."""
import asyncio
import io
import re
import time

import pytest
from pypdf import PdfReader

from app.llm.fake_client import FakeClient
from app.pipeline import tailor, validate
from app.pipeline.fit import render_smallest, trim_to_fit, try_styles
from app.render import pdf as pdf_module
from app.render.pdf import (
    COVER_STYLES,
    _cover_sheet,
    cover_pdf,
    drawable,
    forget_resume_pdfs,
    render_cover_pdf,
    resume_layout,
)
from app.schemas import CoverLetter, FitStyle
from app.timing import FONTS

from .conftest import SAMPLE_JD
from .test_api import save_profile
from .test_letter_in_job import Paced, career, slow_layout, tailored
from .test_pipeline import Priced
from .test_render import long_resume
from .test_speed_extras import pdf_text
from .test_validate import make_output


@pytest.fixture
def resume(profile):
    return validate.build_resume(profile, make_output())


def letter_of(words: int, paragraphs: int = 4, greeting: str = "Dear Hiring Manager,") -> CoverLetter:
    paragraph = " ".join(["Delivered reliable data pipelines for the analytics team."] * words)
    return CoverLetter(greeting=greeting, paragraphs=[paragraph] * paragraphs, closing="Sincerely,",
                       company_name="Fabrikam Robotics")


def lang_of(pdf: bytes) -> str:
    return str(PdfReader(io.BytesIO(pdf)).trailer["/Root"]["/Lang"])


# ------------------------------------------------------------------ sizes tried without fonts
def test_a_size_that_is_only_tried_loads_no_font(profile):
    resume = long_resume(profile, bullets=8)
    loads = FONTS.count
    pages, fill = resume_layout(resume, FitStyle())
    assert pages >= 1 and 0 <= fill <= 1 and FONTS.count == loads  # measured: no font, no drawing
    found = try_styles(resume)  # the old layout's pages, a calibrated step and the design's sizes: one file
    assert found is not None and FONTS.count == loads + 1
    render_smallest(resume)  # two steps tried, the one with fewer pages drawn
    assert FONTS.count == loads + 2
    trim_to_fit(long_resume(profile, bullets=40))  # many layouts while halving, then the one file
    assert FONTS.count == loads + 3


def test_a_long_letter_tries_its_sizes_without_fonts(resume):
    long = letter_of(16)
    assert _cover_sheet(resume, long, "Letter", *COVER_STYLES[0]).pdf.page == 2  # too long at full size
    loads = FONTS.count
    pdf, pages = render_cover_pdf(resume, long)
    assert pages == 1 and FONTS.count == loads + 1 and "Sincerely," in pdf_text(pdf)


def test_a_sheet_draws_what_it_laid_out(profile):
    """The file holds every line the layout noted, drawn where the layout put it, on its page and in reading order, and
    finishing again gives the same file."""
    sheet = pdf_module.resume_sheet(long_resume(profile, bullets=12), FitStyle())
    first, pages = sheet.finish()
    again, _ = sheet.finish()
    reader = PdfReader(io.BytesIO(first))
    assert pages == sheet.pdf.page == len(reader.pages) >= 2
    noted: list[list[str]] = [[]]
    placed: list[list[tuple[str, str]]] = [[]]  # where each piece starts: fpdf2 puts a cell's baseline at y + h/2 + 0.3 size
    for note in sheet._notes:
        if note[0] == "page":
            noted.append([])
            placed.append([])
        elif note[0] == "text":
            _kind, merged, x, y, size, height = note
            noted[-1].append(" ".join(text for text, _style, _width in merged))
            for text, _style, width in merged:
                if drawable(text):
                    placed[-1].append((f"{x:.2f}", f"{sheet.pdf.h - y - 0.5 * height - 0.3 * size:.2f}"))
                x += width
    drawn = [" ".join(page.extract_text().split()) for page in reader.pages]
    assert [" ".join(" ".join(lines).split()) for lines in noted] == drawn
    starts = [re.findall(r"BT (-?[\d.]+) (-?[\d.]+) Td", page.get_contents().get_data().decode("latin-1"))
              for page in reader.pages]
    assert starts == placed and sum(map(len, placed)) > 100
    streams = [[page.get_contents().get_data() for page in PdfReader(io.BytesIO(data)).pages] for data in (first, again)]
    assert streams[0] == streams[1]


# ------------------------------------------------------------------ the letter's request before the layout
def test_the_letters_request_goes_out_before_the_layout_starts(router, db, registry, profile, monkeypatch):
    """The letter starts right after the draft is fitted to its page (a few milliseconds); the PDF's thread starts
    once the letter's request is on its way, so it never slows the letter's preparation."""
    db.set_settings({"resume_time_limit_s": 5})
    client = Paced(FakeClient(registry.get("demo-concise")), {"tailor_output": 0.2, "cover_letter": 0.5})
    router.overrides["demo-concise"] = client
    layouts = slow_layout(monkeypatch, 0.3)
    result = tailored(router, db, profile, cover_letter=True)
    assert result.status == "ok" and result.cover_letter is not None
    [draft], [letter] = client.of("tailor_output"), client.of("cover_letter")
    (layout_start, _layout_end), *_ = layouts
    assert draft["end"] <= letter["start"] <= layout_start  # asked before the layout began
    assert layout_start < letter["end"]  # and still being written while the layout ran


def test_the_letter_starts_after_the_trimming_on_the_final_text(router, db, registry, monkeypatch):
    """No time limit: a draft that needs bullets removed is asked again, shorter, and fitted again. The letter quotes
    each job's first bullets, so it starts only on the final text: after the last fitting (which only measures)."""
    client = Paced(FakeClient(registry.get("demo-detailed")), {"cover_letter": 0.3})
    router.overrides["demo-detailed"] = client
    fits, real = [], tailor.fit_new

    def fit_new(*args, **kwargs):
        fitted = real(*args, **kwargs)
        fits.append(time.perf_counter())
        return fitted

    monkeypatch.setattr(tailor, "fit_new", fit_new)
    result = tailored(router, db, career(9), model="demo-detailed", cover_letter=True, time_limit=0)
    assert result.status == "ok" and result.cover_letter is not None and len(fits) == 2  # the draft, the shorter one
    second, [letter] = client.of("tailor_output")[-1], client.of("cover_letter")
    assert second["end"] <= fits[-1] <= letter["start"]


def test_a_letter_slow_to_ask_holds_the_layout_back_only_briefly(router, db, registry, profile, monkeypatch):
    """The layout waits for the letter's request at most LETTER_START_SECONDS: a letter that takes longer to get
    ready never holds the resume back."""
    db.set_settings({"resume_time_limit_s": 5})
    monkeypatch.setattr(tailor, "LETTER_START_SECONDS", 0.1)
    real = tailor._write_letter

    async def slow_to_ask(*args, **kwargs):
        await asyncio.sleep(1.0)
        return await real(*args, **kwargs)

    monkeypatch.setattr(tailor, "_write_letter", slow_to_ask)
    client = Paced(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = client
    layouts = slow_layout(monkeypatch, 0.0)
    result = tailored(router, db, profile, cover_letter=True)
    assert result.status == "ok" and result.cover_letter is not None
    [draft], [letter] = client.of("tailor_output"), client.of("cover_letter")
    (layout_start, _end), *_ = layouts
    assert layout_start - draft["end"] < 0.6 < letter["start"] - draft["end"]


# ------------------------------------------------------------------ the Settings, read once per run
def counted_settings(db, monkeypatch) -> list[int]:
    reads, real = [0], db.get_settings

    def get_settings():
        reads[0] += 1
        return real()

    monkeypatch.setattr(db, "get_settings", get_settings)
    return reads


def test_a_run_reads_the_settings_once_and_todays_spend_at_each_call(router, db, registry, profile, monkeypatch):
    """A paid model with backups and a daily cost cap, and its letter: the Settings are read once for the whole run
    (before, at every model choice and call); today's spend is still read before each call, so the cap works as
    before."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    calls = []

    class Counted(Priced):
        async def generate_structured(self, **kwargs):
            calls.append(kwargs["schema_name"])
            return await super().generate_structured(**kwargs)

    router.overrides["claude-sonnet-5-5"] = Counted(FakeClient(registry.get("demo-concise")))
    db.set_settings({"daily_cost_cap_usd": 50, "fallback_models": ["demo-detailed"]})
    reads = counted_settings(db, monkeypatch)
    spend, real_cost = [0], db.cost_today

    def cost_today():
        spend[0] += 1
        return real_cost()

    monkeypatch.setattr(db, "cost_today", cost_today)
    result = tailored(router, db, profile, model="claude-sonnet-5-5", cover_letter=True)
    assert result.status == "ok" and result.cover_letter is not None and calls[-1] == "cover_letter"
    assert reads[0] == 1
    assert spend[0] == len(calls) + 1  # before each call, and the letter's check with the draft's cost not yet saved


def test_a_tailoring_request_and_its_job_read_the_settings_once(client, monkeypatch):
    save_profile(client)
    reads = counted_settings(client.app.state.db, monkeypatch)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "cover_letter": True}).json()
    assert run["status"] == "ok" and run["cover_letter"] and reads[0] == 1
    reads[0] = 0
    again = client.post(f"/api/runs/{run['run_id']}/cover-letter", json={})  # Rewrite cover letter: its job too
    assert again.status_code == 200 and reads[0] == 1


def test_a_settings_change_applies_from_the_next_run(router, db, registry):
    db.set_settings({"paper_size": "Letter"})
    with router.keep_settings():
        db.set_settings({"paper_size": "A4"})  # saved while a run is under way
        assert router.settings()["paper_size"] == "Letter"
        with router.keep_settings():  # a block inside another keeps the outer reading
            assert router.settings()["paper_size"] == "Letter"
        kept = router.settings()
        kept["paper_size"] = "Legal"  # a copy: the reading itself stays
        assert router.settings()["paper_size"] == "Letter"
        other = type(router)(registry, db)  # another router reads for itself
        assert other.settings()["paper_size"] == "A4"
    assert router.settings()["paper_size"] == "A4"  # the next run reads it again


# ------------------------------------------------------------------ the letter's PDF, kept for the download
def test_the_letters_preview_makes_its_pdf_once_and_the_download_reuses_it(client):
    """The job makes no letter file (a build alongside the job's last steps delayed its answer); the preview makes it,
    in the run's language, and the download that follows serves the same file without laying the letter out again."""
    save_profile(client)
    loads = FONTS.count
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "cover_letter": True, "spelling": "eu"}).json()
    assert run["status"] == "ok" and run["cover_letter"] and FONTS.count == loads + 1  # the resume's file only
    url = f"/api/runs/{run['run_id']}/cover.pdf"
    preview = client.get(url)
    assert preview.status_code == 200 and FONTS.count == loads + 2 and lang_of(preview.content) == "en-GB"
    assert run["cover_letter"]["greeting"] in pdf_text(preview.content)
    download = client.get(url + "?download=1")
    assert download.content == preview.content and FONTS.count == loads + 2  # kept: no new layout


def test_an_added_letters_download_reuses_its_preview_too(client):
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD}).json()  # Resume only
    added = client.post(f"/api/runs/{run['run_id']}/cover-letter", json={}).json()  # then Add cover letter
    url = f"/api/runs/{run['run_id']}/cover.pdf"
    preview = client.get(url)
    loads = FONTS.count
    assert preview.status_code == 200 and added["cover_letter"]["greeting"] in pdf_text(preview.content)
    assert client.get(url + "?download=1").content == preview.content and FONTS.count == loads
    assert lang_of(preview.content) == "en-US"


def test_a_rewritten_letter_gets_its_own_file(client):
    save_profile(client)
    run = client.post("/api/tailor", json={"jd": SAMPLE_JD, "cover_letter": True}).json()
    url = f"/api/runs/{run['run_id']}/cover.pdf"
    first = client.get(url).content
    db = client.app.state.db
    saved = db.get_run(run["run_id"])
    saved.cover_letter.greeting = "Dear Sam Lee,"  # as a rewrite saves a new letter
    db.update_run(saved)
    rewritten = client.get(url).content
    assert "Dear Sam Lee," in pdf_text(rewritten) and "Dear Sam Lee," not in pdf_text(first)


def test_a_kept_letter_is_the_file_render_cover_pdf_makes(resume):
    letter = letter_of(16)  # needs the second size
    kept = cover_pdf(resume, letter, "A4", "en-GB")
    assert cover_pdf(resume, letter, "A4", "en-GB") is kept  # the second time: the kept file
    made, pages = render_cover_pdf(resume, letter, "A4", "en-GB")
    contents = [[page.get_contents().get_data() for page in PdfReader(io.BytesIO(data)).pages] for data in (kept, made)]
    assert pages == 1 and contents[0] == contents[1] and lang_of(kept) == "en-GB"
    assert cover_pdf(resume, letter, "Letter", "en-GB") != kept  # another paper is another file
    header = resume.model_copy(deep=True)
    header.headline = "Staff Data Engineer"
    assert "Staff Data Engineer" in pdf_text(cover_pdf(header, letter, "A4", "en-GB"))  # another header: made again


def test_deleting_data_forgets_the_kept_letters_too(resume):
    letter = letter_of(3)
    cover_pdf(resume, letter)
    forget_resume_pdfs()
    assert not pdf_module._recent_covers
    loads = FONTS.count
    cover_pdf(resume, letter)
    assert FONTS.count == loads + 1  # made again


# ------------------------------------------------------------------ a pasted job post is checked at once
def test_a_pasted_job_post_is_checked_at_once(client):
    page = client.get("/tailor").text
    assert "const AT_ONCE = new Set(['insertFromPaste', 'insertFromDrop']);" in page
    assert "function postChanged(event) {" in page
    assert "factsTimer = setTimeout(checkFacts, event && AT_ONCE.has(event.inputType) ? 0 : FACTS_WAIT_MS);" in page
    assert "$('#jd').addEventListener('input', postChanged);" in page  # the input event is passed on
