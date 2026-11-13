"""The job post's link (Tailor page, required there): checked as an http or https address, kept with the run, recorded
on every History row, written in the job description file, and never sent to the AI. Every person and employer here is
invented."""
import asyncio
import sqlite3

import pytest

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.main import JOB_LINK_RULE, MAX_JOB_LINK, PlainError, checked_job_link
from app.pipeline.tailor import run_tailor, write_cover_letter
from app.schemas import TailorResult
from app.storage.db import COVER_MARK, Database

from .conftest import SAMPLE_JD
from .test_accounts import add_download, user_id_of
from .test_api import save_profile, tailor

LINK = "https://jobs.example.com/senior-data-engineer?ref=board"


@pytest.mark.parametrize(("typed", "kept"), [
    ("", ""),
    ("  \n", ""),
    (f"  {LINK}  ", LINK),
    ("www.example.com/careers/42", "https://www.example.com/careers/42"),
    ("careers.example.org/jobs/7", "https://careers.example.org/jobs/7"),
    ("HTTP://Careers.Example.org/1", "http://Careers.Example.org/1"),
    ("https://jobs.example.com:8443/a?b=1#c", "https://jobs.example.com:8443/a?b=1#c"),
])
def test_a_job_link_is_trimmed_and_gets_https_when_it_has_none(typed, kept):
    assert checked_job_link(typed) == kept


def test_a_link_of_2000_characters_is_kept_and_one_more_is_refused():
    long = "https://jobs.example.com/" + "a" * (MAX_JOB_LINK - len("https://jobs.example.com/"))
    assert len(long) == MAX_JOB_LINK and checked_job_link(long) == long
    with pytest.raises(PlainError):
        checked_job_link(long + "a")


@pytest.mark.parametrize("typed", [
    "javascript:alert(1)", "JavaScript:alert(1)", "java\tscript:alert(1)", "javascript://jobs.example.com/%0aalert(1)",
    "data:text/html,<b>hi</b>", "vbscript:msgbox(1)", "file:///etc/passwd", "ftp://files.example.com/job",
    "mailto:jobs@example.com", "https:jobs.example.com", "https://", "https:///jobs", "https://jobs", "acme",
    "Acme Corp", "https://jobs.example.com/a b", "https://jobs.example.com/\u200bjob",
    "https://jobs.example.com\\@evil.example/", "https://jobs.example.com:99999/x", "https://jobs.example.com:abc/x",
])
def test_anything_but_a_web_address_is_refused(typed):
    with pytest.raises(PlainError) as refused:
        checked_job_link(typed)
    assert refused.value.status_code == 422 and refused.value.detail == JOB_LINK_RULE


def test_the_api_checks_the_link_and_keeps_it_with_the_run(client, user_client):
    save_profile(user_client)
    bad = tailor(user_client, job_link="javascript:alert(1)")
    assert bad.status_code == 422 and bad.json()["detail"] == JOB_LINK_RULE  # in plain words, for users too
    made = tailor(user_client, company_name="Acme Corp", job_link=f"  {LINK} ")
    assert made.status_code == 200, made.text
    assert "job_link" not in made.json()  # a user's reply keeps its keys
    run_id = made.json()["run_id"]
    assert client.get(f"/api/runs/{run_id}").json()["job_link"] == LINK  # the admin sees it
    save_profile(client)
    assert client.get(f"/api/runs/{tailor(client).json()['run_id']}").json()["job_link"] == ""  # the API needs none
    fake = 'https://jobs.example.com/"cover_letter":{'  # a link can never pass for a cover letter in History's search
    assert COVER_MARK not in TailorResult(requested_model="m", job_link=fake).model_dump_json()


def test_add_or_rewrite_cover_letter_records_a_corrected_link_without_a_new_resume(client):
    save_profile(client)
    run_id = tailor(client, job_link=LINK).json()["run_id"]
    before = client.get(f"/api/runs/{run_id}").json()
    assert client.post(f"/api/runs/{run_id}/cover-letter", json={}).status_code == 200
    assert client.get(f"/api/runs/{run_id}").json()["job_link"] == LINK  # no link sent: the run keeps its own
    rewrite = client.post(f"/api/runs/{run_id}/cover-letter", json={"job_link": "careers.example.org/jobs/7"})
    assert rewrite.status_code == 200, rewrite.text
    after = client.get(f"/api/runs/{run_id}").json()
    assert after["job_link"] == "https://careers.example.org/jobs/7" and after["resume"] == before["resume"]
    resumes = [r for r in client.get("/api/runs").json()["runs"] if r.get("task", "resume") == "resume"]
    assert len(resumes) == 1  # the letter was written again, the resume was not
    bad = client.post(f"/api/runs/{run_id}/cover-letter", json={"job_link": "javascript:alert(1)"})
    assert bad.status_code == 422 and bad.json()["detail"] == JOB_LINK_RULE
    resume = after["resume"]
    edit = {"headline": resume["headline"], "summary": resume["summary"], "skills": resume["skills"],
            "experience": resume["experience"]}
    assert client.put(f"/api/runs/{run_id}/resume", json=edit).status_code == 200
    assert client.get(f"/api/runs/{run_id}").json()["job_link"] == "https://careers.example.org/jobs/7"


class Recording(LLMClient):
    """Keeps every call's prompts and context."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return await self.inner.generate_structured(**kwargs)


def test_the_job_link_is_kept_with_the_run_and_never_sent_to_the_ai(router, db, registry, profile, jd):
    recording = Recording(FakeClient(registry.get("demo-concise")))
    router.overrides["demo-concise"] = recording
    result = asyncio.run(run_tailor(router, db, profile=profile, profile_id=None, jd=jd, model_id="demo-concise",
                                    cover_letter=True, job_link=LINK))
    assert result.status == "ok" and result.job_link == LINK and db.get_run(result.run_id).job_link == LINK
    fresh, error = asyncio.run(write_cover_letter(router, db, result.run_id, jd=jd, model_id="demo-concise"))
    assert error is None and fresh.job_link == LINK and db.get_run(result.run_id).job_link == LINK
    assert recording.calls and not any("jobs.example.com" in repr(call) for call in recording.calls)


def test_history_records_the_job_link_and_keeps_it_after_the_run_is_gone(client, user_client):
    save_profile(user_client)
    run_id = tailor(user_client, company_name="Acme Corp", job_link=LINK).json()["run_id"]
    assert user_client.get(f"/api/runs/{run_id}/resume.pdf?download=1").status_code == 200
    rows = client.get("/api/downloads").json()["downloads"]
    assert [(r["company"], r["job_link"]) for r in rows] == [("Acme Corp", LINK)]
    db = client.app.state.db
    db.delete_user(user_id_of(client, "user@example.com"))
    row = client.get("/api/downloads").json()["downloads"][0]
    assert row["run_kept"] == 0 and row["job_link"] == LINK  # the row keeps what History shows
    add_download(client, "Globex", "Data Engineer", user_id=1)  # a row made before the link was kept
    assert client.get("/api/downloads").json()["downloads"][0]["job_link"] == ""


def test_the_job_description_file_names_the_job_link(client):
    save_profile(client)
    run_id = tailor(client, company_name="Acme Corp", job_link=LINK).json()["run_id"]
    lines = client.get(f"/api/runs/{run_id}/jd.txt").text.splitlines()
    assert lines[0].startswith("Company: Acme Corp | Job title: ")
    assert lines[1:4] == [f"Job link: {LINK}", "", SAMPLE_JD.splitlines()[0]]
    plain = tailor(client, company_name="Acme Corp").json()["run_id"]
    assert client.get(f"/api/runs/{plain}/jd.txt").text.splitlines()[1] == ""  # no link: no line


def test_the_job_link_column_is_added_to_an_older_download_history(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE downloads (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,"
                    " user_id INTEGER, email TEXT NOT NULL DEFAULT '', run_id INTEGER, kind TEXT NOT NULL,"
                    " format TEXT NOT NULL, filename TEXT NOT NULL DEFAULT '', candidate TEXT NOT NULL DEFAULT '',"
                    " company TEXT NOT NULL DEFAULT '', role TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',"
                    " cost_usd REAL NOT NULL DEFAULT 0)")
        raw.execute("INSERT INTO downloads(created_at, user_id, run_id, kind, format, company)"
                    " VALUES('2026-01-05T10:00:00', 1, 7, 'resume', 'pdf', 'Globex')")
    Database(path)
    db = Database(path)  # a second start changes nothing
    db.record_download(user_id=1, email="user@example.com", run_id=8, kind="resume", format="pdf",
                       filename="Taylor_Quinn_Resume.pdf", candidate="Taylor Quinn", company="Acme Corp",
                       role="Data Engineer", model="demo-concise", cost_usd=0.01, job_link=LINK)
    assert [(r["company"], r["job_link"]) for r in db.list_downloads()] == [("Acme Corp", LINK), ("Globex", "")]


def test_the_tailor_page_marks_both_boxes_required_and_sends_the_link(client):
    """Only the page's markup and script text: the browser checks of the boxes are run by hand (the repo has no
    browser tests)."""
    page = client.get("/tailor").text
    assert page.count('<span class="req">required</span>') == 2  # Company and Job link
    assert 'id="company" type="text" maxlength="120" required aria-required="true"' in page
    assert 'id="job-link" type="text" inputmode="url" required aria-required="true"' in page
    assert "<strong>Job link</strong>" in page and 'id="job-link-note"' in page and "maxlength=\"2000\"" not in page
    assert "company_name: company, job_link: link" in page and "body: { job_link: link }" in page
    assert "const link = readLink();  // every button needs the job post's link" in page
    assert "Paste the link of the job post into the Job link box" in page and "starting with https://" in page
    assert "keepLinkWithPost();" in page and "const watched = [box, $('#job-link')];" in page


def test_the_history_page_links_each_job_post_safely(client):
    page = client.get("/history").text
    assert "'Job title', 'Job link', 'File'" in page and "postLink(d.job_link)" in page
    assert "url.protocol !== 'https:' && url.protocol !== 'http:'" in page  # never javascript: or data:
    assert "href: url.href, target: '_blank', rel: 'noopener noreferrer'" in page and "innerHTML" not in page
