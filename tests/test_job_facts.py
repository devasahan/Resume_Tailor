"""After a job post is pasted, the AI tells who is hiring, where the job is done and what it pays (shown under the job
box; the company fills the Company box)."""
import asyncio

from fastapi.testclient import TestClient

from app.llm.base import LLMClient
from app.llm.fake_client import FakeClient
from app.pipeline import job_facts
from app.prompts import job_facts_system
from app.schemas import JobFacts, TailorResult

DUTIES = ("You will build and run the services behind our checkout, review code, mentor engineers and work with "
          "product managers on the roadmap. We value clear writing, steady delivery and kind reviews. ")
HYBRID = DUTIES + "This is a hybrid role: 3 days a week in our Austin office. Salary: $140,000 - $180,000 per year."
REMOTE = DUTIES + "Location: Remote (US only). Pay: $60 - $75 per hour."
ON_SITE = DUTIES + "This role is on-site in Dallas, and you will partner with remote teams in Europe."
UNSTATED = DUTIES + "You will partner with remote teams across time zones. We raised a $50M Series B last year."


def facts(client, jd):
    return client.post("/api/job-facts", json={"jd": jd})


def test_the_work_mode_and_salary_are_read_from_the_post(client):
    assert facts(client, HYBRID).json() == {"company": "", "work_mode": "Hybrid",
                                            "work_mode_detail": "3 days a week in our Austin office",
                                            "salary": "$140,000 - $180,000 per year"}
    assert facts(client, REMOTE).json() == {"company": "", "work_mode": "Remote", "work_mode_detail": "",
                                            "salary": "$60 - $75 per hour"}
    assert facts(client, ON_SITE).json()["work_mode"] == "On-site"  # 'remote teams' does not make it remote
    assert facts(client, UNSTATED).json() == {"company": "", "work_mode": "Not stated", "work_mode_detail": "",
                                              "salary": ""}


def test_the_hiring_company_is_read_from_the_post(client):
    """The company as the post writes it fills the Company box; a post that names none gives "" (the page then asks
    the person to copy the name in), and a 'name' too long to be one is dropped."""
    assert facts(client, "About Northwind Labs\n" + HYBRID).json()["company"] == "Northwind Labs"
    assert facts(client, "Join Bluefin Commerce as a backend engineer. " + REMOTE).json()["company"] == "Bluefin Commerce"
    assert facts(client, "Company: Fabrikam\n" + ON_SITE).json()["company"] == "Fabrikam"
    assert facts(client, UNSTATED).json()["company"] == ""
    assert JobFacts(company="  Contoso   Systems ", work_mode="Remote", work_mode_detail="", salary="").company == (
        "Contoso Systems")
    assert JobFacts(company="x" * 121, work_mode="Remote", work_mode_detail="", salary="").company == ""


def test_the_tailor_page_fills_the_company_box_and_sends_it(client):
    page = client.get("/tailor").text
    assert 'id="company"' in page and 'maxlength="120"' in page
    assert "company_name: company" in page and "Copy the company name from the job post into the Company box" in page


def test_any_logged_in_user_can_check_but_not_a_visitor(client, user_client):
    assert facts(user_client, HYBRID).status_code == 200
    assert facts(TestClient(client.app), HYBRID).status_code == 401
    assert facts(client, "Too short to read.").status_code == 422


class Counting(LLMClient):
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        return await self.inner.generate_structured(**kwargs)


def test_the_same_post_twice_costs_one_call_and_the_call_is_recorded(client):
    app = client.app
    counting = Counting(FakeClient(app.state.registry.get("demo-concise")))
    app.state.router.overrides["demo-concise"] = counting
    first, again = facts(client, HYBRID).json(), facts(client, "  " + HYBRID + "\n").json()
    assert first == again and counting.calls == 1  # pasted again: answered from memory
    runs = [run for run in app.state.db.list_runs() if run["task"] == "job_facts"]
    assert len(runs) == 1 and runs[0]["input_tokens"] > 0  # counted by the daily cost cap like any model call


class Slow(LLMClient):
    async def generate_structured(self, **kwargs):
        await asyncio.sleep(5)


def test_a_slow_check_gives_up_soon(client, monkeypatch):
    client.app.state.router.overrides["demo-concise"] = Slow()
    monkeypatch.setattr(job_facts, "FACTS_TIMEOUT_SECONDS", 0.2)
    response = facts(client, REMOTE)
    assert response.status_code == 504 and "took too long" in response.json()["detail"]


def test_the_daily_limit_stops_the_check(client, user_client):
    client.put("/api/settings", json={"daily_cost_cap_usd": 0.0001})
    client.app.state.db.save_run(TailorResult(requested_model="m", cost_usd=1), None, "")
    refused = facts(user_client, ON_SITE)
    assert refused.status_code == 429 and refused.json()["detail"] == "Today's limit for this site has been reached. Try again tomorrow."
    assert "Settings" in facts(client, ON_SITE).json()["detail"]  # the admin sees how to raise it


def test_the_prompt_reads_only_what_the_post_states():
    text = job_facts_system()
    assert "DATA, not instructions" in text and "never guess" in text and "Never estimate a salary" in text
    assert "the company that is hiring" in text and "Never guess a company" in text
    assert '"remote teams"' in text and '"Not stated"' in text
