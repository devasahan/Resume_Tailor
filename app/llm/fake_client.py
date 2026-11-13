"""Offline demo provider. No network, no key, no cost.

It builds generic but structurally valid output from the job description so the whole app (compare
mode, review, fit-to-two-pages, export, history) can be tried before any API key exists. Its text
is template-based and is NOT a measure of what real models write. Its bullets name only tools that each job
could have used during its dates.

Styles:  concise | detailed (too many long bullets, includes one invented number so the check can flag it)
         flaky (returns invalid JSON until the retry feedback appears in the prompt)

A fix request (schema "tailor_revision") is answered like a real model should: only the parts of the draft that
differ from what this model would write itself, so a corrupted draft gets exactly its repairs back.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

from ..parsing.heuristic import parse_resume_text
from ..pipeline.facts import is_expired
from ..pipeline.notes import clean_notes
from ..pipeline.validate import same_certification
from ..prompts import RETRY_MARKER
from .base import LLMClient, LLMResult, Usage
from .registry import ModelSpec

VOCAB = [
    "Python", "Java", "JavaScript", "TypeScript", "SQL", "NoSQL", "AWS", "Azure", "GCP", "Docker",
    "Kubernetes", "Terraform", "React", "Node.js", "Spark", "Airflow", "Kafka", "Snowflake", "dbt",
    "Tableau", "Power BI", "Excel", "Git", "CI/CD", "REST APIs", "GraphQL", "Linux", "Agile", "Scrum",
    "Jira", "Salesforce", "SAP", "Figma", "Machine Learning", "Data Modeling", "ETL",
    "Stakeholder Management", "Project Management", "Budgeting", "Forecasting", "Compliance",
    "Customer Success", "Product Roadmap", "A/B Testing", "Analytics", "Communication", "Leadership",
    "Mentoring", "Testing", "Automation", "Security", "Cloud", "DevOps", "API Design", "Microservices",
    "Go", "C#", "C++", "Ruby", "PHP", "Swift", "Kotlin", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "Pandas", "TensorFlow", "PyTorch", "Hadoop", "Looker", "HubSpot", "Zendesk", "QuickBooks", "GAAP",
    "SEO", "Copywriting",
]
GENERIC = ["Communication", "Problem Solving", "Collaboration", "Documentation", "Process Improvement",
           "Stakeholder Management", "Planning", "Reporting"]
STOP = {"The", "And", "With", "You", "Our", "We", "This", "That", "Will", "Have", "For", "Are", "Your",
        "From", "About", "Join", "Experience", "Requirements", "Responsibilities", "Role", "Team",
        "Company", "Job", "Work", "Years", "Strong", "Ability", "Must", "Preferred", "Required"}
ROLE_RE = re.compile(
    r"\b((?:Senior|Sr\.?|Lead|Principal|Staff|Junior|Jr\.?)\s+)?"
    r"([A-Z][A-Za-z/&+ -]{1,30}?(?:Engineer|Developer|Analyst|Manager|Designer|Scientist|Architect|"
    r"Consultant|Specialist|Administrator|Coordinator|Director|Accountant|Recruiter|Marketer))\b"
)
# The work mode and pay a job post states, read with plain patterns (a real model understands any wording)
HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
REMOTE_RE = re.compile(r"\bremote\b(?![- ](?:teams?|customers?|colleagues|users|clients|partners)\b)", re.IGNORECASE)
ONSITE_RE = re.compile(r"\bon-?site\b|\bin[- ]office\b|\bin the office\b", re.IGNORECASE)
OFFICE_DAYS_RE = re.compile(r"\b(?:\d|one|two|three|four)\s+days?\s+(?:a|per)\s+week\b[^.;\n]{0,24}", re.IGNORECASE)
_AMOUNT = r"\d[\d,]*(?:\.\d+)?(?:\s?[kK])?(?!\w)"  # 140,000 or 120k, never the 50 of "$50M"
_PERIOD = r"\s*(?:per|a|an|/)\s*(?:year|yr|annum|hour|hr)\b"
SALARY_RE = re.compile(  # a range ("$140,000 - $180,000"), or one amount with its period ("$65 an hour")
    rf"[$£€]\s?{_AMOUNT}\s?(?:-|–|to)\s?[$£€]?\s?{_AMOUNT}(?:{_PERIOD})?|[$£€]\s?{_AMOUNT}{_PERIOD}"
)


def job_facts(jd: str) -> dict:
    """{company, work_mode, work_mode_detail, salary} from plain patterns: hybrid before remote before on-site."""
    if HYBRID_RE.search(jd):
        mode = "Hybrid"
    elif REMOTE_RE.search(jd):
        mode = "Remote"
    elif ONSITE_RE.search(jd):
        mode = "On-site"
    else:
        mode = "Not stated"
    days = OFFICE_DAYS_RE.search(jd) if mode == "Hybrid" else None
    pay = SALARY_RE.search(jd)
    return {"company": guess_company(jd), "work_mode": mode, "work_mode_detail": days.group(0).strip() if days else "",
            "salary": pay.group(0).strip(" .,") if pay else ""}


# Each job's first bullet with estimated figures: the job's biggest win, with its size and its result. The result is
# written in a different shape each time, as the prompt asks: its base in brackets (once), its base in words, or the
# real numbers alone.
LEADS = [
    ("Rebuilt the {k1} pipeline behind the main customer workflow for {size}, cutting failed nightly runs by {result} (from {was} to {now} a month) with staged rollouts and clear ownership.",
     "Rebuild the {k1} pipeline behind the main customer workflow for {size}, cutting failed nightly runs by {result} (from {was} to {now} a month) with staged rollouts and clear ownership."),
    ("Launched {k1} self-service reporting for {size}, so the time from a request to an answer dropped by {result}, from {was} to {now} hours, once shared dashboards replaced spreadsheet hand-offs.",
     "Launch {k1} self-service reporting for {size}, so the time from a request to an answer drops by {result}, from {was} to {now} hours, as shared dashboards replace spreadsheet hand-offs."),
    ("Moved {k1} order processing for {size} onto a versioned, tested setup, and support tickets about missing orders fell from {was} to {now} a week within two quarters.",
     "Move {k1} order processing for {size} onto a versioned, tested setup, so support tickets about missing orders keep falling, from {was} to {now} a week so far this year."),
]
LEAD_SIZES = ["38 internal teams", "12.4K active users", "613 stores", "47 partner companies", "83K customers",
              "23 regional offices", "4.6K merchants", "2.3K clinics", "36 product teams"]
LEAD_RESULTS = ["17.8%", "21%", "12.6%", "8.3%", "19%", "26.1%", "14%", "31.7%", "9.4%"]
# Each result's base, before and after: it agrees with the percentage (101 to 83 is 17.8%), as the prompt asks; a
# result written as the real numbers alone uses only these
LEAD_BASES = [(101, 83), (62, 49), (87, 76), (96, 88), (63, 51), (46, 34), (43, 37), (41, 28), (64, 58)]
LEVELS = ["Junior", "", "Senior", "Lead", "Principal"]
SCOPES = ["implemented", "built", "designed", "owned", "led"]
COMPANY_RE = re.compile(
    r"(?:\bCompany:[ \t]*|\bAbout[ \t]+|\bJoin[ \t]+|\bat[ \t]+)"
    r"([A-Z][A-Za-z0-9&'-]*(?:\.[A-Za-z0-9]+)*(?:[ \t]+[A-Z][A-Za-z0-9&'-]*(?:\.[A-Za-z0-9]+)*){0,3})"
)

CONCISE = [
    ("Migrated legacy {k1} jobs to a cleaner, versioned setup, running old and new side by side until the results matched so the switch caused no downtime for users.",
     "Migrate legacy {k1} jobs to a cleaner, versioned setup, running old and new side by side until the results match so each switch causes no downtime for users."),
    ("Streamlined {k3} reporting by replacing spreadsheet hand-offs with a shared, automated pipeline, so managers saw current numbers each morning instead of weekly exports.",
     "Streamline {k3} reporting by replacing spreadsheet hand-offs with a shared, automated pipeline, so managers see current numbers each morning instead of weekly exports."),
    ("Designed {k1} components for the core customer workflow, choosing simple, testable interfaces so cross-functional teams could ship changes with fewer regressions.",
     "Design {k1} components for the core customer workflow, choosing simple, testable interfaces so cross-functional teams can ship changes with fewer regressions."),
    ("Built {k2} automation for recurring operational work, replacing manual hand-offs and giving the team reliable, repeatable runs it no longer had to babysit.",
     "Build {k2} automation for recurring operational work, replacing manual hand-offs and giving the team reliable, repeatable runs it no longer has to babysit."),
    ("Partnered with product and business stakeholders to turn loosely defined requests into scoped {k3} deliverables with clear acceptance criteria and fewer late changes.",
     "Partner with product and business stakeholders to turn loosely defined requests into scoped {k3} deliverables with clear acceptance criteria and fewer late changes."),
    ("Documented {k1} and {k2} conventions and coached teammates through reviews and pairing, making onboarding faster and work across the team more consistent.",
     "Document {k1} and {k2} conventions and coach teammates through reviews and pairing, making onboarding faster and work across the team more consistent."),
    ("Reworked {k3} processes that caused repeated rework, tracing failure points with the owning teams and rolling out fixes that kept issues from coming back.",
     "Rework {k3} processes that cause repeated rework, tracing failure points with the owning teams and rolling out fixes that keep issues from coming back."),
    ("Introduced {k2} monitoring and alerting for customer-facing services, catching problems before users reported them and feeding post-incident lessons into planning.",
     "Introduce {k2} monitoring and alerting for customer-facing services, catching problems before users report them and feeding post-incident lessons into planning."),
    ("Hardened {k2} deployments with staged rollouts, health checks and quick rollback steps, letting the team release during working hours without late-night follow-ups.",
     "Harden {k2} deployments with staged rollouts, health checks and quick rollback steps, letting the team release during working hours without late-night follow-ups."),
    ("Standardized how {k1} and {k3} work was reviewed and tested, writing checklists and examples that cut back-and-forth in reviews as the team grew.",
     "Standardize how {k1} and {k3} work is reviewed and tested, writing checklists and examples that cut back-and-forth in reviews as the team grows."),
]
DETAILED = [
    ("Migrated a set of legacy {k1} jobs to a versioned, documented setup, running both side by side with the owning teams until every output matched, so the cut-over caused no downtime or data loss.",
     "Migrate legacy {k1} jobs to a versioned, documented setup, running both side by side with the owning teams until every output matches, so each cut-over causes no downtime or data loss."),
    ("Streamlined recurring {k3} reporting by replacing manual spreadsheet hand-offs with a shared automated pipeline and clear ownership, so managers saw current numbers every morning instead of stale weekly exports.",
     "Streamline recurring {k3} reporting by replacing manual spreadsheet hand-offs with a shared automated pipeline and clear ownership, so managers see current numbers every morning instead of stale weekly exports."),
    ("Cut release lead time from 47 to 36 days, a 23.4% drop, by redesigning {k1} workflows across teams, aligning technical decisions with business goals, documenting trade-offs for stakeholders, and keeping every release predictable and well tested.",
     "Cut release lead time from 47 to 36 days, a 23.4% drop, by redesigning {k1} workflows across teams, aligning technical decisions with business goals, documenting trade-offs for stakeholders, and keeping every release predictable and well tested."),
    ("Owned the day-to-day execution of {k2} initiatives from intake to hand-off, coordinating with product, operations and customer-facing teams, tracking risks openly, and communicating status in plain language to leadership.",
     "Own the day-to-day execution of {k2} initiatives from intake to hand-off, coordinating with product, operations and customer-facing teams, tracking risks openly, and communicating status in plain language to leadership."),
    ("Partnered with stakeholders to translate ambiguous requirements into clear {k3} deliverables, validating assumptions early, agreeing on acceptance criteria, and reducing late changes during delivery.",
     "Partner with stakeholders to translate ambiguous requirements into clear {k3} deliverables, validating assumptions early, agreeing on acceptance criteria, and reducing late changes during delivery."),
    ("Mentored colleagues on {k1} and {k2} practices through reviews, pairing sessions and short written guides, raising consistency and shortening onboarding for new team members.",
     "Mentor colleagues on {k1} and {k2} practices through reviews, pairing sessions and short written guides, raising consistency and shortening onboarding for new team members."),
    ("Improved {k3} processes by identifying recurring failure points, proposing practical fixes, and following up with owners until the changes were adopted and measured.",
     "Improve {k3} processes by identifying recurring failure points, proposing practical fixes, and following up with owners until the changes are adopted and measured."),
    ("Monitored {k2} health with simple dashboards and alerts, triaged issues quickly, and wrote post-incident notes that fed back into the team's planning.",
     "Monitor {k2} health with simple dashboards and alerts, triage issues quickly, and write post-incident notes that feed back into the team's planning."),
    ("Hardened {k2} deployments with staged rollouts, automated health checks and one-step rollbacks agreed with operations, letting the team release during working hours without late-night follow-ups.",
     "Harden {k2} deployments with staged rollouts, automated health checks and one-step rollbacks agreed with operations, letting the team release during working hours without late-night follow-ups."),
    ("Standardized how {k1} and {k3} work was reviewed and tested across the group, writing short checklists and worked examples that cut back-and-forth in reviews as the team doubled.",
     "Standardize how {k1} and {k3} work is reviewed and tested across the group, writing short checklists and worked examples that cut back-and-forth in reviews as the team grows."),
]


def extract_keywords(jd: str, limit: int = 16) -> list[str]:
    found: list[tuple[int, str]] = []
    for term in VOCAB:
        flags = 0 if len(term) <= 3 else re.IGNORECASE
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9+#])"
        m = re.search(pattern, jd, flags)
        if m:
            found.append((m.start(), term))
    found.sort()
    keywords = [t for _, t in found]
    if len(keywords) < 6:
        for word in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", jd):
            if word not in STOP and word not in keywords:
                keywords.append(word)
            if len(keywords) >= 8:
                break
    for item in GENERIC:
        if len(keywords) >= 6:
            break
        if item not in keywords:
            keywords.append(item)
    return keywords[:limit]


def guess_role(jd: str) -> tuple[str, str]:
    """Return (seniority prefix, base role)."""
    match = ROLE_RE.search(jd)
    if not match:
        return "", "Professional"
    prefix = (match.group(1) or "").strip().rstrip(".")
    prefix = {"Sr": "Senior", "Jr": "Junior"}.get(prefix, prefix)
    return prefix, match.group(2).strip()


def first_project(notes: str) -> str:
    """The first product or project a person's notes name, as the model reads them (no links or sources): the first
    line's words before a colon, a bracket or a dash, at most five ('Order tracking app (2023): rebuilt its API' ->
    'Order tracking app')."""
    for line in clean_notes(notes).splitlines():
        words = re.split(r"[:(]| - | – ", line.strip(" -*•\t"))[0].split()
        if words:
            return " ".join(words[:5])
    return ""


def guess_company(jd: str) -> str:
    """'... at Stripe', 'About Acme Corp', 'Company: Globex' -> the name; '' when the JD names none."""
    for match in COMPANY_RE.finditer(jd):
        name = match.group(1).strip(" .,'-")
        if name and name.split()[0] not in STOP and name not in VOCAB:
            return name
    return ""


def fits_era(term: str, index: int, end: str) -> bool:
    """Whether a job (the index-th most recent, ending `end`) could have used the term: not a tool that became
    common after the job ended, nor an AI tool outside the most recent jobs (the rules era_problems checks)."""
    from ..pipeline import validate  # late: the pipeline imports the LLM clients

    if index >= validate.AI_RECENT_JOBS and any(validate._mentions_tool(term, t) for t in validate.AI_TOOLS):
        return False
    probe = validate.parse_point(end, is_end=True)
    if probe is None or end.strip().lower() in validate._CURRENT:
        return True
    years = {**validate.COMMON_YEAR, **validate.AI_TOOLS}
    return not any(year > probe[0] and validate._mentions_tool(term, t) for t, year in years.items())


def _title_ladder(prefix: str, base: str, count: int) -> list[str]:
    top = LEVELS.index(prefix) if prefix in LEVELS else 2
    titles = []
    for i in range(count):
        level = LEVELS[max(top - i, 0)]
        titles.append(f"{level} {base}".strip())
    return titles



# Real certifications the demo suggests when the job description mentions these words (name, issuer, words).
DEMO_CERTIFICATIONS = [
    ("AWS Certified Solutions Architect Professional", "Amazon Web Services", ("aws", "amazon web services")),
    ("Microsoft Certified: Azure Solutions Architect Expert", "Microsoft", ("azure",)),
    ("Google Cloud Professional Cloud Architect", "Google Cloud", ("gcp", "google cloud")),
    ("Certified Kubernetes Administrator (CKA)", "The Linux Foundation", ("kubernetes", "k8s")),
    ("Certified Information Systems Security Professional (CISSP)", "ISC2", ("security", "cissp")),
    ("Project Management Professional (PMP)", "Project Management Institute", ("project management", "pmp")),
    ("Certified ScrumMaster (CSM)", "Scrum Alliance", ("scrum", "agile")),
    ("Databricks Certified Data Engineer Professional", "Databricks", ("spark", "databricks", "data engineer")),
]


def demo_certifications(jd: str, own: list[dict]) -> list[dict]:
    """The demo's picks, like a real model's: the candidate's own first (by number), then real certifications the
    job's words point to, then common ones, 4 in all."""
    picks = [{"name": c.get("name", ""), "issuer": c.get("issuer", ""), "candidate_index": i}
             for i, c in enumerate(own) if c.get("name") and not is_expired(str(c.get("expires") or ""))][:4]
    low = jd.lower()
    matched = [row for row in DEMO_CERTIFICATIONS if any(word in low for word in row[2])]
    for name, issuer, _words in matched + [row for row in DEMO_CERTIFICATIONS if row not in matched]:
        if len(picks) == 4:
            break
        if not any(same_certification(name, p["name"]) for p in picks):
            picks.append({"name": name, "issuer": issuer, "candidate_index": -1})
    return picks

class FakeClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.style = spec.settings.get("style", "concise")

    async def generate_structured(self, *, system, user, schema, schema_name, max_tokens, context=None) -> LLMResult:
        await asyncio.sleep(float(os.environ.get("FAKE_DELAY", "0.15")))
        ctx = context or {}
        if self.style == "flaky" and RETRY_MARKER not in user and schema_name != "profile":
            data: dict = {"jd_analysis": {"target_role": "x"}} if schema_name == "tailor_output" else {"greeting": "Hi"}
        elif schema_name == "tailor_output":
            data = self._tailor(ctx)
        elif schema_name == "tailor_revision":
            data = self._revise(ctx)
        elif schema_name == "job_facts":
            data = job_facts(ctx.get("jd", ""))
        elif schema_name == "cover_letter":
            data = self._cover_letter(ctx)
        elif schema_name == "profile":
            data = parse_resume_text(ctx.get("resume_text", "")).model_dump()
        else:
            data = {}
        text = json.dumps(data)
        return LLMResult(
            data=data,
            usage=Usage(max(1, len(system + user) // 4), max(1, len(text) // 4)),
            served_model=self.spec.model,
            raw_text=text,
        )

    # ------------------------------------------------------------------ content
    def _revise(self, ctx: dict) -> dict:
        """What differs between the draft in the context and this model's own clean sections."""
        clean, draft = self._tailor(ctx)["sections"], ctx.get("draft") or {}
        own = {e["company_index"]: e for e in clean["experience"]}
        titles, bullets = [], []
        for exp in draft.get("experience", []):
            mine = own.get(exp["company_index"])
            if mine is None:
                continue
            if mine["job_title"] != exp["job_title"]:
                titles.append({"company_index": exp["company_index"], "job_title": mine["job_title"]})
            for index, (theirs, ours) in enumerate(zip(exp["bullets"], mine["bullets"])):
                if theirs != ours:
                    bullets.append({"company_index": exp["company_index"], "bullet_index": index, "text": ours})
        clean_skills = [skill for group in clean["skills"] for skill in group["skills"]]
        draft_skills = [skill for group in draft.get("skills", []) for skill in group["skills"]]
        missing = [s for s in clean_skills if s.lower() not in {d.lower() for d in draft_skills}]
        changes = [{"old": skill, "new": missing.pop(0) if missing else ""}
                   for skill in draft_skills if skill.lower() not in {c.lower() for c in clean_skills}]
        return {
            "headline": clean["headline"] if clean["headline"] != draft.get("headline") else "",
            "summary": clean["summary"] if clean["summary"] != draft.get("summary") else "",
            "titles": titles, "bullets": bullets, "skill_changes": changes,
        }

    def _tailor(self, ctx: dict) -> dict:
        jd = ctx.get("jd", "")
        companies = ctx.get("profile", {}).get("companies", [])
        keywords = extract_keywords(jd)
        prefix, base = guess_role(jd)
        seniority = prefix or "Mid-level"
        target = f"{prefix} {base}".strip()
        k = (keywords + GENERIC)[:12]
        titles = _title_ladder(prefix, base, len(companies))
        bank = DETAILED if self.style == "detailed" else CONCISE
        if self.style == "detailed":  # deliberately too long, to exercise the fit-to-two-pages steps; asked to
            # shorten, it follows the smaller budget it is given, as a real model would
            counts = [24, 22, 20] if not ctx.get("shorten") else (ctx.get("bullets") or [8, 6, 5])
        else:  # follows the bullet budget in the prompt, like a real model
            counts = ctx.get("bullets") or [4, 3, 3]

        experience = []
        for index, company in enumerate(companies):
            end = str(company.get("end", ""))
            present = end.lower() in ("present", "current", "")
            n = counts[min(index, len(counts) - 1)]
            if len(companies) == 1:  # one job cannot overflow two pages without repeating the whole bank
                n = min(n, len(bank))
            mine = [term for term in k if fits_era(term, index, end)] or GENERIC  # only what this job could use
            lead = first_project(str(company.get("projects", "")))  # the person's own work, from their notes
            bullets = []
            for j in range(n):
                past, now = bank[(index * 2 + j) % len(bank)]
                if j >= len(bank) and "23.4%" in past:  # the one invented figure: once per job
                    past, now = bank[(index * 2 + j + 1) % len(bank)]
                if j == 0 and ctx.get("metrics", "estimate") == "estimate":  # opens with its size and its result
                    past, now = LEADS[index % len(LEADS)]
                template = now if present else past
                bullets.append(template.format(k1=lead if j == 0 and lead else mine[j % len(mine)],
                                               k2=mine[(j + 1) % len(mine)],
                                               k3=mine[(j + 2) % len(mine)], size=LEAD_SIZES[index % len(LEAD_SIZES)],
                                               result=LEAD_RESULTS[index % len(LEAD_RESULTS)],
                                               was=LEAD_BASES[index % len(LEAD_BASES)][0],
                                               now=LEAD_BASES[index % len(LEAD_BASES)][1]))
            experience.append({"company_index": index, "job_title": titles[index], "bullets": bullets})

        groups = [
            {"category": "Core Skills", "skills": k[:6]},
            {"category": "Tools and Platforms", "skills": k[6:12] or k[:3]},
            {"category": "Ways of Working", "skills": ["Cross-functional collaboration", "Clear documentation", "Continuous improvement"]},
        ]
        return {
            "jd_analysis": {
                "company": guess_company(jd),
                "target_role": target,
                "general_role": base,
                "seniority": seniority,
                "domain": "General",
                "must_have_skills": keywords[:6],
                "nice_to_have_skills": keywords[6:10],
                "keywords": keywords,
            },
            # none while the Certifications section is off (Settings): a real model then lists only the job post's
            # own certifications, which are never printed and only keep their names out of the resume
            "certifications": (demo_certifications(jd, ctx.get("profile", {}).get("certifications", []) or [])
                               if ctx.get("certifications_section", True) else []),
            "career_plan": [  # recent jobs show more of the JD, with more scope
                {"company_index": i, "product": "", "focus": k[: max(2, 6 - 2 * i)],  # it knows no companies
                 "scope": SCOPES[max(0, len(SCOPES) - 2 - i)]}
                for i in range(len(companies))
            ],
            "sections": {
                "headline": base,
                "summary": (
                    f"{target} with hands-on experience delivering {k[0]}, {k[1]} and {k[2]} work for product and "
                    f"operations teams in fast-moving organizations. Strong in {k[3]}, {k[4]} and {k[5]}, with a habit of "
                    "turning loosely defined requirements into simple, well-tested deliverables that others can build "
                    "on. Has made day-to-day operations more reliable by automating repetitive steps and adding "
                    "monitoring where problems used to go unnoticed. Brings ownership from planning to hand-off and "
                    "close collaboration with stakeholders and customer-facing teams."
                ),
                "skills": groups,
                "experience": experience,
            },
        }

    def _cover_letter(self, ctx: dict) -> dict:
        company = ctx.get("company_name") or "your company"
        manager = ctx.get("hiring_manager") or "Hiring Manager"
        role = ctx.get("target_role") or "the open role"
        keywords = (ctx.get("keywords") or GENERIC)[:3] + GENERIC[:3]
        return {
            "greeting": f"Dear {manager},",
            "paragraphs": [
                f"I am writing to apply for the {role} position at {company}. My background in {keywords[0]} and {keywords[1]} matches what the role asks for.",
                f"In my recent roles I have worked closely with stakeholders, kept delivery predictable, and built practical experience with {keywords[2]}. I focus on clear communication and on leaving work in a state others can build on.",
                f"I would welcome the chance to discuss how I can contribute to {company}. Thank you for your time and consideration.",
            ],
            "closing": "Sincerely,",
        }
