"""Offline demo provider. No network, no key, no cost.

It builds generic but structurally valid output from the job description so the whole app (compare
mode, review, fit-to-one-page, export, history) can be tried before any API key exists. Its text
is template-based and is NOT a measure of what real models write.

Styles:  concise | detailed (long bullets, includes one invented number so the check can flag it)
         flaky (returns invalid JSON until the retry feedback appears in the prompt)
"""
from __future__ import annotations

import asyncio
import json
import os
import re

from ..parsing.heuristic import parse_resume_text
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
LEVELS = ["Junior", "", "Senior", "Lead", "Principal"]

CONCISE = [
    ("Delivered {k1} solutions for cross-functional teams, improving reliability and turnaround.",
     "Deliver {k1} solutions for cross-functional teams, improving reliability and turnaround."),
    ("Built and maintained {k2} workflows that supported day-to-day operations.",
     "Build and maintain {k2} workflows that support day-to-day operations."),
    ("Partnered with stakeholders to translate requirements into {k3} deliverables.",
     "Partner with stakeholders to translate requirements into {k3} deliverables."),
    ("Documented {k1} practices and coached teammates on {k2}.",
     "Document {k1} practices and coach teammates on {k2}."),
    ("Reviewed and improved {k3} processes to reduce rework.",
     "Review and improve {k3} processes to reduce rework."),
    ("Monitored {k2} performance and resolved issues before they reached customers.",
     "Monitor {k2} performance and resolve issues before they reach customers."),
]
DETAILED = [
    ("Increased delivery speed by 30% by redesigning {k1} workflows across several teams, aligning technical decisions with business goals, documenting trade-offs for stakeholders, and keeping every release predictable and well tested.",
     "Increase delivery speed by 30% by redesigning {k1} workflows across several teams, aligning technical decisions with business goals, documenting trade-offs for stakeholders, and keeping every release predictable and well tested."),
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


def _title_ladder(prefix: str, base: str, count: int) -> list[str]:
    top = LEVELS.index(prefix) if prefix in LEVELS else 2
    titles = []
    for i in range(count):
        level = LEVELS[max(top - i, 0)]
        titles.append(f"{level} {base}".strip())
    return titles


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
        counts = [6, 5, 4] if self.style == "detailed" else [4, 3, 3]
        if ctx.get("shorten"):
            counts = [max(2, c - 2) for c in counts]

        experience = []
        for index, company in enumerate(companies):
            present = str(company.get("end", "")).lower() in ("present", "current", "")
            n = counts[min(index, len(counts) - 1)]
            bullets = []
            for j in range(n):
                past, now = bank[(index * 2 + j) % len(bank)]
                template = now if present else past
                bullets.append(template.format(k1=k[(j) % len(k)], k2=k[(j + 1) % len(k)], k3=k[(j + 2) % len(k)]))
            experience.append({"company_index": index, "job_title": titles[index], "bullets": bullets})

        groups = [
            {"category": "Core Skills", "skills": k[:6]},
            {"category": "Tools and Platforms", "skills": k[6:12] or k[:3]},
            {"category": "Ways of Working", "skills": ["Cross-functional collaboration", "Clear documentation", "Continuous improvement"]},
        ]
        return {
            "jd_analysis": {
                "target_role": target,
                "seniority": seniority,
                "domain": "General",
                "must_have_skills": keywords[:6],
                "nice_to_have_skills": keywords[6:10],
                "keywords": keywords,
                "key_responsibilities": ["Deliver reliable work", "Collaborate across teams", "Communicate progress clearly"],
            },
            "sections": {
                "headline": f"{target} | {k[0]} | {k[1]} | {k[2]}",
                "summary": (
                    f"{target} with hands-on experience in {k[0]}, {k[1]} and {k[2]}. "
                    "Known for turning requirements into dependable results and working closely with stakeholders to deliver on schedule."
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
