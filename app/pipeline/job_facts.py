"""Where a pasted job is done and what it pays: one small AI call, shown under the job box before tailoring."""
from __future__ import annotations

import asyncio

from ..llm.router import Router
from ..llm.schema import strict_json_schema
from ..parsing.resume_parser import record_cost
from ..prompts import job_facts_system, job_facts_user
from ..schemas import JobFacts
from ..storage.db import Database
from .generate import Outcome, generate_validated

FACTS_TIMEOUT_SECONDS = 25.0  # a check that takes longer is given up, so it never holds up the site


def facts_model(router: Router) -> str | None:
    """The cheap model chosen for reading resumes, otherwise the resume model."""
    return router.default_model("parser") or router.default_model("resume")


async def check_job_facts(router: Router, db: Database, jd: str, *, user_id: int | None = None) -> JobFacts:
    """Raises LookupError when no model is available, CostCapError, GenerationFailed, or TimeoutError after
    FACTS_TIMEOUT_SECONDS. What was spent is recorded either way."""
    model_id = facts_model(router)
    if not model_id:
        raise LookupError("No model is available. Add an API key, or choose a model in Settings.")
    outcome = Outcome()
    try:
        await asyncio.wait_for(generate_validated(
            router, [model_id], system=job_facts_system(), user=job_facts_user(jd),
            schema=strict_json_schema(JobFacts), schema_name="job_facts",
            parse=JobFacts.model_validate, context={"jd": jd}, outcome=outcome,
        ), FACTS_TIMEOUT_SECONDS)
    finally:
        record_cost(db, model_id, outcome, user_id, task="job_facts")
    return outcome.value  # type: ignore[return-value]
