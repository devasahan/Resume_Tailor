"""Run the same job description through several models at once (used by the CLI bake-off)."""
from __future__ import annotations

import asyncio

from ..llm.router import Router
from ..schemas import Profile, TailorResult
from ..storage.db import Database
from .tailor import run_tailor


async def compare(
    router: Router,
    db: Database,
    *,
    profile: Profile,
    profile_id: int | None,
    jd: str,
    model_ids: list[str],
    hints: str = "",
    cover_letter: bool = False,
    company_name: str = "",
    hiring_manager: str = "",
    concurrency: int = 6,
    save: bool = True,
    spelling: str = "us",
) -> list[TailorResult]:
    """Fallback is off here: each result must show what that model alone produced."""
    gate = asyncio.Semaphore(concurrency)

    async def one(model_id: str) -> TailorResult:
        async with gate:
            return await run_tailor(
                router, db, profile=profile, profile_id=profile_id, jd=jd, hints=hints, model_id=model_id,
                allow_fallback=False, cover_letter=cover_letter, company_name=company_name,
                hiring_manager=hiring_manager, save=save, spelling=spelling,
            )

    return list(await asyncio.gather(*(one(m) for m in model_ids)))
