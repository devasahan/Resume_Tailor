"""Command-line tools.

  python -m app.cli models                      list models and whether each is usable
  python -m app.cli check --models a,b          one tiny real call per model (cents) to verify key, name, JSON mode
  python -m app.cli bakeoff --jds evals/jds --profile evals/sample_profile.json --models a,b,c
        run every job description through every model, write results.csv, summary.md and the PDFs to out/
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import sys
import time
from pathlib import Path

from .llm.base import LLMError
from .llm.registry import Registry
from .llm.router import Router
from .pipeline.compare import compare
from .render.pdf import render_resume_pdf
from .schemas import Profile
from .storage.db import Database

EST_IN, EST_OUT = 4000, 5000  # tokens per run, the planning assumption used for the cost estimate


def _setup() -> tuple[Registry, Database, Router]:
    registry, db = Registry.load(), Database()
    return registry, db, Router(registry, db)


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in rows) if rows else 0.0


def cmd_models(args) -> int:
    registry, db, _router = _setup()
    regions = db.get_settings()["allowed_regions"]
    print(f"{'id':24} {'ready':6} {'$in/$out per 1M':18} {'region':8} note")
    for spec in registry.all():
        status = registry.status(spec, regions)
        price = "free" if spec.provider == "fake" else f"{spec.price_in:g} / {spec.price_out:g}"
        flag = "yes" if status.available else "no"
        print(f"{spec.id:24} {flag:6} {price:18} {spec.data_region:8} {status.reason}")
    return 0


async def _check(router: Router, ids: list[str]) -> int:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    failures = 0
    for model_id in ids:
        started = time.perf_counter()
        try:
            result, spec = await router.call(
                model_id, system="Reply with one JSON object.", user='Return {"ok": true}.', schema=schema, schema_name="check"
            )
            ms = int((time.perf_counter() - started) * 1000)
            print(f"OK    {model_id:24} answered as {result.served_model!r} in {ms} ms, cost ${spec.cost(result.usage):.5f}")
        except LLMError as exc:
            failures += 1
            print(f"FAIL  {model_id:24} {exc}")
    return failures


def cmd_check(args) -> int:
    registry, db, router = _setup()
    regions = db.get_settings()["allowed_regions"]
    ids = args.models.split(",") if args.models else [s.id for s in registry.available(regions) if s.provider != "fake"]
    if not ids:
        print("No real models are available. Set API keys in .env (see .env.example).")
        return 1
    return 1 if asyncio.run(_check(router, ids)) else 0


def _estimate(registry: Registry, ids: list[str], runs_per_model: int) -> float:
    total = 0.0
    for model_id in ids:
        spec = registry.get(model_id)
        total += runs_per_model * (EST_IN * spec.price_in + EST_OUT * spec.price_out) / 1e6
    return total


async def _bakeoff(args, registry: Registry, db: Database, router: Router, profile: Profile, ids: list[str], jds: list[Path]) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hints = Path(args.hints).read_text(encoding="utf-8") if args.hints else ""
    rows: list[dict] = []
    for jd_path in jds:
        jd = jd_path.read_text(encoding="utf-8")
        print(f"- {jd_path.name}: running {len(ids)} model(s)...")
        results = await compare(router, db, profile=profile, profile_id=None, jd=jd, model_ids=ids, hints=hints,
                                cover_letter=args.cover, save=False)
        for r in results:
            verify = len([f for f in r.flags if f.level == "verify"])
            rows.append({
                "jd": jd_path.name, "model": r.requested_model, "served_model": r.served_model, "status": r.status,
                "one_page": int(r.fits), "pages": r.pages, "keyword_match": r.coverage.score, "to_verify": verify,
                "format_warnings": len(r.ats_warnings), "attempts": r.attempts, "cost_usd": r.cost_usd,
                "seconds": round(r.latency_ms / 1000, 1), "tokens": r.input_tokens + r.output_tokens, "error": r.error[:200],
            })
            if r.status == "ok" and r.resume:
                folder = out / jd_path.stem
                folder.mkdir(exist_ok=True)
                pdf, _ = render_resume_pdf(r.resume, r.style)
                (folder / f"{r.requested_model}.pdf").write_bytes(pdf)
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Bake-off summary", "", f"{len(jds)} job description(s), {len(ids)} model(s).", "",
             "| Model | OK runs | One page | Keyword match | To verify (avg) | Cost per resume | Seconds (avg) | Failures |",
             "|---|---|---|---|---|---|---|---|"]
    for model_id in ids:
        mine = [r for r in rows if r["model"] == model_id]
        ok = [r for r in mine if r["status"] == "ok"]
        lines.append(
            f"| {model_id} | {len(ok)}/{len(mine)} | {sum(r['one_page'] for r in ok)}/{len(ok)} | {_mean(ok, 'keyword_match'):.0%} | "
            f"{_mean(ok, 'to_verify'):.1f} | ${_mean(mine, 'cost_usd'):.4f} | {_mean(ok, 'seconds'):.1f} | {len(mine) - len(ok)} |"
        )
    lines += ["", "Quality is not scored automatically: open the PDFs in each job folder and compare them blind.",
              "Lower 'To verify' means fewer numbers the model invented. Keyword match is a coverage signal, not a quality score."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}/results.csv, {out}/summary.md and PDFs under {out}/<job>/")


def cmd_bakeoff(args) -> int:
    registry, db, router = _setup()
    regions = db.get_settings()["allowed_regions"]
    ids = args.models.split(",")
    usable = [m for m in ids if m in {s.id for s in registry.available(regions)}]
    for missing in sorted(set(ids) - set(usable)):
        print(f"skipping {missing}: not available (no API key, unknown id, or region not allowed)")
    if not usable:
        return 1
    jds = sorted(Path(args.jds).glob("*.txt"))
    if not jds:
        print(f"No .txt job descriptions found in {args.jds}")
        return 1
    profile = Profile.model_validate_json(Path(args.profile).read_text(encoding="utf-8"))
    estimate = _estimate(registry, usable, len(jds))
    print(f"{len(jds)} job description(s) x {len(usable)} model(s) = {len(jds) * len(usable)} runs. "
          f"Estimated cost about ${estimate:.2f} (assumes {EST_IN} input and {EST_OUT} output tokens per run).")
    if estimate > 0 and not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return 1
    asyncio.run(_bakeoff(args, registry, db, router, profile, usable, jds))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list models and availability").set_defaults(fn=cmd_models)
    check = sub.add_parser("check", help="verify keys / model names with one tiny call each")
    check.add_argument("--models", help="comma-separated ids (default: every available real model)")
    check.set_defaults(fn=cmd_check)
    bake = sub.add_parser("bakeoff", help="run job descriptions through several models")
    bake.add_argument("--jds", default="evals/jds", help="folder of .txt job descriptions")
    bake.add_argument("--profile", default="evals/sample_profile.json", help="profile JSON (the fixed facts)")
    bake.add_argument("--models", required=True, help="comma-separated model ids")
    bake.add_argument("--hints", help="text file with real facts about the candidate")
    bake.add_argument("--cover", action="store_true", help="also write cover letters")
    bake.add_argument("--out", default="out/bakeoff")
    bake.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    bake.set_defaults(fn=cmd_bakeoff)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
