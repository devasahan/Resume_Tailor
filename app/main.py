"""FastAPI app: JSON API + server-rendered pages. Single user, no accounts: put your own auth in front
of it before exposing it beyond localhost."""
from __future__ import annotations

import re
import secrets
import unicodedata
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import config
from .llm.base import LLMError
from .llm.registry import ModelSpec, Registry
from .llm.router import Router
from .parsing.resume_parser import MAX_BYTES, extract_text, parse_resume, profile_warnings, read_resume
from .pipeline import validate
from .prompts import METRICS_MODES
from .pipeline.tailor import analysis_terms, fit_report, run_checks, run_tailor, write_cover_letter
from .render.docx import cover_docx, resume_docx
from .render.pdf import render_cover_pdf, render_resume_pdf
from .schemas import ExperienceOut, Profile, SkillGroup, TailoredSections, TailorOutput, TailorResult
from .storage.db import Database

WEB = Path(__file__).parent / "web"
KNOWN_REGIONS = ["US", "EU", "SG", "CN", "UNKNOWN", "DEMO"]
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ------------------------------------------------------------------ request bodies
class TailorRequest(BaseModel):
    jd: str = Field(min_length=40, max_length=20000)
    profile_id: int | None = None  # which uploaded resume; None = the newest one
    hints: str = Field(default="", max_length=4000)
    model: str | None = None  # None = the resume model chosen in Settings
    compare: bool = False  # True: no fallback, no per-task cover-letter model (each model stands on its own)
    cover_letter: bool = False
    company_name: str = Field(default="", max_length=120)
    hiring_manager: str = Field(default="", max_length=120)


class EditRequest(BaseModel):
    headline: str
    summary: str
    skills: list[SkillGroup]
    experience: list[ExperienceOut]
    cover_greeting: str | None = None
    cover_paragraphs: list[str] | None = None
    cover_closing: str | None = None


class CoverRequest(BaseModel):
    # Only needed for runs saved before job descriptions were kept with the run.
    jd: str = Field(default="", max_length=20000)


class CheckRequest(BaseModel):
    models: list[str]


def profile_problem(profile: Profile) -> str | None:
    """Why a profile cannot be used for tailoring, or None."""
    if not profile.name.strip():
        return "your name"
    if not profile.companies or any(not c.name.strip() for c in profile.companies):
        return "any employer"
    return None


def resume_entry(entry: dict, registry: Registry) -> dict:
    """An uploaded resume as the Profile and Tailor pages show it."""
    method = entry["method"]
    if method.startswith("model:"):
        model_id = method.split(":", 1)[1]
        try:
            read_with = registry.get(model_id).label
        except KeyError:
            read_with = model_id
    else:
        read_with = {"rules": "built-in rules", "form": "typed in"}.get(method, "earlier version")
    profile: Profile = entry["profile"]
    return {
        "id": entry["id"], "created_at": entry["created_at"], "filename": entry["filename"], "method": method,
        "read_with": read_with, "profile": profile.model_dump(),
        "warnings": entry["notes"] + profile_warnings(profile),
    }


UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def cross_site(request: Request) -> bool:
    """True for a state-changing request sent by another website. Browsers label every request with
    Sec-Fetch-Site and send Origin on cross-origin posts; a plain HTML form on any other page (which needs no
    CORS preflight, e.g. a multipart upload) is caught here. Scripts and tests send neither header."""
    site = request.headers.get("sec-fetch-site")
    if site and site not in ("same-origin", "none"):
        return True
    origin = request.headers.get("origin")
    if origin and origin != "null":
        return urlsplit(origin).netloc.lower() != request.headers.get("host", "").lower()
    return origin == "null"


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    token = config.admin_token()
    if token and not secrets.compare_digest(x_admin_token or "", token):
        raise HTTPException(401, "Admin token required (set it on the Settings page).")


def file_part(text: str, limit: int = 50) -> str:
    """'José Álvarez' -> 'Jose_Alvarez' (accents become plain letters, everything else becomes _)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")[:limit].strip("_")


def download_name(result: TailorResult, ext: str, cover: bool = False) -> str:
    """Company applied to, name on the resume, target role: 'Stripe_Jordan_Rivera_Senior_Data_Engineer.pdf'.

    The company typed on the tailor form wins over the one read from the JD; missing parts are left out.
    """
    assert result.resume is not None
    analysis = result.analysis
    company = result.company_name or (analysis.company if analysis else "")
    role = analysis.target_role if analysis else ""
    if not role and result.resume.experience:
        role = result.resume.experience[0].job_title
    parts = [file_part(company, 40), file_part(result.resume.profile.name, 40), file_part(role, 60)]
    if cover:
        parts.append("Cover_Letter")
    base = "_".join(p for p in parts if p) or "Resume"
    return f"{base}.{ext}"


def model_info(registry: Registry, spec: ModelSpec, allowed_regions: list[str]) -> dict:
    status = registry.status(spec, allowed_regions)
    return {
        "id": spec.id, "label": spec.label, "provider": spec.provider, "model": spec.model,
        "price_in": spec.price_in, "price_out": spec.price_out, "data_region": spec.data_region,
        "verified": spec.verified, "notes": spec.notes, "available": status.available, "reason": status.reason,
        "demo": spec.provider == "fake",
        "key_hint": config.key_hint(spec.key_env) if spec.key_env else "",
    }


def validate_settings(registry: Registry, values: dict) -> dict:
    out: dict = {}
    for key, value in values.items():
        try:
            if key in ("resume_model", "cover_letter_model", "parser_model"):
                if value in (None, ""):
                    out[key] = None
                elif key == "parser_model" and value == config.RULES_READER:
                    out[key] = value
                else:
                    registry.get(value)
                    out[key] = value
            elif key == "fallback_models":
                for model_id in value:
                    registry.get(model_id)
                out[key] = list(dict.fromkeys(value))
            elif key == "daily_cost_cap_usd":
                out[key] = max(0.0, float(value))
            elif key == "allowed_regions":
                bad = set(value) - set(KNOWN_REGIONS)
                if bad:
                    raise ValueError(f"unknown region(s): {', '.join(sorted(bad))}")
                out[key] = list(value)
            elif key == "retention_days":
                out[key] = max(0, min(3650, int(value)))
            elif key == "metrics_mode":
                if value not in METRICS_MODES:
                    raise ValueError(f"metrics_mode must be one of {', '.join(METRICS_MODES)}")
                out[key] = value
            elif key == "paper_size":
                if value not in ("Letter", "A4"):
                    raise ValueError("paper_size must be Letter or A4")
                out[key] = value
            else:
                raise ValueError(f"unknown setting '{key}'")
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(422, f"Invalid value for '{key}': {exc}") from exc
    return out


def create_app(db: Database | None = None, registry: Registry | None = None, router: Router | None = None) -> FastAPI:
    db = db or Database()
    registry = registry or Registry.load()
    router = router or Router(registry, db)
    db.purge(db.get_settings()["retention_days"])

    app = FastAPI(title="Resume Tailor")
    app.state.db, app.state.registry, app.state.router = db, registry, router
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        if request.method in UNSAFE_METHODS and cross_site(request):
            return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; frame-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'",
            )
        return response

    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
    templates = Jinja2Templates(directory=WEB / "templates")

    def page(request: Request, name: str, active: str, **ctx):
        return templates.TemplateResponse(request, name, {"active": active, **ctx})

    def regions() -> list[str]:
        return db.get_settings()["allowed_regions"]

    def load_run(run_id: int):
        result = db.get_run(run_id)
        if result is None:
            raise HTTPException(404, "Run not found (it may have passed the retention period).")
        return result

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return RedirectResponse("/tailor")

    @app.get("/profile")
    def profile_page(request: Request):
        return page(request, "profile.html", "profile")

    @app.get("/tailor")
    def tailor_page(request: Request):
        return page(request, "tailor.html", "tailor")

    @app.get("/history")
    def history_page(request: Request):
        return page(request, "history.html", "history")

    @app.get("/settings")
    def settings_page(request: Request):
        return page(request, "settings.html", "settings")

    @app.get("/run/{run_id}")
    def run_page(request: Request, run_id: int):
        load_run(run_id)
        return page(request, "run.html", "history", run_id=run_id)

    # ------------------------------------------------------------------ models + settings
    @app.get("/api/models")
    def api_models():
        allowed = regions()
        return {"models": [model_info(registry, s, allowed) for s in registry.all()], "regions": KNOWN_REGIONS}

    @app.get("/api/settings")
    def api_settings():
        settings = db.get_settings()
        return {
            "settings": settings,
            "resolved": {t: router.default_model(t) for t in ("resume", "cover_letter", "parser")},
            "automatic": {t: router.default_model(t, automatic=True) for t in ("resume", "cover_letter", "parser")},
            "cost_today": round(db.cost_today(), 4),
            "admin_token_required": bool(config.admin_token()),
        }

    @app.put("/api/settings", dependencies=[Depends(require_admin)])
    def api_set_settings(values: dict):
        updated = db.set_settings(validate_settings(registry, values))
        db.purge(updated["retention_days"])
        return api_settings()

    @app.post("/api/models/check", dependencies=[Depends(require_admin)])
    async def api_check(body: CheckRequest):
        """One tiny real call per model: verifies the key, the model name and the JSON mode. Costs cents."""
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
        results = []
        for model_id in body.models:
            started = time.perf_counter()
            try:
                result, spec = await router.call(
                    model_id, system="Reply with one JSON object.", user='Return {"ok": true}.',
                    schema=schema, schema_name="check",
                )
                results.append({"id": model_id, "ok": True, "served_model": result.served_model,
                                "ms": int((time.perf_counter() - started) * 1000), "cost_usd": round(spec.cost(result.usage), 6)})
            except LLMError as exc:
                results.append({"id": model_id, "ok": False, "error": str(exc)[:300]})
        return {"results": results}

    # ------------------------------------------------------------------ profile
    @app.post("/api/profile/parse")
    async def api_parse(file: UploadFile = File(...), model: str = Form("")):
        data = await file.read(MAX_BYTES + 1)  # never buffer more than the limit
        try:
            text = extract_text(data, file.filename or "")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        model_id = None if model in ("", "rules") else model
        if model_id:
            try:
                router.usable_spec(model_id)
            except LLMError as exc:
                raise HTTPException(422, str(exc)) from exc
        profile, method, warnings = await parse_resume(router, db, text, model_id)
        return {"profile": profile.model_dump(), "method": method, "warnings": warnings, "text_chars": len(text)}

    @app.get("/api/profile")
    def api_get_profile():
        found = db.latest_profile()
        if not found:
            return {"profile": None, "warnings": []}
        profile_id, profile = found
        return {"id": profile_id, "profile": profile.model_dump(), "warnings": profile_warnings(profile)}

    @app.post("/api/profile")
    def api_save_profile(profile: Profile):
        profile.name = profile.name.strip()
        if not profile.name:
            raise HTTPException(422, "Name is required.")
        if not profile.companies:
            raise HTTPException(422, "Add at least one employer.")
        if any(not c.name.strip() for c in profile.companies):
            raise HTTPException(422, "Every employer needs a name.")
        profile_id = db.save_profile(profile, method="form")
        db.purge(db.get_settings()["retention_days"])
        return {"id": profile_id, "warnings": profile_warnings(profile)}

    # ------------------------------------------------------------------ uploaded resumes
    @app.post("/api/resumes")
    async def api_upload_resume(file: UploadFile = File(...)):
        """Read an uploaded base resume with the Settings reader and save it. Nothing to check or confirm."""
        data = await file.read(MAX_BYTES + 1)  # never buffer more than the limit
        original = Path(file.filename or "resume").name
        suffix = Path(original).suffix[:10]
        filename = original if len(original) <= 120 else Path(original).stem[: 120 - len(suffix)] + suffix
        try:
            text = extract_text(data, original)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        profile, method, notes = await read_resume(router, db, text, router.default_model("parser"))
        profile.name = profile.name.strip()
        missing = profile_problem(profile)
        if missing:
            tip = (" Try a PDF or DOCX with selectable text, or a different Base-resume reader in Settings."
                   if method == "rules" else " Try a PDF or DOCX with selectable text.")
            raise HTTPException(422, f"Could not find {missing} in {filename}, so it was not saved.{tip}")
        profile_id = db.save_profile(profile, filename=filename, method=method, notes=notes)
        db.purge(db.get_settings()["retention_days"])
        entry = db.profile_entry(profile_id)
        assert entry is not None
        return {"resume": resume_entry(entry, registry)}

    @app.get("/api/resumes")
    def api_list_resumes():
        return {"resumes": [resume_entry(e, registry) for e in db.list_profiles()]}

    @app.delete("/api/resumes/{profile_id}", dependencies=[Depends(require_admin)])
    def api_delete_resume(profile_id: int):
        if not db.delete_profile(profile_id):
            raise HTTPException(404, "Resume not found.")
        return {"deleted": True}

    # ------------------------------------------------------------------ tailor
    @app.post("/api/tailor")
    async def api_tailor(body: TailorRequest):
        if body.profile_id is not None:
            chosen_profile = db.get_profile(body.profile_id)
            if chosen_profile is None:
                raise HTTPException(404, "That resume was not found (it may have been deleted). Pick another one.")
            profile_id, profile = body.profile_id, chosen_profile
        else:
            found = db.latest_profile()
            if not found:
                raise HTTPException(400, "Upload a resume first (Profile page).")
            profile_id, profile = found
        model_id = body.model or router.default_model("resume")
        if not model_id:
            raise HTTPException(422, "No model is available. Add an API key, or choose a model in Settings.")
        try:
            router.usable_spec(model_id)
        except LLMError as exc:
            raise HTTPException(422, str(exc)) from exc
        cover_model = None
        if body.cover_letter and not body.compare:
            chosen = db.get_settings().get("cover_letter_model")
            if chosen:
                try:
                    router.usable_spec(chosen)
                    cover_model = chosen
                except LLMError:
                    cover_model = None
        result = await run_tailor(
            router, db, profile=profile, profile_id=profile_id, jd=body.jd, hints=body.hints, model_id=model_id,
            allow_fallback=not body.compare, cover_letter=body.cover_letter, cover_model_id=cover_model,
            company_name=body.company_name.strip(), hiring_manager=body.hiring_manager.strip(),
        )
        return result.model_dump()

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def api_runs(limit: int = 100):
        return {"runs": db.list_runs(min(limit, 500)), "cost_today": round(db.cost_today(), 4)}

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: int):
        return load_run(run_id).model_dump()

    @app.put("/api/runs/{run_id}/resume")
    async def api_edit_run(run_id: int, body: EditRequest):
        result = load_run(run_id)
        if result.resume is None or result.analysis is None:
            raise HTTPException(409, "This run has no resume to edit.")
        try:
            merged = validate.build_resume(
                result.resume.profile,  # fixed facts always come from the stored profile, never from the request
                TailorOutput(
                    jd_analysis=result.analysis,
                    sections=TailoredSections(
                        headline=body.headline, summary=body.summary, skills=body.skills, experience=body.experience
                    ),
                ),
            )
            if result.cover_letter and body.cover_paragraphs is not None:
                greeting, paragraphs, closing = validate.clean_cover_letter(
                    body.cover_greeting or result.cover_letter.greeting,
                    body.cover_paragraphs,
                    body.cover_closing or result.cover_letter.closing,
                )
                result.cover_letter.greeting, result.cover_letter.paragraphs, result.cover_letter.closing = (
                    greeting, paragraphs, closing,
                )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result.resume = merged
        style, pages, fits, notes = await fit_report(merged, db.get_settings()["paper_size"])
        latest = db.get_run(run_id)  # a cover letter may have been written while this edit was in flight
        if latest is not None and body.cover_paragraphs is None:
            result.cover_letter, result.cover_cost_usd = latest.cover_letter, latest.cover_cost_usd
        result.style, result.pages, result.fits = style, pages, fits
        result.notes = [n for n in result.notes if not n.startswith(("Font reduced", "Still"))] + notes
        run_checks(result, result.jd or analysis_terms(result.analysis))
        db.update_run(result)
        return result.model_dump()

    @app.post("/api/runs/{run_id}/cover-letter")
    async def api_make_cover_letter(run_id: int, body: CoverRequest):
        """Write (or rewrite) the cover letter for a tailored resume. The JSON body is required on purpose:
        a cross-site form cannot send one, so another website cannot spend money through this endpoint."""
        result = load_run(run_id)
        if result.status != "ok" or result.resume is None or result.analysis is None:
            raise HTTPException(409, "This run has no resume to write a cover letter from.")
        jd = result.jd or body.jd.strip()
        if len(jd) < 40:
            raise HTTPException(
                409, "This run was made before job descriptions were kept. Tailor the resume again, then make "
                     "the cover letter.")
        model_id = router.default_model("cover_letter")
        if not model_id:
            raise HTTPException(422, "No model is available for cover letters. Choose one in Settings.")
        try:
            router.usable_spec(model_id)
        except LLMError as exc:
            raise HTTPException(422, str(exc)) from exc
        fresh, error = await write_cover_letter(router, db, run_id, jd=jd, model_id=model_id)
        if error:
            raise HTTPException(502, error)
        return fresh.model_dump()

    def _resume_pdf(result):
        pdf, _pages = render_resume_pdf(result.resume, result.style)
        return pdf

    @app.get("/api/runs/{run_id}/resume.pdf")
    def api_resume_pdf(run_id: int, download: int = 0):
        result = load_run(run_id)
        if result.resume is None:
            raise HTTPException(404, "No resume in this run.")
        disposition = "attachment" if download else "inline"
        name = download_name(result, "pdf")
        return Response(_resume_pdf(result), media_type="application/pdf",
                        headers={"Content-Disposition": f'{disposition}; filename="{name}"'})

    @app.get("/api/runs/{run_id}/resume.docx")
    def api_resume_docx(run_id: int):
        result = load_run(run_id)
        if result.resume is None:
            raise HTTPException(404, "No resume in this run.")
        name = download_name(result, "docx")
        return Response(resume_docx(result.resume, result.style), media_type=DOCX,
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/runs/{run_id}/cover.pdf")
    def api_cover_pdf(run_id: int, download: int = 0):
        result = load_run(run_id)
        if result.resume is None or result.cover_letter is None:
            raise HTTPException(404, "No cover letter in this run.")
        pdf, _ = render_cover_pdf(result.resume, result.cover_letter, result.style.paper)
        disposition = "attachment" if download else "inline"
        name = download_name(result, "pdf", cover=True)
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'{disposition}; filename="{name}"'})

    @app.get("/api/runs/{run_id}/cover.docx")
    def api_cover_docx(run_id: int):
        result = load_run(run_id)
        if result.resume is None or result.cover_letter is None:
            raise HTTPException(404, "No cover letter in this run.")
        name = download_name(result, "docx", cover=True)
        return Response(cover_docx(result.resume, result.cover_letter, result.style.paper), media_type=DOCX,
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.delete("/api/data", dependencies=[Depends(require_admin)])
    def api_delete_all():
        """Privacy control: remove the stored profile and every run."""
        db.purge_all()
        return {"deleted": True}

    return app
