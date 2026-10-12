"""FastAPI app: JSON API + server-rendered pages, behind a login.

Two roles. A user uploads resumes, tailors them and downloads the PDFs; every resume and run is visible only
to its owner. The admin (the ADMIN_EMAIL account from .env) also has History (every download), Users (approve
sign-ups) and Settings. New sign-ups wait until the admin approves them.
"""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
import unicodedata
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote, urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import auth, config
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


class LoginRequest(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1000)
    next: str = Field(default="", max_length=500)


class SignupRequest(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1000)
    confirm: str = Field(max_length=1000)


class PasswordChange(BaseModel):
    current: str = Field(max_length=1000)
    new: str = Field(max_length=1000)


PUBLIC_PATHS = {"/login", "/signup", "/api/login", "/api/signup", "/api/logout"}
LIMIT_MESSAGE = "Today's limit for this site has been reached. Try again tomorrow."
BUSY_MESSAGE = "The service is busy or unavailable right now. Try again in a few minutes."
_TECHNICAL = (
    "api key", "settings", ".env", "model", "provider", "region", "http ", "quota", "rate limit", "token",
    "json", "invalid structure", "empty answer", "response shape", "declined", "unexpected error", "network",
)


# Tailoring can take minutes when a model is slow or busy, longer than hosting proxies wait for an answer
# (PythonAnywhere then shows "Gateway Time-out"). So a request waits at most JOB_WAIT_SECONDS; slower work goes on
# in the background and the page asks GET /api/jobs/{id} every few seconds until it ends.
JOB_WAIT_SECONDS = 20.0
JOB_LOST_MINUTES = 30  # a job still running after this was stopped by a restart (Reload) and will never end
JOB_LOST = "The server restarted before this finished. Try again."


class JobFailed(Exception):
    """Background work that ended without a result. The message is the admin's; users see failure_message."""


class PlainError(HTTPException):
    """A message the app wrote itself and that is fine for every user as it is, even when it contains their own
    text (a file name): the error handler never rewrites it."""


def failure_message(text: str) -> str:
    """A failed model call as a user sees it: the daily limit, or a plain 'try again'. Never the model, the
    provider, the fallback chain or what the model answered."""
    return LIMIT_MESSAGE if "cost cap" in text.lower() else BUSY_MESSAGE


def public_error(text: str, names: frozenset[str] = frozenset()) -> str:
    """What a user (not the admin) is told for a refused request: no model names, key hints or Settings advice."""
    low = text.lower()
    if "cost cap" in low:
        return LIMIT_MESSAGE
    if any(word in low for word in _TECHNICAL) or any(name in low for name in names):
        return BUSY_MESSAGE
    return text


def public_run(result: TailorResult) -> dict:
    """A run as a user sees it: only what the Tailor page needs (no models, costs, notes or attempts)."""
    return {
        "run_id": result.run_id, "status": result.status, "error": failure_message(result.error) if result.error else "",
        "pages": result.pages, "fits": result.fits, "coverage": result.coverage.model_dump(),
        "cover_letter": result.cover_letter.model_dump() if result.cover_letter else None,
        "created_at": result.created_at,
    }


def safe_next(target: str | None) -> str:
    """A local path to go to after login, or ''. Never another site ('//evil.example', 'https://...')."""
    target = (target or "").strip()
    if not target.startswith("/") or target.startswith(("//", "/\\")) or "\\" in target:
        return ""
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or target.startswith(("/api/", "/static/", "/login", "/signup")):
        return ""
    return target


def profile_problem(profile: Profile) -> str | None:
    """Why a profile cannot be used for tailoring, or None."""
    if not profile.name.strip():
        return "your name"
    if not profile.companies or any(not c.name.strip() for c in profile.companies):
        return "any employer"
    return None


def resume_entry(entry: dict, registry: Registry, admin: bool) -> dict:
    """An uploaded resume as the Profile and Tailor pages show it. Users do not see which model read it."""
    method = entry["method"]
    profile: Profile = entry["profile"]
    if not admin:
        return {
            "id": entry["id"], "created_at": entry["created_at"], "filename": entry["filename"], "method": "",
            "read_with": "", "profile": profile.model_dump(), "warnings": profile_warnings(profile),
        }
    if method.startswith("model:"):
        model_id = method.split(":", 1)[1]
        try:
            read_with = registry.get(model_id).label
        except KeyError:
            read_with = model_id
    else:
        read_with = {"rules": "built-in rules", "form": "typed in"}.get(method, "earlier version")
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
        # Behind a proxy (PythonAnywhere) the site's own name may arrive as X-Forwarded-Host. A web page cannot set
        # that header on a request to another site, so accepting it keeps other websites out.
        hosts = {request.headers.get("host", "")} | {h.strip() for h in request.headers.get("x-forwarded-host", "").split(",")}
        return urlsplit(origin).netloc.lower() not in {h.lower() for h in hosts if h}
    return origin == "null"


def forwarded_proto(request: Request) -> str:
    """The scheme the browser used, as the proxy in front of the app reports it ('' when there is no proxy)."""
    return request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()


HTTPS_PROBE = "rt_https"  # a Secure cookie: browsers send it only over https://
log = logging.getLogger("resume_tailor")
_proxy_logged = False


def needs_https_redirect(request: Request) -> bool:
    """FORCE_HTTPS=1: send a plain-http visitor to https://. Some proxies (PythonAnywhere's) report "http" even
    when the browser used https; a Secure cookie in the request proves it was https, because browsers never send
    one over http. So a visitor is redirected at most once, never in a loop."""
    if not config.force_https() or forwarded_proto(request) != "http":
        return False
    return not (request.cookies.get(HTTPS_PROBE) or request.cookies.get(auth.COOKIE))


def log_proxy_headers(request: Request) -> None:
    """Once per process, what the proxy says about the scheme: helps when HTTPS behaves oddly on a host."""
    global _proxy_logged
    if _proxy_logged:
        return
    _proxy_logged = True
    names = ("x-forwarded-proto", "x-forwarded-host", "x-forwarded-ssl", "x-forwarded-port", "x-scheme", "forwarded")
    seen = {n: request.headers.get(n) for n in names if request.headers.get(n)}
    log.warning("FORCE_HTTPS redirect: scheme=%s host=%s proxy headers=%s", request.url.scheme,
                request.headers.get("host", ""), seen)


def current_user(request: Request) -> auth.User:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Please log in.")
    return user


def admin_user(user: auth.User = Depends(current_user)) -> auth.User:
    if not user.is_admin:
        raise HTTPException(403, "Only the admin can do that.")
    return user


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
    auth.bootstrap_admin(db)

    model_names = frozenset(n.lower() for spec in registry.all() for n in (spec.id, spec.label, spec.model) if n)
    app = FastAPI(title="Resume Tailor", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.db, app.state.registry, app.state.router = db, registry, router

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        if needs_https_redirect(request):  # FORCE_HTTPS=1 on a server
            log_proxy_headers(request)
            if request.method not in ("GET", "HEAD"):
                return JSONResponse({"detail": "Use the https:// address of this site."}, status_code=403)
            redirect = RedirectResponse(str(request.url.replace(scheme="https")), status_code=307)
            # If the browser was on https already (the proxy's "http" was wrong), it now sends this cookie back
            # and gets the page; over real http the cookie is never sent, so it is redirected as it should be.
            redirect.set_cookie(HTTPS_PROBE, "1", max_age=365 * 86400, path="/", secure=True, httponly=True,
                                samesite="lax")
            return redirect
        if request.method in UNSAFE_METHODS and cross_site(request):
            return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
        path = request.url.path
        if path.startswith("/static/"):
            request.state.user = None
            response = await call_next(request)
        else:
            user = auth.session_user(db, request.cookies.get(auth.COOKIE))
            request.state.user = user
            if user is None and path not in PUBLIC_PATHS:
                if path.startswith("/api/"):
                    response = JSONResponse({"detail": "Please log in."}, status_code=401)
                else:
                    target = safe_next(path + (f"?{request.url.query}" if request.url.query else ""))
                    response = RedirectResponse("/login" + (f"?next={quote(target)}" if target else ""), 303)
            else:
                response = await call_next(request)
            response.headers.setdefault("Cache-Control", "no-store")  # nothing personal stays in a shared browser
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if config.force_https():  # browsers then use https:// for this site by themselves for a year
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; frame-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'",
            )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        """Users get plain messages; the admin keeps the technical detail (model, key hint, Settings advice)."""
        user = getattr(request.state, "user", None)
        detail = exc.detail
        if isinstance(detail, str) and not isinstance(exc, PlainError) and not (user and user.is_admin):
            detail = failure_message(detail) if exc.status_code >= 500 else public_error(detail, model_names)
        return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
    templates = Jinja2Templates(directory=WEB / "templates")

    def page(request: Request, name: str, active: str, **ctx):
        user = getattr(request.state, "user", None)
        pending = db.count_users("pending") if user and user.is_admin else 0
        return templates.TemplateResponse(request, name, {"active": active, "user": user, "pending": pending, **ctx})

    def regions() -> list[str]:
        return db.get_settings()["allowed_regions"]

    def load_run(run_id: int, user: auth.User, *, write: bool = False) -> TailorResult:
        """A run its owner may read and change. The admin may read anyone's (History links) but changes only
        their own. Runs saved before accounts existed belong to the admin. Anything else is 'not found'."""
        found = db.get_run_owned(run_id)
        if found is not None:
            owner, result = found
            own = owner == user.id or (owner is None and user.is_admin)
            if own or (user.is_admin and not write):
                return result
        raise HTTPException(404, "Run not found (it may have passed the retention period).")

    def viewed(result: TailorResult, user: auth.User) -> dict:
        return result.model_dump() if user.is_admin else public_run(result)

    def job_error(message: str, user: auth.User) -> HTTPException:
        return HTTPException(502, message if user.is_admin else failure_message(message))

    running: set[asyncio.Task] = set()  # a reference keeps a job that nobody waits for from being garbage collected

    def forget(task: asyncio.Task) -> None:
        running.discard(task)
        if not task.cancelled():
            task.exception()  # retrieved, so asyncio does not warn; run() has recorded it already

    async def answer_soon(user: auth.User, kind: str, work):
        """The run work() makes, if it is ready within JOB_WAIT_SECONDS. Otherwise 202 {"job", "status":
        "running"}: the work goes on, and GET /api/jobs/{job} gives the same answer once it ends."""
        job_id = secrets.token_urlsafe(16)
        db.create_job(job_id, user.id, kind)

        async def run() -> TailorResult:
            try:
                result = await work()
            except JobFailed as exc:
                db.finish_job(job_id, run_id=None, error=str(exc))
                raise
            except asyncio.CancelledError:  # the server is stopping (Reload): tell the page at once
                db.finish_job(job_id, run_id=None, error=JOB_LOST)
                raise
            except Exception as exc:
                log.exception("Background %s failed", kind)
                db.finish_job(job_id, run_id=None, error=f"Unexpected error: {type(exc).__name__}")
                raise
            db.finish_job(job_id, run_id=result.run_id)
            return result

        task = asyncio.create_task(run())
        running.add(task)
        task.add_done_callback(forget)
        await asyncio.wait({task}, timeout=JOB_WAIT_SECONDS)  # never cancels the task, even if this request ends
        if not task.done():
            return JSONResponse({"job": job_id, "status": "running"}, status_code=202)
        try:
            return viewed(task.result(), user)
        except JobFailed as exc:
            raise job_error(str(exc), user) from None

    def log_download(user: auth.User, run_id: int, result: TailorResult, kind: str, fmt: str, filename: str) -> None:
        analysis = result.analysis
        db.record_download(
            user_id=user.id, email=user.email, run_id=run_id, kind=kind, format=fmt, filename=filename,
            candidate=result.resume.profile.name if result.resume else "",
            company=result.company_name or (analysis.company if analysis else ""),
            role=analysis.target_role if analysis else "",
            model=result.served_model or result.requested_model,
            cost_usd=result.cost_usd + result.cover_cost_usd,
        )

    def file_response(content: bytes, media_type: str, disposition: str, name: str) -> Response:
        return Response(content, media_type=media_type, headers={
            "Content-Disposition": f'{disposition}; filename="{name}"', "Cache-Control": "private, no-store"})

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return RedirectResponse("/tailor")

    @app.get("/login")
    def login_page(request: Request, next: str = ""):
        if request.state.user:
            return RedirectResponse(safe_next(next) or "/tailor", 303)
        return page(request, "login.html", "login", next=safe_next(next),
                    admin_ready=bool(auth.normalize_email(config.admin_credentials()[0])))

    @app.get("/signup")
    def signup_page(request: Request):
        if request.state.user:
            return RedirectResponse("/tailor", 303)
        return page(request, "signup.html", "signup")

    @app.get("/profile")
    def profile_page(request: Request):
        return page(request, "profile.html", "profile")

    @app.get("/tailor")
    def tailor_page(request: Request):
        return page(request, "tailor.html", "tailor")

    @app.get("/account")
    def account_page(request: Request):
        return page(request, "account.html", "account")

    def admin_page(request: Request, name: str, active: str):
        if not request.state.user.is_admin:
            return RedirectResponse("/tailor", 303)
        return page(request, name, active)

    @app.get("/history")
    def history_page(request: Request):
        return admin_page(request, "history.html", "history")

    @app.get("/users")
    def users_page(request: Request):
        return admin_page(request, "users.html", "users")

    @app.get("/settings")
    def settings_page(request: Request):
        return admin_page(request, "settings.html", "settings")

    @app.get("/run/{run_id}")
    def run_page(run_id: int):
        """The Review and edit page was removed: tailoring is final. Old links go to the Tailor page."""
        return RedirectResponse("/tailor")

    # ------------------------------------------------------------------ accounts
    @app.post("/api/signup")
    def api_signup(body: SignupRequest):
        """The same answer for a new or an existing email. Emails are not verified, so while an account still
        waits for approval the newest sign-up's password wins: signing up first with someone else's address
        cannot keep them out (the admin approves the person, not the first request)."""
        email = auth.normalize_email(body.email)
        if not email:
            raise HTTPException(422, "Enter a valid email address.")
        problem = auth.password_problem(body.password)
        if problem:
            raise HTTPException(422, problem)
        if body.password != body.confirm:
            raise HTTPException(422, "The two passwords do not match.")
        if db.count_users("pending") >= auth.MAX_PENDING:
            raise HTTPException(429, "Too many sign-ups are waiting for approval. Try again later.")
        hashed = auth.hash_password(body.password)
        is_admin_address = email == auth.normalize_email(config.admin_credentials()[0])  # never a sign-up
        if not is_admin_address and db.create_user(email, hashed, role="user", status="pending") is None:
            db.renew_pending_password(email, hashed)
        return {"ok": True, "message": "Request received. You can log in as soon as the admin approves your account."}

    @app.post("/api/login")
    def api_login(body: LoginRequest, request: Request):
        user, error = auth.login(db, body.email, body.password)
        if user is None:
            status = 429 if error == auth.LOCKED else 403 if error in (auth.PENDING, auth.DISABLED) else 401
            raise HTTPException(status, error)
        auth.end_session(db, request.cookies.get(auth.COOKIE))  # never keep a session id from before the login
        response = JSONResponse({"ok": True, "next": safe_next(body.next) or "/tailor"})
        response.set_cookie(auth.COOKIE, auth.start_session(db, user.id),
                            **auth.cookie_options(request.headers, request.url.scheme))
        return response

    @app.post("/api/logout")
    def api_logout(request: Request):
        auth.end_session(db, request.cookies.get(auth.COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(auth.COOKIE, path="/")
        return response

    @app.get("/api/me")
    def api_me(user: auth.User = Depends(current_user)):
        return {"id": user.id, "email": user.email, "role": user.role}

    @app.post("/api/account/password")
    def api_change_password(body: PasswordChange, request: Request, user: auth.User = Depends(current_user)):
        if user.is_admin:
            raise HTTPException(409, "The admin password comes from the .env file: change ADMIN_PASSWORD there and restart.")
        if not auth.claim_attempt(db, user.id):  # the same limit as logging in, so a borrowed session cannot guess
            raise HTTPException(429, auth.LOCKED)
        row = db.get_user(user.id)
        if row is None or not auth.verify_password(body.current, row["password_hash"]):
            raise HTTPException(403, "The current password is not correct.")
        problem = auth.password_problem(body.new)
        if problem:
            raise HTTPException(422, problem)
        db.set_password_hash(user.id, auth.hash_password(body.new))  # ends every session, this one too
        response = JSONResponse({"ok": True})
        response.set_cookie(auth.COOKIE, auth.start_session(db, user.id),
                            **auth.cookie_options(request.headers, request.url.scheme))
        return response

    @app.get("/api/users", dependencies=[Depends(admin_user)])
    def api_users():
        return {"users": db.list_users(), "pending": db.count_users("pending")}

    def change_status(user_id: int, status: str, only_from: str) -> dict:
        if not db.set_user_status(user_id, status, only_from=only_from):
            raise HTTPException(409, "That account cannot be changed this way (it may have changed already).")
        return {"ok": True}

    @app.post("/api/users/{user_id}/approve", dependencies=[Depends(admin_user)])
    def api_approve(user_id: int):
        return change_status(user_id, "active", "pending")

    @app.post("/api/users/{user_id}/disable", dependencies=[Depends(admin_user)])
    def api_disable(user_id: int):
        return change_status(user_id, "disabled", "active")

    @app.post("/api/users/{user_id}/enable", dependencies=[Depends(admin_user)])
    def api_enable(user_id: int):
        return change_status(user_id, "active", "disabled")

    @app.post("/api/users/{user_id}/reset-password", dependencies=[Depends(admin_user)])
    def api_reset_password(user_id: int):
        """A one-time temporary password for the admin to pass on; the user changes it on the Account page."""
        row = db.get_user(user_id)
        if row is None or row["role"] != "user":
            raise HTTPException(404, "Account not found.")
        temporary = auth.temporary_password()
        db.set_password_hash(user_id, auth.hash_password(temporary))
        return {"ok": True, "password": temporary}

    @app.delete("/api/users/{user_id}", dependencies=[Depends(admin_user)])
    def api_delete_user(user_id: int):
        """Reject a sign-up, or remove an account with its resumes and runs (download history stays)."""
        if not db.delete_user(user_id):
            raise HTTPException(404, "Account not found.")
        return {"deleted": True}

    @app.get("/api/downloads", dependencies=[Depends(admin_user)])
    def api_downloads(user_id: int | None = None, limit: int = 200):
        return {"downloads": db.list_downloads(user_id=user_id, limit=max(1, min(limit, 1000))),
                "accounts": db.download_accounts(), "cost_today": round(db.cost_today(), 4)}

    # ------------------------------------------------------------------ models + settings (admin)
    @app.get("/api/models", dependencies=[Depends(admin_user)])
    def api_models():
        allowed = regions()
        return {"models": [model_info(registry, s, allowed) for s in registry.all()], "regions": KNOWN_REGIONS}

    def settings_view() -> dict:
        settings = db.get_settings()
        return {
            "settings": settings,
            "resolved": {t: router.default_model(t) for t in ("resume", "cover_letter", "parser")},
            "automatic": {t: router.default_model(t, automatic=True) for t in ("resume", "cover_letter", "parser")},
            "cost_today": round(db.cost_today(), 4),
        }

    @app.get("/api/settings", dependencies=[Depends(admin_user)])
    def api_settings():
        return settings_view()

    @app.put("/api/settings", dependencies=[Depends(admin_user)])
    def api_set_settings(values: dict):
        updated = db.set_settings(validate_settings(registry, values))
        db.purge(updated["retention_days"])
        return settings_view()

    @app.post("/api/models/check", dependencies=[Depends(admin_user)])
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

    # ------------------------------------------------------------------ profile (older endpoints, admin only)
    @app.post("/api/profile/parse")
    async def api_parse(file: UploadFile = File(...), model: str = Form(""), user: auth.User = Depends(admin_user)):
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
        profile, method, warnings = await parse_resume(router, db, text, model_id, user_id=user.id)
        return {"profile": profile.model_dump(), "method": method, "warnings": warnings, "text_chars": len(text)}

    @app.get("/api/profile")
    def api_get_profile(user: auth.User = Depends(admin_user)):
        found = db.latest_profile(user_id=user.id)
        if not found:
            return {"profile": None, "warnings": []}
        profile_id, profile = found
        return {"id": profile_id, "profile": profile.model_dump(), "warnings": profile_warnings(profile)}

    @app.post("/api/profile")
    def api_save_profile(profile: Profile, user: auth.User = Depends(admin_user)):
        profile.name = profile.name.strip()
        if not profile.name:
            raise HTTPException(422, "Name is required.")
        if not profile.companies:
            raise HTTPException(422, "Add at least one employer.")
        if any(not c.name.strip() for c in profile.companies):
            raise HTTPException(422, "Every employer needs a name.")
        profile_id = db.save_profile(profile, user_id=user.id, method="form")
        db.purge(db.get_settings()["retention_days"])
        return {"id": profile_id, "warnings": profile_warnings(profile)}

    # ------------------------------------------------------------------ uploaded resumes (each user's own)
    @app.post("/api/resumes")
    async def api_upload_resume(file: UploadFile = File(...), user: auth.User = Depends(current_user)):
        """Read an uploaded base resume with the Settings reader and save it. Nothing to check or confirm."""
        data = await file.read(MAX_BYTES + 1)  # never buffer more than the limit
        original = Path(file.filename or "resume").name
        suffix = Path(original).suffix[:10]
        filename = original if len(original) <= 120 else Path(original).stem[: 120 - len(suffix)] + suffix
        try:
            text = extract_text(data, original)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        profile, method, notes = await read_resume(router, db, text, router.default_model("parser"), user_id=user.id)
        profile.name = profile.name.strip()
        missing = profile_problem(profile)
        if missing:
            tip = (" Try a PDF or DOCX with selectable text, or a different Base-resume reader in Settings."
                   if method == "rules" and user.is_admin else " Try a PDF or DOCX with selectable text.")
            raise PlainError(422, f"Could not find {missing} in {filename}, so it was not saved.{tip}")
        profile_id = db.save_profile(profile, user_id=user.id, filename=filename, method=method, notes=notes)
        db.purge(db.get_settings()["retention_days"])
        entry = db.profile_entry(profile_id, user_id=user.id)
        assert entry is not None
        return {"resume": resume_entry(entry, registry, user.is_admin)}

    @app.get("/api/resumes")
    def api_list_resumes(user: auth.User = Depends(current_user)):
        return {"resumes": [resume_entry(e, registry, user.is_admin) for e in db.list_profiles(user_id=user.id)]}

    @app.delete("/api/resumes/{profile_id}")
    def api_delete_resume(profile_id: int, user: auth.User = Depends(current_user)):
        if not db.delete_profile(profile_id, user_id=user.id):
            raise HTTPException(404, "Resume not found.")
        return {"deleted": True}

    # ------------------------------------------------------------------ tailor
    @app.post("/api/tailor")
    async def api_tailor(body: TailorRequest, user: auth.User = Depends(current_user)):
        if body.profile_id is not None:
            chosen_profile = db.get_profile(body.profile_id, user_id=user.id)
            if chosen_profile is None:
                raise HTTPException(404, "That resume was not found (it may have been deleted). Pick another one.")
            profile_id, profile = body.profile_id, chosen_profile
        else:
            found = db.latest_profile(user_id=user.id)
            if not found:
                raise HTTPException(400, "Upload a resume first (Profile page).")
            profile_id, profile = found
        compare = body.compare and user.is_admin  # users never pick a model: the admin's Settings decide
        model_id = (body.model if user.is_admin else None) or router.default_model("resume")
        if not model_id:
            raise HTTPException(422, "No model is available. Add an API key, or choose a model in Settings.")
        try:
            router.usable_spec(model_id)
        except LLMError as exc:
            raise HTTPException(422, str(exc)) from exc
        cover_model = None
        if body.cover_letter and not compare:
            chosen = db.get_settings().get("cover_letter_model")
            if chosen:
                try:
                    router.usable_spec(chosen)
                    cover_model = chosen
                except LLMError:
                    cover_model = None
        return await answer_soon(user, "tailor", lambda: run_tailor(
            router, db, profile=profile, profile_id=profile_id, jd=body.jd, hints=body.hints, model_id=model_id,
            allow_fallback=not compare, cover_letter=body.cover_letter, cover_model_id=cover_model,
            company_name=body.company_name.strip(), hiring_manager=body.hiring_manager.strip(), user_id=user.id,
        ))

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str, user: auth.User = Depends(current_user)):
        """Slow work started by POST /api/tailor or /api/runs/{id}/cover-letter: still running, or its answer."""
        job = db.get_job(job_id, user.id) if len(job_id) <= 64 else None
        if job is None:
            raise HTTPException(404, "Not found.")
        if job["status"] == "running":
            lost = auth.utc(auth.now_utc() - timedelta(minutes=JOB_LOST_MINUTES))
            if job["created_at"] >= lost:
                return JSONResponse({"job": job_id, "status": "running"}, status_code=202)
            job["error"] = JOB_LOST
        if job["error"]:
            raise job_error(job["error"], user)
        return viewed(load_run(job["run_id"], user), user)

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs", dependencies=[Depends(admin_user)])
    def api_runs(limit: int = 100):
        return {"runs": db.list_runs(min(limit, 500)), "cost_today": round(db.cost_today(), 4)}

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: int, user: auth.User = Depends(current_user)):
        return viewed(load_run(run_id, user), user)

    @app.put("/api/runs/{run_id}/resume")
    async def api_edit_run(run_id: int, body: EditRequest, user: auth.User = Depends(admin_user)):
        result = load_run(run_id, user, write=True)
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
    async def api_make_cover_letter(run_id: int, body: CoverRequest, user: auth.User = Depends(current_user)):
        """Write (or rewrite) the cover letter for a tailored resume. The JSON body is required on purpose:
        a cross-site form cannot send one, so another website cannot spend money through this endpoint."""
        result = load_run(run_id, user, write=True)
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

        async def work() -> TailorResult:
            fresh, error = await write_cover_letter(router, db, run_id, jd=jd, model_id=model_id, user_id=user.id)
            if error:
                raise JobFailed(error)
            return fresh

        return await answer_soon(user, "cover_letter", work)

    # ------------------------------------------------------------------ files (a Download click is logged)
    @app.get("/api/runs/{run_id}/resume.pdf")
    def api_resume_pdf(run_id: int, download: int = 0, user: auth.User = Depends(current_user)):
        result = load_run(run_id, user)
        if result.resume is None:
            raise HTTPException(404, "No resume in this run.")
        pdf, _pages = render_resume_pdf(result.resume, result.style)
        name = download_name(result, "pdf")
        if download:
            log_download(user, run_id, result, "resume", "pdf", name)
        return file_response(pdf, "application/pdf", "attachment" if download else "inline", name)

    @app.get("/api/runs/{run_id}/resume.docx")
    def api_resume_docx(run_id: int, user: auth.User = Depends(admin_user)):  # users download PDFs only
        result = load_run(run_id, user)
        if result.resume is None:
            raise HTTPException(404, "No resume in this run.")
        name = download_name(result, "docx")
        log_download(user, run_id, result, "resume", "docx", name)
        return file_response(resume_docx(result.resume, result.style), DOCX, "attachment", name)

    @app.get("/api/runs/{run_id}/cover.pdf")
    def api_cover_pdf(run_id: int, download: int = 0, user: auth.User = Depends(current_user)):
        result = load_run(run_id, user)
        if result.resume is None or result.cover_letter is None:
            raise HTTPException(404, "No cover letter in this run.")
        pdf, _ = render_cover_pdf(result.resume, result.cover_letter, result.style.paper)
        name = download_name(result, "pdf", cover=True)
        if download:
            log_download(user, run_id, result, "cover_letter", "pdf", name)
        return file_response(pdf, "application/pdf", "attachment" if download else "inline", name)

    @app.get("/api/runs/{run_id}/cover.docx")
    def api_cover_docx(run_id: int, user: auth.User = Depends(admin_user)):  # users download PDFs only
        result = load_run(run_id, user)
        if result.resume is None or result.cover_letter is None:
            raise HTTPException(404, "No cover letter in this run.")
        name = download_name(result, "docx", cover=True)
        log_download(user, run_id, result, "cover_letter", "docx", name)
        return file_response(cover_docx(result.resume, result.cover_letter, result.style.paper), DOCX, "attachment", name)

    @app.delete("/api/data", dependencies=[Depends(admin_user)])
    def api_delete_all():
        """Privacy control: remove every resume, run and download record (accounts and settings stay)."""
        db.purge_all()
        return {"deleted": True}

    return app
