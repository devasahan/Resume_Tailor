"""FastAPI app: JSON API + server-rendered pages, behind a login.

Two roles. A user uploads resumes, tailors them and downloads the PDFs; every resume and run is visible only
to its owner. The admin (the ADMIN_EMAIL account from .env) also has History (every download), Users (approve
sign-ups) and Settings. New sign-ups wait until the admin approves them.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
import unicodedata
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Mount

from . import auth, config, timing
from .llm.base import CostCapError, LLMError
from .llm.registry import ModelSpec, Registry
from .llm.router import Router
from .parsing.heuristic import MAX_OWN_CERTIFICATIONS, read_certification_list
from .parsing.resume_parser import MAX_BYTES, extract_text, parse_resume, profile_warnings, read_resume
from .pipeline import validate
from .pipeline.generate import GenerationFailed
from .pipeline.job_facts import check_job_facts
from .prompts import METRICS_MODES
from .prompts import read_all as read_prompt_files
from .pipeline.tailor import analysis_terms, fit_report, refit, run_checks, run_tailor, write_cover_letter
from .render.docx import cover_docx, resume_docx
from .render.pdf import person_name, render_cover_pdf, render_resume_pdf
from .schemas import (
    MAX_CERTIFICATIONS,
    Certification,
    Company,
    ExperienceOut,
    FitStyle,
    Profile,
    Resume,
    ResumeExperience,
    SkillGroup,
    TailoredSections,
    TailorOutput,
    TailorResult,
)
from .storage.db import Database

WEB = Path(__file__).parent / "web"


def asset_version(folder: Path) -> str:
    """A fingerprint of the script and style files. Pages ask for /static/app.js?v=<it>, so after an update every
    browser loads the new files instead of an old cached copy (an old app.js with a new page breaks the buttons)."""
    digest = hashlib.sha256()
    for name in ("app.js", "style.css"):
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()[:12]


KNOWN_REGIONS = ["US", "EU", "SG", "CN", "UNKNOWN", "DEMO"]
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ------------------------------------------------------------------ request bodies
class JobFactsRequest(BaseModel):
    jd: str = Field(min_length=40, max_length=20000)


FACTS_CACHE_SIZE = 200  # answers kept by job-post text, so pasting the same post again costs nothing


class TailorRequest(BaseModel):
    jd: str = Field(min_length=40, max_length=20000)
    profile_id: int | None = None  # which uploaded resume; None = the newest one
    hints: str = Field(default="", max_length=4000)
    model: str | None = None  # None = the resume model chosen in Settings
    compare: bool = False  # True: no fallback, no per-task cover-letter model (each model stands on its own)
    cover_letter: bool = False
    company_name: str = Field(default="", max_length=120)
    hiring_manager: str = Field(default="", max_length=120)
    spelling: Literal["us", "eu"] = "us"  # the US style / EU style (British spelling) buttons


class EditRequest(BaseModel):
    headline: str
    summary: str
    skills: list[SkillGroup]
    experience: list[ExperienceOut]
    cover_greeting: str | None = None
    cover_paragraphs: list[str] | None = None
    cover_closing: str | None = None


class CertificationList(BaseModel):
    text: str = Field(max_length=20000)  # one certification per line, as pasted on the Profile page
    # The list the page showed when it was opened: if it changed since (a tick on the Tailor page), nothing is saved
    loaded: list[Certification] | None = Field(default=None, max_length=500)


class StaleList(Exception):
    """The saved list changed after the page was opened."""


class CertificationTicks(BaseModel):
    have: list[bool] = Field(max_length=MAX_CERTIFICATIONS)  # one "I have it" tick per row, in the page's order


class CoverRequest(BaseModel):
    # Only needed for runs saved before job descriptions were kept with the run.
    jd: str = Field(default="", max_length=20000)


class CheckRequest(BaseModel):
    models: list[str]


class KeyRequest(BaseModel):
    value: str = Field(max_length=600)


# An API key as one .env line: letters, digits and . _ - + / = : only (no spaces, quotes, '#' or line breaks)
KEY_VALUE = re.compile(r"[A-Za-z0-9._\-+/=:]{8,512}")


def key_names(registry: Registry) -> list[str]:
    """The key variables the models need, in models.yaml order."""
    return list(dict.fromkeys(spec.key_env for spec in registry.all() if spec.key_env))


def keys_view(registry: Registry) -> list[dict]:
    """For the Settings page: each key variable, the models it opens, and whether it is set. Never the key itself,
    only its hint (the provider's marker and its last 4 characters)."""
    return [{"name": name, "models": [spec.label for spec in registry.all() if spec.key_env == name],
             "set": bool(config.key_hint(name)), "hint": config.key_hint(name),
             "in_env_file": name in config.FROM_DOTENV} for name in key_names(registry)]


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
# in the background and the page asks GET /api/jobs/{id} every few seconds until it ends. While it runs, the job
# writes a heartbeat; a restart (Reload) ends the process without warning, and the missing heartbeat tells the page.
JOB_WAIT_SECONDS = 20.0
# On PythonAnywhere work done while no request is open runs slowly or not at all: the Web-tab server (uWSGI) has no
# thread support, and on the uvicorn site the log showed steps after the first 20 s taking 3-4 times longer. A check on
# a running job therefore waits up to this long for it to end, so the job runs while a request is open. Under the
# POST's 20 s, so no proxy times out.
JOB_POLL_WAIT_SECONDS = 15.0
JOB_BEAT_SECONDS = 15.0
JOB_LOST_SECONDS = 90.0  # no heartbeat for this long: the process that ran the job is gone
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
        "created_at": result.created_at, "spelling": result.spelling or "us",
        "certifications": [row.model_dump() for row in result.certifications],
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


def show_app_log() -> None:
    """Send the app's own INFO lines (one per model call: how long it took and how many tokens) to stderr, which
    hosts keep as the server log. Called by the server entry points; it does nothing once the 'app' logger has a
    level, so `python -m app` keeps its console quiet unless LOG_LEVEL is info or debug."""
    logger = logging.getLogger("app")
    if logger.level == logging.NOTSET:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


HTTPS_PROBE = "rt_https"  # a Secure cookie: browsers send it only over https://
log = logging.getLogger("resume_tailor")
timing_log = logging.getLogger(__name__)  # under "app": its INFO lines reach the server log (show_app_log)


def warm_up() -> None:
    """Do the first tailoring's one-time work when the server starts: read the prompt files (kept in memory), and lay
    out a small made-up resume as a PDF, which loads the font and page-layout code and brings the font files into the
    system's file cache (fpdf2 still reads and parses them for every layout; the job lines count those loads). On a
    host's network storage that is slow the first time (PythonAnywhere). Called by the server entry points
    (app/asgi.py, app/wsgi.py); a failure is logged and never keeps the site from starting."""
    began = time.perf_counter()
    try:
        prompt_files = read_prompt_files()
        sample = Resume(
            profile=Profile(name="Sample Person", companies=[Company(name="Sample Company", start="Jan 2020", end="Present")]),
            headline="Software Engineer", summary="Builds reliable web services.",
            skills=[SkillGroup(category="Languages", skills=["Python"])],
            experience=[ResumeExperience(company_index=0, job_title="Software Engineer",
                                         bullets=["Built a sample service used by 1,000 people, cutting load time by 20%."])],
        )
        render_resume_pdf(sample, FitStyle())
    except Exception as exc:
        log.exception("Warm-up failed (%s); the first tailoring will do this work instead", type(exc).__name__)
        return
    timing_log.info("warm-up done in %.1fs (%d prompt files, the font and page layout code)",
                    time.perf_counter() - began, prompt_files)
SLOW_REQUEST_SECONDS = 2.0  # a request this slow is named in the server log, with its time


class SlowRequestLog:
    """Names each request that took SLOW_REQUEST_SECONDS or more in the server log: its method, its route as the app
    declares it (/api/runs/{run_id}/resume.pdf, never the address the browser sent, which could hold anything) and
    its seconds. Job checks (/api/jobs/...) are left out: they wait for the job on purpose."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        began = time.perf_counter()
        try:
            await self.app(scope, receive, send)
        finally:
            took = time.perf_counter() - began
            if took >= SLOW_REQUEST_SECONDS and not scope["path"].startswith("/api/jobs/"):
                timing_log.info("slow request %s %s: %.1fs", scope["method"], route_name(scope), took)


def route_name(scope) -> str:
    """The route a request went to, as the app declares it: FastAPI records the API route it matched; a mounted folder
    (/static) is named by its mount. Otherwise '(no route)': an unknown address, or a login redirect."""
    route = getattr(scope.get("route"), "path", None)
    if route:
        return route
    for mounted in getattr(scope.get("app"), "routes", ()):
        if isinstance(mounted, Mount) and scope["path"].startswith(mounted.path + "/"):
            return mounted.path
    return "(no route)"
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


def document_word(result: TailorResult) -> str:
    """What the run calls the document: a CV in EU style (British usage), a resume otherwise."""
    return "CV" if result.spelling == "eu" else "Resume"


def document_lang(result: TailorResult) -> str:
    """The language the PDF and DOCX files declare: British English for an EU style run."""
    return "en-GB" if result.spelling == "eu" else "en-US"


NO_JD = ("No job description was saved with this run. Runs tailored before the app kept the job description "
         "with each run have none.")


def job_names(result: TailorResult) -> tuple[str, str]:
    """A run's company and job title as History records them (the company typed on the Tailor form wins)."""
    analysis = result.analysis
    return result.company_name or (analysis.company if analysis else ""), analysis.target_role if analysis else ""


def jd_text(result: TailorResult) -> str:
    """A run's job description as plain text: one header line, then the text as it was pasted, or NO_JD."""
    company, role = (" ".join(name.split()) or "-" for name in job_names(result))
    return f"Company: {company} | Job title: {role}\n\n{result.jd.strip() or NO_JD}\n"


def download_name(result: TailorResult, ext: str, cover: bool = False) -> str:
    """The usual pattern for job applications: the name on the resume, then the document. 'Jordan_Rivera_Resume.pdf'
    and 'Jordan_Rivera_Cover_Letter.pdf'. No company in it, so a file sent to the wrong company does not show it."""
    assert result.resume is not None
    name = file_part(person_name(result.resume.profile.name), 60)
    kind = "Cover_Letter" if cover else document_word(result)
    return f"{name}_{kind}.{ext}" if name else f"{kind}.{ext}"


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
        if not path.startswith("/static/") and config.refresh_dotenv():
            router.forget_clients()  # .env changed (a key saved by another worker, or edited by hand)
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

    app.add_middleware(SlowRequestLog)  # added last, so it is the outermost: it times the whole request

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
    templates.env.globals["asset_version"] = asset_version(WEB / "static")
    templates.env.globals["local_zone"] = config.LOCAL_ZONE  # History's days are the same days as the cost cap's

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

    # This process's unfinished jobs. The reference also keeps a task that nobody waits for from being garbage
    # collected. On the Web-tab website (uWSGI without threads) the event loop moves only while a request is served,
    # so a job here is never reported lost for a late heartbeat, and a check on it waits (JOB_POLL_WAIT_SECONDS).
    running: dict[str, asyncio.Task] = {}
    app.state.jobs = running

    def forget(job_id: str, task: asyncio.Task) -> None:
        running.pop(job_id, None)
        if not task.cancelled():
            task.exception()  # retrieved, so asyncio does not warn; run() has recorded it already

    async def answer_soon(user: auth.User, kind: str, work):
        """The run work(steps) makes, if it is ready within JOB_WAIT_SECONDS. Otherwise 202 {"job", "status":
        "running"}: the work goes on, and GET /api/jobs/{job} gives the same answer once it ends. The work marks its
        steps, and the server log gets one line with each step's time and the CPU used (see app/timing.py)."""
        job_id = secrets.token_urlsafe(16)
        db.create_job(job_id, user.id, kind)

        async def beat() -> None:
            while True:
                await asyncio.sleep(JOB_BEAT_SECONDS)
                db.touch_job(job_id)

        async def run() -> TailorResult:
            before, steps = timing.snapshot(), timing.Steps()
            beating = asyncio.create_task(beat())
            try:
                try:
                    result = await work(steps)
                except JobFailed:  # the work ended normally with an error to show: its steps are all marked
                    raise
                except BaseException:
                    steps.mark("until the error")  # from the last step the work finished to where it stopped
                    raise
                finally:  # before the job's own bookkeeping below, so the steps add up to the time logged
                    beating.cancel()
                    timing_log.info("%s job finished in %.1fs: %s", kind, steps.total(), timing.report(steps, before))
            except JobFailed as exc:
                db.finish_job(job_id, run_id=None, error=str(exc))
                raise
            except asyncio.CancelledError:  # Ctrl+C; a host's restart ends the process before this, see beat()
                db.finish_job(job_id, run_id=None, error=JOB_LOST)
                raise
            except Exception as exc:
                log.exception("Background %s failed", kind)
                db.finish_job(job_id, run_id=None, error=f"Unexpected error: {type(exc).__name__}")
                raise
            if not db.finish_job(job_id, run_id=result.run_id) and kind == "tailor" and result.run_id is not None:
                db.delete_run(result.run_id)  # the account (or all data) was deleted meanwhile: keep nothing of it
            return result

        task = asyncio.create_task(run())
        running[job_id] = task
        task.add_done_callback(lambda done: forget(job_id, done))
        await asyncio.wait({task}, timeout=JOB_WAIT_SECONDS)  # never cancels the task, even if this request ends
        if not task.done():
            return JSONResponse({"job": job_id, "status": "running"}, status_code=202)
        try:
            return viewed(task.result(), user)
        except JobFailed as exc:
            raise job_error(str(exc), user) from None

    def log_download(user: auth.User, run_id: int, result: TailorResult, kind: str, fmt: str, filename: str) -> None:
        company, role = job_names(result)
        db.record_download(
            user_id=user.id, email=user.email, run_id=run_id, kind=kind, format=fmt, filename=filename,
            candidate=result.resume.profile.name if result.resume else "",
            company=company, role=role,
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
    def api_downloads(user_id: int | None = None, company: str = "", role: str = "", limit: int = 200,
                      lists: int = 0):
        """History: the newest matching downloads (one account; part of the company and of the job title, any case),
        their totals and how many match. lists=1 (the page's first load) adds the filters' choices: the active
        accounts, and the companies and job titles already in History."""
        company, role = company.strip()[:200], role.strip()[:200]
        answer = {
            "downloads": db.list_downloads(user_id=user_id, company=company, role=role, limit=max(1, min(limit, 1000))),
            "totals": db.download_totals(user_id=user_id, company=company, role=role),
            "cost_today": round(db.cost_today(), 4),
        }
        if lists:
            answer.update(accounts=db.download_accounts(), **db.download_names())
        return answer

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

    # ------------------------------------------------------------------ API keys (admin): kept only in .env
    def known_key(name: str) -> str:
        if name not in key_names(registry):
            raise HTTPException(404, "Unknown key name.")
        return name

    @app.get("/api/keys", dependencies=[Depends(admin_user)])
    def api_keys():
        return {"keys": keys_view(registry)}

    @app.put("/api/keys/{name}", dependencies=[Depends(admin_user)])
    def api_set_key(name: str, body: KeyRequest):
        """Save a pasted API key in .env and use it at once: no restart. The key is never sent back."""
        name, value = known_key(name), body.value.strip()
        if not KEY_VALUE.fullmatch(value):
            raise HTTPException(422, "That does not look like an API key. Paste only the key itself: no spaces, "
                                     "quotes or line breaks.")
        try:
            config.set_env_value(name, value)
        except OSError as error:
            raise HTTPException(500, f"The key could not be saved in .env ({type(error).__name__}).") from None
        router.forget_clients()
        return {"keys": keys_view(registry)}

    @app.delete("/api/keys/{name}", dependencies=[Depends(admin_user)])
    def api_delete_key(name: str):
        """Take a key out of .env; its models stop at once. A key set outside .env is left alone."""
        name = known_key(name)
        if name not in config.FROM_DOTENV:
            if config.key_hint(name):
                raise HTTPException(409, "This key is set as a system environment variable, not in .env: remove it "
                                         "where it was set.")
            return {"keys": keys_view(registry)}
        try:
            config.set_env_value(name, "")
        except OSError as error:
            raise HTTPException(500, f"The key could not be removed from .env ({type(error).__name__}).") from None
        router.forget_clients()
        return {"keys": keys_view(registry)}

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

    @app.put("/api/resumes/{profile_id}/certifications")
    def api_resume_certifications(profile_id: int, body: CertificationList, user: auth.User = Depends(current_user)):
        """The certifications the person has passed, pasted on the Profile page (they replace the saved list). For
        each job the AI then picks the best ones that fit, up to 4, and they are printed without asking."""
        certifications = read_certification_list(body.text)
        if body.text.strip() and not certifications:  # never wipe the saved list by mistake
            raise HTTPException(422, "No certification could be read. Put one per line, for example: Certified "
                                     "Kubernetes Application Developer (CKAD), The Linux Foundation, 2024")
        if len(certifications) > MAX_OWN_CERTIFICATIONS:
            raise HTTPException(422, f"That is {len(certifications)} certifications. Keep the ones you have passed "
                                     f"(at most {MAX_OWN_CERTIFICATIONS}).")

        def replace(profile: Profile) -> None:
            if body.loaded is not None and profile.certifications != body.loaded:
                raise StaleList
            profile.certifications = certifications

        try:
            changed = db.change_profile(profile_id, user_id=user.id, change=replace)
        except StaleList:
            raise HTTPException(409, "This list was changed in another tab (a tick on the Tailor page). Reload this "
                                     "page and save again.") from None
        if changed is None:
            raise HTTPException(404, "Resume not found.")
        entry = db.profile_entry(profile_id, user_id=user.id)
        assert entry is not None
        return {"resume": resume_entry(entry, registry, user.is_admin)}

    @app.delete("/api/resumes/{profile_id}")
    def api_delete_resume(profile_id: int, user: auth.User = Depends(current_user)):
        if not db.delete_profile(profile_id, user_id=user.id):
            raise HTTPException(404, "Resume not found.")
        return {"deleted": True}

    # ------------------------------------------------------------------ work mode and salary of a pasted job post
    facts_cache: OrderedDict[str, dict] = OrderedDict()

    @app.post("/api/job-facts")
    async def api_job_facts(body: JobFactsRequest, user: auth.User = Depends(current_user)):
        """Where the pasted job is done and what it pays, as the AI reads the job post (shown under the job box)."""
        key = hashlib.sha256(body.jd.strip().encode("utf-8")).hexdigest()
        if key in facts_cache:
            facts_cache.move_to_end(key)
            return facts_cache[key]
        try:
            facts = await check_job_facts(router, db, body.jd, user_id=user.id)
        except LookupError as exc:
            raise HTTPException(422, str(exc) if user.is_admin else public_error(str(exc))) from exc
        except CostCapError as exc:
            raise HTTPException(429, str(exc) if user.is_admin else public_error(str(exc))) from exc
        except TimeoutError as exc:
            raise HTTPException(504, "The check took too long. Try again in a minute.") from exc
        except (GenerationFailed, LLMError) as exc:
            raise HTTPException(502, "The check did not work this time. Try again in a minute.") from exc
        facts_cache[key] = facts.model_dump()
        while len(facts_cache) > FACTS_CACHE_SIZE:
            facts_cache.popitem(last=False)
        return facts_cache[key]

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
        return await answer_soon(user, "tailor", lambda steps: run_tailor(
            router, db, profile=profile, profile_id=profile_id, jd=body.jd, hints=body.hints, model_id=model_id,
            allow_fallback=not compare, cover_letter=body.cover_letter, cover_model_id=cover_model,
            company_name=body.company_name.strip(), hiring_manager=body.hiring_manager.strip(), user_id=user.id,
            spelling=body.spelling, steps=steps,
        ))

    @app.get("/api/jobs/{job_id}")
    async def api_job(job_id: str, user: auth.User = Depends(current_user)):
        """Slow work started by POST /api/tailor or /api/runs/{id}/cover-letter: still running, or its answer. A job
        running in this process is given up to JOB_POLL_WAIT_SECONDS to end first (see JOB_POLL_WAIT_SECONDS)."""
        job = db.get_job(job_id, user.id) if len(job_id) <= 64 else None
        if job is None:
            raise HTTPException(404, "Not found.")
        if job["status"] == "running" and job_id in running:
            await asyncio.wait({running[job_id]}, timeout=JOB_POLL_WAIT_SECONDS)
            job = db.get_job(job_id, user.id) or job
        if job["status"] == "running":
            lost = auth.utc(auth.now_utc() - timedelta(seconds=JOB_LOST_SECONDS))
            if job_id in running or max(job["created_at"], job["beat_at"]) >= lost:
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
                from_model=False,  # the user's own words: kept as typed, run_checks notes what to look at
            )
            # the printed certifications are not part of the text: they stay as ticked (and count in the fit below)
            merged.certifications = [cert.model_copy() for cert in result.resume.certifications]
            result.untrimmed = []  # the edited text is the whole text now
            if result.cover_letter and body.cover_paragraphs is not None:
                greeting, paragraphs, closing = validate.clean_cover_letter(
                    body.cover_greeting or result.cover_letter.greeting,
                    body.cover_paragraphs,
                    body.cover_closing or result.cover_letter.closing,
                    from_model=False,
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

    @app.put("/api/runs/{run_id}/certifications")
    async def api_certifications(run_id: int, body: CertificationTicks, user: auth.User = Depends(current_user)):
        """The "I have it" ticks on the Tailor page, one per certification the AI picked for this job. A ticked one
        is printed on the resume (after Education) and remembered on the base resume for the next jobs; an
        unticked one comes off both. Only ticks that changed are applied, so an older run's rows never undo a tick
        made on another run. The page is fitted again: a bullet is removed only if two pages need it (and the
        reply says so)."""
        result = load_run(run_id, user, write=True)
        if result.status != "ok" or result.resume is None or result.analysis is None or not result.certifications:
            raise HTTPException(409, "This resume has no certifications to choose from.")
        if len(body.have) != len(result.certifications):
            raise HTTPException(422, f"Send one tick for each of the {len(result.certifications)} certifications.")
        changed = [(row, have) for row, have in zip(result.certifications, body.have) if row.have != have]
        if not changed:
            return viewed(result, user)
        added = [Certification(name=row.name, issuer=row.issuer, year=row.year) for row, have in changed if have]
        dropped = [row.name for row, have in changed if not have]
        profile_id, owner = db.run_source(run_id)
        if profile_id is not None:  # remembered for the next jobs with this base resume (if it still exists)
            db.change_profile(profile_id, user_id=owner,
                              change=lambda profile: validate.change_certifications(profile, added, dropped))
        rows = [row.model_copy(update={"have": have}) for row, have in zip(result.certifications, body.have)]
        resume = result.resume.model_copy(deep=True)
        if result.untrimmed:  # start again from the bullets an earlier tick removed
            resume.experience = [exp.model_copy(deep=True) for exp in result.untrimmed]
        validate.change_certifications(resume.profile, added, dropped)
        resume.certifications = validate.printed_certifications(rows)
        validate.assert_certifications_held(resume)
        whole = [exp.model_copy(deep=True) for exp in resume.experience]
        resume, style, pages, fits, notes, removed = await refit(resume, result.style.paper)
        fresh = db.get_run(run_id) or result  # a cover letter may have been written while this was in flight
        fresh.resume, fresh.certifications = resume, rows
        fresh.untrimmed = whole if removed else []
        fresh.style, fresh.pages, fresh.fits = style, pages, fits
        fresh.notes = [n for n in fresh.notes if not n.startswith(("Font reduced", "Still"))] + notes
        run_checks(fresh, fresh.jd or analysis_terms(fresh.analysis))
        db.update_run(fresh)
        view = viewed(fresh, user)
        if removed:
            total = sum(removed.values())
            parts = [f"{count} bullet{'' if count == 1 else 's'} from {name}" for name, count in removed.items()]
            view["message"] = (f"To keep two pages, {' and '.join(parts)} {'was' if total == 1 else 'were'} "
                               "removed. Untick a certification to bring them back.")
            view["message_kind"] = "warn"
        elif result.untrimmed:
            view["message"], view["message_kind"] = "The bullets removed earlier are back.", "ok"
        return view

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

        async def work(steps: timing.Steps) -> TailorResult:
            fresh, error = await write_cover_letter(router, db, run_id, jd=jd, model_id=model_id, user_id=user.id,
                                                    steps=steps)
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
        pdf, _pages = render_resume_pdf(result.resume, result.style, document_word(result))
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
        return file_response(resume_docx(result.resume, result.style, document_word(result)), DOCX, "attachment", name)

    @app.get("/api/runs/{run_id}/cover.pdf")
    def api_cover_pdf(run_id: int, download: int = 0, user: auth.User = Depends(current_user)):
        result = load_run(run_id, user)
        if result.resume is None or result.cover_letter is None:
            raise HTTPException(404, "No cover letter in this run.")
        pdf, _ = render_cover_pdf(result.resume, result.cover_letter, result.style.paper, document_lang(result))
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
        letter = cover_docx(result.resume, result.cover_letter, result.style.paper, document_lang(result))
        return file_response(letter, DOCX, "attachment", name)

    @app.get("/api/runs/{run_id}/jd.txt")
    def api_run_jd(run_id: int, user: auth.User = Depends(current_user)):
        """The job description a run was made for, as plain text (History opens it in a new tab). Its owner or the
        admin may read it, like the PDFs; like a preview, it is never counted as a download."""
        text = jd_text(load_run(run_id, user))
        return file_response(text.encode("utf-8"), "text/plain; charset=utf-8", "inline",
                             f"Job_Description_{run_id}.txt")

    @app.delete("/api/data", dependencies=[Depends(admin_user)])
    def api_delete_all():
        """Privacy control: remove every resume, run and download record (accounts and settings stay)."""
        db.purge_all()
        return {"deleted": True}

    return app
