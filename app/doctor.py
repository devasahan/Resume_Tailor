"""`python -m app.doctor`: check that this computer can run Resume Tailor, and say how to fix what cannot.

Output is plain ASCII so it prints correctly in any terminal, including the Windows console.
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
from dataclasses import dataclass

REQUIRED = [
    ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("jinja2", "jinja2"), ("pydantic", "pydantic"),
    ("yaml", "pyyaml"), ("anthropic", "anthropic"), ("httpx", "httpx"), ("fpdf", "fpdf2"),
    ("docx", "python-docx"), ("pypdf", "pypdf"), ("pdfplumber", "pdfplumber"), ("multipart", "python-multipart"),
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    fatal: bool = True  # False = worth knowing, but the app still runs


def check_python() -> Check:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info >= (3, 11):
        return Check("Python", True, version)
    return Check("Python", False, f"{version} is too old", "Install Python 3.11 or newer from https://www.python.org/downloads/")


def check_libraries() -> Check:
    missing = []
    for module, package in REQUIRED:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    if not missing:
        return Check("Libraries", True, "all installed")
    return Check("Libraries", False, "missing: " + ", ".join(missing), "Run: pip install -r requirements.txt")


def check_pdf_and_docx() -> Check:
    try:
        from .render.docx import resume_docx
        from .render.pdf import render_resume_pdf
        from .schemas import Company, Contact, FitStyle, Profile, Resume, ResumeExperience, SkillGroup

        profile = Profile(name="Test Person", contact=Contact(email="test@example.com"), companies=[Company(name="Example Co", start="Jan 2020", end="Present", mode="Remote")])
        resume = Resume(
            profile=profile, headline="Test Role | Skill", summary="A short test summary.",
            skills=[SkillGroup(category="Core", skills=["One", "Two"])],
            experience=[ResumeExperience(company_index=0, job_title="Test Role", bullets=["Did a thing."])],
        )
        pdf, pages = render_resume_pdf(resume, FitStyle())
        resume_docx(resume, FitStyle())
        if pages != 1 or not pdf.startswith(b"%PDF"):
            return Check("PDF and DOCX export", False, "the test PDF was not one valid page", "Reinstall: pip install --force-reinstall fpdf2")
        return Check("PDF and DOCX export", True, "made a test PDF and DOCX")
    except Exception as exc:
        return Check("PDF and DOCX export", False, f"{type(exc).__name__}: {exc}", "Run: pip install -r requirements.txt")


def check_data_dir() -> Check:
    try:
        from . import config

        folder = config.data_dir()
        probe = folder / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return Check("Data folder", True, f"writable: {folder}")
    except Exception as exc:
        return Check("Data folder", False, f"{type(exc).__name__}: {exc}", "Set DATA_DIR in .env to a folder you can write to")


def check_models() -> list[Check]:
    try:
        from .llm.registry import Registry

        registry = Registry.load()
    except Exception as exc:
        return [Check("Model list (models.yaml)", False, f"{type(exc).__name__}: {exc}", "Fix models.yaml, or restore it from the project")]
    from . import config

    ready = [s for s in registry.all() if s.provider != "fake" and s.has_key()]
    checks = [Check("Model list (models.yaml)", True, f"{len(registry.all())} models listed")]
    if ready:
        names = sorted({s.key_env for s in ready})
        detail = "; ".join(f"{n} = {config.key_hint(n)}" for n in names)
        checks.append(Check("API keys", True, f"{detail} -> ready: " + ", ".join(s.label for s in ready)))
    else:
        checks.append(Check(
            "API keys", False, "none found, so the app starts in demo mode (offline demo models only)",
            "To use real models: copy .env.example to .env, add a key, restart. See README.", fatal=False,
        ))
    return checks


def check_admin() -> Check:
    from . import auth, config  # config reads .env

    email, password = config.admin_credentials()
    fix = "Open .env (Windows: notepad .env), add ADMIN_EMAIL=your email and ADMIN_PASSWORD=your password, save, start again."
    if not auth.normalize_email(email):
        return Check("Admin account", False, "ADMIN_EMAIL is not set (or not an email), so nobody can log in as admin",
                     fix, fatal=False)
    if not password:
        return Check("Admin account", True, f"{email} (ADMIN_PASSWORD is empty: the password saved earlier stays)")
    if len(password) < auth.MIN_PASSWORD:
        return Check("Admin account", False, f"ADMIN_PASSWORD is shorter than {auth.MIN_PASSWORD} characters",
                     "Choose a longer password in .env, then start again.", fatal=False)
    return Check("Admin account", True, email)


def port_problem(host: str, port: int) -> str | None:
    """None if the port can be used, otherwise a message that says what to do."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "posix":  # on Windows this option would hide a port that is really in use
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        return None
    except OSError:
        return (
            f"Port {port} is already in use (maybe Resume Tailor is already running in another window).\n"
            f"  To use another port:  Windows: set PORT={port + 1}   then start again\n"
            f"                        macOS/Linux: PORT={port + 1} ./start.sh"
        )
    finally:
        probe.close()


def check_port(host: str, port: int) -> Check:
    problem = port_problem(host, port)
    if problem is None:
        return Check("Port", True, f"{port} is free")
    return Check("Port", False, problem.splitlines()[0], "Close the other copy, or use another port (see the message when starting)", fatal=False)


def run_checks(host: str = "127.0.0.1", port: int = 8000) -> list[Check]:
    checks = [check_python(), check_libraries()]
    if checks[-1].ok:  # the next checks import the libraries
        checks.append(check_pdf_and_docx())
        checks.append(check_data_dir())
        checks.extend(check_models())
        checks.append(check_admin())
    checks.append(check_port(host, port))
    return checks


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    quiet = "--quiet" in args
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    checks = run_checks(host, port)
    fatal = [c for c in checks if not c.ok and c.fatal]
    if not quiet:
        print("Resume Tailor setup check")
    for check in checks:
        if quiet and check.ok:
            continue
        mark = "[ok]" if check.ok else ("[!!]" if check.fatal else "[--]")
        print(f"  {mark} {check.name}: {check.detail}")
        if not check.ok and check.fix:
            print(f"       -> {check.fix}")
    if fatal:
        print("\nSomething needs fixing before the app can run (see the lines marked [!!]).")
        return 1
    if not quiet:
        print("\nAll set. Start the app with:  python -m app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
