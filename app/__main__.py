"""`python -m app` starts Resume Tailor and opens it in your browser (http://127.0.0.1:8000).

Environment (all optional): PORT, HOST, OPEN_BROWSER=0 (do not open the browser), LOG_LEVEL.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import webbrowser

from . import doctor


def main() -> int:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    problem = doctor.port_problem(host, port)
    if problem:
        print(problem)
        return 1

    import uvicorn

    from .llm.registry import Registry

    shown_host = "127.0.0.1" if host in ("", "0.0.0.0") else host
    url = f"http://{shown_host}:{port}"
    registry = Registry.load()
    if not any(s.provider != "fake" and s.has_key() for s in registry.all()):
        print("No API keys found: starting in demo mode (offline demo models). See README to add keys.")
    print(f"Resume Tailor is running at {url}\nPress Ctrl+C to stop.")
    if os.environ.get("OPEN_BROWSER", "1") != "0":
        threading.Timer(1.5, webbrowser.open, [url]).start()
    level = os.environ.get("LOG_LEVEL", "warning")
    if level.lower() not in ("info", "debug", "trace"):
        logging.getLogger("app").setLevel(logging.WARNING)  # a quiet console: no per-call timing lines
    try:
        uvicorn.run("app.asgi:app", host=host, port=port, log_level=level)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
