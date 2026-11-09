"""WSGI entry point, for hosts that run WSGI apps (PythonAnywhere's Web tab): `from app.wsgi import application`.

The app itself is ASGI (FastAPI); a2wsgi runs it inside a WSGI server. Under WSGI each worker handles one
request at a time, so prefer `uvicorn app.asgi:app` where the host offers it.
"""
from __future__ import annotations

import os
import threading

from a2wsgi import ASGIMiddleware

from .main import create_app, show_app_log, warm_up

show_app_log()
warm_up()  # the first tailoring's one-time work, at import: forked workers start with it done (it starts no thread)

_bridge: ASGIMiddleware | None = None
_bridge_pid = 0
_lock = threading.Lock()


def application(environ, start_response):
    """a2wsgi runs the app on an event-loop thread. A thread does not survive a fork, and WSGI servers often load
    this file once and then fork their workers, so each worker process builds the app and its bridge on its first
    request (building the app at import would also start threads in the process that forks)."""
    global _bridge, _bridge_pid
    if _bridge is None or _bridge_pid != os.getpid():
        with _lock:
            if _bridge is None or _bridge_pid != os.getpid():
                show_app_log()
                _bridge, _bridge_pid = ASGIMiddleware(create_app()), os.getpid()
    return _bridge(environ, start_response)
