"""WSGI entry point, for hosts that run WSGI apps (PythonAnywhere's Web tab): `from app.wsgi import application`.

The app itself is ASGI (FastAPI); a2wsgi runs it inside a WSGI server. Under WSGI each worker handles one
request at a time, so prefer `uvicorn app.asgi:app` where the host offers it.
"""
from a2wsgi import ASGIMiddleware

from .asgi import app

application = ASGIMiddleware(app)
