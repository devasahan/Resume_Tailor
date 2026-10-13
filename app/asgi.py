"""`uvicorn app.asgi:app` entry point."""
from .main import create_app, show_app_log

show_app_log()
app = create_app()
