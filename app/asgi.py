"""`uvicorn app.asgi:app` entry point."""
from .main import create_app

app = create_app()
