"""`uvicorn app.asgi:app` entry point."""
from .main import create_app, show_app_log, warm_up

show_app_log()
app = create_app()
warm_up()  # the first tailoring's one-time work, now, while the server starts
