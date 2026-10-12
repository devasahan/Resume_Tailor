import os

os.environ.setdefault("FAKE_DELAY", "0")  # demo models answer instantly in tests

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.llm.registry import Registry
from app.llm.router import Router
from app.main import create_app
from app.schemas import Company, Contact, Education, Profile
from app.storage.db import Database

SAMPLE_JD = (
    "Senior Data Engineer. You will build ETL pipelines with Python, SQL, Airflow and Spark on AWS and Snowflake, "
    "mentor engineers, and work with stakeholders. We use Agile, CI/CD and Docker. 5+ years of experience required."
)

SAMPLE_RESUME_TEXT = """Jordan Rivera
Austin, TX | jordan.rivera@example.com | (555) 010-2030 | linkedin.com/in/jordanrivera

EXPERIENCE
Northwind Labs | Remote
Mar 2022 - Present
Contoso Systems - Dallas, TX | Onsite  Jun 2019 - Feb 2022
Fabrikam Inc. (Hybrid)  Jul 2016 - May 2019

EDUCATION
University of Texas at Austin
B.S. in Computer Science, 2016
"""


@pytest.fixture
def profile() -> Profile:
    return Profile(
        name="Jordan Rivera",
        contact=Contact(email="jordan@example.com", phone="(555) 010-2030", linkedin="linkedin.com/in/jordanrivera", location="Austin, TX"),
        companies=[
            Company(name="Northwind Labs", start="Mar 2022", end="Present", mode="Remote"),
            Company(name="Contoso Systems", start="Jun 2019", end="Feb 2022", mode="Onsite", location="Dallas, TX"),
            Company(name="Fabrikam", start="Jul 2016", end="May 2019", mode="Hybrid"),
        ],
        education=[Education(school="University of Texas at Austin", degree="B.S.", field="Computer Science", end="2016")],
    )


@pytest.fixture
def jd() -> str:
    return SAMPLE_JD


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def registry() -> Registry:
    return Registry.load()


@pytest.fixture
def router(registry, db) -> Router:
    return Router(registry, db)


ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin-test-password"
USER_PASSWORD = "user-test-password"


@pytest.fixture(autouse=True)
def test_accounts(monkeypatch):
    """Test credentials for the admin account, never the developer's real .env values; a cheap scrypt setting
    so hashing does not slow the suite down."""
    monkeypatch.setenv("ADMIN_EMAIL", ADMIN_EMAIL)
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN_PASSWORD)
    monkeypatch.delenv("COOKIE_SECURE", raising=False)
    monkeypatch.setattr(auth, "SCRYPT_N", 2**4)


def log_in(client: TestClient, email: str, password: str) -> TestClient:
    response = client.post("/api/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return client


def add_user(client: TestClient, email: str = "user@example.com", status: str = "active") -> int:
    """An account made directly in the database (as if signed up and approved)."""
    user_id = client.app.state.db.create_user(email, auth.hash_password(USER_PASSWORD), role="user", status=status)
    assert user_id is not None
    return user_id


@pytest.fixture
def client(tmp_path):
    """The web app on a fresh database, logged in as the admin. Uploads are read with the rules, so a developer's
    .env keys are never used."""
    db = Database(tmp_path / "api.db")
    db.set_settings({"parser_model": "rules"})
    return log_in(TestClient(create_app(db=db)), ADMIN_EMAIL, ADMIN_PASSWORD)


@pytest.fixture
def user_client(client):
    """A second browser on the same app, logged in as an approved user."""
    add_user(client)
    return log_in(TestClient(client.app), "user@example.com", USER_PASSWORD)
