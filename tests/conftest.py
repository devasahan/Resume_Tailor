import os

os.environ.setdefault("FAKE_DELAY", "0")  # demo models answer instantly in tests

import pytest

from app.llm.registry import Registry
from app.llm.router import Router
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
