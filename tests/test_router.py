import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfoNotFoundError

import pytest

from app.llm.base import CostCapError, ProviderError
from app.schemas import TailorResult
from app.storage import db as db_module
from app.storage.db import day_bounds


def test_only_demo_models_without_keys(registry, router, monkeypatch):
    for spec in registry.all():
        if spec.key_env:
            monkeypatch.delenv(spec.key_env, raising=False)
    available = {s.id for s in registry.available(["US", "EU", "SG", "DEMO"])}
    assert available == {"demo-concise", "demo-detailed", "demo-flaky"}
    assert router.default_model("resume") == "demo-concise"
    assert router.default_model("parser") is None  # Automatic with only demo models: read with the rules
    router.db.set_settings({"parser_model": "demo-concise"})
    assert router.default_model("parser") == "demo-concise"  # an explicit choice is honoured
    router.db.set_settings({"parser_model": "rules"})
    assert router.default_model("parser") is None


def test_key_and_region_gating(registry, router, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert router.default_model("resume") == "claude-sonnet-5-5"  # first by priority once a key exists
    regions = router.settings()["allowed_regions"]
    assert "CN" not in regions
    assert not registry.status(registry.get("deepseek-v4-flash"), regions).available  # key set, region blocked
    router.db.set_settings({"allowed_regions": [*regions, "CN"]})
    assert registry.status(registry.get("deepseek-v4-flash"), router.settings()["allowed_regions"]).available


def test_user_choice_overrides_priority_and_unavailable_choice_is_ignored(router, monkeypatch):
    router.db.set_settings({"resume_model": "demo-detailed", "cover_letter_model": "claude-sonnet-5-5"})
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert router.default_model("resume") == "demo-detailed"
    assert router.default_model("cover_letter") == "demo-detailed"  # chosen model has no key, falls back to resume's


def test_unknown_or_unavailable_model_raises(router, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="no API key"):
        router.usable_spec("claude-sonnet-5-5")
    with pytest.raises(ProviderError, match="Unknown model"):
        router.usable_spec("nope")


def test_fallback_chain_skips_unavailable_and_duplicates(router, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    router.db.set_settings({"fallback_models": ["grok-4.3", "demo-detailed", "demo-concise", "demo-detailed"]})
    assert router.chain("demo-concise", True) == ["demo-concise", "demo-detailed"]
    assert router.chain("demo-concise", False) == ["demo-concise"]


def test_cost_cap(router, db):
    db.set_settings({"daily_cost_cap_usd": 1.0})
    router.check_cost_cap()
    db.save_run(TailorResult(requested_model="m", cost_usd=1.5), None, "")
    with pytest.raises(CostCapError):
        router.check_cost_cap()
    db.set_settings({"daily_cost_cap_usd": 0})
    router.check_cost_cap()  # 0 = no cap


# ------------------------------------------------------------------ the day is the owner's day in Arlington, Texas
def run_at(db, created_at, cost):
    run_id = db.save_run(TailorResult(requested_model="m", cost_usd=cost), None, "")
    with sqlite3.connect(db.path) as raw:
        raw.execute("UPDATE runs SET created_at = ? WHERE id = ?", (created_at, run_id))


def at(text):
    return datetime.fromisoformat(text)  # Python 3.11+ reads the "Z"


def test_spent_today_starts_at_midnight_texas_time(db):
    """Reported: at 10 PM in Texas 'Spent today' already counted the next (UTC) day."""
    run_at(db, "2026-10-14T04:59:59Z", 1.0)  # 11:59:59 PM CDT, Tue Oct 13
    run_at(db, "2026-10-14T05:00:00Z", 0.2)  # midnight CDT, Wed Oct 14
    run_at(db, "2026-10-15T03:00:00Z", 0.3)  # 10 PM CDT, Wed Oct 14 (UTC already says Oct 15)
    run_at(db, "2026-10-15T05:00:00Z", 4.0)  # midnight CDT, Thu Oct 15
    assert db.cost_today(at("2026-10-15T03:30:00Z")) == pytest.approx(0.5)
    assert db.cost_today(at("2026-10-14T04:00:00Z")) == pytest.approx(1.0)


@pytest.mark.parametrize("now,bounds", [
    ("2026-03-08T12:00:00Z", ("2026-03-08T06:00:00Z", "2026-03-09T05:00:00Z")),  # spring forward: a 23-hour day
    ("2026-11-01T12:00:00Z", ("2026-11-01T05:00:00Z", "2026-11-02T06:00:00Z")),  # fall back: a 25-hour day
    ("2026-01-15T05:59:59Z", ("2026-01-14T06:00:00Z", "2026-01-15T06:00:00Z")),  # winter (CST), just before midnight
])
def test_texas_days_around_daylight_saving(now, bounds):
    assert day_bounds(at(now)) == bounds


def test_the_cost_cap_follows_the_texas_day(router, db, monkeypatch):
    db.set_settings({"daily_cost_cap_usd": 1.0})
    run_at(db, "2026-10-15T03:00:00Z", 1.0)  # 10 PM CDT
    monkeypatch.setattr(db_module, "utc_now", lambda: at("2026-10-15T04:00:00Z"))  # 11 PM CDT, the same day
    with pytest.raises(CostCapError):
        router.check_cost_cap()
    monkeypatch.setattr(db_module, "utc_now", lambda: at("2026-10-15T05:00:00Z"))  # midnight: a new day
    router.check_cost_cap()


def test_without_time_zone_data_the_day_is_the_utc_day(db, monkeypatch):
    def missing(name):
        raise ZoneInfoNotFoundError(name)

    monkeypatch.setattr(db_module, "ZoneInfo", missing)
    db_module.local_zone.cache_clear()
    try:
        assert day_bounds(at("2026-10-15T03:00:00Z")) == ("2026-10-15T00:00:00Z", "2026-10-16T00:00:00Z")
        run_at(db, "2026-10-15T03:00:00Z", 0.25)
        assert db.cost_today(at("2026-10-15T23:00:00Z")) == pytest.approx(0.25)
    finally:
        db_module.local_zone.cache_clear()


def test_history_uses_the_same_zone_as_the_server(client):
    assert 'const ZONE = "America/Chicago";' in client.get("/history").text



def test_gemini_flash_is_the_automatic_resume_model_once_its_key_is_set(router, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert router.default_model("resume") == "gemini-3.8-flash"
    router.db.set_settings({"resume_model": "claude-sonnet-5-5"})
    assert router.default_model("resume") == "claude-sonnet-5-5"  # an explicit choice still wins
