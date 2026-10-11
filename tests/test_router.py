import pytest

from app.llm.base import CostCapError, ProviderError
from app.schemas import TailorResult


def test_only_demo_models_without_keys(registry, router, monkeypatch):
    for spec in registry.all():
        if spec.key_env:
            monkeypatch.delenv(spec.key_env, raising=False)
    available = {s.id for s in registry.available(["US", "EU", "SG", "DEMO"])}
    assert available == {"demo-concise", "demo-detailed", "demo-flaky"}
    assert router.default_model("resume") == "demo-concise"
    assert router.default_model("parser") == "demo-concise"


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
