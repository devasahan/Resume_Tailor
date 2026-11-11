"""API keys pasted on the Settings page: kept only in .env, used at once, never sent back."""
import os

import pytest
from fastapi.testclient import TestClient

from app import config

OPENAI_KEY = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-ZyXw"
TOUCHED = ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "COOKIE_SECURE")


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """A .env of its own (never the real one); every variable these tests set is put back afterwards."""
    path = tmp_path / ".env"
    path.write_text("# my settings\nCOOKIE_SECURE=\nGEMINI_API_KEY=AIzaSyOLDOLDOLDOLD1234\nOPENAI_API_KEY=\n",
                    encoding="utf-8")
    for name in TOUCHED:
        monkeypatch.setenv(name, "x")  # remembers the variable as it was, so the undo restores it
        monkeypatch.delenv(name)
    monkeypatch.setenv("ENV_FILE", str(path))
    monkeypatch.setattr(config, "FROM_DOTENV", set())
    monkeypatch.setattr(config, "_INHERITED", {})
    monkeypatch.setattr(config, "_ENV_STAMP", None)
    return path


def model(client, model_id):
    return next(m for m in client.get("/api/models").json()["models"] if m["id"] == model_id)


def keys(response) -> dict:
    return {k["name"]: k for k in response.json()["keys"]}


def test_a_pasted_key_is_saved_in_env_and_works_at_once(client, env_file):
    assert not model(client, "gpt-6.1-sol")["available"]
    now = keys(client.get("/api/keys"))
    assert now["OPENAI_API_KEY"]["set"] is False and now["OPENAI_API_KEY"]["models"] == ["GPT-6.1 Sol"]
    assert now["GEMINI_API_KEY"]["hint"] == "AIza-...1234 (from .env)"  # .env was read on the first request
    saved = client.put("/api/keys/OPENAI_API_KEY", json={"value": f"  {OPENAI_KEY}  "})
    assert saved.status_code == 200 and OPENAI_KEY not in saved.text
    openai = keys(saved)["OPENAI_API_KEY"]
    assert openai["set"] and openai["in_env_file"] and openai["hint"] == "sk-proj-...ZyXw (from .env)"
    assert model(client, "gpt-6.1-sol")["available"]  # no restart
    assert env_file.read_text(encoding="utf-8") == (
        f"# my settings\nCOOKIE_SECURE=\nGEMINI_API_KEY=AIzaSyOLDOLDOLDOLD1234\nOPENAI_API_KEY={OPENAI_KEY}\n")
    for path in ("/api/keys", "/api/models", "/api/settings", "/settings"):
        assert OPENAI_KEY[8:-4] not in client.get(path).text, path  # never sent back


def test_a_new_key_replaces_the_old_one_and_clients_start_again(client, env_file):
    router = client.app.state.router
    client.get("/api/keys")
    router._clients["gemini-3.8-flash"] = object()  # a client made with the old key
    assert client.put("/api/keys/GEMINI_API_KEY", json={"value": "AIzaSyNEWNEWNEWNEW5678"}).status_code == 200
    assert router._clients == {}
    text = env_file.read_text(encoding="utf-8")
    assert text.count("GEMINI_API_KEY=") == 1 and "GEMINI_API_KEY=AIzaSyNEWNEWNEWNEW5678\n" in text
    assert os.environ["GEMINI_API_KEY"] == "AIzaSyNEWNEWNEWNEW5678"


@pytest.mark.parametrize("name,value,status", [
    ("NOT_A_KEY", "sk-proj-abcdefgh12345678", 404),
    ("ADMIN_PASSWORD", "sk-proj-abcdefgh12345678", 404),  # only the models' key variables
    ("OPENAI_API_KEY", "sk proj with spaces 1234", 422),
    ("OPENAI_API_KEY", "short", 422),
    ("OPENAI_API_KEY", "sk-proj-abc\nADMIN_PASSWORD=hacked", 422),  # never a second .env line
    ("OPENAI_API_KEY", '"sk-proj-quoted1234"', 422),
    ("OPENAI_API_KEY", "sk-proj-abc#comment1234", 422),
])
def test_a_wrong_key_name_or_value_is_refused_and_env_stays(client, env_file, name, value, status):
    before = env_file.read_text(encoding="utf-8")
    assert client.put(f"/api/keys/{name}", json={"value": value}).status_code == status
    assert env_file.read_text(encoding="utf-8") == before


def test_only_the_admin_can_see_or_change_keys(client, user_client, env_file):
    assert user_client.get("/api/keys").status_code == 403
    assert user_client.put("/api/keys/OPENAI_API_KEY", json={"value": OPENAI_KEY}).status_code == 403
    assert user_client.delete("/api/keys/GEMINI_API_KEY").status_code == 403
    assert TestClient(client.app).get("/api/keys").status_code == 401
    other_site = client.put("/api/keys/OPENAI_API_KEY", json={"value": OPENAI_KEY},
                            headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert other_site.status_code == 403
    assert "OPENAI_API_KEY=\n" in env_file.read_text(encoding="utf-8")


def test_removing_a_key_takes_its_models_off_at_once(client, env_file, monkeypatch):
    client.get("/api/keys")  # .env read: Gemini is set
    assert model(client, "gemini-3.8-flash")["available"]
    removed = client.delete("/api/keys/GEMINI_API_KEY")
    assert removed.status_code == 200 and keys(removed)["GEMINI_API_KEY"]["set"] is False
    assert "GEMINI_API_KEY" not in env_file.read_text(encoding="utf-8")
    assert not model(client, "gemini-3.8-flash")["available"]
    monkeypatch.setenv("XAI_API_KEY", "xai-systemsystem1234")  # set outside .env: not ours to remove
    assert client.delete("/api/keys/XAI_API_KEY").status_code == 409


def test_removing_a_pasted_key_brings_back_the_system_one(client, env_file, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-SYSTEMKEY-sys1")
    assert client.put("/api/keys/ANTHROPIC_API_KEY", json={"value": "sk-ant-api03-PASTEDKEY-new2"}).status_code == 200
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-api03-PASTEDKEY-new2"
    assert client.delete("/api/keys/ANTHROPIC_API_KEY").status_code == 200
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-api03-SYSTEMKEY-sys1"


def test_a_key_saved_by_another_worker_is_read_on_the_next_request(client, env_file):
    client.get("/api/keys")
    text = env_file.read_text(encoding="utf-8").replace("OPENAI_API_KEY=\n", f"OPENAI_API_KEY={OPENAI_KEY}\n")
    env_file.write_text(text, encoding="utf-8")  # as another process (or a hand edit) would
    assert model(client, "gpt-6.1-sol")["available"]
    env_file.write_text(text.replace(f"OPENAI_API_KEY={OPENAI_KEY}\n", ""), encoding="utf-8")
    assert not model(client, "gpt-6.1-sol")["available"]  # gone from .env: gone from the app


@pytest.mark.parametrize("value,hint", [
    (OPENAI_KEY, "sk-proj-...ZyXw"), ("sk-ant-api03-SECRETSECRET-ab12", "sk-ant-api03-...ab12"),
    ("xai-SECRETSECRETSECRET9876", "xai-...9876"), ("AIzaSySECRETSECRET4321", "AIza-...4321"),
])
def test_a_hint_never_shows_the_random_part_of_a_key(monkeypatch, value, hint):
    monkeypatch.setenv("ZZ_HINT_KEY", value)
    assert config.key_hint("ZZ_HINT_KEY") == f"{hint} (from system environment variable)"


def test_a_new_env_file_is_readable_by_its_owner_only_and_an_old_one_keeps_its_mode(client, env_file):
    env_file.unlink()
    assert client.put("/api/keys/OPENAI_API_KEY", json={"value": OPENAI_KEY}).status_code == 200
    assert env_file.read_text(encoding="utf-8") == f"OPENAI_API_KEY={OPENAI_KEY}\n"
    if os.name == "posix":
        assert env_file.stat().st_mode & 0o777 == 0o600
        env_file.chmod(0o640)
        assert client.put("/api/keys/GEMINI_API_KEY", json={"value": "AIzaSyNEWNEWNEWNEW5678"}).status_code == 200
        assert env_file.stat().st_mode & 0o777 == 0o640


def test_a_save_that_is_cut_off_keeps_the_old_env_and_leaves_no_copy(client, env_file, monkeypatch):
    before = env_file.read_text(encoding="utf-8")

    def disk_full(*args):
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(config.os, "replace", disk_full)
        failed = client.put("/api/keys/OPENAI_API_KEY", json={"value": OPENAI_KEY})
    assert failed.status_code == 500 and OPENAI_KEY not in failed.text
    assert env_file.read_text(encoding="utf-8") == before
    assert not list(env_file.parent.glob(".env.saving*"))  # no copy of the keys left behind
    assert not model(client, "gpt-6.1-sol")["available"]
    assert ".env.saving*" in (config.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()  # never committed


def test_the_settings_page_has_the_api_keys_panel(client, env_file):
    page = client.get("/settings").text
    assert "<h2>API keys</h2>" in page and '<tbody id="key-rows">' in page
    assert "type: DOTS ? 'text' : 'password'" in page  # dots, and no browser offer to save it as the site's password
    assert "Keys are kept only in the .env file on the server" in page
