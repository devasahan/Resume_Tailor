"""The OpenAI-compatible adapter over a real TCP connection to a local server (no mock transport)."""
import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.llm.base import ProviderError
from app.llm.openai_compat_client import OpenAICompatClient
from app.llm.registry import ModelSpec

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


@pytest.fixture
def server():
    seen: list[dict] = []
    behaviour = {"reject_schema": False, "reject_effort": False, "status": 200}
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        seen.append({"body": body, "auth": request.headers.get("authorization")})
        if behaviour["status"] != 200:
            return JSONResponse({"error": "nope"}, status_code=behaviour["status"])
        if behaviour["reject_effort"] and "reasoning_effort" in body:
            return JSONResponse({"error": "unknown parameter reasoning_effort"}, status_code=400)
        if behaviour["reject_schema"] and body.get("response_format", {}).get("type") == "json_schema":
            return JSONResponse({"error": "json_schema unsupported"}, status_code=400)
        return JSONResponse({
            "model": "served-model-v2",
            "choices": [{"message": {"role": "assistant", "content": "```json\n" + json.dumps({"a": 7}) + "\n```"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 9},
        })

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}/v1", "seen": seen, "behaviour": behaviour}
    srv.should_exit = True
    thread.join(timeout=5)


def client_for(server, **settings):
    spec = ModelSpec(id="m", label="M", provider="openai_compat", model="m-1", base_url=server["url"],
                     key_env="WIRE_KEY", price_in=1, price_out=2, settings=settings)
    return OpenAICompatClient(spec)


def call(client):
    return asyncio.run(client.generate_structured(system="sys", user="usr", schema=SCHEMA, schema_name="thing", max_tokens=777))


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("WIRE_KEY", "wire-secret")


def test_round_trip_over_real_http(server):
    result = call(client_for(server, json_mode="json_schema", reasoning_effort="low"))
    sent = server["seen"][0]
    assert sent["auth"] == "Bearer wire-secret"
    assert sent["body"]["max_tokens"] == 777 and sent["body"]["reasoning_effort"] == "low"
    assert sent["body"]["response_format"]["json_schema"]["strict"] is True
    assert result.data == {"a": 7}  # fenced JSON accepted
    assert (result.usage.input_tokens, result.usage.output_tokens, result.served_model) == (20, 9, "served-model-v2")


def test_degrades_across_real_400s_and_remembers_across_calls(server):
    server["behaviour"].update(reject_effort=True, reject_schema=True)
    client = client_for(server, json_mode="json_schema", reasoning_effort="low")
    assert call(client).data == {"a": 7}
    assert len(server["seen"]) == 4  # schema+effort -> object+effort -> schema -> json_object works
    server["seen"].clear()
    assert call(client).data == {"a": 7}
    assert len(server["seen"]) == 1 and server["seen"][0]["body"]["response_format"] == {"type": "json_object"}


def test_auth_failure_is_reported_with_status(server):
    server["behaviour"]["status"] = 401
    with pytest.raises(ProviderError) as info:
        call(client_for(server, json_mode="json_object"))
    assert info.value.status == 401 and len(server["seen"]) == 1


def test_the_gpt_entry_sends_what_openai_expects(server, monkeypatch):
    from dataclasses import replace

    from app.llm.registry import Registry

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    gpt = Registry.load().get("gpt-6.1-sol")
    call(OpenAICompatClient(replace(gpt, base_url=server["url"])))  # its own settings, to a local server
    sent = server["seen"][0]
    assert sent["auth"] == "Bearer openai-secret"
    body = sent["body"]
    assert body["model"] == "gpt-6.1-sol" and body["max_completion_tokens"] == 777 and "max_tokens" not in body
    assert body["reasoning_effort"] == "low" and body["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in body  # OpenAI's reasoning models accept only the default
