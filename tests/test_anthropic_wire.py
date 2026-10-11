"""Point the REAL Anthropic SDK at a local mock of the Messages API.

This checks what the fake-client tests cannot: the SDK accepts the parameters we pass, serialises
them the way the API documents, parses a response, and raises its real exception classes.
"""
import asyncio
import socket
import threading
import time

import anthropic
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.llm.anthropic_client import FALLBACK_BETA, AnthropicClient
from app.llm.base import ProviderError, RefusedError
from app.llm.registry import ModelSpec

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


def message_json(text='{"a": 1}', stop="end_turn", model="claude-sonnet-5-5"):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 34},
    }


@pytest.fixture
def server():
    seen: list[dict] = []
    behaviour = {"beta_status": 200, "text": '{"a": 1}', "stop": "end_turn"}
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        beta_header = request.headers.get("anthropic-beta", "")
        seen.append({"body": body, "beta": beta_header, "key": request.headers.get("x-api-key"), "query": str(request.url.query)})
        if FALLBACK_BETA in beta_header and behaviour["beta_status"] != 200:
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks is not available"}},
                status_code=behaviour["beta_status"],
            )
        return JSONResponse(message_json(behaviour["text"], behaviour["stop"]))

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
    yield {"url": f"http://127.0.0.1:{port}", "seen": seen, "behaviour": behaviour}
    srv.should_exit = True
    thread.join(timeout=5)


def make_client(server, **settings):
    spec = ModelSpec(id="claude-sonnet-5-5", label="Claude", provider="anthropic", model="claude-sonnet-5-5",
                     key_env="X", price_in=2, price_out=10, settings=settings)
    sdk = anthropic.AsyncAnthropic(api_key="test-key", base_url=server["url"], max_retries=0)
    return AnthropicClient(spec, client=sdk)


def call(client):
    return asyncio.run(client.generate_structured(system="sys", user="usr", schema=SCHEMA, schema_name="t", max_tokens=1234))


def test_request_on_the_wire_without_fallback(server):
    result = call(make_client(server, effort="low", refusal_fallback=False))
    sent = server["seen"][0]
    body = sent["body"]
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 1234 and body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "usr"}]
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice", "fallbacks"):
        assert forbidden not in body
    assert sent["key"] == "test-key" and FALLBACK_BETA not in sent["beta"]
    assert result.data == {"a": 1} and (result.usage.input_tokens, result.usage.output_tokens) == (12, 34)
    assert result.served_model == "claude-sonnet-5-5"


def test_request_on_the_wire_with_refusal_fallback(server):
    call(make_client(server, effort="medium", refusal_fallback=True))
    sent = server["seen"][0]
    assert FALLBACK_BETA in sent["beta"] and sent["body"]["fallbacks"] == "default"
    assert sent["body"]["output_config"]["format"]["type"] == "json_schema"  # structured output survives the beta path


def test_real_bad_request_on_fallback_beta_retries_plain_and_succeeds(server):
    server["behaviour"]["beta_status"] = 400
    result = call(make_client(server, refusal_fallback=True))
    assert len(server["seen"]) == 2 and "fallbacks" in server["seen"][0]["body"] and "fallbacks" not in server["seen"][1]["body"]
    assert result.data == {"a": 1}


def test_real_refusal_and_truncation_stop_reasons(server):
    server["behaviour"]["stop"] = "refusal"
    with pytest.raises(RefusedError):
        call(make_client(server))
    server["behaviour"]["stop"] = "max_tokens"
    with pytest.raises(Exception, match="max_output_tokens"):
        call(make_client(server))


def test_real_connection_error_becomes_provider_error():
    spec = ModelSpec(id="c", label="Claude", provider="anthropic", model="claude-sonnet-5-5", key_env="X")
    sdk = anthropic.AsyncAnthropic(api_key="k", base_url="http://127.0.0.1:9", max_retries=0, timeout=2)
    with pytest.raises(ProviderError, match="connection"):
        call(AnthropicClient(spec, client=sdk))
