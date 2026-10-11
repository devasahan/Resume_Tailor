import asyncio
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.llm.anthropic_client import FALLBACK_BETA, AnthropicClient
from app.llm.base import InvalidOutputError, ProviderError, RefusedError, TruncatedError
from app.llm.openai_compat_client import OpenAICompatClient
from app.llm.registry import ModelSpec

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


def run(coro):
    return asyncio.run(coro)


def spec(provider="openai_compat", **settings):
    return ModelSpec(
        id="x", label="X Model", provider=provider, model="m1", base_url="https://api.example.com/v1",
        key_env="TEST_KEY", price_in=1.0, price_out=2.0, settings=settings,
    )


def ok_payload(content='{"a": 1}', finish="stop", model="m1-served"):
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def client_with(handler, **settings):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    c = OpenAICompatClient(spec(**settings), transport=httpx.MockTransport(handler), sleep=fake_sleep)
    c.sleeps = sleeps
    return c


def call(client):
    return run(client.generate_structured(system="sys", user="usr", schema=SCHEMA, schema_name="thing", max_tokens=500))


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-123")


# ------------------------------------------------------------------ OpenAI-compatible adapter
def test_openai_compat_request_shape_and_parsing():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=ok_payload())

    result = call(client_with(handler, json_mode="json_schema", reasoning_effort="low"))
    request = seen[0]
    body = json.loads(request.content)
    assert request.url == "https://api.example.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-123"
    assert body["model"] == "m1" and body["max_tokens"] == 500 and body["reasoning_effort"] == "low"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert result.data == {"a": 1}
    assert (result.usage.input_tokens, result.usage.output_tokens) == (11, 7)
    assert result.served_model == "m1-served"


def test_degrades_when_provider_rejects_features_and_remembers():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body or body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "unsupported"})
        return httpx.Response(200, json=ok_payload())

    client = client_with(handler, json_mode="json_schema", reasoning_effort="low")
    call(client)
    assert len(bodies) == 3  # schema+effort -> schema -> json_object works
    assert bodies[-1]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in bodies[-1]["messages"][0]["content"]  # schema moved into the prompt
    bodies.clear()
    call(client)
    assert len(bodies) == 1  # the working variant is remembered


def test_fenced_json_is_accepted():
    client = client_with(lambda r: httpx.Response(200, json=ok_payload('```json\n{"a": 5}\n```')), json_mode="json_object")
    assert call(client).data == {"a": 5}


def test_truncated_refused_and_garbage_outputs():
    with pytest.raises(TruncatedError):
        call(client_with(lambda r: httpx.Response(200, json=ok_payload(finish="length"))))
    with pytest.raises(RefusedError):
        call(client_with(lambda r: httpx.Response(200, json=ok_payload(finish="content_filter"))))
    with pytest.raises(InvalidOutputError):
        call(client_with(lambda r: httpx.Response(200, json=ok_payload("sorry, no json"))))
    with pytest.raises(InvalidOutputError):
        call(client_with(lambda r: httpx.Response(200, json={"unexpected": True})))


def test_transient_errors_are_retried_then_succeed():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(429, text="slow down") if len(attempts) == 1 else httpx.Response(200, json=ok_payload())

    client = client_with(handler, json_mode="json_object")
    assert call(client).data == {"a": 1}
    assert client.sleeps == [1.5]


def test_auth_error_is_not_retried_and_keeps_status():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(ProviderError) as info:
        call(client_with(handler, json_mode="json_object"))
    assert info.value.status == 401 and len(attempts) == 1


def test_persistent_server_error_raises_after_retries():
    client = client_with(lambda r: httpx.Response(503, text="down"), json_mode="json_object")
    with pytest.raises(ProviderError):
        call(client)
    assert len(client.sleeps) == 2


# ------------------------------------------------------------------ Anthropic adapter
def anthropic_client(responses, calls, **settings):
    queue = list(responses)

    class Messages:
        def __init__(self, kind):
            self.kind = kind

        async def create(self, **kwargs):
            calls.append((self.kind, kwargs))
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    fake = SimpleNamespace(messages=Messages("plain"), beta=SimpleNamespace(messages=Messages("beta")))
    return AnthropicClient(spec("anthropic", **settings), client=fake)


def message(text='{"a": 2}', stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop, stop_details=None, model="claude-sonnet-5-5",
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=40),
    )


def test_anthropic_request_uses_structured_output_and_fallback():
    calls = []
    client = anthropic_client([message()], calls, effort="low", refusal_fallback=True)
    result = call(client)
    kind, kwargs = calls[0]
    assert kind == "beta" and kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert kwargs["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["max_tokens"] == 500 and kwargs["system"] == "sys"
    assert "temperature" not in kwargs and "tool_choice" not in kwargs  # both rejected by Sonnet 5.5
    assert kwargs["messages"] == [{"role": "user", "content": "usr"}]  # no assistant prefill
    assert result.data == {"a": 2} and result.usage.input_tokens == 100 and result.served_model == "claude-sonnet-5-5"


def test_anthropic_plain_endpoint_when_fallback_off():
    calls = []
    call(anthropic_client([message()], calls, refusal_fallback=False))
    assert calls[0][0] == "plain" and "fallbacks" not in calls[0][1]


def test_anthropic_retries_without_fallback_if_api_rejects_it():
    class Rejected(anthropic.BadRequestError):
        def __init__(self):
            Exception.__init__(self, "fallbacks is not supported")

    calls = []
    result = call(anthropic_client([Rejected(), message()], calls, refusal_fallback=True))
    assert [c[0] for c in calls] == ["beta", "plain"] and result.data == {"a": 2}


def test_anthropic_refusal_truncation_and_bad_json():
    with pytest.raises(RefusedError):
        call(anthropic_client([message(stop="refusal")], []))
    with pytest.raises(TruncatedError):
        call(anthropic_client([message(stop="max_tokens")], []))
    with pytest.raises(InvalidOutputError):
        call(anthropic_client([message(text="not json")], []))


def test_token_limit_field_name_is_configurable():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=ok_payload())

    call(client_with(handler, json_mode="json_object", max_tokens_param="max_completion_tokens"))
    assert seen[0]["max_completion_tokens"] == 500 and "max_tokens" not in seen[0]
