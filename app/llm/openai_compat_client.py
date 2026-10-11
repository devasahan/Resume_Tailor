"""One adapter for every provider that serves POST {base_url}/chat/completions.

Used for Gemini, Grok, Mistral, DeepSeek, Qwen, GLM, Kimi (see models.yaml). Providers differ in
what they accept (json_schema vs json_object, reasoning_effort), so a request that gets HTTP
400/422 is retried with progressively fewer features until one works; the working variant is
remembered for the rest of the process.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from .base import (
    InvalidOutputError,
    LLMClient,
    LLMResult,
    ProviderError,
    RefusedError,
    TruncatedError,
    Usage,
)
from .registry import ModelSpec
from .schema import extract_json

TRANSIENT = {408, 409, 429, 500, 502, 503, 504}
RETRY_DELAYS = (1.5, 4.0)


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        spec: ModelSpec,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        self.spec = spec
        self._transport = transport
        self._sleep = sleep
        self._start = 0  # index of the first variant worth trying (learned at runtime)

    # ------------------------------------------------------------------ request building
    def _variants(self) -> list[tuple[str, bool]]:
        mode = self.spec.settings.get("json_mode", "json_object")
        effort = bool(self.spec.settings.get("reasoning_effort"))
        ladder = [(mode, effort)]
        if effort:
            ladder.append((mode, False))
        if mode == "json_schema":
            ladder.append(("json_object", False))
        if mode != "none":
            ladder.append(("none", False))
        seen, out = set(), []
        for item in ladder:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _body(self, variant, system, user, schema, schema_name, max_tokens) -> dict:
        kind, with_effort = variant
        if kind != "json_schema":
            system = (
                f"{system}\n\nReturn ONLY one JSON object (no markdown, no commentary) that follows "
                f"this JSON Schema:\n{json.dumps(schema)}"
            )
        body: dict = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.spec.settings.get("max_tokens_param", "max_tokens"): max_tokens,
        }
        if kind == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        elif kind == "json_object":
            body["response_format"] = {"type": "json_object"}
        if with_effort:
            body["reasoning_effort"] = self.spec.settings["reasoning_effort"]
        return body

    # ------------------------------------------------------------------ transport
    async def _post(self, http: httpx.AsyncClient, body: dict) -> httpx.Response:
        url = f"{self.spec.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.spec.api_key()}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                resp = await http.post(url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = ProviderError(f"{self.spec.label}: network error ({type(exc).__name__})")
            else:
                if resp.status_code not in TRANSIENT:
                    return resp
                last = ProviderError(
                    f"{self.spec.label}: HTTP {resp.status_code}: {resp.text[:200]}", resp.status_code
                )
            if attempt < len(RETRY_DELAYS):
                await self._sleep(RETRY_DELAYS[attempt])
        assert last is not None
        raise last

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str,
        max_tokens: int,
        context: dict | None = None,
    ) -> LLMResult:
        variants = self._variants()
        resp: httpx.Response | None = None
        async with httpx.AsyncClient(
            transport=self._transport, timeout=httpx.Timeout(240.0, connect=15.0)
        ) as http:
            for index in range(self._start, len(variants)):
                body = self._body(variants[index], system, user, schema, schema_name, max_tokens)
                resp = await self._post(http, body)
                if resp.status_code in (400, 422) and index < len(variants) - 1:
                    continue  # the provider rejected a feature: degrade and retry
                self._start = index
                break
        assert resp is not None
        if resp.status_code >= 400:
            raise ProviderError(f"{self.spec.label}: HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
        return self._parse(resp)

    # ------------------------------------------------------------------ response parsing
    def _parse(self, resp: httpx.Response) -> LLMResult:
        try:
            payload = resp.json()
            choice = payload["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidOutputError(f"{self.spec.label}: unexpected response shape") from exc
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        if message.get("refusal") or finish == "content_filter":
            raise RefusedError(f"{self.spec.label} declined the request")
        if finish == "length":
            raise TruncatedError(
                f"{self.spec.label} hit the token limit before finishing the JSON "
                "(reasoning tokens count too: raise max_output_tokens or lower reasoning_effort in models.yaml)"
            )
        text = message.get("content") or ""
        if not text.strip():
            raise InvalidOutputError(f"{self.spec.label} returned an empty answer")
        try:
            data = extract_json(text)
        except json.JSONDecodeError as exc:
            raise InvalidOutputError(f"{self.spec.label} returned text that is not valid JSON: {exc}") from exc
        usage = payload.get("usage") or {}
        return LLMResult(
            data=data,
            usage=Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
            served_model=str(payload.get("model") or self.spec.model),
            raw_text=text,
        )
