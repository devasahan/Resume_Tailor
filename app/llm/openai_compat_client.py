"""One adapter for every provider that serves POST {base_url}/chat/completions.

Used for Gemini, Grok, Mistral, DeepSeek, Qwen, GLM, Kimi (see models.yaml). Providers differ in
what they accept (json_schema vs json_object, reasoning_effort), so a request that gets HTTP
400/422 is retried with progressively fewer features until one works; the variant that worked is
remembered for the rest of the process. A rejected API key is not a missing feature: some providers
(Gemini) answer it with HTTP 400, so it is recognised by its message and reported at once.
"""
from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urlsplit

import httpx

from .. import config

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
KEY_ERROR_MARKERS = ("api key not valid", "api_key_invalid", "missing or invalid authorization", "invalid api key",
                     "incorrect api key", "invalid x-api-key")


QUOTA_MARKERS = ("resource_exhausted", "exceeded your current quota", "insufficient_quota")
# Where the owner sees a key's plan, billing and limits, per API host
BILLING = {
    "generativelanguage.googleapis.com": ("Google", "aistudio.google.com"),
    "api.openai.com": ("OpenAI", "platform.openai.com (Settings > Billing)"),
}


def is_quota_error(resp: httpx.Response) -> bool:
    """The key's quota is used up (Google's RESOURCE_EXHAUSTED, OpenAI's insufficient_quota): not a short hiccup, so
    it is not retried. A plain 429 (too many requests at once) still is."""
    return resp.status_code == 429 and any(marker in resp.text.lower() for marker in QUOTA_MARKERS)


def _error_body(resp: httpx.Response) -> dict:
    try:
        payload = resp.json()
    except ValueError:
        return {}
    if isinstance(payload, list):  # Google sometimes wraps it in a list
        payload = payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def error_message(resp: httpx.Response) -> str:
    """The provider's own words from its JSON error ({"error": {"message": ...}}, also in a list), else the start of
    the body: readable, not raw JSON."""
    error = _error_body(resp).get("error")
    message = error.get("message") if isinstance(error, dict) else error if isinstance(error, str) else None
    return " ".join(str(message or resp.text).split())[:300]


def first_sentence(text: str) -> str:
    """The provider's first sentence ("You exceeded your current quota, please check your plan and billing details."),
    at most 160 characters: the rest is links and quota metric names."""
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0][:160]


def retry_after(resp: httpx.Response) -> float | None:
    """Seconds until the quota lets a request through, when the provider says: a Retry-After header ("33"), or
    Google's RetryInfo ("retryDelay": "33s"). Only plain seconds count (never a date, a negative number or "inf")."""
    error = _error_body(resp).get("error")
    details = error.get("details") if isinstance(error, dict) else None
    delays = [resp.headers.get("retry-after", "")]
    delays += [detail.get("retryDelay", "") for detail in (details if isinstance(details, list) else [])
               if isinstance(detail, dict)]
    for delay in delays:
        match = re.fullmatch(r"(\d{1,9}(?:\.\d+)?)s?", str(delay).strip())
        if match:
            return float(match.group(1))
    return None


def is_key_error(resp: httpx.Response) -> bool:
    """401/403 always; a 400 only when its message says the key is missing or invalid (Gemini does this)."""
    if resp.status_code in (401, 403):
        return True
    return resp.status_code == 400 and any(marker in resp.text.lower() for marker in KEY_ERROR_MARKERS)
RETRY_DELAYS = (1.5, 4.0)  # network errors, rate limits (429) and other short hiccups
# "This model is currently experiencing high demand" (503) and other server-side failures usually pass within a
# minute, for paid keys too. Tailoring runs as a background job, so the page keeps waiting ("Still tailoring...").
BUSY = {500, 502, 503, 504}
BUSY_RETRY_DELAYS = (3.0, 8.0, 15.0, 30.0)


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
        """Most capable first. When only the schema format is refused, keep the reasoning setting (dropping it
        would switch some providers to a costlier default amount of thinking). With `keep_effort` the reasoning setting
        is never dropped: a provider that refuses it fails the call (and Run check) instead of thinking at its own,
        slower default."""
        mode = self.spec.settings.get("json_mode", "json_object")
        effort = bool(self.spec.settings.get("reasoning_effort"))
        keep = effort and bool(self.spec.settings.get("keep_effort"))
        ladder = [(mode, effort)]
        if mode == "json_schema" and effort:
            ladder.append(("json_object", True))
        if effort and not keep:
            ladder.append((mode, False))
        if mode == "json_schema":
            ladder.append(("json_object", keep))
        if mode != "none":
            ladder.append(("none", keep))
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
        attempt = 0
        while True:
            try:
                resp = await http.post(url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = ProviderError(f"{self.spec.label}: network error ({type(exc).__name__})")
                delays = RETRY_DELAYS
            else:
                if resp.status_code not in TRANSIENT or is_quota_error(resp):
                    return resp
                last = ProviderError(f"{self.spec.label}: HTTP {resp.status_code}: {error_message(resp)[:200]}",
                                     resp.status_code)
                delays = BUSY_RETRY_DELAYS if resp.status_code in BUSY else RETRY_DELAYS
            if attempt >= len(delays):
                raise last
            await self._sleep(delays[attempt])
            attempt += 1

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
                if is_key_error(resp):
                    break  # no feature to drop: the key itself was refused
                if resp.status_code in (400, 422) and index < len(variants) - 1:
                    continue  # the provider rejected a feature: degrade and retry
                if resp.status_code < 400:
                    self._start = index  # remember only a variant that actually worked
                break
        assert resp is not None
        if resp.status_code >= 400 and is_quota_error(resp):
            provider, where = BILLING.get(urlsplit(self.spec.base_url).hostname or "",
                                          ("the provider", "the provider's site"))
            raise ProviderError(
                f"{self.spec.label}: the API key's quota is used up ({provider}: '{first_sentence(error_message(resp))}'). "
                f"Check the plan and billing at {where}, or wait until it resets.",
                resp.status_code, quota=True, retry_after=retry_after(resp))
        if resp.status_code >= 400:
            hint = ""
            if is_key_error(resp):
                hint = (f" -> The API key was rejected. In use: {config.key_hint(self.spec.key_env) or 'none'}. "
                        "Put a valid key in .env and restart.")
            raise ProviderError(f"{self.spec.label}: HTTP {resp.status_code}: {error_message(resp)}{hint}",
                                resp.status_code)
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
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or 0)
        # Thinking tokens are billed as output. Some providers (Gemini) leave them out of completion_tokens but
        # include them in total_tokens, so the larger of the two readings is the billed output.
        output = max(completion, total - prompt)
        # how much of it was thinking: the provider's own count (OpenAI-style), else what only the total holds (Gemini)
        details = usage.get("completion_tokens_details") or {}
        thinking = int(details.get("reasoning_tokens") or 0) or (max(0, total - prompt - completion) if total else 0)
        return LLMResult(
            data=data,
            usage=Usage(prompt, output, thinking),
            served_model=str(payload.get("model") or self.spec.model),
            raw_text=text,
        )
