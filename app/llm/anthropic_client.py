"""Claude via the official Anthropic SDK (never an OpenAI-compatible shim)."""
from __future__ import annotations

import json

import anthropic

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

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient(LLMClient):
    def __init__(self, spec: ModelSpec, client: anthropic.AsyncAnthropic | None = None):
        self.spec = spec
        self._client = client or anthropic.AsyncAnthropic(
            api_key=spec.api_key() or None, max_retries=2, timeout=240.0
        )

    def _params(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        return {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Sonnet 5.5: thinking cannot be disabled and budget_tokens is rejected, so depth is
            # controlled only through effort. Forced tool_choice is rejected too, hence the schema
            # goes through output_config.format instead of a forced tool call.
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": self.spec.settings.get("effort", "medium"),
                "format": {"type": "json_schema", "schema": schema},
            },
        }

    async def _create(self, params: dict, use_fallback: bool):
        if use_fallback:
            # Safety-declined requests are re-run server-side on Anthropic's default fallback model.
            return await self._client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **params
            )
        return await self._client.messages.create(**params)

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
        params = self._params(system, user, schema, max_tokens)
        use_fallback = bool(self.spec.settings.get("refusal_fallback"))
        try:
            try:
                resp = await self._create(params, use_fallback)
            except anthropic.BadRequestError:
                # The optional fallback feature may be what the API objected to (unsupported on this
                # account or platform). A rejected request is never billed, so retry without it: if the
                # request itself is wrong, the plain call raises the real error.
                if not use_fallback:
                    raise
                resp = await self._create(params, False)
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}", exc.status_code) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Anthropic connection error: {exc}") from exc

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None)
            raise RefusedError(f"Claude declined the request{f' ({category})' if category else ''}")
        if resp.stop_reason == "max_tokens":
            raise TruncatedError("Claude hit the max_tokens limit before finishing the JSON (raise max_output_tokens in models.yaml)")

        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = extract_json(text)
        except json.JSONDecodeError as exc:
            raise InvalidOutputError(f"Claude returned text that is not valid JSON: {exc}") from exc
        return LLMResult(
            data=data,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            served_model=resp.model,
            raw_text=text,
        )
