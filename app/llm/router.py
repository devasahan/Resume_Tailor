"""Chooses and calls models. The model for each task comes from Settings, read at each call, or once for a whole
tailoring run, cover letter job or tailoring request (keep_settings): a change takes effect from the next one (work
already running finishes with the Settings it started with)."""
from __future__ import annotations

import asyncio
import copy
import functools
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .. import config
from ..storage.db import Database
from .anthropic_client import AnthropicClient
from .base import CostCapError, LLMClient, LLMResult, ProviderError
from .fake_client import FakeClient
from .openai_compat_client import OpenAICompatClient
from .registry import ModelSpec, Registry

# (router, its Settings as read once) inside a keep_settings block: asyncio tasks and asyncio.to_thread copy it
_KEPT: ContextVar[tuple[Router, dict] | None] = ContextVar("kept_settings", default=None)
CHECK_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                "additionalProperties": False}
# A check (Settings > Run check, `python -m app.cli check`) asks a model that answered "busy" once more after this
# long: a passing spike (Google's 503 "high demand") does not show as a broken key or model
CHECK_BUSY_PAUSE = 3.0


def keeps_settings(work):
    """For an async function whose first argument is the Router: the Settings are read once for the whole call (see
    Router.keep_settings)."""

    @functools.wraps(work)
    async def kept(router: Router, *args, **kwargs):
        with router.keep_settings():
            return await work(router, *args, **kwargs)

    return kept


class Router:
    def __init__(self, registry: Registry, db: Database):
        self.registry = registry
        self.db = db
        self._clients: dict[str, LLMClient] = {}
        self.overrides: dict[str, LLMClient] = {}  # tests inject fakes here, keyed by model id

    def forget_clients(self) -> None:
        """An API key changed: the next call builds its client again, with the key as it is now (the Anthropic SDK
        keeps the key it was made with). A call already running finishes with the client it has. A model resting
        because its old key's quota was used up is asked again at once: a new key has its own quota."""
        self._clients.clear()
        from ..pipeline import generate  # here: generate imports this module

        generate._RESTING_UNTIL.clear()

    # ------------------------------------------------------------------ selection
    def settings(self) -> dict:
        """The Settings as saved, or as read at the start of the keep_settings block this runs in."""
        kept = _KEPT.get()
        if kept is not None and kept[0] is self:
            return copy.deepcopy(kept[1])
        return self.db.get_settings()

    @contextmanager
    def keep_settings(self) -> Iterator[None]:
        """Read the Settings once for the block: every model choice and check inside it, and in the tasks and threads
        it starts, uses that reading (a run then reads the database once for them, not before every model call; the
        daily cost cap still reads today's spend at each call). A block inside another keeps the outer reading."""
        kept = _KEPT.get()
        if kept is not None and kept[0] is self:
            yield
            return
        token = _KEPT.set((self, self.db.get_settings()))
        try:
            yield
        finally:
            _KEPT.reset(token)

    def default_model(self, task: str, automatic: bool = False) -> str | None:
        """task: resume | cover_letter | parser. Falls back to the first available model.

        For the parser, None means "read with the built-in rules": when the user chose Rules only, or when
        Automatic would only find an offline demo model (whose reading is the same rules anyway)."""
        s = self.settings()
        chosen = None if automatic else s.get(f"{task}_model")  # automatic: what Automatic would pick
        if task == "parser" and chosen == config.RULES_READER:
            return None
        if chosen:
            try:
                if self.registry.status(self.registry.get(chosen), s["allowed_regions"]).available:
                    return chosen
            except KeyError:
                pass
        if task == "cover_letter":
            return self.default_model("resume", automatic)
        priority = config.PARSER_PRIORITY if task == "parser" else config.DEFAULT_PRIORITY
        picked = self.registry.pick_default(priority, s["allowed_regions"])
        if task == "parser" and picked and self.registry.get(picked).provider == "fake":
            return None
        return picked

    def chain(self, first: str, allow_fallback: bool) -> list[str]:
        ids = [first]
        if allow_fallback:
            s = self.settings()
            for model_id in s.get("fallback_models", []):
                if model_id not in ids:
                    try:
                        if self.registry.status(self.registry.get(model_id), s["allowed_regions"]).available:
                            ids.append(model_id)
                    except KeyError:
                        pass
        return ids

    # ------------------------------------------------------------------ calling
    def _client(self, spec: ModelSpec) -> LLMClient:
        if spec.id in self.overrides:
            return self.overrides[spec.id]
        if spec.id not in self._clients:
            if spec.provider == "fake":
                self._clients[spec.id] = FakeClient(spec)
            elif spec.provider == "anthropic":
                self._clients[spec.id] = AnthropicClient(spec)
            elif spec.provider == "openai_compat":
                self._clients[spec.id] = OpenAICompatClient(spec)
            else:
                raise ProviderError(f"Unknown provider '{spec.provider}' for model {spec.id}")
        return self._clients[spec.id]

    def usable_spec(self, model_id: str) -> ModelSpec:
        try:
            spec = self.registry.get(model_id)
        except KeyError as exc:
            raise ProviderError(str(exc)) from exc
        status = self.registry.status(spec, self.settings()["allowed_regions"])
        if not status.available:
            raise ProviderError(f"{spec.label} is not available: {status.reason}")
        return spec

    def check_cost_cap(self, extra: float = 0.0) -> None:
        """Stop when today's spend reached the daily cap in Settings. `extra`: spent but not saved yet (a draft whose
        run is saved only after its cover letter), so it counts too."""
        cap = float(self.settings().get("daily_cost_cap_usd") or 0)
        if cap > 0 and self.db.cost_today() + extra >= cap:
            raise CostCapError(f"Daily cost cap of ${cap:.2f} reached. Raise it in Settings to continue.")

    async def call(
        self, model_id: str, *, system: str, user: str, schema: dict, schema_name: str, context: dict | None = None
    ) -> tuple[LLMResult, ModelSpec]:
        spec = self.usable_spec(model_id)
        self.check_cost_cap()
        result = await self._client(spec).generate_structured(
            system=system, user=user, schema=schema, schema_name=schema_name,
            max_tokens=spec.max_output_tokens, context=context,
        )
        return result, spec

    async def check(self, model_id: str) -> tuple[LLMResult, ModelSpec]:
        """One tiny real call that verifies the key, the model name and the JSON mode (it costs cents). A "busy" answer
        is asked once more after CHECK_BUSY_PAUSE seconds; one still busy fails with the provider's own words."""
        ask = functools.partial(self.call, model_id, system="Reply with one JSON object.", user='Return {"ok": true}.',
                                schema=CHECK_SCHEMA, schema_name="check")
        try:
            return await ask()
        except ProviderError as exc:
            if not exc.busy:
                raise
        await asyncio.sleep(CHECK_BUSY_PAUSE)
        return await ask()
