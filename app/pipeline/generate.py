"""One reliable "ask a model for structured data" step: validate, retry once with feedback, then fall back."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Generic, TypeVar

from ..llm.base import CostCapError, InvalidOutputError, LLMError, Usage
from ..llm.router import Router
from ..prompts import retry_suffix

T = TypeVar("T")
log = logging.getLogger(__name__)

MIN_FOLLOW_UP_SECONDS = 6.0  # a follow-up (the rewrite pass) is not started with less time than this left


@dataclass(frozen=True)
class Ask(Generic[T]):
    """One request: the prompt, the schema, and how to read the reply (`parse` raises ValueError when unusable)."""
    user: str
    schema: dict
    schema_name: str
    parse: Callable[[dict], T]
    context: dict | None = None
    required: bool = False  # a follow-up that runs even past the deadline (it removes a certification claim)


class GenerationFailed(Exception):
    """Every model in the chain failed. `usage_cost` still records what was spent."""

    def __init__(self, message: str, outcome: "Outcome"):
        super().__init__(message)
        self.outcome = outcome


@dataclass
class Outcome(Generic[T]):
    value: T | None = None
    served_model: str = ""
    final_model_id: str = ""
    chain: list[str] = field(default_factory=list)  # model ids tried, in order
    attempts: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    follow_up: str = ""  # "skipped" or "stopped" when the deadline left no time for the follow-up, else ""

    def add_usage(self, usage: Usage, price_cost: float) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost_usd += price_cost


async def generate_validated(
    router: Router,
    model_ids: list[str],
    *,
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    parse: Callable[[dict], T],
    context: dict | None = None,
    outcome: Outcome | None = None,
    refine: Callable[[T], Ask[T] | None] | None = None,
    deadline: float | None = None,
) -> Outcome[T]:
    """Try each model in turn. Per model: up to 2 attempts, the second one told what was wrong.

    `parse` turns the raw dict into the final object and raises ValueError when it is not usable.
    Provider errors, refusals and truncation skip straight to the next model (no same-model retry).
    `refine` may return a follow-up request for a VALID first answer (style, not correctness), with its own
    prompt, schema and parser (a small "fix only this" answer instead of a full rewrite). It is asked once, and if
    that second answer fails for any reason the first one is kept.
    `deadline` (a time.perf_counter() value) limits only that follow-up: it is not started with less than
    MIN_FOLLOW_UP_SECONDS left, and it is stopped at the deadline; the first answer is kept either way. A follow-up
    marked `required` ignores the deadline.
    """
    out = outcome or Outcome()
    first = Ask(user, schema, schema_name, parse, context)
    for model_id in model_ids:
        out.chain.append(model_id)
        ask = first
        kept: tuple | None = None  # (value, served_model) of a valid first answer under refinement
        time_left: float | None = None  # how long the follow-up may take
        for attempt in (1, 2):
            out.attempts += 1
            started = time.perf_counter()
            try:
                call = router.call(
                    model_id, system=system, user=ask.user, schema=ask.schema, schema_name=ask.schema_name,
                    context=ask.context,
                )
                result, spec = await (asyncio.wait_for(call, time_left) if time_left is not None else call)
            except TimeoutError:  # the follow-up ran past the deadline: the first answer is kept
                log.info("%s %s attempt %d stopped at the time limit after %.1fs", model_id, ask.schema_name,
                         attempt, time.perf_counter() - started)
                out.follow_up = "stopped"
                break
            except CostCapError:
                if kept is not None:
                    break
                raise
            except InvalidOutputError as exc:  # no usable JSON came back: no usage info either
                out.errors.append(f"{model_id}: {exc}")
                if attempt == 1:
                    ask = replace(first, user=user + retry_suffix(str(exc)))
                    out.retries += 1
                    continue
                break
            except LLMError as exc:
                out.errors.append(f"{model_id}: {exc}")
                break
            log.info("%s %s attempt %d: %.1fs, %d tokens in, %d out", model_id, ask.schema_name, attempt,
                     time.perf_counter() - started, result.usage.input_tokens, result.usage.output_tokens)
            out.add_usage(result.usage, spec.cost(result.usage))
            try:
                value = ask.parse(result.data)
            except ValueError as exc:  # includes pydantic.ValidationError
                message = str(exc)[:700]
                out.errors.append(f"{model_id}: invalid structure: {message}")
                if attempt == 1:
                    ask = replace(first, user=user + retry_suffix(message))
                    out.retries += 1
                    continue
                break
            if attempt == 1 and refine is not None:
                follow_up = refine(value)
                if follow_up is not None and not follow_up.required and deadline is not None:
                    time_left = deadline - time.perf_counter()
                    if time_left < MIN_FOLLOW_UP_SECONDS:
                        log.info("%s %s follow-up skipped: %.1fs left of the time limit", model_id,
                                 follow_up.schema_name, time_left)
                        out.follow_up, follow_up = "skipped", None
                if follow_up is not None:
                    kept, ask = (value, result.served_model), follow_up
                    out.retries += 1
                    continue
            out.value, out.served_model, out.final_model_id = value, result.served_model, model_id
            return out
        if kept is not None:  # the refinement failed: the first answer was valid, so use it
            out.value, out.served_model = kept
            out.final_model_id = model_id
            return out
    raise GenerationFailed("; ".join(out.errors) or "No model could be used.", out)
