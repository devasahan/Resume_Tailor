"""One reliable "ask a model for structured data" step: validate, retry once with feedback, then fall back."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from ..llm.base import CostCapError, InvalidOutputError, LLMError, Usage
from ..llm.router import Router
from ..prompts import retry_suffix

T = TypeVar("T")


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
    refine: Callable[[T], str | None] | None = None,
) -> Outcome[T]:
    """Try each model in turn. Per model: up to 2 attempts, the second one told what was wrong.

    `parse` turns the raw dict into the final object and raises ValueError when it is not usable.
    Provider errors, refusals and truncation skip straight to the next model (no same-model retry).
    `refine` may return feedback for a VALID first answer (style, not correctness): the model is asked
    once to improve it, and if that second answer fails for any reason the first one is kept.
    """
    out = outcome or Outcome()
    for model_id in model_ids:
        out.chain.append(model_id)
        prompt = user
        kept: tuple | None = None  # (value, served_model) of a valid first answer under refinement
        for attempt in (1, 2):
            out.attempts += 1
            try:
                result, spec = await router.call(
                    model_id, system=system, user=prompt, schema=schema, schema_name=schema_name, context=context
                )
            except CostCapError:
                if kept is not None:
                    break
                raise
            except InvalidOutputError as exc:  # no usable JSON came back: no usage info either
                out.errors.append(f"{model_id}: {exc}")
                if attempt == 1:
                    prompt = user + retry_suffix(str(exc))
                    out.retries += 1
                    continue
                break
            except LLMError as exc:
                out.errors.append(f"{model_id}: {exc}")
                break
            out.add_usage(result.usage, spec.cost(result.usage))
            try:
                value = parse(result.data)
            except ValueError as exc:  # includes pydantic.ValidationError
                message = str(exc)[:700]
                out.errors.append(f"{model_id}: invalid structure: {message}")
                if attempt == 1:
                    prompt = user + retry_suffix(message)
                    out.retries += 1
                    continue
                break
            if attempt == 1 and refine is not None:
                feedback = refine(value)
                if feedback:
                    kept = (value, result.served_model)
                    prompt = user + retry_suffix(feedback)
                    out.retries += 1
                    continue
            out.value, out.served_model, out.final_model_id = value, result.served_model, model_id
            return out
        if kept is not None:  # the refinement failed: the first answer was valid, so use it
            out.value, out.served_model = kept
            out.final_model_id = model_id
            return out
    raise GenerationFailed("; ".join(out.errors) or "No model could be used.", out)
