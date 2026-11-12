"""One reliable "ask a model for structured data" step: validate, retry once with feedback, then fall back."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Generic, TypeVar

from ..llm.base import CostCapError, InvalidOutputError, LLMError, ProviderError, Usage
from ..llm.router import Router
from ..prompts import retry_suffix

T = TypeVar("T")
log = logging.getLogger(__name__)

MIN_FOLLOW_UP_SECONDS = 6.0  # a follow-up (the rewrite pass) is not started with less time than this left
FOLLOW_UP_SHARE = 0.5  # a follow-up (a fix, not a whole draft) takes at least about this share of the first call
# How long the last follow-up took, per (model, kind of answer), in this process; one that was stopped at the time
# limit counts as at least the time it ran. A follow-up that cannot finish in time is not started at all.
_FOLLOW_UP_SECONDS: dict[tuple[str, str], float] = {}
# A model whose API key's quota is used up rests this long, or less when the provider says when it resets: while
# another model follows it in the chain it is skipped, since its provider refuses every call until then. A model that
# was slow, busy or unreachable is tried first again in the next run: most resumes should come from the chosen model.
QUOTA_REST_SECONDS = 3600.0
_RESTING_UNTIL: dict[str, float] = {}  # model id -> time.monotonic() until which it is skipped


def _rest(model_id: str, why: str, seconds: float) -> None:
    _RESTING_UNTIL[model_id] = time.monotonic() + seconds
    span = f"{seconds:.0f} seconds" if seconds < 120 else f"{seconds / 60:.0f} minutes"
    log.info("%s rests for %s (%s): the next model in the chain is used meanwhile", model_id, span, why)


def resting(model_id: str) -> bool:
    until = _RESTING_UNTIL.get(model_id)
    if until is not None and time.monotonic() >= until:
        _RESTING_UNTIL.pop(model_id, None)
        return False
    return until is not None


def record_sent(out: Outcome, router: Router, model_id: str, system: str, user: str) -> Usage:
    """A call stopped at the time limit had already sent its whole prompt, which the provider bills: count it
    (about 4 characters a token, as the demo models do) in the run's cost, Spent today and the daily cost cap.
    Returns the tokens counted."""
    sent = Usage(input_tokens=(len(system) + len(user)) // 4)
    try:
        cost = router.registry.get(model_id).cost(sent)
    except KeyError:
        cost = 0.0
    out.add_usage(sent, cost)
    return sent


def add_call(out: Outcome, model_id: str, kind: str, seconds: float, usage: Usage | None, result: str) -> None:
    """One AI attempt in the run's timing record: the model, the kind of answer asked for, how long it took, its tokens
    (in, out with thinking, of which thinking) and how it ended: "ok", "stopped" (at the time limit, after its
    prompt was sent), "no answer" (in time), "failed" or "unusable"."""
    usage = usage or Usage()
    out.calls.append({"model": model_id, "kind": kind, "seconds": round(seconds, 2), "input": usage.input_tokens,
                      "output": usage.output_tokens, "thinking": usage.thinking_tokens, "result": result})


def follow_up_seconds(model_id: str, schema_name: str, first_call_seconds: float) -> float:
    """About how long a follow-up will take: at least MIN_FOLLOW_UP_SECONDS, half the first call, and as long as
    the last one for this model took."""
    return max(MIN_FOLLOW_UP_SECONDS, FOLLOW_UP_SHARE * first_call_seconds,
               _FOLLOW_UP_SECONDS.get((model_id, schema_name), 0.0))


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
    first_call_seconds: float = 0.0  # how long the first answer took (a whole new draft takes about as long)
    calls: list[dict] = field(default_factory=list)  # one entry per attempt, see add_call (the admin's timing line)

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
    switch_after: float | None = None,
    last_after: float | None = None,
) -> Outcome[T]:
    """Try each model in turn. Per model: up to 2 attempts, the second one told what was wrong.

    `parse` turns the raw dict into the final object and raises ValueError when it is not usable.
    Provider errors, refusals and truncation skip straight to the next model (no same-model retry).
    `refine` may return a follow-up request for a VALID first answer (style, not correctness), with its own
    prompt, schema and parser (a small "fix only this" answer instead of a full rewrite). It is asked once, and if
    that second answer fails for any reason the first one is kept.
    `deadline` (a time.perf_counter() value) limits only that follow-up: it is not started when it would not finish
    in time (see follow_up_seconds), and it is stopped at the deadline; the first answer is kept either way. A
    follow-up marked `required` ignores the deadline.
    `switch_after` (seconds) stops waiting for a model's answer when another model follows it in the chain, and
    `last_after` for the last model: that bounds a hanging request and the provider's own retry pauses. A model whose
    API key's quota is used up rests (QUOTA_REST_SECONDS): it is skipped while another model follows it (the last
    model is always tried). A slow or busy model never rests: the next run tries it first again.
    """
    out = outcome or Outcome()
    first = Ask(user, schema, schema_name, parse, context)
    for position, model_id in enumerate(model_ids):
        last = position == len(model_ids) - 1
        if not last and resting(model_id):
            log.info("%s %s skipped: its API key's quota is used up for now, so %s is used", model_id, schema_name,
                     model_ids[position + 1])
            continue
        answer_limit = last_after if last else switch_after
        out.chain.append(model_id)
        ask = first
        kept: tuple | None = None  # (value, served_model) of a valid first answer under refinement
        time_left: float | None = None  # how long the follow-up may take
        first_call = 0.0  # seconds the first answer took
        for attempt in (1, 2):
            out.attempts += 1
            started = time.perf_counter()
            try:
                call = router.call(
                    model_id, system=system, user=ask.user, schema=ask.schema, schema_name=ask.schema_name,
                    context=ask.context,
                )
                # a follow-up: the time left before the deadline, else (no time limit, or a required fix) the
                # model's longest wait, so no call is ever waited for without an end
                limit = time_left if kept is not None and time_left is not None else answer_limit
                result, spec = await (asyncio.wait_for(call, limit) if limit is not None else call)
            except TimeoutError:
                ran = time.perf_counter() - started
                sent = record_sent(out, router, model_id, system, ask.user)  # the provider got the whole prompt
                add_call(out, model_id, ask.schema_name, ran, sent, "stopped" if kept is not None else "no answer")
                if kept is not None and time_left is not None:  # past the deadline: the first answer is kept
                    log.info("%s %s attempt %d stopped at the time limit after %.1fs", model_id, ask.schema_name,
                             attempt, ran)
                    key = (model_id, ask.schema_name)
                    _FOLLOW_UP_SECONDS[key] = max(_FOLLOW_UP_SECONDS.get(key, 0.0), ran)  # it needed at least this
                    out.follow_up = "stopped"
                    break
                if kept is not None:  # no answer within the model's longest wait: the first answer is kept
                    log.info("%s %s attempt %d: no answer within %.0fs, the first answer is kept", model_id,
                             ask.schema_name, attempt, ran)
                    break
                out.errors.append(f"{model_id}: did not answer within {limit:.0f} seconds")
                log.info("%s %s attempt %d: no answer within %.0fs%s", model_id, ask.schema_name, attempt, ran,
                         "" if last else f", switching to {model_ids[position + 1]}")
                break
            except CostCapError:  # stopped before the prompt was sent
                add_call(out, model_id, ask.schema_name, time.perf_counter() - started, None, "failed")
                if kept is not None:
                    break
                raise
            except InvalidOutputError as exc:  # no usable JSON came back: no usage info either
                add_call(out, model_id, ask.schema_name, time.perf_counter() - started, None, "unusable")
                out.errors.append(f"{model_id}: {exc}")
                log.info("%s %s attempt %d failed after %.1fs: %s", model_id, ask.schema_name, attempt,
                         time.perf_counter() - started, str(exc)[:200])
                if attempt == 1:
                    ask = replace(first, user=user + retry_suffix(str(exc)))
                    out.retries += 1
                    continue
                break
            except LLMError as exc:
                add_call(out, model_id, ask.schema_name, time.perf_counter() - started, None, "failed")
                out.errors.append(f"{model_id}: {exc}")
                log.info("%s %s attempt %d failed after %.1fs: %s", model_id, ask.schema_name, attempt,
                         time.perf_counter() - started, str(exc)[:200])
                if isinstance(exc, ProviderError) and exc.quota:
                    _rest(model_id, "its API key's quota is used up", QUOTA_REST_SECONDS if exc.retry_after is None
                          else min(exc.retry_after, QUOTA_REST_SECONDS))
                break
            took = time.perf_counter() - started
            thinking = f" (thinking {result.usage.thinking_tokens})" if result.usage.thinking_tokens else ""
            log.info("%s %s attempt %d: %.1fs, %d tokens in, %d out%s", model_id, ask.schema_name, attempt, took,
                     result.usage.input_tokens, result.usage.output_tokens, thinking)
            if kept is not None:  # a follow-up that finished: how long the next one may take
                _FOLLOW_UP_SECONDS[(model_id, ask.schema_name)] = took
            elif attempt == 1:
                first_call = out.first_call_seconds = took
            out.add_usage(result.usage, spec.cost(result.usage))
            try:
                value = ask.parse(result.data)
            except ValueError as exc:  # includes pydantic.ValidationError
                add_call(out, model_id, ask.schema_name, took, result.usage, "unusable")
                message = str(exc)[:700]
                out.errors.append(f"{model_id}: invalid structure: {message}")
                log.info("%s %s attempt %d: unusable answer: %s", model_id, ask.schema_name, attempt, message[:200])
                if attempt == 1:
                    ask = replace(first, user=user + retry_suffix(message))
                    out.retries += 1
                    continue
                break
            add_call(out, model_id, ask.schema_name, took, result.usage, "ok")
            if attempt == 1 and refine is not None:
                follow_up = refine(value)
                if follow_up is not None and not follow_up.required and deadline is not None:
                    time_left = deadline - time.perf_counter()
                    needed = follow_up_seconds(model_id, follow_up.schema_name, first_call)
                    if time_left < needed:
                        log.info("%s %s follow-up skipped: %.1fs left of the time limit, it needs about %.1fs",
                                 model_id, follow_up.schema_name, time_left, needed)
                        # One slow rewrite (a provider hiccup) must not block every later one: when only the
                        # remembered time stood in the way, it shrinks, so a later run tries again.
                        key = (model_id, follow_up.schema_name)
                        remembered = _FOLLOW_UP_SECONDS.get(key)
                        enough = max(MIN_FOLLOW_UP_SECONDS, FOLLOW_UP_SHARE * first_call)
                        if remembered is not None and remembered > time_left >= enough:
                            _FOLLOW_UP_SECONDS[key] = remembered * 0.8
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
