"""One reliable "ask a model for structured data" step: validate, retry once with feedback, then fall back."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Generic, TypeVar
from urllib.parse import urlsplit

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
# A model that answers "busy" (Google's 503 "This model is currently experiencing high demand", another 500/502/504, a
# plain 429, Anthropic's 529 "overloaded") hands over at once to the next model in the chain. When no model answered in
# a round and some were busy, those are asked again, the chosen model first, after each of these pauses in turn:
# a provider's "busy" usually passes within seconds to a minute, and taking turns finds the first model that is free.
# A busy follow-up (the rewrite pass) is asked again after the same pauses, while its own time allows.
# Busy answers and these pauses may take BUSY_GIVE_UP_SECONDS in all (about a minute and a half, instead of a minute of
# waiting per model): no pause starts that would end past it, and the error then says plainly that the provider is
# busy. A model's whole turn counts when it ends busy (an unusable answer, then "busy"), as it is asked again; time
# spent on other failures (a slow answer that never came) does not count: those models are not asked again.
BUSY_ROUND_PAUSES = (2.0, 5.0, 10.0, 15.0, 20.0, 25.0)
BUSY_GIVE_UP_SECONDS = 100.0
# The pause between rounds and the clock that measures BUSY_GIVE_UP_SECONDS (tests replace both)
_sleep = asyncio.sleep
_clock = time.monotonic
# Who runs a model's API, by its host (Anthropic's client needs no base_url): named when every model was busy, so the
# admin knows whose side the problem is on
COMPANIES = {
    "generativelanguage.googleapis.com": "Google", "api.openai.com": "OpenAI", "api.x.ai": "xAI",
    "api.mistral.ai": "Mistral", "api.deepseek.com": "DeepSeek", "api.z.ai": "Z.ai",
    "dashscope-intl.aliyuncs.com": "Alibaba Cloud", "api.moonshot.ai": "Moonshot AI",
}


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
    # The latest attempt's model and when it started (time.perf_counter()): while it has no entry in `calls` yet, that
    # call is on its way (see on_its_way)
    asking: str = ""
    asking_since: float = 0.0
    # The latest "busy" answer, as in `errors` ("model: error"): what a step stopped during a pause between rounds was
    # waiting out (a model asked after it in that round may have failed another way)
    busy_error: str = ""

    def add_usage(self, usage: Usage, price_cost: float) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cost_usd += price_cost

    def on_its_way(self) -> bool:
        """A call was started and has not ended: every attempt that ends gets its entry in `calls` (add_call), so a
        step stopped from outside during one has one attempt more than entries. Between calls (a pause between rounds
        of busy answers) nothing is on its way."""
        return self.attempts > len(self.calls)


def company(router: Router, model_id: str) -> str:
    """The company whose API serves this model ("Google"), or "" when it is not known."""
    try:
        spec = router.registry.get(model_id)
    except KeyError:
        return ""
    if spec.provider == "anthropic":
        return "Anthropic"
    return COMPANIES.get(urlsplit(spec.base_url).hostname or "", "")


def model_label(router: Router, model_id: str) -> str:
    try:
        return router.registry.get(model_id).label
    except KeyError:
        return model_id


def _in_words(seconds: float) -> str:
    """'45 s', '1 min 32 s', '2 min'."""
    whole = max(0, round(seconds))
    minutes, rest = divmod(whole, 60)
    if not minutes:
        return f"{rest} s"
    return f"{minutes} min {rest} s" if rest else f"{minutes} min"


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def busy_summary(router: Router, busy: dict[str, ProviderError], asked: dict[str, int], seconds: float,
                 chain: list[str] | None = None, rested: list[str] | None = None) -> str:
    """Every model asked was busy to the end, in plain words for the admin: 'Google's AI service is overloaded right
    now: Gemini 3.8 Flash and Gemini 3.5 Flash-Lite answered "busy" (HTTP 503) every time they were asked (5 times
    each in 1 min 32 s). This is on Google's side, not a problem in the app: try again in a few minutes. A backup
    model from another company (Settings > Fallback models) keeps the app working when Google is busy.' A plain 429
    (too many requests) may be the rate limit of the app's API key at that company or an overload (some providers
    answer 429 when they are overloaded): both are named. `rested`: models of the `chain` that were not asked because
    their API key's quota is used up for now; the backup advice is given only when every model of the chain is from
    the busy company."""
    owners = {company(router, model_id) for model_id in busy}
    owner = next(iter(owners)) if len(owners) == 1 else ""
    if owner:
        who, side = f"{owner}'s AI service is", f"{owner}'s side"
    elif len(owners) > 1:
        who, side = "The AI services are", "the AI companies' side"
    else:
        who, side = "The AI service is", "the AI company's side"
    statuses = sorted({error.status for error in busy.values() if error.status})
    http = f" (HTTP {_and([str(status) for status in statuses])})" if statuses else ""
    counts = {asked[model_id] for model_id in busy}
    times, many = max(counts), len(busy) > 1
    how_often = "" if times == 1 else (
        f" every time {'they were' if many else 'it was'} asked ({'' if len(counts) == 1 else 'up to '}{times} times"
        f"{' each' if many and len(counts) == 1 else ''} in {_in_words(seconds)})")
    names = _and([model_label(router, model_id) for model_id in busy])
    both = " (an overload, or the rate limit of the app's API key)"  # a 429 can be either: the app cannot tell
    state, cause = ("overloaded or limiting this app's requests", both) if 429 in statuses else ("overloaded", "")
    said = "too many requests" if statuses == [429] else "busy"
    text = (f"{who} {state} right now: {names} answered \"{said}\"{http}{how_often}. This is on {side}{cause}, not a "
            "problem in the app: try again in a few minutes.")
    rested = [model_id for model_id in rested or [] if model_id not in busy]
    if rested:
        one = len(rested) == 1
        whose = "its API key's quota is" if one else "their API keys' quotas are"
        text += (f" {_and([model_label(router, model_id) for model_id in rested])} {'was' if one else 'were'} not "
                 f"asked, because {whose} used up for now.")
    if len(owners) == 1 and all(company(router, model_id) == owner for model_id in chain or busy):
        # every model of the chain from one company: a backup from another one would have answered
        text += (" A backup model from another company (Settings > Fallback models) keeps the app working when "
                 f"{owner or 'that company'} is busy.")
    return text


def failure_text(router: Router, errors: list[str], asked: dict[str, int], busy: dict[str, ProviderError | None],
                 seconds: float, chain: list[str] | None = None, rested: list[str] | None = None) -> str:
    """What the admin reads when no model answered (users see only "busy, try again", app/main.py). With one round,
    every failure in order, as each model's attempts went. After more rounds, each model's last error, with how often
    it was asked. When every model asked was busy to the end, a plain summary first (busy_summary: `chain` is every
    model of the step, `rested` the ones skipped because their quota is used up), then each model's last answer."""
    if not errors:
        return "No model could be used."
    last: dict[str, str] = {}
    for error in errors:
        model_id = next((model_id for model_id in asked if error.startswith(f"{model_id}: ")), None)
        if model_id is not None:
            last[model_id] = error
    ended_busy = {model_id: error for model_id, error in busy.items() if error is not None}
    if asked and len(ended_busy) == len(asked):
        summary = busy_summary(router, ended_busy, asked, seconds, chain, rested)
        return f"{summary} Last answers: {'; '.join(last.values())}"
    if max(asked.values(), default=1) <= 1:
        return "; ".join(errors)
    return "; ".join(f"{error} (asked {asked[model_id]} times)" if asked[model_id] > 1 else error
                     for model_id, error in last.items())


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
    busy_give_up: float | None = None,
) -> Outcome[T]:
    """Try each model in turn. Per model: up to 2 attempts, the second one told what was wrong.

    `parse` turns the raw dict into the final object and raises ValueError when it is not usable.
    Provider errors, refusals and truncation skip straight to the next model (no same-model retry).
    `refine` may return a follow-up request for a VALID first answer (style, not correctness), with its own
    prompt, schema and parser (a small "fix only this" answer instead of a full rewrite). It is asked once (a "busy"
    answer is asked again, see below), and if that second answer fails the first one is kept.
    `deadline` (a time.perf_counter() value) limits only that follow-up: it is not started when it would not finish
    in time (see follow_up_seconds), and it is stopped at the deadline; the first answer is kept either way. A
    follow-up marked `required` ignores the deadline.
    `switch_after` (seconds) stops waiting for a model's answer when another model follows it in the round, and
    `last_after` for the last model: that bounds a hanging request. A model that answers "busy" (ProviderError.busy:
    Google's 503 "high demand" and the like) hands over to the next model at once. When a whole round answered
    nothing and some models were busy, only those are asked again, in chain order (the chosen model first), after
    the next of BUSY_ROUND_PAUSES. A busy follow-up is asked again after the same pauses while there is time for it
    before the deadline, checked again when the pause ends (a required one, or one with no deadline: as long as the
    busy answers allow). Busy answers (a model's whole turn, when it ends busy) and these pauses may take
    `busy_give_up` seconds in all (BUSY_GIVE_UP_SECONDS when not given): no pause starts that would end past it, and
    the error then starts with a plain summary (failure_text). A model whose API key's quota
    is used up rests (QUOTA_REST_SECONDS): it is skipped while another model follows it (the last model is always
    tried). A slow or busy model never rests: the next run tries it first again.
    """
    out = outcome or Outcome()
    first = Ask(user, schema, schema_name, parse, context)
    began, errors_before = _clock(), len(out.errors)
    give_up = BUSY_GIVE_UP_SECONDS if busy_give_up is None else busy_give_up
    busy_spent = 0.0  # seconds of turns that ended busy and of the pauses after them, held against give_up
    asked: dict[str, int] = {}  # model id -> how many rounds asked it for a first answer
    busy: dict[str, ProviderError | None] = {}  # model id -> its last error, when that error said "busy"
    rested: list[str] = []  # models skipped because their API key's quota is used up for now
    todo = list(model_ids)
    for round_no, pause in enumerate((0.0, *BUSY_ROUND_PAUSES)):
        if round_no:  # no model answered in the last round: ask again only the ones that were busy
            todo = [model_id for model_id in todo if busy.get(model_id) is not None]
            if not todo:
                break
            if busy_spent + pause > give_up:
                log.info("%s: still busy after %.0fs of busy answers and pauses (%s): no more rounds", schema_name,
                         busy_spent, ", ".join(todo))
                break
            log.info("%s: every model asked was busy (%s), asking again in %gs", schema_name, ", ".join(todo),
                     pause)
            await _sleep(pause)
            busy_spent += pause
        for position, model_id in enumerate(todo):
            last = position == len(todo) - 1
            if not last and resting(model_id):
                log.info("%s %s skipped: its API key's quota is used up for now, so %s is used", model_id,
                         schema_name, todo[position + 1])
                busy[model_id] = None  # resting since the last round (another call used up the quota): not again
                if model_id not in rested:
                    rested.append(model_id)
                continue
            answer_limit = last_after if last else switch_after
            if model_id not in out.chain:
                out.chain.append(model_id)
            asked[model_id] = asked.get(model_id, 0) + 1
            busy[model_id] = None
            ask = first
            kept: tuple | None = None  # (value, served_model) of a valid first answer under refinement
            time_left: float | None = None  # how long the follow-up may take
            first_call = 0.0  # seconds the first answer took
            follow_up_pauses = iter(BUSY_ROUND_PAUSES)  # before a busy follow-up is asked again
            turn_began = _clock()  # a turn that ends busy counts in full: the model is asked again
            attempt = 0
            while attempt < 2:
                attempt += 1
                out.attempts += 1
                started, clocked = time.perf_counter(), _clock()
                out.asking, out.asking_since = model_id, started
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
                             "" if last else f", switching to {todo[position + 1]}")
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
                    if isinstance(exc, ProviderError) and exc.busy:
                        # the follow-up's own try, else the whole turn (an unusable first answer, then "busy")
                        busy_spent += _clock() - (clocked if kept is not None else turn_began)
                        out.busy_error = out.errors[-1]
                        if kept is None:
                            busy[model_id] = exc  # asked again in the next round, if there is one
                        else:  # the follow-up: asked again after a pause while its own time allows
                            pause = next(follow_up_pauses, None)
                            # under a deadline (time_left is set) only when the next try can still finish before it
                            fits = time_left is None or deadline is None or (
                                deadline - time.perf_counter()
                                >= (pause or 0.0) + follow_up_seconds(model_id, ask.schema_name, first_call))
                            if pause is not None and busy_spent + pause <= give_up and fits:
                                log.info("%s %s follow-up was busy, asking again in %gs", model_id, ask.schema_name,
                                         pause)
                                await _sleep(pause)
                                busy_spent += pause
                                if time_left is not None and deadline is not None:
                                    time_left = deadline - time.perf_counter()
                                    needed = follow_up_seconds(model_id, ask.schema_name, first_call)
                                    if time_left < needed:  # the pause ended late (the event loop stood still)
                                        log.info("%s %s follow-up not asked again: %.1fs left of the time limit, it "
                                                 "needs about %.1fs", model_id, ask.schema_name, time_left, needed)
                                        out.follow_up = "stopped"  # the first answer is kept, nothing more is sent
                                        break
                                attempt -= 1  # the same follow-up again: still the second attempt
                                continue
                            log.info("%s %s follow-up still busy: the first answer is kept", model_id,
                                     ask.schema_name)
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
    raise GenerationFailed(failure_text(router, out.errors[errors_before:], asked, busy, _clock() - began,
                                        list(model_ids), rested), out)
