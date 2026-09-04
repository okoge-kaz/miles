"""Generate stateful Tau three rollouts with the official Gym environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from experiments.src.environments.common.observations import (
    append_loss_masked_tokens,
    append_tool_observation,
)
from experiments.src.environments.tau_bench.continuation import (
    TAU_CONTINUATION_KEY,
    build_continuation,
    validate_continuation,
)
from experiments.src.environments.tau_bench.runtime import (
    DEFAULT_NVIDIA_MODEL,
    TauEnvironmentError,
    TauSession,
    TauStep,
    TauUserConfig,
)
from experiments.src.environments.tau_bench.task_identity import TAU_VERIFIER
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_ATTEMPT_METADATA_KEYS = (
    "messages",
    "tau_done",
    "tau_infrastructure_error",
    "tau_overhead",
    "tau_policy_action_error",
    "tau_reward_info",
    "tau_simulation_run",
    "tau_turns",
)
_POLICY_ACTION_ERROR_RESPONSE_TAIL_CHARS = 1024


class PolicyActionParseError(ValueError):
    """A model-generated policy action that cannot be parsed safely."""


@dataclass(frozen=True)
class PolicyAction:
    """One parsed policy action in both Tau and chat-history representations."""

    tau_action: str
    message: dict[str, Any]


@dataclass(frozen=True)
class PolicyTurnResponse:
    """One completed SGLang request that has not yet been applied to a Sample."""

    payload: dict[str, Any] | None
    output: dict[str, Any] | None
    halt_reason: str | None
    halt_status: Sample.Status | None
    started_at: float
    finished_at: float


@dataclass(frozen=True)
class TimedTauReset:
    """A Tau reset result and the interval occupied by DB/event-log restore."""

    value: Any
    started_at: float
    finished_at: float

    @property
    def seconds(self) -> float:
        return self.finished_at - self.started_at


@dataclass
class TauOverhead:
    """Optional detailed timing for one Tau environment lifecycle."""

    enabled: bool
    reset_seconds: float = 0.0
    tool_wait_seconds: float = 0.0
    user_simulator_wait_seconds: float = 0.0
    terminal_wait_seconds: float = 0.0
    close_seconds: float = 0.0
    tool_steps: int = 0
    user_simulator_steps: int = 0
    terminal_steps: int = 0
    resume_overlap_attempts: int = 0
    resume_policy_request_seconds: float = 0.0
    resume_db_prefill_overlap_seconds: float = 0.0
    resume_db_restore_unhidden_seconds: float = 0.0

    def record_reset(
        self,
        seconds: float,
        *,
        policy_request_seconds: float | None = None,
        overlap_seconds: float = 0.0,
    ) -> None:
        if self.enabled:
            self.reset_seconds += seconds
            if policy_request_seconds is not None:
                self.resume_overlap_attempts += 1
                self.resume_policy_request_seconds += policy_request_seconds
                self.resume_db_prefill_overlap_seconds += overlap_seconds
                self.resume_db_restore_unhidden_seconds += max(0.0, seconds - overlap_seconds)

    def record_step(self, action: PolicyAction, seconds: float) -> None:
        if not self.enabled:
            return
        tool_calls = action.message.get("tool_calls") or []
        if not tool_calls:
            self.user_simulator_wait_seconds += seconds
            self.user_simulator_steps += 1
            return
        function = tool_calls[0].get("function") or {}
        if function.get("name") == "done":
            self.terminal_wait_seconds += seconds
            self.terminal_steps += 1
            return
        self.tool_wait_seconds += seconds
        self.tool_steps += 1

    def record_close(self, seconds: float) -> None:
        if self.enabled:
            self.close_seconds += seconds

    def summary(self) -> dict[str, float | int]:
        total = (
            self.reset_seconds
            + self.tool_wait_seconds
            + self.user_simulator_wait_seconds
            + self.terminal_wait_seconds
            + self.close_seconds
        )
        summary = {
            "reset_seconds": self.reset_seconds,
            "tool_wait_seconds": self.tool_wait_seconds,
            "user_simulator_wait_seconds": self.user_simulator_wait_seconds,
            "terminal_wait_seconds": self.terminal_wait_seconds,
            "close_seconds": self.close_seconds,
            "environment_total_seconds": total,
            "tool_steps": self.tool_steps,
            "user_simulator_steps": self.user_simulator_steps,
            "terminal_steps": self.terminal_steps,
        }
        if self.resume_overlap_attempts:
            summary.update(
                {
                    "resume_overlap_attempts": self.resume_overlap_attempts,
                    "resume_policy_request_seconds": self.resume_policy_request_seconds,
                    "resume_db_prefill_overlap_seconds": self.resume_db_prefill_overlap_seconds,
                    "resume_db_restore_unhidden_seconds": self.resume_db_restore_unhidden_seconds,
                    "environment_unhidden_seconds": max(
                        0.0,
                        total - self.resume_db_prefill_overlap_seconds,
                    ),
                }
            )
        return summary


def _tokenize_user_observation(tokenizer: Any, content: str) -> list[int]:
    user = {"role": "user", "content": "dummy"}
    assistant = {
        "role": "assistant",
        "content": "acknowledged",
        "reasoning_content": " ",
    }
    messages_without = [user, assistant]
    messages_with = [*messages_without, {"role": "user", "content": content}]
    tokens_without = tokenizer.apply_chat_template(
        messages_without,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    tokens_with = tokenizer.apply_chat_template(
        messages_with,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if tokens_with[: len(tokens_without)] != tokens_without:
        raise RuntimeError("Tau three user observation is not append-only under this chat template")
    return list(tokens_with[len(tokens_without) :])


def append_user_observation(sample: Sample, tokenizer: Any, content: str, max_response_len: int) -> bool:
    """Append a loss-masked user turn and return whether it fit the budget."""

    token_ids = _tokenize_user_observation(tokenizer, content)
    return append_loss_masked_tokens(sample, tokenizer, token_ids, max_response_len)


def _parse_policy_action(parser: Any, response: str, turn: int) -> PolicyAction:
    try:
        normal_text, calls = parser.parse_non_stream(response)
    except Exception as error:
        # Parser implementations consume untrusted model text and do not all
        # validate JSON value types before using them. Normalize ordinary parser
        # exceptions to a failed trajectory instead of killing the rollout worker.
        raise PolicyActionParseError(f"tool-call parser raised {type(error).__name__}: {error}") from error
    if not isinstance(normal_text, str):
        raise PolicyActionParseError(f"tool-call parser returned non-string text: {type(normal_text).__name__}")
    if not isinstance(calls, list):
        raise PolicyActionParseError(f"tool-call parser returned non-list calls: {type(calls).__name__}")
    if not calls:
        content = normal_text.strip()
        if not content:
            raise PolicyActionParseError("Tau three policy produced neither text nor a tool call")
        return PolicyAction(tau_action=content, message={"role": "assistant", "content": content})

    if len(calls) > 1:
        logger.warning("Tau three permits one agent tool call per turn; using the first of %d", len(calls))
    call = calls[0]
    name = getattr(call, "name", None)
    if not isinstance(name, str) or not name:
        raise PolicyActionParseError(f"Tau three tool name must be a non-empty string, got {type(name).__name__}")
    parameters = getattr(call, "parameters", None)
    if parameters is not None and not isinstance(parameters, str):
        raise PolicyActionParseError(
            f"Tau three tool arguments for {name!r} must be JSON text, got {type(parameters).__name__}"
        )
    try:
        arguments = json.loads(parameters or "{}")
    except (json.JSONDecodeError, TypeError) as error:
        raise PolicyActionParseError(f"Tau three tool arguments for {name!r} are invalid JSON: {error}") from error
    if not isinstance(arguments, dict):
        raise PolicyActionParseError(f"Tau three tool arguments for {name!r} are not an object")
    call_id = str(getattr(call, "id", "") or f"tau3-{turn}-{name}")
    tau_call = {
        "id": call_id,
        "name": name,
        "arguments": arguments,
        "requestor": "assistant",
    }
    message = {
        "role": "assistant",
        "content": normal_text.strip() or None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, separators=(",", ":"))},
            }
        ],
    }
    return PolicyAction(tau_action=json.dumps(tau_call, separators=(",", ":")), message=message)


def _user_config(input: GenerateFnInput) -> TauUserConfig:
    args = input.args
    sample_index = input.sample.index if input.sample.index is not None else 0
    return TauUserConfig(
        provider=args.tau_user_provider,
        model=args.tau_user_model,
        max_tokens=args.tau_user_max_tokens,
        temperature=args.tau_user_temperature,
        top_p=args.tau_user_top_p,
        timeout=args.tau_user_request_timeout,
        retries=args.tau_user_max_retries,
        seed=int(args.rollout_seed) + int(sample_index),
    )


def _initialize_sample(
    input: GenerateFnInput,
    sample: Sample,
    reset: Any,
) -> tuple[list[dict[str, Any]], Any]:
    messages = [{"role": "system", "content": reset.system_prompt}, *reset.messages]
    tokenizer = input.state.tokenizer
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=reset.tools,
        return_dict=False,
    )
    sample.prompt = messages
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.reward = None
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.status = Sample.Status.PENDING
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = {
        **metadata,
        "messages": list(messages),
        "tau_user_provider": input.args.tau_user_provider,
        "tau_user_model": input.args.tau_user_model,
    }
    return messages, create_tool_call_parser(reset.tools, input.args.tau_tool_call_parser)


def _append_observations(
    sample: Sample,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    max_response_len: int,
) -> bool:
    for observation in observations:
        role = observation.get("role")
        if role == "user":
            appended = append_user_observation(
                sample,
                tokenizer,
                str(observation.get("content") or ""),
                max_response_len,
            )
        elif role == "tool":
            appended = append_tool_observation(sample, tokenizer, observation, max_response_len)
        else:
            raise RuntimeError(f"unexpected Tau three observation role {role!r}")
        if not appended:
            return False
        messages.append(observation)
    sample.metadata["messages"] = list(messages)
    return True


async def _request_policy_turn(
    input: GenerateFnInput,
    sample: Sample,
    url: str,
) -> PolicyTurnResponse:
    """Issue one policy request without parsing or applying its response."""

    started_at = time.monotonic()
    remaining = input.args.rollout_max_response_len - sample.response_length
    if remaining <= 0:
        return PolicyTurnResponse(
            payload=None,
            output=None,
            halt_reason="budget",
            halt_status=Sample.Status.TRUNCATED,
            started_at=started_at,
            finished_at=time.monotonic(),
        )
    sampling_params = {**input.sampling_params, "max_new_tokens": remaining}
    payload, halt_status = compute_request_payload(input.args, sample.tokens, sampling_params)
    if payload is None:
        return PolicyTurnResponse(
            payload=None,
            output=None,
            halt_reason="context",
            halt_status=halt_status,
            started_at=started_at,
            finished_at=time.monotonic(),
        )
    output = await post(url, payload, headers=compute_routing_headers(input.args, sample))
    return PolicyTurnResponse(
        payload=payload,
        output=output,
        halt_reason=None,
        halt_status=None,
        started_at=started_at,
        finished_at=time.monotonic(),
    )


async def _finish_policy_turn(
    input: GenerateFnInput,
    sample: Sample,
    parser: Any,
    turn: int,
    response: PolicyTurnResponse,
    *,
    response_prefix: str,
) -> tuple[PolicyAction | None, str | None, str]:
    """Apply a completed request only after the Tau environment is ready."""

    if response.halt_reason is not None:
        if response.halt_status is None:
            raise AssertionError("Tau policy preflight halt is missing a Sample status")
        sample.status = response.halt_status
        return None, response.halt_reason, response_prefix
    if response.payload is None or response.output is None:
        raise AssertionError("Tau policy request has neither a halt reason nor an output")
    await update_sample_from_response(
        input.args,
        sample,
        response.payload,
        response.output,
        update_loss_mask=True,
    )
    output = response.output
    response_text = response_prefix + str(output.get("text") or "")
    finish_type = output.get("meta_info", {}).get("finish_reason", {}).get("type")
    if finish_type == "abort":
        sample.reward = None
        return None, "abort", response_text
    if finish_type == "length":
        sample.reward = 0.0
        return None, "length", response_text
    try:
        action = _parse_policy_action(parser, response_text, turn)
    except PolicyActionParseError as error:
        response_tail = response_text[-_POLICY_ACTION_ERROR_RESPONSE_TAIL_CHARS:]
        parser_error = error.__cause__ or error
        sample.metadata["tau_policy_action_error"] = {
            "turn": turn + 1,
            "error_type": type(parser_error).__name__,
            "reason": str(error),
            "response_tail": response_tail,
        }
        logger.warning(
            "Tau three policy action parse failed sample=%s turn=%d: %s; response_tail=%r",
            sample.index,
            turn + 1,
            error,
            response_tail,
        )
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        return None, "action", response_text
    return action, None, response_text


async def _generate_turn(
    input: GenerateFnInput,
    sample: Sample,
    parser: Any,
    url: str,
    turn: int,
    *,
    response_prefix: str = "",
) -> tuple[PolicyAction | None, str | None, str]:
    response = await _request_policy_turn(input, sample, url)
    return await _finish_policy_turn(
        input,
        sample,
        parser,
        turn,
        response,
        response_prefix=response_prefix,
    )


async def _timed_session_reset(session: TauSession) -> TimedTauReset:
    started_at = time.monotonic()
    value = await asyncio.to_thread(session.reset)
    return TimedTauReset(
        value=value,
        started_at=started_at,
        finished_at=time.monotonic(),
    )


def _overlap_seconds(left: TimedTauReset, right: PolicyTurnResponse) -> float:
    return max(
        0.0,
        min(left.finished_at, right.finished_at) - max(left.started_at, right.started_at),
    )


async def _restore_session(
    input: GenerateFnInput,
    sample: Sample,
    session: TauSession,
    continuation: dict[str, Any] | None,
    url: str | None,
) -> tuple[TimedTauReset, PolicyTurnResponse | None, float]:
    """Optionally overlap continuation restore with its first policy request."""

    reset_task = asyncio.create_task(_timed_session_reset(session))
    overlap_enabled = bool(
        continuation is not None
        and getattr(input.args, "tau_overlap_db_restore_with_prefill", False)
        and not getattr(input.state, "aborted", False)
        and continuation["policy_turns"] < input.args.tau_max_turns
    )
    if not overlap_enabled:
        return await reset_task, None, 0.0

    if url is None:
        raise AssertionError("Tau resume overlap requires a policy URL")
    request_task = asyncio.create_task(_request_policy_turn(input, sample, url))
    try:
        reset = await reset_task
    except BaseException:
        request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)
        raise
    response = await request_task
    return reset, response, _overlap_seconds(reset, response)


def _sample_continuation(sample: Sample) -> dict[str, Any] | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    value = metadata.get(TAU_CONTINUATION_KEY)
    if value is None:
        return None
    continuation = validate_continuation(value)
    prefix_tokens = sample.response_length - continuation["turn_start_response_length"]
    if prefix_tokens != continuation["policy_prefix_response_tokens"]:
        raise ValueError(
            "Tau inflight continuation policy-token boundary differs from the saved Sample"
        )
    if prefix_tokens > len(sample.tokens):
        raise ValueError("Tau inflight continuation has more response tokens than Sample tokens")
    return continuation


def _resume_sample(
    input: GenerateFnInput,
    sample: Sample,
    reset: Any,
    continuation: dict[str, Any],
) -> tuple[list[dict[str, Any]], Any]:
    if sample.status not in {Sample.Status.ABORTED, Sample.Status.PENDING}:
        raise ValueError(f"Tau inflight continuation has unsupported sample status {sample.status}")
    messages = [{"role": "system", "content": reset.system_prompt}, *reset.messages]
    sample.reward = None
    sample.metadata["messages"] = list(messages)
    sample.metadata["tau_user_provider"] = input.args.tau_user_provider
    sample.metadata["tau_user_model"] = input.args.tau_user_model
    sample.metadata[TAU_CONTINUATION_KEY] = continuation
    return messages, create_tool_call_parser(reset.tools, input.args.tau_tool_call_parser)


def _record_continuation(
    sample: Sample,
    session: TauSession,
    *,
    policy_turns: int,
    turn_start_response_length: int,
    policy_response_prefix: str,
) -> None:
    prefix_tokens = sample.response_length - turn_start_response_length
    if prefix_tokens < 0:
        raise RuntimeError("Tau policy turn starts after the current response boundary")
    sample.metadata[TAU_CONTINUATION_KEY] = build_continuation(
        session.snapshot(),
        policy_turns=policy_turns,
        turn_start_response_length=turn_start_response_length,
        policy_response_prefix=policy_response_prefix,
        policy_prefix_response_tokens=prefix_tokens,
    )
    sample.status = Sample.Status.ABORTED
    sample.reward = None


def _record_terminal_metadata(sample: Sample, step: TauStep | None, turns: int) -> None:
    sample.metadata["tau_turns"] = turns
    sample.metadata["tau_done"] = bool(step and step.terminated)
    sample.metadata["tau_reward_info"] = None if step is None else step.reward_info
    sample.metadata["tau_simulation_run"] = None if step is None else step.simulation_run


def _record_infrastructure_failure(sample: Sample, error: TauEnvironmentError) -> None:
    sample.status = Sample.Status.ABORTED
    sample.reward = None
    sample.metadata["tau_infrastructure_error"] = {
        "operation": error.operation,
        "reason": error.reason,
    }


def _log_overhead(sample: Sample, overhead: TauOverhead) -> None:
    if not overhead.enabled:
        return
    summary = overhead.summary()
    sample.metadata["tau_overhead"] = summary
    logger.info(
        "Tau overhead sample=%s reset=%.3fs tool=%.3fs user_simulator=%.3fs "
        "terminal=%.3fs close=%.3fs total=%.3fs",
        sample.index,
        summary["reset_seconds"],
        summary["tool_wait_seconds"],
        summary["user_simulator_wait_seconds"],
        summary["terminal_wait_seconds"],
        summary["close_seconds"],
        summary["environment_total_seconds"],
    )
    if summary.get("resume_overlap_attempts", 0):
        logger.info(
            "Tau resume overlap sample=%s attempts=%d policy_request=%.3fs "
            "db_prefill_overlap=%.3fs db_restore_unhidden=%.3fs",
            sample.index,
            summary["resume_overlap_attempts"],
            summary["resume_policy_request_seconds"],
            summary["resume_db_prefill_overlap_seconds"],
            summary["resume_db_restore_unhidden_seconds"],
        )


async def _generate_tau(
    input: GenerateFnInput,
    *,
    session_factory: Callable[..., TauSession] | None = None,
) -> GenerateFnOutput:
    args = input.args
    if args.partial_rollout:
        raise ValueError("Tau three does not support partial rollout")

    sample = deepcopy(input.sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    continuation = _sample_continuation(sample)
    sample.metadata = {key: value for key, value in metadata.items() if key not in _ATTEMPT_METADATA_KEYS}
    if continuation is not None:
        sample.metadata[TAU_CONTINUATION_KEY] = continuation
    else:
        # A recycled prompt without a continuation starts a new environment
        # trajectory. Clear every generated-token field before reset so stale
        # provenance from the failed attempt cannot be attributed to the new
        # response. Inflight continuations deliberately retain these fields.
        sample.reset_for_retry()
    session: TauSession | None = None
    last_step: TauStep | None = None
    turns = 0
    overhead = TauOverhead(enabled=bool(getattr(args, "tau_log_overhead", False)))
    try:
        factory_started = time.monotonic()
        factory = TauSession if session_factory is None else session_factory
        session = factory(
            sample.metadata,
            _user_config(input),
            max_steps=args.tau_max_steps,
            continuation=continuation,
        )
        factory_seconds = time.monotonic() - factory_started
        overlap_configured = bool(
            continuation is not None
            and getattr(args, "tau_overlap_db_restore_with_prefill", False)
        )
        url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
            if overlap_configured
            else None
        )
        reset_result, pending_response, overlap_seconds = await _restore_session(
            input,
            sample,
            session,
            continuation,
            url,
        )
        reset = reset_result.value
        if url is None:
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        if overhead.enabled:
            reset_seconds = factory_seconds + reset_result.seconds
            policy_request_seconds = None
            if pending_response is not None:
                policy_request_seconds = pending_response.finished_at - pending_response.started_at
            sample.non_generation_time += max(0.0, reset_seconds - overlap_seconds)
            overhead.record_reset(
                reset_seconds,
                policy_request_seconds=policy_request_seconds,
                overlap_seconds=overlap_seconds,
            )
        if continuation is None:
            messages, parser = _initialize_sample(input, sample, reset)
            turn_start_response_length = sample.response_length
            response_prefix = ""
        else:
            messages, parser = _resume_sample(input, sample, reset, continuation)
            turns = continuation["policy_turns"]
            turn_start_response_length = continuation["turn_start_response_length"]
            response_prefix = continuation["policy_response_prefix"]

        if input.state.aborted and pending_response is None:
            _record_continuation(
                sample,
                session,
                policy_turns=turns,
                turn_start_response_length=turn_start_response_length,
                policy_response_prefix=response_prefix,
            )

        for turn in range(turns, args.tau_max_turns):
            if (
                sample.status == Sample.Status.ABORTED
                and input.state.aborted
                and pending_response is None
            ):
                break
            if pending_response is None:
                action, halt_reason, response_text = await _generate_turn(
                    input,
                    sample,
                    parser,
                    url,
                    turn,
                    response_prefix=response_prefix,
                )
            else:
                action, halt_reason, response_text = await _finish_policy_turn(
                    input,
                    sample,
                    parser,
                    turn,
                    pending_response,
                    response_prefix=response_prefix,
                )
                pending_response = None
            if halt_reason is not None:
                if halt_reason == "abort":
                    _record_continuation(
                        sample,
                        session,
                        policy_turns=turn,
                        turn_start_response_length=turn_start_response_length,
                        policy_response_prefix=response_text,
                    )
                else:
                    sample.metadata.pop(TAU_CONTINUATION_KEY, None)
                break
            if action is None:
                raise AssertionError("Tau three turn returned neither an action nor a halt reason")
            sample.metadata.pop(TAU_CONTINUATION_KEY, None)
            turns = turn + 1
            messages.append(action.message)
            sample.metadata["messages"] = list(messages)

            transition_started = time.monotonic() if overhead.enabled else None
            try:
                last_step = await asyncio.to_thread(session.step, action.tau_action)
            finally:
                if transition_started is not None:
                    transition_seconds = time.monotonic() - transition_started
                    sample.non_generation_time += transition_seconds
                    overhead.record_step(action, transition_seconds)
            if last_step.terminated or last_step.truncated:
                sample.reward = last_step.reward
                sample.status = Sample.Status.COMPLETED if last_step.terminated else Sample.Status.TRUNCATED
                break
            if not _append_observations(
                sample,
                input.state.tokenizer,
                messages,
                last_step.observations,
                args.rollout_max_response_len,
            ):
                sample.reward = 0.0
                break
            turn_start_response_length = sample.response_length
            response_prefix = ""
            if input.state.aborted:
                _record_continuation(
                    sample,
                    session,
                    policy_turns=turns,
                    turn_start_response_length=turn_start_response_length,
                    policy_response_prefix=response_prefix,
                )
                break

        if sample.status == Sample.Status.ABORTED:
            sample.reward = None
        elif sample.reward is None:
            sample.reward = 0.0
            sample.status = Sample.Status.TRUNCATED
    except TauEnvironmentError as error:
        logger.warning(
            "Tau infrastructure failure during %s (%s); recycling the sample",
            error.operation,
            error.reason,
        )
        _record_infrastructure_failure(sample, error)
    finally:
        if session is not None:
            close_started = time.monotonic() if overhead.enabled else None
            try:
                await asyncio.to_thread(session.close)
            except Exception as error:
                logger.warning("Tau three session cleanup failed: %s", type(error).__name__)
            if close_started is not None:
                close_seconds = time.monotonic() - close_started
                sample.non_generation_time += close_seconds
                overhead.record_close(close_seconds)

    _log_overhead(sample, overhead)
    _record_terminal_metadata(sample, last_step, turns)
    sample.validate()
    return GenerateFnOutput(samples=sample)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Generate one Tau three environment trajectory."""

    metadata = input.sample.metadata if isinstance(input.sample.metadata, dict) else {}
    if metadata.get("verifier") != TAU_VERIFIER:
        raise ValueError(f"Tau three generator rejects verifier {metadata.get('verifier')!r}")
    return await _generate_tau(input)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tau-max-turns", type=int, default=30)
    parser.add_argument("--tau-max-steps", type=int, default=200)
    parser.add_argument("--tau-user-provider", choices=("gemini", "nvidia"), default="nvidia")
    parser.add_argument("--tau-user-model", type=str, default=DEFAULT_NVIDIA_MODEL)
    parser.add_argument("--tau-user-max-tokens", type=int, default=512)
    parser.add_argument("--tau-user-temperature", type=float, default=0.7)
    parser.add_argument("--tau-user-top-p", type=float, default=0.95)
    parser.add_argument("--tau-user-request-timeout", type=float, default=120.0)
    parser.add_argument("--tau-user-max-retries", type=int, default=4)
    parser.add_argument("--tau-tool-call-parser", type=str, default="qwen25")
    parser.add_argument(
        "--tau-overlap-db-restore-with-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "On inflight resume, overlap official DB/event-log restoration with the first "
            "SGLang prefill/decode request. The generated action is parsed and applied only "
            "after message-history and DB-hash validation succeeds."
        ),
    )
    parser.add_argument(
        "--tau-log-overhead",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Log and retain per-episode Tau reset, tool, user-simulator, terminal, "
            "and cleanup timing. Reset includes construction and the initial user request; "
            "user-simulator timing includes official orchestrator overhead. Also populates "
            "Sample.non_generation_time for the shared rollout timing metrics."
        ),
    )


generate.add_arguments = _add_arguments
