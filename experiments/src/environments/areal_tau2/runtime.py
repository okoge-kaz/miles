"""AReaL task/DB injection for the pinned official Tau2 Gym lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

from experiments.src.environments.tau_bench.continuation import task_with_continuation
from experiments.src.environments.tau_bench.runtime import TauSession, TauUserConfig, verify_tau_runtime
from experiments.src.environments.tau_bench.task_identity import TAU_COMMIT, TAU_PACKAGE_VERSION
from experiments.src.protocols.areal_tau2 import (
    AREAL_TAU2_DATASET,
    AREAL_TAU2_DB_SHA256,
    AREAL_TAU2_DOMAINS,
    AREAL_TAU2_INTERACTION_MODE,
    AREAL_TAU2_POLICY,
    AREAL_TAU2_REVISION,
    AREAL_TAU2_VERIFIER,
)

DEFAULT_AREAL_TAU2_ROOT = Path(os.environ.get("AREAL_TAU2_ROOT", "/data/areal-tau2-data"))


def canonical_digest(value: Any) -> str:
    """Hash a JSON-compatible value independent of formatting."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_db_path(dataset_root: Path, relative_path: str) -> Path:
    if relative_path not in AREAL_TAU2_DB_SHA256:
        raise ValueError(f"unsupported AReaL Tau2 DB path {relative_path!r}")
    root = dataset_root.resolve(strict=True)
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"AReaL Tau2 DB escapes dataset root: {relative_path!r}")
    return path


def _db_class(domain: str) -> Any:
    if domain == "airline":
        from tau2.domains.airline.data_model import FlightDB

        return FlightDB
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB

        return RetailDB
    if domain == "telecom":
        from tau2.domains.telecom.data_model import TelecomDB

        return TelecomDB
    raise ValueError(f"unsupported AReaL Tau2 domain {domain!r}")


@lru_cache(maxsize=len(AREAL_TAU2_DB_SHA256))
def _verified_db_template(
    domain: str,
    dataset_root: str,
    relative_path: str,
    expected_sha256: str,
) -> Any:
    path = _resolve_db_path(Path(dataset_root), relative_path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"AReaL Tau2 DB digest mismatch for {relative_path}: "
            f"actual={actual_sha256}, expected={expected_sha256}"
        )
    return _db_class(domain).load(path)


def create_domain_environment(
    domain: str,
    *,
    dataset_root: Path,
    db_relative_path: str,
    db_sha256: str,
) -> Any:
    """Create one isolated official domain environment from a verified DB template."""

    template = _verified_db_template(
        domain,
        str(dataset_root.resolve(strict=True)),
        db_relative_path,
        db_sha256,
    )
    database = template.model_copy(deep=True)
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment
    elif domain == "retail":
        from tau2.domains.retail.environment import get_environment
    elif domain == "telecom":
        from tau2.domains.telecom.environment import get_environment
    else:
        raise ValueError(f"unsupported AReaL Tau2 domain {domain!r}")
    return get_environment(db=database, solo_mode=False)


def load_task(task_data: dict[str, Any]) -> Any:
    """Validate one serialized AReaL task against the pinned Tau schema."""

    from tau2.data_model.tasks import RewardType, Task

    task = Task.model_validate(task_data)
    criteria = task.evaluation_criteria
    if criteria is None:
        raise ValueError(f"AReaL Tau2 task {task.id!r} has no evaluation criteria")
    if RewardType.NL_ASSERTION in criteria.reward_basis:
        raise ValueError("AReaL Tau2 RL does not admit LLM-judged NL_ASSERTION reward")
    return task


def _user_message_from_assistant(assistant_message: Any) -> Any:
    """Atomically preserve a Tau user simulator's text or tool calls."""

    from tau2.data_model.message import ToolCall, UserMessage

    tool_calls = None
    if assistant_message.tool_calls:
        tool_calls = [
            ToolCall(
                id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                requestor="user",
            )
            for tool_call in assistant_message.tool_calls
        ]
    return UserMessage(
        role="user",
        content=assistant_message.content,
        tool_calls=tool_calls,
        cost=assistant_message.cost,
        usage=assistant_message.usage,
        raw_data=assistant_message.raw_data,
    )


def _assistant_message_has_payload(assistant_message: Any) -> bool:
    """Return whether a generated user turn has text or tool calls."""

    content = assistant_message.content
    return bool(isinstance(content, str) and content.strip()) or bool(
        assistant_message.tool_calls
    )


def _generate_user_message(
    generate_fn: Any,
    *,
    model: str,
    messages: list[Any],
    tools: list[Any] | None,
    llm_args: dict[str, Any],
) -> Any:
    """Retry semantically empty provider responses before constructing a user turn."""

    empty_retries = int(llm_args.get("num_retries", 0))
    if empty_retries < 0:
        raise ValueError("Tau user simulator retries must be non-negative")
    base_seed = llm_args.get("seed")
    for attempt in range(empty_retries + 1):
        request_args = dict(llm_args)
        if attempt and isinstance(base_seed, int):
            request_args["seed"] = base_seed + attempt
        assistant_message = generate_fn(
            model=model,
            messages=messages,
            tools=tools,
            call_name="user_simulator_response",
            **request_args,
        )
        if _assistant_message_has_payload(assistant_message):
            return _user_message_from_assistant(assistant_message)
        if attempt < empty_retries:
            logger.warning(
                "Tau user simulator returned an empty response; retrying ({}/{})",
                attempt + 1,
                empty_retries,
            )

    return _user_message_from_assistant(assistant_message)


@lru_cache(maxsize=1)
def _user_simulator_type() -> type[Any]:
    """Return the pinned Tau simulator with atomic tool-call construction."""

    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage
    from tau2.user.user_simulator import UserSimulator
    from tau2.utils.llm_utils import generate

    class AReaLTau2UserSimulator(UserSimulator):
        def _generate_next_message(self, message: Any, state: Any) -> Any:
            if isinstance(message, AssistantMessage) and message.is_audio:
                raise ValueError(
                    "Assistant message cannot be audio. Use VoiceUserSimulator instead."
                )
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif isinstance(message, ToolMessage):
                state.messages.append(message)
            elif message.has_content() or message.is_tool_call():
                state.messages.append(message)

            return _generate_user_message(
                generate,
                model=self.llm,
                messages=state.system_messages + state.flip_roles(),
                tools=self.tools,
                llm_args=self.llm_args,
            )

    return AReaLTau2UserSimulator


def compute_expected_state(
    task_data: dict[str, Any],
    *,
    domain: str,
    dataset_root: Path,
    db_relative_path: str,
    db_sha256: str,
) -> tuple[str | None, str | None, list[str]]:
    """Replay the official gold trajectory once during preparation."""

    task = load_task(task_data)
    environment = create_domain_environment(
        domain,
        dataset_root=dataset_root,
        db_relative_path=db_relative_path,
        db_sha256=db_sha256,
    )
    initial = task.initial_state
    environment.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=list(initial.message_history or []) if initial else [],
    )
    replay_errors = []
    for action in task.evaluation_criteria.actions or []:
        try:
            environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
        except Exception as error:
            # Official Tau evaluation warns and continues when a gold action
            # fails. Preserve that terminal-state rule and record the failure.
            replay_errors.append(f"{action.action_id}:{action.name}:{type(error).__name__}")
    return environment.get_db_hash(), environment.get_user_db_hash(), replay_errors


def _validate_metadata(metadata: dict[str, Any]) -> None:
    required_equal = {
        "source": "areal-tau2-rl",
        "verifier": AREAL_TAU2_VERIFIER,
        "interaction_mode": AREAL_TAU2_INTERACTION_MODE,
        "environment_policy": AREAL_TAU2_POLICY,
        "dataset_repo": AREAL_TAU2_DATASET,
        "dataset_revision": AREAL_TAU2_REVISION,
        "tau_package_version": TAU_PACKAGE_VERSION,
        "tau_commit": TAU_COMMIT,
    }
    for key, expected in required_equal.items():
        if metadata.get(key) != expected:
            raise ValueError(f"AReaL Tau2 metadata {key} must equal {expected!r}")
    if metadata.get("stateful_environment") is not True or metadata.get("user_simulator") is not True:
        raise ValueError("AReaL Tau2 requires a stateful environment and user simulator")
    if metadata.get("eval_only") is not False:
        raise ValueError("AReaL Tau2 training generator rejects evaluation-only rows")

    domain = str(metadata.get("tau_domain") or "")
    if domain not in AREAL_TAU2_DOMAINS:
        raise ValueError(f"unsupported AReaL Tau2 domain {domain!r}")
    relative_path = str(metadata.get("tau_db_path") or "")
    expected_db_digest = AREAL_TAU2_DB_SHA256.get(relative_path)
    if expected_db_digest is None or metadata.get("tau_db_sha256") != expected_db_digest:
        raise ValueError(f"AReaL Tau2 DB identity mismatch for {relative_path!r}")
    task_data = metadata.get("tau_task")
    if not isinstance(task_data, dict) or canonical_digest(task_data) != metadata.get("tau_task_sha256"):
        raise ValueError("AReaL Tau2 task digest does not match serialized task metadata")
    for key in ("tau_expected_agent_db_hash", "tau_expected_user_db_hash"):
        if key not in metadata:
            raise ValueError(f"AReaL Tau2 metadata is missing {key}")


def _environment_reward(task: Any, environment: Any, metadata: dict[str, Any]) -> Any:
    from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
    from tau2.data_model.tasks import RewardType

    agent_match = environment.get_db_hash() == metadata["tau_expected_agent_db_hash"]
    user_match = environment.get_user_db_hash() == metadata["tau_expected_user_db_hash"]
    db_reward = float(agent_match and user_match)
    assertion_checks = []
    assertion_reward = 1.0
    for assertion in task.evaluation_criteria.env_assertions or []:
        met = environment.run_env_assertion(assertion, raise_assertion_error=False)
        check = EnvAssertionCheck(env_assertion=assertion, met=met, reward=float(met))
        assertion_checks.append(check)
        assertion_reward *= check.reward

    reward = 1.0
    breakdown = {}
    basis = set(task.evaluation_criteria.reward_basis)
    if RewardType.DB in basis:
        breakdown[RewardType.DB] = db_reward
        reward *= db_reward
    if RewardType.ENV_ASSERTION in basis:
        breakdown[RewardType.ENV_ASSERTION] = assertion_reward
        reward *= assertion_reward
    return RewardInfo(
        reward=reward,
        db_check=DBCheck(db_match=bool(db_reward), db_reward=db_reward),
        env_assertions=assertion_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=breakdown,
    )


def evaluate_terminal_state(simulation: Any, task: Any, environment: Any, metadata: dict[str, Any]) -> Any:
    """Apply official Tau reward components to the live final environment."""

    from tau2.data_model.simulation import RewardInfo, TerminationReason
    from tau2.data_model.tasks import RewardType
    from tau2.environment.toolkit import get_tool_types
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator

    if simulation.termination_reason not in {TerminationReason.AGENT_STOP, TerminationReason.USER_STOP}:
        return RewardInfo(
            reward=0.0,
            reward_basis=None,
            info={"note": f"Simulation terminated prematurely: {simulation.termination_reason.value}"},
        )
    if task.evaluation_criteria is None:
        return RewardInfo(reward=1.0, reward_basis=None, info={"note": "No evaluation criteria"})

    environment_info = _environment_reward(task, environment, metadata)
    tool_types = get_tool_types(environment.tools) if environment.tools is not None else {}
    if environment.user_tools is not None:
        tool_types.update(get_tool_types(environment.user_tools))
    action_info = ActionEvaluator.calculate_reward(task, simulation.messages, tool_types=tool_types)
    communicate_info = CommunicateEvaluator.calculate_reward(task, simulation.messages)

    basis = set(task.evaluation_criteria.reward_basis)
    if RewardType.NL_ASSERTION in basis:
        raise ValueError("AReaL Tau2 terminal reward does not admit NL_ASSERTION")
    reward = 1.0
    breakdown = {}
    for enabled, result in (
        (bool(basis & {RewardType.DB, RewardType.ENV_ASSERTION}), environment_info),
        (RewardType.ACTION in basis, action_info),
        (RewardType.COMMUNICATE in basis, communicate_info),
    ):
        if not enabled:
            continue
        reward *= result.reward
        breakdown.update(result.reward_breakdown or {})
    return RewardInfo(
        reward=reward,
        db_check=environment_info.db_check,
        env_assertions=environment_info.env_assertions,
        action_checks=action_info.action_checks,
        communicate_checks=communicate_info.communicate_checks,
        reward_basis=task.evaluation_criteria.reward_basis,
        reward_breakdown=breakdown,
        info={
            "env": environment_info.info,
            "communicate": communicate_info.info,
            "action": action_info.info,
        },
    )


def _create_gym_environment(
    metadata: dict[str, Any],
    task: Any,
    user_model: str,
    user_args: dict[str, Any],
    max_steps: int,
    max_errors: int,
) -> Any:
    from tau2.gym.gym_agent import AgentGymEnv, GymAgent
    from tau2.orchestrator.orchestrator import Orchestrator

    domain = str(metadata["tau_domain"])
    dataset_root = DEFAULT_AREAL_TAU2_ROOT
    user_simulator_type = _user_simulator_type()

    class AReaLTau2GymEnv(AgentGymEnv):
        def _get_task(self) -> Any:
            return task

        def _get_orchestrator(self) -> Any:
            environment = create_domain_environment(
                domain,
                dataset_root=dataset_root,
                db_relative_path=str(metadata["tau_db_path"]),
                db_sha256=str(metadata["tau_db_sha256"]),
            )
            self._episode_environment = environment
            agent = GymAgent(tools=environment.get_tools(), domain_policy=environment.get_policy())
            try:
                user_tools = environment.get_user_tools(include=task.user_tools) or None
            except ValueError:
                user_tools = None
            user = user_simulator_type(
                tools=user_tools,
                instructions=task.user_scenario,
                llm=self.user_llm,
                llm_args=self.user_llm_args,
            )
            return Orchestrator(
                domain=domain,
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=self.max_steps,
                max_errors=max_errors,
                solo_mode=False,
            )

        def _get_reward(self) -> tuple[float, str]:
            if self._simulation_run is None:
                return 0.0, json.dumps({})
            reward_info = evaluate_terminal_state(
                self._simulation_run,
                task,
                self._episode_environment,
                metadata,
            )
            return reward_info.reward, reward_info.model_dump_json(indent=2)

    return AReaLTau2GymEnv(
        domain=domain,
        task_id=str(metadata["source_row_id"]),
        max_steps=max_steps,
        solo_mode=False,
        user_llm=user_model,
        user_llm_args=user_args,
        all_messages_as_observation=True,
    )


class AReaLTau2Session(TauSession):
    """Own one AReaL task, isolated DB snapshot, and official user simulator."""

    def __init__(
        self,
        metadata: dict[str, Any],
        user: TauUserConfig,
        *,
        max_steps: int,
        continuation: dict[str, Any] | None = None,
    ) -> None:
        verify_tau_runtime()
        _validate_metadata(metadata)
        self._set_continuation(continuation)
        task = load_task(metadata["tau_task"])
        instructions = task.user_scenario.instructions
        if getattr(instructions, "domain", None) != metadata["tau_domain"]:
            raise ValueError("AReaL Tau2 task domain differs from metadata")
        if continuation is not None:
            task = task_with_continuation(task, continuation)
        user_model, user_args = user.litellm_config()
        self._environment = _create_gym_environment(
            metadata=metadata,
            task=task,
            user_model=user_model,
            user_args=user_args,
            max_steps=self.remaining_max_steps(max_steps),
            max_errors=self.remaining_max_errors(),
        )
        self._terminated = False
        self._history_length = 0


__all__ = [
    "AReaLTau2Session",
    "DEFAULT_AREAL_TAU2_ROOT",
    "canonical_digest",
    "compute_expected_state",
    "create_domain_environment",
    "evaluate_terminal_state",
    "file_sha256",
    "load_task",
]
