"""Thin lifecycle adapter over the pinned official Tau three Gym runtime."""

from __future__ import annotations

import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, NoReturn

from experiments.src.environments.tau_bench.continuation import (
    message_history_digest,
    validate_continuation,
)
from experiments.src.environments.tau_bench.task_identity import (
    TAU_PACKAGE_VERSION,
    validate_task_identity,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_NVIDIA_MODEL = "us/gcp/google/eccn-gemini-2.5-flash-lite"
DEFAULT_NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"


class TauEnvironmentError(RuntimeError):
    """A transient failure in the official Tau environment lifecycle."""

    def __init__(self, operation: str, reason: str) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(f"Tau environment {operation} failed ({reason})")


class TauContinuationError(RuntimeError):
    """A non-retryable mismatch while capturing or restoring Tau state."""

    def __init__(self, operation: str, reason: str) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(f"Tau continuation {operation} failed ({reason})")


@dataclass(frozen=True)
class TauUserConfig:
    """Official user-simulator model configuration without serialized secrets."""

    provider: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    timeout: float = 120.0
    retries: int = 4
    seed: int = 0

    def litellm_config(self) -> tuple[str, dict[str, Any]]:
        """Resolve a provider model and process-environment credential."""

        common = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "num_retries": self.retries,
            "seed": self.seed,
        }
        if self.provider == "nvidia":
            api_key = _required_secret("NVIDIA_INFERENCE_API_KEY")
            base_url = os.environ.get("NVIDIA_INFERENCE_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
            return f"openai/{self.model}", {**common, "api_key": api_key, "api_base": base_url}
        if self.provider == "gemini":
            api_key = _required_secret("GEMINI_API_KEY")
            return f"gemini/{self.model}", {**common, "api_key": api_key}
        raise ValueError(f"unsupported Tau three user provider {self.provider!r}")


@dataclass(frozen=True)
class TauReset:
    """Initial policy-facing state returned by the official environment."""

    system_prompt: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


@dataclass(frozen=True)
class TauStep:
    """One official environment transition."""

    observations: list[dict[str, Any]]
    reward: float
    terminated: bool
    truncated: bool
    reward_info: dict[str, Any]
    simulation_run: dict[str, Any]


def _required_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required in the process environment; Tau Slurm wrappers "
            "load only allowlisted .env values at job startup"
        )
    return value


def verify_tau_runtime() -> None:
    """Reject a runtime that is not the package release paired with task metadata."""

    try:
        installed = importlib.metadata.version("tau2")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Tau three runtime package tau2 is not installed") from error
    if installed != TAU_PACKAGE_VERSION:
        raise RuntimeError(
            f"Tau three runtime version mismatch: installed={installed}, expected={TAU_PACKAGE_VERSION}"
        )


def _official_task(metadata: dict[str, Any]) -> Any:
    # Tau three is optional outside its dedicated runtime image.
    from tau2.runner.helpers import load_tasks

    domain = str(metadata.get("tau_domain") or "")
    split = str(metadata.get("tau_split") or "")
    task_id = str(metadata.get("tau_task_id") or "")
    matches = [task for task in load_tasks(domain, split) if str(task.id) == task_id]
    if len(matches) != 1:
        raise ValueError(f"Tau three metadata resolves to {len(matches)} tasks: {domain}/{split}/{task_id}")
    validate_task_identity(metadata, matches[0])
    return matches[0]


def _system_prompt(policy: str) -> str:
    # Use the official scaffold text rather than maintaining a local copy.
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    return SYSTEM_PROMPT.format(domain_policy=policy, agent_instruction=AGENT_INSTRUCTION)


def _message_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    # The official converter handles User/Assistant/Tool messages. Flatten the
    # official MultiTool container before conversion.
    from tau2.data_model.message import MultiToolMessage
    from tau2.utils.llm_utils import to_litellm_messages

    flattened = []
    for message in messages:
        if isinstance(message, MultiToolMessage):
            flattened.extend(message.tool_messages)
        else:
            flattened.append(message)
    return to_litellm_messages(flattened)


def _external_agent_observations(messages: list[Any]) -> list[Any]:
    """Remove actions echoed by the official GymAgent observation history."""

    # GymAgent exposes its complete agent-side state at every boundary. Usually
    # the just-submitted AssistantMessage is the first item in the delta, but
    # terminal/error finalization can place it after another new message. Every
    # assistant message in this delta was produced by the externally controlled
    # policy, so only user/tool messages are new policy-facing observations.
    return [message for message in messages if getattr(message, "role", None) != "assistant"]


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _raise_environment_error(operation: str, error: Exception) -> NoReturn:
    if isinstance(error, TauEnvironmentError):
        raise error
    raise TauEnvironmentError(operation, type(error).__name__) from error


class TauSession:
    """Own one official AgentGymEnv and its orchestrator thread."""

    def __init__(
        self,
        metadata: dict[str, Any],
        user: TauUserConfig,
        *,
        max_steps: int,
        continuation: dict[str, Any] | None = None,
    ) -> None:
        if continuation is not None:
            raise ValueError("The held-out Tau session does not admit training continuation state")
        verify_tau_runtime()
        task = _official_task(metadata)
        user_model, user_args = user.litellm_config()

        # AgentGymEnv is the public Tau three RL interface. Its observation
        # messages are exposed through the pinned GymAgent lifecycle object.
        from tau2.gym.gym_agent import AgentGymEnv

        self._environment = AgentGymEnv(
            domain=str(metadata["tau_domain"]),
            task_id=str(task.id),
            max_steps=max_steps,
            solo_mode=False,
            user_llm=user_model,
            user_llm_args=user_args,
            all_messages_as_observation=True,
        )
        self._terminated = False
        self._history_length = 0
        self._set_continuation(None)

    def _set_continuation(self, continuation: dict[str, Any] | None) -> None:
        self._continuation = None if continuation is None else validate_continuation(continuation)
        self._step_count_offset = 0 if continuation is None else continuation["orchestrator_step_count"]
        self._error_count_offset = 0 if continuation is None else continuation["orchestrator_num_errors"]

    def remaining_max_steps(self, max_steps: int) -> int:
        """Return the exact post-resume step budget."""

        remaining = max_steps - self._step_count_offset
        if remaining <= 0:
            raise ValueError("Tau inflight continuation exhausted the orchestrator step budget")
        return remaining

    def remaining_max_errors(self, max_errors: int = 10) -> int:
        """Return the exact post-resume tool-error budget."""

        remaining = max_errors - self._error_count_offset
        if remaining <= 0:
            raise ValueError("Tau inflight continuation exhausted the orchestrator error budget")
        return remaining

    def _history(self) -> list[Any]:
        agent = self._environment._agent
        if agent is None:
            raise RuntimeError("Tau three Gym agent is not initialized")
        return list(agent.observation)

    def reset(self) -> TauReset:
        """Start the official simulation and return the exact agent scaffold."""

        try:
            _observation, info = self._environment.reset()
        except Exception as error:
            if getattr(self, "_continuation", None) is not None:
                raise TauContinuationError("restore", type(error).__name__) from error
            _raise_environment_error("reset", error)
        if self._environment._simulation_done.is_set():
            raise TauEnvironmentError(
                "reset",
                "official simulation ended before the first policy turn",
            )
        history = self._history()
        self._validate_restored_continuation()
        self._history_length = len(history)
        tools = [tool.openai_schema for tool in info["tools"]]
        return TauReset(
            system_prompt=_system_prompt(str(info["policy"])),
            messages=_message_dicts(history),
            tools=tools,
        )

    def _validate_restored_continuation(self) -> None:
        continuation = self._continuation
        if continuation is None:
            return
        orchestrator = self._environment._orchestrator
        agent = self._environment._agent
        if orchestrator is None or agent is None or not agent.is_agent_turn:
            raise TauContinuationError(
                "restore",
                "official simulation did not restore at an agent boundary",
            )
        history = orchestrator.get_trajectory()
        if message_history_digest(history) != continuation["message_history_sha256"]:
            raise TauContinuationError(
                "restore",
                "official message history differs after event-log replay",
            )
        environment = orchestrator.environment
        if environment.get_db_hash() != continuation["agent_db_hash"]:
            raise TauContinuationError("restore", "agent DB hash differs after event-log replay")
        if environment.get_user_db_hash() != continuation["user_db_hash"]:
            raise TauContinuationError("restore", "user DB hash differs after event-log replay")

    def snapshot(self) -> dict[str, Any]:
        """Capture a safe official event-log boundary for inflight continuation."""

        orchestrator = self._environment._orchestrator
        agent = self._environment._agent
        if orchestrator is None or agent is None or not agent.is_agent_turn:
            raise TauContinuationError("snapshot", "Tau session is not waiting at an agent boundary")
        if orchestrator.done or self._environment._simulation_done.is_set():
            raise TauContinuationError("snapshot", "Tau session is already terminal")
        history = orchestrator.get_trajectory()
        serialized_history = [message.model_dump(mode="json") for message in history]
        return {
            "message_history": serialized_history,
            "message_history_sha256": message_history_digest(serialized_history),
            "agent_db_hash": orchestrator.environment.get_db_hash(),
            "user_db_hash": orchestrator.environment.get_user_db_hash(),
            "orchestrator_step_count": self._step_count_offset + orchestrator.step_count,
            "orchestrator_num_errors": self._error_count_offset + orchestrator.num_errors,
        }

    def step(self, action: str) -> TauStep:
        """Advance one policy turn and return only new external observations."""

        try:
            _observation, reward, terminated, truncated, info = self._environment.step(action)
        except Exception as error:
            _raise_environment_error("step", error)
        reward_info = _parse_json_object(info.get("reward_info"))
        simulation_run = _parse_json_object(info.get("simulation_run"))
        if (terminated or truncated) and (not reward_info or not simulation_run):
            raise TauEnvironmentError(
                "step",
                "official simulation ended without reward or trajectory data",
            )

        history = self._history()
        new_messages = history[self._history_length :]
        self._history_length = len(history)
        new_messages = _external_agent_observations(new_messages)
        self._terminated = bool(terminated or truncated)
        return TauStep(
            observations=_message_dicts(new_messages),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            reward_info=reward_info,
            simulation_run=simulation_run,
        )

    def close(self) -> None:
        """Finish an abandoned Gym episode so its daemon thread cannot accumulate."""

        if self._terminated or self._environment._simulation_done.is_set():
            self._terminated = True
            return
        try:
            self._environment.step('{"name":"done","arguments":{}}')
        finally:
            self._terminated = True
