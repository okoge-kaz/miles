"""Fail-closed validation of Harbor's repository-verifier result contract."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_PUBLIC_REPORT_COUNTS = frozenset(
    {
        "actual_count",
        "expected_count",
        "failed_count",
        "mismatched_count",
        "missing_count",
        "passed_count",
        "unexpected_count",
    }
)
_PUBLIC_METRIC_LIMITS = {
    "agent_run_time": (86_400.0, False),
    "agent_setup_time": (86_400.0, False),
    "cost_usd": (1_000_000.0, False),
    "env_setup_time": (86_400.0, False),
    "eval_time": (86_400.0, False),
    "n_cache_tokens": (1_000_000_000.0, True),
    "n_input_tokens": (1_000_000_000.0, True),
    "n_output_tokens": (1_000_000_000.0, True),
    "total_time": (86_400.0, False),
    "total_tool_time": (86_400.0, False),
    "turns": (1_000_000.0, True),
}


def _public_eval_report(value: Mapping[str, Any], reward: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "reward": reward,
        "resolved": reward == 1.0,
    }
    for key in _PUBLIC_REPORT_COUNTS:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            report[key] = item
    return report


def _public_agent_metrics(value: Mapping[str, Any]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for key, (maximum, integer_only) in _PUBLIC_METRIC_LIMITS.items():
        item = value.get(key)
        if (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and 0.0 <= float(item) <= maximum
            and (not integer_only or isinstance(item, int))
        ):
            metrics[key] = item
    return metrics


@dataclass(frozen=True)
class HarborSWEOutcome:
    """A verifier-scored Harbor trial returned to Miles."""

    reward: float
    exit_status: str
    eval_report: dict[str, Any]
    agent_metrics: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HarborSWEOutcome:
        exit_status = str(value.get("exit_status") or "")
        if exit_status != "Submitted":
            raise ValueError(f"Harbor SWE trial is ungraded: exit_status={exit_status!r}")
        reward = value.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("Harbor SWE result has no numeric verifier reward")
        normalized_reward = float(reward)
        if not math.isfinite(normalized_reward) or normalized_reward not in {0.0, 1.0}:
            raise ValueError(f"Harbor SWE verifier reward is not binary: {reward!r}")
        eval_report = value.get("eval_report")
        if not isinstance(eval_report, dict):
            raise ValueError("Harbor SWE result has no eval_report object")
        if not eval_report:
            raise ValueError("Harbor SWE result has no verifier report evidence")
        report_reward = eval_report.get("reward")
        if isinstance(report_reward, bool) or not isinstance(report_reward, (int, float)):
            raise ValueError("Harbor SWE eval_report has no numeric reward")
        if not math.isfinite(float(report_reward)) or float(report_reward) != normalized_reward:
            raise ValueError("Harbor SWE top-level and eval_report rewards disagree")
        agent_metrics = value.get("agent_metrics") or {}
        if not isinstance(agent_metrics, dict):
            raise ValueError("Harbor SWE result agent_metrics must be an object")
        return cls(
            reward=normalized_reward,
            exit_status=exit_status,
            eval_report=_public_eval_report(eval_report, normalized_reward),
            agent_metrics=_public_agent_metrics(agent_metrics),
        )


@dataclass(frozen=True)
class SWEEvaluationTrial:
    instance_id: str
    trial_index: int
    status_code: int
    elapsed_seconds: float
    reward: float | None
    exit_status: str
    eval_report: dict[str, Any]
    agent_metrics: dict[str, Any]
    error: str | None


def summarize_trials(
    results: list[SWEEvaluationTrial],
    *,
    task_count: int,
) -> dict[str, Any]:
    """Aggregate scored trials while keeping infrastructure failures separate."""
    graded = [result for result in results if result.reward is not None]
    by_task: dict[str, list[SWEEvaluationTrial]] = defaultdict(list)
    for result in graded:
        by_task[result.instance_id].append(result)
    return {
        "task": "repository_swe",
        "verifier": "Harbor task-specific executable verifier",
        "tasks": task_count,
        "trials": len(results),
        "graded_trials": len(graded),
        "infrastructure_failures": len(results) - len(graded),
        "mean_reward": sum(result.reward or 0.0 for result in graded) / len(graded) if graded else 0.0,
        "exact_success_rate": (
            sum(result.reward == 1.0 for result in graded) / len(graded) if graded else 0.0
        ),
        "task_any_success_rate": (
            sum(any(result.reward == 1.0 for result in task_results) for task_results in by_task.values())
            / task_count
            if task_count
            else 0.0
        ),
    }
