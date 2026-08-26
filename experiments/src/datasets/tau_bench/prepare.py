"""Prepare held-out Tau Bench tasks and a Nemotron agentic training blend."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from experiments.src.environments.tau_bench.compat import install_litellm_import_stub
from experiments.src.environments.tau_bench.task_identity import (
    TAU_COMMIT,
    _task_digest,
    _task_dict,
    validate_task_identity,
)

TAU_SPLITS = (("retail", "train"), ("retail", "test"), ("airline", "test"))


def load_tau_rows(env_name: str, split: str, *, eval_only: bool) -> list[dict[str, Any]]:
    """Load one official Tau split through its pinned environment package."""

    install_litellm_import_stub()
    from tau_bench.envs import get_env

    env = get_env(
        env_name=env_name,
        user_strategy="human",
        user_model="offline",
        task_split=split,
        task_index=0,
    )
    rows = []
    for index, task in enumerate(env.tasks):
        task_data = _task_dict(task)
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Run the stateful Tau Bench {env_name} task {index}.",
                    }
                ],
                "label": "official Tau Bench state transition and output checks",
                "metadata": {
                    "source": f"tau-bench-{env_name}",
                    "verifier": "tau_bench_environment",
                    "tau_env": env_name,
                    "tau_split": split,
                    "tau_task_index": index,
                    "tau_task_sha256": _task_digest(task_data),
                    "tau_commit": TAU_COMMIT,
                    "eval_only": eval_only,
                },
            }
        )
    return rows


def _reset_environment(env: Any, task_index: int) -> Any:
    task = env.tasks[task_index]
    env.task_index = task_index
    env.task = task
    env.data = env.data_load_func()
    env.actions = []
    return task


def _official_reward(env: Any, task_index: int, actions: Iterable[Any]) -> float:
    _reset_environment(env, task_index)
    for action in actions:
        env.step(action)
    return float(env.calculate_reward().reward)


def partition_reward_verified_rows(
    rows: Iterable[dict[str, Any]], env: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep tasks whose official reward accepts ground truth but rejects no-op."""

    verified = []
    rejected = []
    for row in rows:
        metadata = row.get("metadata") or {}
        task_index = int(metadata["tau_task_index"])
        task = env.tasks[task_index]
        validate_task_identity(metadata, task)

        no_op_reward = _official_reward(env, task_index, ())
        ground_truth_reward = _official_reward(env, task_index, task.actions)
        reward_audit = {
            "no_op_reward": no_op_reward,
            "ground_truth_reward": ground_truth_reward,
        }
        if ground_truth_reward != 1.0:
            raise ValueError(f"Tau task {task_index} rejects its official ground-truth trajectory")
        if no_op_reward != 0.0:
            rejected.append({"task_index": task_index, **reward_audit})
            continue

        verified_metadata = {
            **metadata,
            "tau_reward_verified": True,
            "tau_reward_audit": reward_audit,
        }
        verified.append({**row, "metadata": verified_metadata})
    return verified, rejected


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def reservoir_sample(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    """Select a reproducible unbiased sample without loading the source file."""

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen = 0
    for seen, row in enumerate(_read_jsonl(path), start=1):
        metadata = row.get("metadata") or {}
        if metadata.get("eval_only") is True:
            raise ValueError(f"training source contains eval_only row: {path}:{seen}")
        if len(selected) < count:
            selected.append(row)
            continue
        replacement = rng.randrange(seen)
        if replacement < count:
            selected[replacement] = row
    if seen < count:
        raise ValueError(f"{path} has {seen} rows; cannot sample {count}")
    return selected


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return count


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    split_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for env_name, split in TAU_SPLITS:
        rows = load_tau_rows(env_name, split, eval_only=split != "train")
        split_rows[(env_name, split)] = rows
        output = output_dir / f"tau1-{env_name}-{split}-miles.jsonl"
        counts[output.name] = _write_jsonl(output, rows)

    tau_train = split_rows[("retail", "train")]
    if len(tau_train) != args.expected_tau_rows:
        raise ValueError(
            f"Tau retail train has {len(tau_train)} rows, expected {args.expected_tau_rows}; "
            "change --expected-tau-rows explicitly if the pinned task set changes"
        )

    install_litellm_import_stub()
    from tau_bench.envs import get_env

    retail_env = get_env(
        env_name="retail",
        user_strategy="human",
        user_model="offline",
        task_split="train",
        task_index=0,
    )
    verified_tau_train, rejected_tau_train = partition_reward_verified_rows(tau_train, retail_env)
    verified_output = output_dir / "tau1-retail-train-reward-verified-miles.jsonl"
    counts[verified_output.name] = _write_jsonl(verified_output, verified_tau_train)

    rows_per_source = len(verified_tau_train)
    if rows_per_source == 0:
        raise ValueError("no Tau retail tasks passed official reward verification")
    conv_rows = reservoir_sample(Path(args.conv_tooluse), rows_per_source, args.seed)
    fncall_rows = reservoir_sample(Path(args.fncall_pivot), rows_per_source, args.seed + 1)
    mixed_rows = [*verified_tau_train, *conv_rows, *fncall_rows]
    random.Random(args.seed + 2).shuffle(mixed_rows)
    mixed_output = output_dir / "nemotron3-agentic-tau-retail-train.jsonl"
    counts[mixed_output.name] = _write_jsonl(mixed_output, mixed_rows)

    summary = {
        "tau_commit": TAU_COMMIT,
        "expected_tau_rows": args.expected_tau_rows,
        "rows_per_source": rows_per_source,
        "seed": args.seed,
        "counts": counts,
        "tau_reward_audit": {
            "accepted": rows_per_source,
            "rejected": len(rejected_tau_train),
            "rejected_tasks": rejected_tau_train,
        },
        "mixed_sources": {
            "tau-bench-retail-reward-verified": rows_per_source,
            "nemotron-conversational-tool-use": len(conv_rows),
            "nemotron-function-calling-pivot": len(fncall_rows),
        },
    }
    summary_path = output_dir / "prepare-summary.json"
    _write_jsonl(summary_path, [summary])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conv-tooluse", required=True)
    parser.add_argument("--fncall-pivot", required=True)
    parser.add_argument("--expected-tau-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    summary = prepare(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
