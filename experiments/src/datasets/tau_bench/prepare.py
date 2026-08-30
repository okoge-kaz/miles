"""Materialize the pinned held-out Tau three evaluation split."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.src.environments.tau_bench.task_identity import (
    TAU_COMMIT,
    TAU_DOMAINS,
    TAU_RELEASE,
    TAU_VERIFIER,
    task_digest,
    task_dict,
)

EXPECTED_COUNTS = {
    ("retail", "train"): 74,
    ("retail", "test"): 40,
    ("retail", "base"): 114,
    ("airline", "train"): 30,
    ("airline", "test"): 20,
    ("airline", "base"): 50,
    ("telecom", "train"): 74,
    ("telecom", "test"): 40,
    ("telecom", "base"): 114,
}

NON_EVAL_OUTPUTS = tuple(
    [
        f"tau3-{domain}-{split}-miles.jsonl"
        for domain in TAU_DOMAINS
        for split in ("train", "base")
    ]
    + [f"tau3-{split}-miles.jsonl" for split in ("train", "base")]
)


def _load_tasks(domain: str, split: str) -> list[Any]:
    # Tau three is an optional, container-pinned runtime dependency.
    from tau2.runner.helpers import load_tasks

    return load_tasks(task_set_name=domain, task_split_name=split)


def load_tau_rows(domain: str, split: str) -> list[dict[str, Any]]:
    """Load one official split and convert it to Miles prompt metadata."""

    tasks = _load_tasks(domain, split)
    expected = EXPECTED_COUNTS[(domain, split)]
    if len(tasks) != expected:
        raise ValueError(
            f"Tau three {domain}/{split} has {len(tasks)} tasks, expected {expected}; "
            "update the release pin and count contract together"
        )
    eval_only = split == "test"
    return [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": f"Run the stateful Tau three {domain} task {task.id}.",
                }
            ],
            "label": "official Tau three terminal reward",
            "metadata": {
                "source": f"tau3-{domain}",
                "verifier": TAU_VERIFIER,
                "tau_domain": domain,
                "tau_split": split,
                "tau_task_id": str(task.id),
                "tau_task_sha256": task_digest(task_dict(task)),
                "tau_release": TAU_RELEASE,
                "tau_commit": TAU_COMMIT,
                "eval_only": eval_only,
                "official_compat_only": split == "base",
            },
        }
        for task in tasks
    ]


def validate_split_contract(rows_by_split: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    """Ensure held-out task IDs are disjoint and base is their exact union."""

    for domain in TAU_DOMAINS:
        ids = {
            split: {
                str(row["metadata"]["tau_task_id"])
                for row in rows_by_split[(domain, split)]
            }
            for split in ("train", "test", "base")
        }
        overlap = ids["train"] & ids["test"]
        if overlap:
            raise ValueError(f"Tau three {domain} train/test overlap: {sorted(overlap)}")
        if ids["base"] != ids["train"] | ids["test"]:
            raise ValueError(f"Tau three {domain} base is not the train/test union")


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _remove_non_eval_outputs(output_dir: Path) -> None:
    """Remove exact Tau train/base artifacts from the evaluation directory."""

    for name in NON_EVAL_OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)


def prepare(output_dir: Path) -> dict[str, Any]:
    """Validate all official splits, but write only held-out test tasks."""

    rows_by_split = {
        (domain, split): load_tau_rows(domain, split)
        for domain in TAU_DOMAINS
        for split in ("train", "test", "base")
    }
    validate_split_contract(rows_by_split)
    _remove_non_eval_outputs(output_dir)

    counts: dict[str, int] = {}
    for domain in TAU_DOMAINS:
        path = output_dir / f"tau3-{domain}-test-miles.jsonl"
        rows = rows_by_split[(domain, "test")]
        counts[path.name] = _write_jsonl(path, rows)
    combined = [row for domain in TAU_DOMAINS for row in rows_by_split[(domain, "test")]]
    path = output_dir / "tau3-test-miles.jsonl"
    counts[path.name] = _write_jsonl(path, combined)

    summary = {
        "tau_release": TAU_RELEASE,
        "tau_commit": TAU_COMMIT,
        "counts": counts,
        "evaluation_policy": {
            "benchmark_split": "test",
            "base_contains_training_tasks": True,
            "train_and_base_materialized": False,
            "training_dataset": "inclusionAI/AReaL-tau2-data",
            "training_split": "tau2_rl_train.jsonl",
            "training_environment": "stateful_multi_turn_user_simulator_environment",
        },
    }
    _write_json(output_dir / "prepare-summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    summary = prepare(parse_args().output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
