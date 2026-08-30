"""Prepare the pinned AReaL Tau2 RL tasks with terminal-state targets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from experiments.src.datasets.common.io import read_rows
from experiments.src.environments.areal_tau2.runtime import (
    canonical_digest,
    compute_expected_state,
    file_sha256,
    load_task,
)
from experiments.src.environments.tau_bench.task_identity import TAU_COMMIT, TAU_PACKAGE_VERSION
from experiments.src.protocols.areal_tau2 import (
    AREAL_TAU2_DATASET,
    AREAL_TAU2_DB_SHA256,
    AREAL_TAU2_EXPECTED_COUNTS,
    AREAL_TAU2_EXPECTED_ROWS,
    AREAL_TAU2_INTERACTION_MODE,
    AREAL_TAU2_POLICY,
    AREAL_TAU2_RAW_SHA256,
    AREAL_TAU2_REVISION,
    AREAL_TAU2_VERIFIER,
)

StateResolver = Callable[..., tuple[str | None, str | None, list[str]]]
_TASK_FIELDS = ("id", "description", "user_scenario", "ticket", "initial_state", "evaluation_criteria")


def _task_data(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    value = {key: row[key] for key in _TASK_FIELDS if key in row}
    criteria = value.get("evaluation_criteria")
    if isinstance(criteria, str):
        value["evaluation_criteria"] = json.loads(criteria)
    task = load_task(value)
    instructions = task.user_scenario.instructions
    domain = str(getattr(instructions, "domain", ""))
    if domain not in AREAL_TAU2_EXPECTED_COUNTS:
        raise ValueError(f"AReaL Tau2 task {task.id!r} has unsupported domain {domain!r}")
    return task.model_dump(mode="json"), domain


def adapt_areal_tau2(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_root: Path,
    state_resolver: StateResolver = compute_expected_state,
) -> dict[str, Any]:
    """Convert one complete AReaL task without treating its non-unique ID as identity."""

    task_data, domain = _task_data(row)
    db_relative_path = str(row.get("db_path") or "")
    try:
        db_sha256 = AREAL_TAU2_DB_SHA256[db_relative_path]
    except KeyError as error:
        raise ValueError(f"AReaL Tau2 row {row_index} has unknown DB {db_relative_path!r}") from error
    expected_agent_hash, expected_user_hash, replay_errors = state_resolver(
        task_data,
        domain=domain,
        dataset_root=dataset_root,
        db_relative_path=db_relative_path,
        db_sha256=db_sha256,
    )
    source_task_id = str(task_data["id"])
    source_row_id = f"{domain}:{row_index:04d}:{source_task_id}"
    criteria = task_data["evaluation_criteria"]
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"Run AReaL Tau2 training task {source_row_id} with a user simulator.",
            }
        ],
        "label": json.dumps(
            {"terminal_reward_basis": criteria["reward_basis"]},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "metadata": {
            "source": "areal-tau2-rl",
            "verifier": AREAL_TAU2_VERIFIER,
            "interaction_mode": AREAL_TAU2_INTERACTION_MODE,
            "environment_policy": AREAL_TAU2_POLICY,
            "stateful_environment": True,
            "user_simulator": True,
            "eval_only": False,
            "dataset_repo": AREAL_TAU2_DATASET,
            "dataset_revision": AREAL_TAU2_REVISION,
            "source_row_index": row_index,
            "source_row_id": source_row_id,
            "source_task_id": source_task_id,
            "source_row_sha256": canonical_digest(row),
            "source_pattern_id": row.get("pattern_id"),
            "source_original_id": row.get("original_id"),
            "source_reward": row.get("reward"),
            "source_op": row.get("op"),
            "source_correct": row.get("correct"),
            "tau_domain": domain,
            "tau_task": task_data,
            "tau_task_sha256": canonical_digest(task_data),
            "tau_db_path": db_relative_path,
            "tau_db_sha256": db_sha256,
            "tau_expected_agent_db_hash": expected_agent_hash,
            "tau_expected_user_db_hash": expected_user_hash,
            "tau_gold_replay_errors": replay_errors,
            "tau_package_version": TAU_PACKAGE_VERSION,
            "tau_commit": TAU_COMMIT,
        },
    }


def _adapt_worker(item: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
    row_index, row, dataset_root = item
    return adapt_areal_tau2(row, row_index=row_index, dataset_root=Path(dataset_root))


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return count


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def validate_source_assets(dataset_root: Path, input_path: Path) -> None:
    """Reject any source file or DB snapshot outside the pinned dataset revision."""

    root = dataset_root.resolve(strict=True)
    source = input_path.resolve(strict=True)
    if source != root / "tau2_rl_train.jsonl":
        raise ValueError(f"AReaL Tau2 input must be {root / 'tau2_rl_train.jsonl'}")
    actual_source_digest = file_sha256(source)
    if actual_source_digest != AREAL_TAU2_RAW_SHA256:
        raise ValueError(
            f"AReaL Tau2 raw task digest mismatch: actual={actual_source_digest}, "
            f"expected={AREAL_TAU2_RAW_SHA256}"
        )
    for relative_path, expected_digest in AREAL_TAU2_DB_SHA256.items():
        path = (root / relative_path).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"AReaL Tau2 DB escapes dataset root: {relative_path}")
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            raise ValueError(f"AReaL Tau2 DB digest mismatch for {relative_path}")


def _schedule_summary(rows: int) -> dict[str, Any]:
    schedules = {}
    for rollout_batch_size in (16, 32, 63, 64, 192):
        schedules[str(rollout_batch_size)] = {
            str(epochs): (rows * epochs + rollout_batch_size - 1) // rollout_batch_size
            for epochs in (5, 6)
        }
    return {
        "definition": "optimizer updates = ceil(rows * epochs / rollout_batch_size)",
        "updates_by_rollout_batch_size_and_epoch": schedules,
    }


def _summarize_output(path: Path) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    replay_error_tasks = 0
    replay_errors = 0
    for row in read_rows([path]):
        metadata = row["metadata"]
        counts[str(metadata["tau_domain"])] += 1
        errors = metadata["tau_gold_replay_errors"]
        replay_error_tasks += int(bool(errors))
        replay_errors += len(errors)
    return counts, replay_error_tasks, replay_errors


def prepare(
    dataset_root: Path,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Validate, normalize, and materialize all 1,982 RL tasks in source order."""

    if workers < 1:
        raise ValueError("workers must be positive")
    validate_source_assets(dataset_root, input_path)
    source_rows = list(read_rows([input_path]))
    if len(source_rows) != AREAL_TAU2_EXPECTED_ROWS:
        raise ValueError(
            f"AReaL Tau2 has {len(source_rows)} RL rows, expected {AREAL_TAU2_EXPECTED_ROWS}"
        )
    work = [(index, row, str(dataset_root)) for index, row in enumerate(source_rows, start=1)]
    if workers == 1:
        converted: Iterable[dict[str, Any]] = map(_adapt_worker, work)
        count = _atomic_write_jsonl(output_path, converted)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            count = _atomic_write_jsonl(output_path, executor.map(_adapt_worker, work, chunksize=4))

    domain_counts, replay_error_tasks, replay_errors = _summarize_output(output_path)
    if dict(domain_counts) != AREAL_TAU2_EXPECTED_COUNTS:
        raise ValueError(f"AReaL Tau2 domain counts differ from contract: {dict(domain_counts)}")
    summary = {
        "dataset_repo": AREAL_TAU2_DATASET,
        "dataset_revision": AREAL_TAU2_REVISION,
        "raw_sha256": AREAL_TAU2_RAW_SHA256,
        "db_sha256": AREAL_TAU2_DB_SHA256,
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "rows": count,
        "domain_counts": dict(domain_counts),
        "verifier": AREAL_TAU2_VERIFIER,
        "interaction_mode": AREAL_TAU2_INTERACTION_MODE,
        "environment_policy": AREAL_TAU2_POLICY,
        "stateful_environment": True,
        "user_simulator": True,
        "tau_package_version": TAU_PACKAGE_VERSION,
        "tau_commit": TAU_COMMIT,
        "gold_replay_error_tasks": replay_error_tasks,
        "gold_replay_errors": replay_errors,
        "schedule": _schedule_summary(count),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare(
        args.dataset_root,
        args.input,
        args.output,
        args.summary,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
