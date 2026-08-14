"""Analyze selection and discarded-compute telemetry from rollout debug dumps.

The scalar log is the low-overhead, always-on view.  ``--dump-details`` adds
primitive attempt records that make joint questions such as
``P(consumed | response length, reward, difficulty)`` answerable offline.  This
tool flattens those group records, reconciles admitted rows with the final
postprocessed loss input, and optionally joins policy-loss diagnostics.

The reported conditional rates are observational.  They identify selection
bias and candidate mechanisms; causal claims require an intervention such as a
randomized staleness bound, queue policy, or node ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROLLOUT_DEBUG_KEY = "rollout_fn_debug"
TERMINAL_DISPOSITIONS = {
    "consumed",
    "stale_recycle",
    "stale_recycled",
    "aborted_recycle",
    "aborted_recycled",
    "dynamic_filter_drop",
    "dynamic_filter_dropped",
    "age_cutoff_dropped",
    "queue_evicted",
}
RECYCLED_DISPOSITIONS = {"stale_recycle", "stale_recycled", "aborted_recycle", "aborted_recycled"}
DROPPED_DISPOSITIONS = {
    "dynamic_filter_drop",
    "dynamic_filter_dropped",
    "age_cutoff_dropped",
    "queue_evicted",
    "postprocess_trimmed",
}
ANALYSIS_FIELDS = (
    "response_length",
    "generation_duration_seconds",
    "reward",
    "difficulty",
    "prompt_pass_rate",
    "pre_queue_active",
    "pre_queue_group_wait",
    "pre_queue_postprocess",
    "in_queue_staleness",
    "queue_wait_seconds",
)


def _at(values: Any, index: int) -> Any:
    return values[index] if isinstance(values, list) and index < len(values) else None


def _nonnegative_delta(end: Any, start: Any, *, scale: float = 1.0) -> float | int | None:
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
        return None
    return (end - start) / scale


def normalize_queue_eviction_record(record: dict[str, Any], *, schema_version: int | None) -> dict[str, Any]:
    """Adapt the canonical queue-lifecycle eviction to the richer row schema."""
    sample_indices = record.get("sample_indices") or []
    sample_count = len(sample_indices)
    group_index = record.get("group_index")
    retry_count = record.get("retry_count")
    attempt_id = (
        f"{group_index}:{retry_count}"
        if group_index is not None and isinstance(retry_count, int)
        else f"queue:{record.get('attempt_id')}"
    )
    completion_min = record.get("completion_version_min")
    completion_max = record.get("completion_version_max")
    ready_version = record.get("ready_version")
    decision_version = record.get("decision_version")
    queue_wait_seconds = _nonnegative_delta(
        record.get("decision_time_ns"),
        record.get("enqueue_time_ns"),
        scale=1e9,
    )
    postprocess = _nonnegative_delta(ready_version, completion_max)
    in_queue = _nonnegative_delta(decision_version, ready_version)
    numeric_rewards = [value for value in (record.get("reward_values") or []) if isinstance(value, (int, float))]
    return {
        "schema_version": schema_version,
        "disposition": "queue_evicted",
        "reason_code": "queue_capacity_evicted",
        "group_index": group_index,
        "sample_indices": sample_indices,
        "recycle_count_before": retry_count,
        "generation_attempt_id": attempt_id,
        "versions": {
            "reference": record.get("reference_version"),
            "generation_completion": completion_max,
            "group_ready": ready_version,
            "queue_put": record.get("queue_put_version"),
            "drain": decision_version,
        },
        "response_lengths": record.get("response_lengths") or [],
        # The canonical queue record predates exact per-sample lifecycle wall
        # boundaries. Missing is preferable to charging reward latency to
        # generation for queue-evicted rows.
        "generation_duration_seconds": [None] * sample_count,
        "rewards": record.get("reward_values") or [None] * sample_count,
        "group_reward_mean": statistics.fmean(numeric_rewards) if numeric_rewards else None,
        "group_reward_variance": statistics.pvariance(numeric_rewards) if numeric_rewards else None,
        "pre_queue_active": [None] * sample_count,
        # The compact queue record has only group min/max completion versions,
        # not each sample's completion boundary. Leave this unidentified rather
        # than assigning the whole spread to every sample.
        "pre_queue_group_wait": [None] * sample_count,
        "pre_queue_postprocess": [postprocess] * sample_count,
        "in_queue_staleness": [in_queue] * sample_count,
        "queue_wait_seconds": [queue_wait_seconds] * sample_count,
    }


def flatten_record(record: dict[str, Any], *, rollout_id: int, dump_root: str, source: str) -> list[dict[str, Any]]:
    """Expand one group-level debug record to one row per generated sample."""
    indices = record.get("sample_indices") or []
    versions = record.get("versions") or {}
    rows = []
    for index, sample_index in enumerate(indices):
        row = {
            "rollout_id": rollout_id,
            "dump_root": dump_root,
            "source": source,
            "schema_version": record.get("schema_version"),
            "group_index": record.get("group_index"),
            "prompt_id": record.get("prompt_id", record.get("group_index")),
            "sample_index": sample_index,
            "generation_attempt_id": record.get("generation_attempt_id"),
            "generation_attempt_number": record.get("recycle_count_before"),
            "disposition": record.get("disposition"),
            "reason_code": record.get("reason_code"),
            "reference_mode": record.get("reference_mode"),
            "bound": record.get("bound"),
            "reference_version": versions.get("reference"),
            "generation_completion_version": versions.get("generation_completion"),
            "group_ready_version": versions.get("group_ready"),
            "queue_put_version": versions.get("queue_put"),
            "drain_version": versions.get("drain"),
            "response_length": _at(record.get("response_lengths"), index),
            "generation_duration_seconds": _at(record.get("generation_duration_seconds"), index),
            "reward": _at(record.get("rewards"), index),
            "difficulty": _at(record.get("difficulty"), index),
            "prompt_pass_rate": _at(record.get("prompt_pass_rates"), index),
            "group_reward_mean": record.get("group_reward_mean"),
            "group_reward_variance": record.get("group_reward_variance"),
            "pre_queue_active": _at(record.get("pre_queue_active"), index),
            "pre_queue_group_wait": _at(record.get("pre_queue_group_wait"), index),
            "pre_queue_postprocess": _at(record.get("pre_queue_postprocess"), index),
            "in_queue_staleness": _at(record.get("in_queue_staleness"), index),
            "queue_wait_seconds": _at(record.get("queue_wait_seconds"), index),
            "loss_input_tokens": _at(record.get("loss_input_tokens"), index),
            "training_step": record.get("training_step"),
            "straggler_collateral": sample_index in set(record.get("straggler_collateral_sample_indices") or []),
        }
        for component, value in (record.get("waste") or {}).items():
            row[f"group_waste_{component}"] = value
        rows.append(row)
    return rows


def reconcile_attempt_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one terminal outcome per attempt/sample, including postprocess trims."""
    admitted: dict[tuple[Any, ...], dict[str, Any]] = {}
    terminal: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in records:
        key = (
            row["dump_root"],
            row["rollout_id"],
            row["generation_attempt_id"],
            row["sample_index"],
        )
        disposition = row.get("disposition")
        if disposition == "admitted":
            admitted[key] = row
        elif disposition in TERMINAL_DISPOSITIONS:
            terminal[key] = row

    for key, row in admitted.items():
        if key in terminal:
            continue
        trimmed = dict(row)
        trimmed["disposition"] = "postprocess_trimmed"
        trimmed["reason_code"] = "postprocess_trimmed"
        trimmed["loss_input_tokens"] = 0
        terminal[key] = trimmed

    rows = []
    for row in terminal.values():
        disposition = row["disposition"]
        row = dict(row)
        row["accepted"] = disposition == "consumed"
        row["recycled"] = disposition in RECYCLED_DISPOSITIONS
        row["dropped"] = disposition in DROPPED_DISPOSITIONS
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["rollout_id"],
            str(row.get("generation_attempt_id")),
            -1 if row.get("sample_index") is None else row["sample_index"],
        ),
    )


def load_attempt_rows(dump_roots: Iterable[Path]) -> list[dict[str, Any]]:
    flat_rows = []
    for root in dump_roots:
        for path in sorted((root / "rollout_data").glob("[0-9]*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            rollout_id = int(payload.get("rollout_id", path.stem))
            debug = (payload.get("metadata") or {}).get(ROLLOUT_DEBUG_KEY) or {}
            for record in debug.get("records") or []:
                if record.get("disposition") != "queue_evicted":
                    continue
                flat_rows.extend(
                    flatten_record(
                        normalize_queue_eviction_record(record, schema_version=debug.get("schema_version")),
                        rollout_id=rollout_id,
                        dump_root=str(root),
                        source=str(path),
                    )
                )
            records = (debug.get("recycle_compute") or {}).get("records") or []
            for record in records:
                flat_rows.extend(
                    flatten_record(
                        record,
                        rollout_id=rollout_id,
                        dump_root=str(root),
                        source=str(path),
                    )
                )
    return reconcile_attempt_rows(flat_rows)


def _tensor_values(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().float().reshape(-1)]
    if isinstance(value, list):
        return [float(item) for item in value]
    return []


def load_policy_diagnostics(dump_roots: Iterable[Path]) -> dict[tuple[Any, ...], dict[str, float]]:
    """Aggregate debug calls by stable rollout/attempt/sample identity."""
    accumulators: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for root in dump_roots:
        for path in sorted((root / "policy_loss_debug").glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            parallel = payload.get("parallel") or {}
            if parallel.get("tp_rank", 0) != 0:
                continue
            for sample in payload.get("samples") or []:
                key = (
                    str(root),
                    sample.get("training_step"),
                    sample.get("generation_attempt_id"),
                    sample.get("sample_index"),
                    sample.get("optimizer_step_id", 0),
                )
                if any(value is None for value in key[:4]):
                    continue
                for field in (
                    "response_token_count_local",
                    "pre_loss_token_count_local",
                    "final_loss_token_count_local",
                    "ppo_clip_count_local",
                    "importance_clip_count_local",
                    "sequence_policy_rollout_log_ratio_local",
                    "absolute_pg_contribution_local",
                    "sample_staleness",
                ):
                    value = sample.get(field)
                    if isinstance(value, (int, float)):
                        accumulators[key][field].append(float(value))
                final_loss = _tensor_values(sample.get("final_pg_loss"))
                final_mask = _tensor_values(sample.get("final_local_loss_mask"))
                if (
                    "absolute_pg_contribution_local" not in sample
                    and final_loss
                    and len(final_loss) == len(final_mask)
                ):
                    contribution = sum(abs(loss) * mask for loss, mask in zip(final_loss, final_mask, strict=True))
                    accumulators[key]["absolute_pg_contribution_local"].append(contribution)

    step_diagnostics = {}
    for key, fields in accumulators.items():
        response_tokens = sum(fields.get("response_token_count_local", []))
        pre_loss_tokens = sum(fields.get("pre_loss_token_count_local", []))
        final_loss_tokens = sum(fields.get("final_loss_token_count_local", []))
        row = {
            "response_tokens": response_tokens,
            "pre_loss_tokens": pre_loss_tokens,
            "final_loss_tokens": final_loss_tokens,
            "ppo_clip_count": sum(fields.get("ppo_clip_count_local", [])),
            "importance_clip_count": sum(fields.get("importance_clip_count_local", [])),
            "sequence_policy_rollout_log_ratio": sum(fields.get("sequence_policy_rollout_log_ratio_local", [])),
            "absolute_pg_contribution": sum(fields.get("absolute_pg_contribution_local", [])),
        }
        if fields.get("sample_staleness"):
            row["sample_staleness"] = statistics.fmean(fields["sample_staleness"])
        step_diagnostics[key] = row

    grouped_steps: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for key, row in step_diagnostics.items():
        grouped_steps[key[:4]].append(row)

    diagnostics = {}
    for key, steps in grouped_steps.items():
        response_tokens = sum(step["response_tokens"] for step in steps)
        pre_loss_tokens = sum(step["pre_loss_tokens"] for step in steps)
        final_loss_tokens = sum(step["final_loss_tokens"] for step in steps)
        row = {
            "ppo_clip_fraction": (
                sum(step["ppo_clip_count"] for step in steps) / pre_loss_tokens if pre_loss_tokens > 0 else 0.0
            ),
            "importance_clip_fraction": (
                sum(step["importance_clip_count"] for step in steps) / pre_loss_tokens if pre_loss_tokens > 0 else 0.0
            ),
            "mask_fraction": (1.0 - final_loss_tokens / response_tokens if response_tokens > 0 else 0.0),
            "sequence_policy_rollout_log_ratio": statistics.fmean(
                step["sequence_policy_rollout_log_ratio"] for step in steps
            ),
            "absolute_pg_contribution": sum(step["absolute_pg_contribution"] for step in steps),
            "optimizer_updates_observed": float(len(steps)),
        }
        staleness_values = [step["sample_staleness"] for step in steps if "sample_staleness" in step]
        if staleness_values:
            row["sample_staleness"] = statistics.fmean(staleness_values)
        diagnostics[key] = row
    return diagnostics


def join_policy_diagnostics(rows: list[dict[str, Any]], diagnostics: dict[tuple[Any, ...], dict[str, float]]) -> None:
    for row in rows:
        key = (
            row.get("dump_root"),
            row.get("training_step"),
            row.get("generation_attempt_id"),
            row.get("sample_index"),
        )
        row.update(diagnostics.get(key, {}))


def _numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def distribution(values: Iterable[Any]) -> dict[str, float | int]:
    numeric = sorted(_numeric(values))
    if not numeric:
        return {"count": 0}
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "min": numeric[0],
        "p50": _percentile(numeric, 0.50),
        "p90": _percentile(numeric, 0.90),
        "p95": _percentile(numeric, 0.95),
        "p99": _percentile(numeric, 0.99),
        "max": numeric[-1],
    }


def _quantile_edges(rows: list[dict[str, Any]], field: str, num_bins: int) -> list[float]:
    values = sorted(_numeric(row.get(field) for row in rows))
    if not values:
        return []
    return sorted({_percentile(values, index / num_bins) for index in range(1, num_bins)})


def _bin_index(value: float, edges: list[float]) -> int:
    return sum(value > edge for edge in edges)


def conditional_acceptance(rows: list[dict[str, Any]], field: str, *, num_bins: int) -> list[dict[str, Any]]:
    edges = _quantile_edges(rows, field, num_bins)
    cells: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            cells[_bin_index(float(value), edges)].append(row)
    output = []
    for index, cell in sorted(cells.items()):
        accepted = sum(bool(row["accepted"]) for row in cell)
        output.append(
            {
                "bin": index,
                "lower_exclusive": edges[index - 1] if index > 0 else None,
                "upper_inclusive": edges[index] if index < len(edges) else None,
                "count": len(cell),
                "accepted": accepted,
                "acceptance_rate": accepted / len(cell),
            }
        )
    return output


def joint_acceptance(
    rows: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    num_bins: int,
    min_cell: int,
) -> list[dict[str, Any]]:
    usable_fields = tuple(field for field in fields if len(_numeric(row.get(field) for row in rows)) >= min_cell)
    if not usable_fields:
        return []
    edges = {field: _quantile_edges(rows, field, num_bins) for field in usable_fields}
    cells: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values = [row.get(field) for field in usable_fields]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            continue
        key = tuple(_bin_index(float(value), edges[field]) for field, value in zip(usable_fields, values, strict=True))
        cells[key].append(row)
    output = []
    for key, cell in sorted(cells.items()):
        if len(cell) < min_cell:
            continue
        accepted = sum(bool(row["accepted"]) for row in cell)
        output.append(
            {
                "bins": dict(zip(usable_fields, key, strict=True)),
                "count": len(cell),
                "accepted": accepted,
                "acceptance_rate": accepted / len(cell),
            }
        )
    return output


def summarize(rows: list[dict[str, Any]], *, num_bins: int, min_joint_cell: int) -> dict[str, Any]:
    populations = {
        "generated": rows,
        "consumed": [row for row in rows if row["accepted"]],
        "recycled": [row for row in rows if row["recycled"]],
        "dropped": [row for row in rows if row["dropped"]],
    }
    reason_counts: dict[str, int] = defaultdict(int)
    reason_tokens: dict[str, int] = defaultdict(int)
    for row in rows:
        reason = str(row.get("reason_code"))
        reason_counts[reason] += 1
        if isinstance(row.get("response_length"), int):
            reason_tokens[reason] += row["response_length"]
    return {
        "semantics": {
            "accepted": "present in the final postprocessed training loss input",
            "generated": "one terminal row per generation attempt and sample",
            "causal_scope": "conditional rates are observational associations; causal effects require intervention",
        },
        "counts": {
            "generated": len(rows),
            "consumed": len(populations["consumed"]),
            "recycled": len(populations["recycled"]),
            "dropped": len(populations["dropped"]),
        },
        "useful_sample_efficiency": (len(populations["consumed"]) / len(rows) if rows else 0.0),
        "reason_sample_counts": dict(sorted(reason_counts.items())),
        "reason_response_tokens": dict(sorted(reason_tokens.items())),
        "population_distributions": {
            population: {field: distribution(row.get(field) for row in population_rows) for field in ANALYSIS_FIELDS}
            for population, population_rows in populations.items()
        },
        "acceptance_by_feature": {
            field: conditional_acceptance(rows, field, num_bins=num_bins)
            for field in (
                "response_length",
                "reward",
                "difficulty",
                "prompt_pass_rate",
                "generation_duration_seconds",
            )
        },
        "acceptance_by_length_reward_difficulty": joint_acceptance(
            rows,
            fields=("response_length", "reward", "difficulty"),
            num_bins=num_bins,
            min_cell=min_joint_cell,
        ),
    }


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-details", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None, help="Summary JSON path; stdout if omitted")
    parser.add_argument("--rows-out", type=Path, default=None, help="Flattened .jsonl or .csv output")
    parser.add_argument("--num-bins", type=int, default=4)
    parser.add_argument("--min-joint-cell", type=int, default=5)
    parser.add_argument("--skip-policy-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.num_bins > 0
    assert args.min_joint_cell > 0
    rows = load_attempt_rows(args.dump_details)
    if not args.skip_policy_debug:
        join_policy_diagnostics(rows, load_policy_diagnostics(args.dump_details))
    summary = summarize(
        rows,
        num_bins=args.num_bins,
        min_joint_cell=args.min_joint_cell,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    if args.rows_out is not None:
        write_rows(rows, args.rows_out)


if __name__ == "__main__":
    main()
