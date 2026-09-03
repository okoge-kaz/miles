#!/usr/bin/env python3
"""Predict fully-async producer/consumer balance and mean training staleness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

QUEUE_RECYCLE = "queue-recycle"
QUEUE_MAX = "queue-max"
QUEUE_DROP = "queue-drop"
QUEUE_POLICIES = (QUEUE_RECYCLE, QUEUE_MAX, QUEUE_DROP)
ARM_PATTERN = re.compile(r"s(?P<staleness>\d+)-t(?P<trainer>\d+)r(?P<rollout>\d+)")


@dataclass(frozen=True)
class ModelInputs:
    """One deterministic fluid-model operating point."""

    train_compute_seconds: float
    rollout_groups_per_second: float
    batch_groups: int
    concurrency_groups: float
    active_group_fraction: float = 1.0
    group_latency_seconds: float | None = None
    queue_policy: str = QUEUE_RECYCLE
    queue_capacity_groups: int | None = None
    max_weight_staleness: float | None = None
    rho_tolerance: float = 0.02


@dataclass(frozen=True)
class Prediction:
    """Fluid prediction; caps are envelopes rather than distributional means."""

    train_capacity_groups_per_second: float
    rollout_groups_per_second: float
    rho: float
    actual_updates_per_second: float
    effective_active_groups: float
    effective_group_latency_seconds: float
    group_latency_source: str
    pre_queue_staleness: float
    in_queue_staleness: float | None
    natural_staleness: float | None
    stationary_interval_low: float | None
    stationary_interval_high: float | None
    linear_growth_per_training_step: float
    max_weight_staleness_cap: float | None
    queue_capacity_staleness_cap: float | None
    effective_staleness_cap: float | None
    steady_staleness_or_cap: float | None
    steps_to_cap: float | None
    regime: str
    prediction_kind: str


@dataclass(frozen=True)
class Fit:
    """Ordinary least-squares fit y = slope * x + intercept."""

    observations: int
    slope: float
    intercept: float
    r_squared: float


@dataclass(frozen=True)
class _Balance:
    """Shared producer/consumer quantities used by all queue policies."""

    train_capacity: float
    rho: float
    actual_updates_per_second: float
    active_groups: float
    group_latency_seconds: float
    group_latency_source: str
    pre_queue_staleness: float


def _validate_inputs(inputs: ModelInputs) -> None:
    if inputs.train_compute_seconds <= 0:
        raise ValueError("train_compute_seconds must be positive")
    if inputs.rollout_groups_per_second <= 0:
        raise ValueError("rollout_groups_per_second must be positive")
    if inputs.batch_groups < 1:
        raise ValueError("batch_groups must be positive")
    if inputs.concurrency_groups <= 0:
        raise ValueError("concurrency_groups must be positive")
    if not 0 < inputs.active_group_fraction <= 1:
        raise ValueError("active_group_fraction must be in (0, 1]")
    if inputs.group_latency_seconds is not None and inputs.group_latency_seconds <= 0:
        raise ValueError("group_latency_seconds must be positive")
    if inputs.queue_policy not in QUEUE_POLICIES:
        raise ValueError(f"unknown queue policy: {inputs.queue_policy}")
    if inputs.queue_capacity_groups is not None and inputs.queue_capacity_groups < 1:
        raise ValueError("queue_capacity_groups must be positive")
    if inputs.max_weight_staleness is not None and (
        math.isnan(inputs.max_weight_staleness) or inputs.max_weight_staleness < 0
    ):
        raise ValueError("max_weight_staleness must be nonnegative or infinity")
    if (
        inputs.queue_policy == QUEUE_DROP
        and inputs.max_weight_staleness is not None
        and math.isfinite(inputs.max_weight_staleness)
    ):
        raise ValueError("queue-drop does not use max_weight_staleness")
    if not 0 <= inputs.rho_tolerance < 1:
        raise ValueError("rho_tolerance must be in [0, 1)")


def _queue_drop_iqs(*, rho: float, queue_factor: float, tolerance: float) -> float | None:
    if rho < 1 - tolerance:
        return rho
    if rho > 1 + tolerance:
        return (2 * queue_factor + rho - 1) / (2 * rho)
    return None


def _finite_min(*values: float | None) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return min(finite) if finite else None


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _compute_balance(inputs: ModelInputs) -> _Balance:
    train_capacity = inputs.batch_groups / inputs.train_compute_seconds
    rho = inputs.rollout_groups_per_second / train_capacity
    actual_updates_per_second = min(
        1 / inputs.train_compute_seconds,
        inputs.rollout_groups_per_second / inputs.batch_groups,
    )
    active_groups = inputs.concurrency_groups * inputs.active_group_fraction
    if inputs.group_latency_seconds is None:
        group_latency = active_groups / inputs.rollout_groups_per_second
        latency_source = "little-law from active groups / group rate"
    else:
        group_latency = inputs.group_latency_seconds
        latency_source = "measured"
    return _Balance(
        train_capacity=train_capacity,
        rho=rho,
        actual_updates_per_second=actual_updates_per_second,
        active_groups=active_groups,
        group_latency_seconds=group_latency,
        group_latency_source=latency_source,
        pre_queue_staleness=group_latency * actual_updates_per_second,
    )


def _common_prediction_fields(inputs: ModelInputs, balance: _Balance) -> dict[str, Any]:
    return {
        "train_capacity_groups_per_second": balance.train_capacity,
        "rollout_groups_per_second": inputs.rollout_groups_per_second,
        "rho": balance.rho,
        "actual_updates_per_second": balance.actual_updates_per_second,
        "effective_active_groups": balance.active_groups,
        "effective_group_latency_seconds": balance.group_latency_seconds,
        "group_latency_source": balance.group_latency_source,
        "pre_queue_staleness": balance.pre_queue_staleness,
    }


def _predict_queue_drop(inputs: ModelInputs, balance: _Balance) -> Prediction:
    queue_factor = (inputs.queue_capacity_groups or inputs.batch_groups) / inputs.batch_groups
    in_queue = _queue_drop_iqs(
        rho=balance.rho,
        queue_factor=queue_factor,
        tolerance=inputs.rho_tolerance,
    )
    total = balance.pre_queue_staleness + in_queue if in_queue is not None else None
    return Prediction(
        **_common_prediction_fields(inputs, balance),
        in_queue_staleness=in_queue,
        natural_staleness=total,
        stationary_interval_low=total,
        stationary_interval_high=total,
        linear_growth_per_training_step=0.0,
        max_weight_staleness_cap=None,
        queue_capacity_staleness_cap=queue_factor,
        effective_staleness_cap=None,
        steady_staleness_or_cap=total,
        steps_to_cap=None,
        regime="critical" if total is None else "queue-drop stationary",
        prediction_kind="closed-form mean" if total is not None else "rho≈1 boundary",
    )


def _predict_underloaded_fifo(
    inputs: ModelInputs,
    balance: _Balance,
    *,
    handoff: float,
    natural: float,
    queue_cap: float | None,
    effective_cap: float | None,
) -> Prediction:
    if effective_cap is not None and effective_cap < natural:
        regime = "bound-limited feedback"
        predicted = effective_cap
        kind = "cap envelope; rejection changes the input rate"
    else:
        regime = "rollout-limited stationary"
        predicted = natural
        kind = "deterministic lower-envelope mean"
    return Prediction(
        **_common_prediction_fields(inputs, balance),
        in_queue_staleness=handoff,
        natural_staleness=natural,
        stationary_interval_low=min(predicted, natural),
        stationary_interval_high=min(
            predicted + 1.0,
            effective_cap if effective_cap is not None else math.inf,
        ),
        linear_growth_per_training_step=0.0,
        max_weight_staleness_cap=_finite_or_none(inputs.max_weight_staleness),
        queue_capacity_staleness_cap=queue_cap,
        effective_staleness_cap=effective_cap,
        steady_staleness_or_cap=predicted,
        steps_to_cap=None,
        regime=regime,
        prediction_kind=kind,
    )


def _predict_critical_fifo(
    inputs: ModelInputs,
    balance: _Balance,
    *,
    handoff: float,
    natural: float,
    queue_cap: float | None,
    effective_cap: float | None,
) -> Prediction:
    return Prediction(
        **_common_prediction_fields(inputs, balance),
        in_queue_staleness=handoff,
        natural_staleness=natural,
        stationary_interval_low=natural,
        stationary_interval_high=effective_cap,
        linear_growth_per_training_step=0.0,
        max_weight_staleness_cap=_finite_or_none(inputs.max_weight_staleness),
        queue_capacity_staleness_cap=queue_cap,
        effective_staleness_cap=effective_cap,
        steady_staleness_or_cap=None,
        steps_to_cap=None,
        regime="critical",
        prediction_kind="no unique deterministic stationary queue at rho≈1",
    )


def _predict_overloaded_fifo(
    inputs: ModelInputs,
    balance: _Balance,
    *,
    handoff: float,
    natural: float,
    queue_cap: float | None,
    effective_cap: float | None,
) -> Prediction:
    growth = 1 - 1 / balance.rho
    if effective_cap is None:
        steps_to_cap = None
        regime = "unbounded linear growth"
        predicted = None
        kind = "non-stationary"
    else:
        steps_to_cap = max(0.0, effective_cap - natural) / growth
        regime = "cap-limited"
        predicted = effective_cap
        kind = "max-staleness or queue-capacity envelope"
    return Prediction(
        **_common_prediction_fields(inputs, balance),
        in_queue_staleness=handoff,
        natural_staleness=natural,
        stationary_interval_low=None,
        stationary_interval_high=None,
        linear_growth_per_training_step=growth,
        max_weight_staleness_cap=_finite_or_none(inputs.max_weight_staleness),
        queue_capacity_staleness_cap=queue_cap,
        effective_staleness_cap=effective_cap,
        steady_staleness_or_cap=predicted,
        steps_to_cap=steps_to_cap,
        regime=regime,
        prediction_kind=kind,
    )


def predict(inputs: ModelInputs) -> Prediction:
    """Predict deterministic balance, stationary lag, or a cap-limited envelope."""
    _validate_inputs(inputs)
    balance = _compute_balance(inputs)

    if inputs.queue_policy == QUEUE_DROP:
        return _predict_queue_drop(inputs, balance)

    handoff = 1.0 if inputs.queue_policy == QUEUE_RECYCLE else 0.0
    natural = balance.pre_queue_staleness + handoff
    queue_cap = (
        natural + inputs.queue_capacity_groups / inputs.batch_groups
        if inputs.queue_capacity_groups is not None
        else None
    )
    effective_cap = _finite_min(inputs.max_weight_staleness, queue_cap)

    prediction_args = {
        "handoff": handoff,
        "natural": natural,
        "queue_cap": queue_cap,
        "effective_cap": effective_cap,
    }
    if balance.rho < 1 - inputs.rho_tolerance:
        return _predict_underloaded_fifo(inputs, balance, **prediction_args)
    if balance.rho <= 1 + inputs.rho_tolerance:
        return _predict_critical_fifo(inputs, balance, **prediction_args)
    return _predict_overloaded_fifo(inputs, balance, **prediction_args)


def predicted_trajectory(prediction: Prediction, steps: int) -> list[dict[str, float | int]]:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    initial = prediction.natural_staleness or 0.0
    rows = []
    for step in range(steps + 1):
        value = initial + prediction.linear_growth_per_training_step * step
        if prediction.effective_staleness_cap is not None:
            value = min(value, prediction.effective_staleness_cap)
        rows.append({"training_step_since_warmup": step, "predicted_training_staleness": value})
    return rows


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(rows: list[dict[str, str]], field: str) -> float | None:
    values = [_optional_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return statistics.median(finite) if finite else None


def _clean_history_rows(rows: list[dict[str, str]], exclude_after_resume: int) -> list[dict[str, str]]:
    clean = []
    cooldown = 0
    for row in sorted(rows, key=lambda item: int(item["training_step"])):
        if int(float(row.get("resume_boundary") or 0)):
            cooldown = exclude_after_resume
        if cooldown > 0:
            cooldown -= 1
            continue
        clean.append(row)
    return clean


def _group_rate(
    rows: list[dict[str, str]],
    *,
    samples_per_group: int,
) -> tuple[float, str]:
    direct = _median(rows, "throughput/generated_groups_per_second")
    if direct is not None and direct > 0:
        return direct, "direct generated-group counter"
    token_rate = _median(rows, "throughput/generated_tokens_per_second")
    generated_length = _median(rows, "queue/selection/generated/sample_length/mean")
    if generated_length is not None:
        source = "token proxy using generated sample length"
    else:
        generated_length = _median(rows, "rollout/response_len/mean")
        source = "token proxy using trained sample length (selection-sensitive)"
    if token_rate is None or generated_length is None or generated_length <= 0:
        raise ValueError("history has neither direct group rate nor usable token-rate proxy")
    return token_rate / (samples_per_group * generated_length), source


def _parse_arm(arm: str) -> tuple[int, int, int] | None:
    match = ARM_PATTERN.fullmatch(arm)
    if match is None:
        return None
    return tuple(int(match.group(name)) for name in ("staleness", "trainer", "rollout"))


def summarize_history(
    path: Path,
    *,
    window: int,
    exclude_after_resume: int,
    batch_groups: int,
    samples_per_group: int,
    concurrency_groups: float,
    queue_policy: str,
    queue_capacity_groups: int | None,
    backpressure_fraction: float,
) -> list[dict[str, Any]]:
    if window < 1:
        raise ValueError("window must be positive")
    if exclude_after_resume < 0:
        raise ValueError("exclude_after_resume must be nonnegative")
    if samples_per_group < 1:
        raise ValueError("samples_per_group must be positive")
    if not 0 <= backpressure_fraction <= 1:
        raise ValueError("backpressure_fraction must be in [0, 1]")
    with path.open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in history:
        if _parse_arm(row["arm"]) is not None:
            by_arm[row["arm"]].append(row)

    summaries = []
    for arm, arm_rows in sorted(by_arm.items()):
        staleness, trainer_nodes, rollout_nodes = _parse_arm(arm) or (0, 0, 0)
        clean = _clean_history_rows(arm_rows, exclude_after_resume)
        selected = clean[-window:]
        if not selected:
            continue
        train_seconds = _median(selected, "perf/train_time")
        if train_seconds is None:
            continue
        group_rate, rate_source = _group_rate(selected, samples_per_group=samples_per_group)
        measured_concurrency = _median(selected, "queue/occupancy/max_in_flight_groups")
        measured_active_fraction = _median(selected, "queue/active_group_capacity_time_mean")
        active_fraction = measured_active_fraction if measured_active_fraction is not None else 1.0
        capacity = _median(selected, "queue/occupancy/capacity_groups")
        point_capacity = int(capacity) if capacity is not None else queue_capacity_groups
        prediction = predict(
            ModelInputs(
                train_compute_seconds=train_seconds,
                rollout_groups_per_second=group_rate,
                batch_groups=batch_groups,
                concurrency_groups=measured_concurrency or concurrency_groups,
                active_group_fraction=active_fraction,
                queue_policy=queue_policy,
                queue_capacity_groups=point_capacity,
                max_weight_staleness=float(staleness),
            )
        )
        step_seconds = _median(selected, "perf/step_time")
        backpressure_seconds = _median(selected, "queue/rollout_backpressure_seconds") or 0.0
        denominator = _median(selected, "throughput/window_seconds") or step_seconds
        backpressure_ratio = backpressure_seconds / denominator if denominator else 0.0
        loss_tokens = _median(selected, "train/final_loss_tokens")
        summaries.append(
            {
                "arm": arm,
                "max_weight_staleness": staleness,
                "trainer_nodes": trainer_nodes,
                "rollout_nodes": rollout_nodes,
                "observations": len(selected),
                "first_training_step": int(selected[0]["training_step"]),
                "last_training_step": int(selected[-1]["training_step"]),
                "train_compute_seconds": train_seconds,
                "train_loss_tokens": loss_tokens,
                "train_loss_tokens_per_second": loss_tokens / train_seconds if loss_tokens is not None else None,
                "rollout_groups_per_second": group_rate,
                "rollout_group_rate_source": rate_source,
                "rollout_rate_capacity_censored": int(backpressure_ratio >= backpressure_fraction),
                "backpressure_fraction": backpressure_ratio,
                "observed_training_staleness": _median(selected, "staleness/total/mean"),
                "observed_pre_queue_staleness": _median(selected, "staleness/pre_queue/mean"),
                "observed_in_queue_staleness": _median(selected, "staleness/in_queue/mean"),
                "observed_train_wait_seconds": _median(selected, "perf/train_wait_time"),
                **{f"predicted_{key}": value for key, value in asdict(prediction).items()},
            }
        )
    return summaries


def linear_fit(pairs: list[tuple[float, float]]) -> Fit:
    if len(pairs) < 2:
        raise ValueError("at least two points are required for a fit")
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("fit x values are constant")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in pairs)
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1 - residual / total if total > 0 else 1.0
    return Fit(observations=len(pairs), slope=slope, intercept=intercept, r_squared=r_squared)


def scaling_fits(points: list[dict[str, Any]], fit_staleness: int) -> dict[str, Any]:
    selected = [point for point in points if point["max_weight_staleness"] == fit_staleness]
    train_pairs = [(1 / point["trainer_nodes"], point["train_compute_seconds"]) for point in selected]
    rollout_pairs = [
        (1 / point["rollout_nodes"], 1 / point["rollout_groups_per_second"])
        for point in selected
        if not point["rollout_rate_capacity_censored"]
    ]
    return {
        "fit_staleness": fit_staleness,
        "training_model": "train_seconds(T) = slope / T + intercept",
        "training_fit": asdict(linear_fit(train_pairs)),
        "rollout_model": "1 / groups_per_second(R) = slope / R + intercept",
        "rollout_fit": asdict(linear_fit(rollout_pairs)),
        "rollout_fit_excludes_capacity_censored_points": True,
    }


def _fit_train_seconds(fit: Fit, trainer_nodes: int) -> float:
    return fit.slope / trainer_nodes + fit.intercept


def _fit_rollout_rate(fit: Fit, rollout_nodes: int) -> float:
    reciprocal = fit.slope / rollout_nodes + fit.intercept
    if reciprocal <= 0:
        raise ValueError(f"rollout fit predicts a nonpositive reciprocal rate at R={rollout_nodes}")
    return 1 / reciprocal


def ratio_candidates(
    fits: dict[str, Any],
    *,
    total_nodes: int,
    trainer_nodes: list[int],
    batch_groups: int,
    concurrency_groups: float,
    queue_policy: str,
    queue_capacity_groups: int | None,
    max_weight_staleness: float | None,
) -> list[dict[str, Any]]:
    if total_nodes < 2:
        raise ValueError("total_nodes must be at least two")
    train_fit = Fit(**fits["training_fit"])
    rollout_fit = Fit(**fits["rollout_fit"])
    candidates = []
    for trainer in trainer_nodes:
        rollout = total_nodes - trainer
        if rollout < 1:
            continue
        train_seconds = _fit_train_seconds(train_fit, trainer)
        group_rate = _fit_rollout_rate(rollout_fit, rollout)
        prediction = predict(
            ModelInputs(
                train_compute_seconds=train_seconds,
                rollout_groups_per_second=group_rate,
                batch_groups=batch_groups,
                concurrency_groups=concurrency_groups,
                queue_policy=queue_policy,
                queue_capacity_groups=queue_capacity_groups,
                max_weight_staleness=max_weight_staleness,
            )
        )
        candidates.append(
            {
                "trainer_nodes": trainer,
                "rollout_nodes": rollout,
                "ratio": f"{trainer}:{rollout}",
                "predicted_train_compute_seconds": train_seconds,
                "predicted_rollout_groups_per_second": group_rate,
                "rho": prediction.rho,
                "predicted_optimizer_updates_per_hour": prediction.actual_updates_per_second * 3600,
                "predicted_training_staleness_or_cap": prediction.steady_staleness_or_cap,
                "staleness_regime": prediction.regime,
            }
        )
    if candidates:
        best = max(candidates, key=lambda row: row["predicted_optimizer_updates_per_hour"])
        for candidate in candidates:
            candidate["recommended_for_throughput"] = int(candidate is best)
    return candidates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_int_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return result


def _point_inputs(args: argparse.Namespace) -> ModelInputs:
    return ModelInputs(
        train_compute_seconds=args.train_compute_seconds,
        rollout_groups_per_second=args.rollout_groups_per_second,
        batch_groups=args.batch_groups,
        concurrency_groups=args.concurrency_groups,
        active_group_fraction=args.active_group_fraction,
        group_latency_seconds=args.group_latency_seconds,
        queue_policy=args.queue_policy,
        queue_capacity_groups=args.queue_capacity_groups,
        max_weight_staleness=_finite_or_none(args.max_weight_staleness),
        rho_tolerance=args.rho_tolerance,
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-compute-seconds", type=float, required=True)
    parser.add_argument("--rollout-groups-per-second", type=float, required=True)
    parser.add_argument("--batch-groups", type=int, required=True)
    parser.add_argument("--concurrency-groups", type=float, required=True)
    parser.add_argument("--active-group-fraction", type=float, default=1.0)
    parser.add_argument("--group-latency-seconds", type=float)
    parser.add_argument("--queue-policy", choices=QUEUE_POLICIES, default=QUEUE_RECYCLE)
    parser.add_argument("--queue-capacity-groups", type=int, help="omit for an unbounded completed-group queue")
    parser.add_argument("--max-weight-staleness", type=float, help="omit or pass inf for no weight-staleness bound")
    parser.add_argument("--rho-tolerance", type=float, default=0.02)


def _run_point(args: argparse.Namespace) -> None:
    prediction = predict(_point_inputs(args))
    result = {"inputs": asdict(_point_inputs(args)), "prediction": asdict(prediction)}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.trajectory_csv is not None:
        args.trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.trajectory_csv, predicted_trajectory(prediction, args.steps))


def _run_history(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = summarize_history(
        args.history_csv,
        window=args.window,
        exclude_after_resume=args.exclude_after_resume,
        batch_groups=args.batch_groups,
        samples_per_group=args.samples_per_group,
        concurrency_groups=args.concurrency_groups,
        queue_policy=args.queue_policy,
        queue_capacity_groups=args.queue_capacity_groups,
        backpressure_fraction=args.backpressure_fraction,
    )
    fits = scaling_fits(points, args.fit_staleness)
    candidates = ratio_candidates(
        fits,
        total_nodes=args.total_nodes,
        trainer_nodes=args.allowed_trainer_nodes,
        batch_groups=args.batch_groups,
        concurrency_groups=args.concurrency_groups,
        queue_policy=args.queue_policy,
        queue_capacity_groups=args.queue_capacity_groups,
        max_weight_staleness=float(args.fit_staleness),
    )
    _write_csv(args.output_dir / "point-predictions.csv", points)
    _write_csv(args.output_dir / "ratio-candidates.csv", candidates)
    summary = {
        "history_csv": str(args.history_csv),
        "window": args.window,
        "exclude_after_resume": args.exclude_after_resume,
        "fits": fits,
        "ratio_candidates": candidates,
        "rate_warning": (
            "Rows without throughput/generated_groups_per_second use a token-rate proxy. "
            "That proxy is selection-sensitive when generated sample length is unavailable."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    point = subparsers.add_parser("point", help="Predict one measured operating point")
    _add_model_arguments(point)
    point.add_argument("--steps", type=int, default=300)
    point.add_argument("--trajectory-csv", type=Path)
    point.set_defaults(run=_run_point)

    history = subparsers.add_parser("history", help="Analyze a W&B training-history CSV")
    history.add_argument("--history-csv", type=Path, required=True)
    history.add_argument("--output-dir", type=Path, required=True)
    history.add_argument("--window", type=int, default=50)
    history.add_argument("--exclude-after-resume", type=int, default=10)
    history.add_argument("--batch-groups", type=int, required=True)
    history.add_argument("--samples-per-group", type=int, required=True)
    history.add_argument("--concurrency-groups", type=float, required=True)
    history.add_argument("--queue-policy", choices=QUEUE_POLICIES, default=QUEUE_RECYCLE)
    history.add_argument("--queue-capacity-groups", type=int)
    history.add_argument("--backpressure-fraction", type=float, default=0.05)
    history.add_argument("--fit-staleness", type=int, required=True)
    history.add_argument("--total-nodes", type=int, required=True)
    history.add_argument("--allowed-trainer-nodes", type=_parse_int_list, required=True)
    history.set_defaults(run=_run_history)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
