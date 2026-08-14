"""Audit one run against what the equivalence protocol needs, before spending GPU time.

Every check names the analysis it unlocks, so the output is a list of
consequences rather than a list of files. ``FAIL`` means a headline number
cannot be produced at all; ``WARN`` means it can, but with a weaker claim than
the design asks for -- usually a confidence interval that is missing one of its
three variance sources.

Standard library only, so it runs on a login node with no environment at all:

    python -m experiments.src.offpolicy_acceleration.check_logging \\
      --dump-details /ckpt/training/math/dapo-math/Qwen3-4B/<tag>/dump
    python -m experiments.src.offpolicy_acceleration.check_logging \\
      --slurm-log experiments/outputs/training/.../<job>.log

Exit status is 1 when any check FAILs, so it can gate a sweep submission.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.src.offpolicy_acceleration import log_source

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]

# How many evaluations the plateau test plus the consecutive-crossing rule need
# before it can return anything but "not converged".
MIN_EVALS = 8


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    unlocks: str
    fix: str = ""


def audit(dump_dir: Path | None, slurm_log: Path | None) -> list[Check]:
    checks: list[Check] = []
    records, source = _load_records(dump_dir, slurm_log, checks)
    if records is None:
        return checks

    metrics_seen = {key for record in records for key in record["metrics"]}
    checks.append(
        Check("metric stream", "PASS", f"{len(records)} records from {source}", "every time series in the study")
    )
    checks += _wall_clock_checks(dump_dir, records)
    checks += _eval_checks(dump_dir, records, metrics_seen)
    checks += _guard_checks(records, metrics_seen)
    checks += _drift_checks(metrics_seen)
    checks += _lag_checks(dump_dir, metrics_seen)
    checks += _telemetry_checks(dump_dir, records, metrics_seen)
    checks += _factor_checks(dump_dir)
    checks += _engine_checks(dump_dir)
    checks += _resume_checks(records)
    return checks


def _load_records(dump_dir, slurm_log, checks) -> tuple[list[dict[str, Any]] | None, str]:
    if dump_dir is not None and (dump_dir / "dashboard" / "metrics.jsonl").is_file():
        path = dump_dir / "dashboard" / "metrics.jsonl"
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        return [json.loads(line) for line in lines], "dashboard/metrics.jsonl"
    if slurm_log is not None and slurm_log.is_file():
        return log_source.merge_step_records(log_source.parse_log(slurm_log)), slurm_log.name
    checks.append(
        Check(
            "metric stream",
            "FAIL",
            "neither dashboard/metrics.jsonl nor a readable --slurm-log",
            "nothing",
            "pass --use-miles-dashboard, or keep the Slurm job log",
        )
    )
    return None, ""


def _wall_clock_checks(dump_dir: Path | None, records: list[dict[str, Any]]) -> list[Check]:
    meta_path = (dump_dir / "dashboard" / "meta.json") if dump_dir else None
    if meta_path and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        args = meta.get("args", {})
        gpus = int(args.get("actor_num_nodes", 0)) * int(args.get("actor_num_gpus_per_node", 0)) + (
            0 if args.get("colocate") else int(args.get("rollout_num_gpus", 0))
        )
        gpu_check = Check(
            "GPU count (GPU-hours axis)",
            "PASS" if gpus else "WARN",
            f"{gpus} GPUs from meta.json" if gpus else "meta.json args snapshot has no usable GPU count",
            "GPU-hours to equivalence, the secondary metric",
            "" if gpus else "pass --total-gpus to extract_run.py",
        )
        return [
            Check("wall-clock origin", "PASS", f"meta.json start_ts {meta['start_ts']:.0f}", "wall-clock speedup"),
            gpu_check,
        ]
    return [
        Check(
            "wall-clock origin",
            "WARN",
            "no meta.json; the origin becomes the first metric line",
            "wall-clock speedup, minus a constant startup offset",
            "compare only against arms measured the same way, or pass --use-miles-dashboard",
        ),
        Check(
            "GPU count (GPU-hours axis)",
            "WARN",
            "no meta.json args snapshot",
            "GPU-hours to equivalence",
            "pass --total-gpus to extract_run.py",
        ),
    ]


def _eval_checks(dump_dir: Path | None, records, metrics_seen) -> list[Check]:
    benchmarks = sorted(
        key.removeprefix("eval/")
        for key in metrics_seen
        if key.startswith("eval/") and "/" not in key.removeprefix("eval/") and "-" not in key and key != "eval/step"
    )
    n_evals = sum(1 for r in records if any(k.startswith("eval/") for k in r["metrics"]))
    checks = [
        Check(
            "held-out evaluations",
            "PASS" if n_evals >= MIN_EVALS else "FAIL" if n_evals < 2 else "WARN",
            f"{n_evals} evaluation points on {', '.join(benchmarks) or 'no benchmark'}",
            "Q(t), the whole protocol",
            "" if n_evals >= MIN_EVALS else f"needs >= {MIN_EVALS} for a plateau test; lower --eval-interval",
        )
    ]

    dumps = sorted((dump_dir / "rollout_data").glob("eval_*.pt")) if dump_dir else []
    checks.append(
        Check(
            "per-prompt eval rewards",
            "PASS" if dumps else "WARN",
            f"{len(dumps)} eval dumps" if dumps else "only the logged scalar mean per benchmark",
            "paired-over-prompts bootstrap; separating prompt from rollout variance",
            (
                ""
                if dumps
                else "pass --dump-details; log_eval_rollout_data logs only the mean, so the "
                "dumps are the only per-prompt record"
            ),
        )
    )
    return checks


def _guard_checks(records, metrics_seen) -> list[Check]:
    entropy = [r["metrics"]["train/entropy_loss"] for r in records if "train/entropy_loss" in r["metrics"]]
    live = any(abs(float(v)) > 0 for v in entropy)
    checks = [
        Check(
            "entropy collapse guard",
            "PASS" if live else "WARN",
            "train/entropy_loss varies" if live else "train/entropy_loss is identically 0",
            "the entropy criterion of the convergence definition",
            (
                ""
                if live
                else "pass --observe-training-entropy; without it "
                "calculate_entropy is False (losses.py:99) and the metric is a constant 0"
            ),
        )
    ]
    for metric, unlocks in (
        ("rollout/raw_reward", "the reward-decline criterion"),
        ("rollout/truncated_ratio", "the truncation guard, and the response-length confound"),
        ("train/kl_loss", "the KL criterion"),
    ):
        checks.append(
            Check(
                f"guard {metric}",
                "PASS" if metric in metrics_seen else "WARN",
                "logged" if metric in metrics_seen else "absent",
                unlocks,
            )
        )
    return checks


def _drift_checks(metrics_seen) -> list[Check]:
    """The off-policy drift axis, which the standard log covers in full."""
    wanted = {
        "train/train_rollout_logprob_abs_diff": "the train/rollout mismatch floor, and drift above it",
        "train/train_rollout_kl": "KL(rollout || train), the distribution-level mismatch",
        "train/ess_ratio": "ESS of the PPO inner-loop weights only; identically 1.0 when "
        "NUM_STEPS_PER_ROLLOUT=1, so it does NOT measure staleness",
        "train/tis": "exp(train - rollout) per token: the importance weight staleness actually moves",
        "train/rollout_token_level_ess": "ESS over tokens within a sequence",
        "train/rollout_sequence_level_ess": "ESS over sequences in the batch -- VCPO eq. 4",
        "train/pg_clipfrac": "how much of the update the clip is discarding",
        "train/tis_abs": "the TIS correction's magnitude, when --use-tis is on",
    }
    return [
        Check(
            f"drift {metric}",
            "PASS" if metric in metrics_seen else "WARN",
            "logged" if metric in metrics_seen else "absent",
            unlocks,
        )
        for metric, unlocks in wanted.items()
    ]


def _lag_checks(dump_dir: Path | None, metrics_seen) -> list[Check]:
    exact = "rollout/fully_async/avg_staleness" in metrics_seen
    distribution = any(
        key in metrics_seen for key in ("rollout/fully_async/staleness_p90", "staleness/bound/rollout/p90")
    )
    checks = [
        Check(
            "realized lag distribution (logged)",
            "PASS" if distribution else "WARN",
            "staleness percentiles logged" if distribution else "only avg/max, no percentiles",
            "P(L) without reading any dump",
            "" if distribution else "requires the staleness-percentile metrics in fully_async_rollout.py",
        ),
        Check(
            "staleness reference version",
            "PASS" if "rollout/fully_async/current_weight_version" in metrics_seen else "WARN",
            (
                "current_weight_version logged"
                if "rollout/fully_async/current_weight_version" in metrics_seen
                else "absent: a missing staleness metric is indistinguishable from a dead router query"
            ),
            "telling 'never stale' apart from 'never measured'",
        ),
        Check(
            "wasted generation (tokens)",
            "PASS" if "rollout/fully_async/wasted_token_frac" in metrics_seen else "WARN",
            (
                "token-level waste logged"
                if "rollout/fully_async/wasted_token_frac" in metrics_seen
                else "only group counts, no token volume"
            ),
            "sample- and token-efficiency claims alongside the wall-clock ones",
        ),
        Check(
            "realized staleness (exact mean)",
            "PASS" if exact else "FAIL",
            "avg_staleness logged" if exact else "avg_staleness never logged",
            "reporting realized lag next to the configured bound",
            (
                ""
                if exact
                else "two independent causes: --max-weight-staleness must be set at all "
                "(use 1000000 for the unbounded arm, not unset), AND the router /model_info query must "
                "succeed -- when it does not, `current` is None, the cap silently enforces nothing and "
                "no staleness metric is emitted. Check the log for "
                "'--max-weight-staleness cannot be enforced'"
            ),
        ),
        Check(
            "version bracket",
            "PASS" if "rollout/weight_version/min" in metrics_seen else "WARN",
            "weight_version min/mean/max logged" if "rollout/weight_version/min" in metrics_seen else "absent",
            "a bracket on the lag when the exact mean is missing",
        ),
    ]
    dumps = [p for p in (dump_dir / "rollout_data").glob("*.pt")] if dump_dir else []
    train_dumps = [p for p in dumps if not p.stem.startswith("eval_")]
    checks.append(
        Check(
            "realized lag distribution P(L)",
            "PASS" if train_dumps else "WARN",
            (
                f"{len(train_dumps)} rollout dumps carry per-sample weight_versions"
                if train_dumps
                else "no rollout dumps; only per-step aggregates"
            ),
            "percentiles and the tail of P(L), which the logged mean/max cannot give",
            "" if train_dumps else "pass --dump-details",
        )
    )
    return checks


def _telemetry_checks(dump_dir: Path | None, records, metrics_seen) -> list[Check]:
    """Audit the additive async-compute diagnostics without redefining legacy lag."""
    useful = {
        "rollout/fully_async/useful_rollout/generated_tokens",
        "rollout/fully_async/useful_rollout/loss_input_tokens",
        "rollout/fully_async/useful_rollout/efficiency",
        "rollout/fully_async/useful_rollout/accounting_error_tokens",
    }
    waste = {
        f"rollout/fully_async/waste/all_discarded/{component}"
        for component in ("decode_tokens", "prefill_uncached_tokens", "tool_env_seconds", "reward_seconds")
    }
    reasons = {
        f"rollout/fully_async/recycle_reason/{reason}/groups"
        for reason in (
            "stale_at_generation_completion",
            "stale_during_reward_finalize",
            "stale_during_queue_backpressure",
            "stale_in_output_queue",
            "actor_weight_sync_overlap",
        )
    } | {
        "rollout/fully_async/recycle_aux/group_straggler_collateral/groups",
        "staleness/late_stale_trained/forward_handoff_groups",
    }
    selection = {f"selection_bias/{population}/samples" for population in ("generated", "consumed", "recycled")}
    prequeue = {
        f"staleness/pre_queue_phase/version/{phase}/sequence_mean"
        for phase in ("active", "group_wait", "postprocess", "total")
    } | {
        "staleness/pre_queue_phase/version/identity_max_abs_error",
        "staleness/pre_queue_phase/exact_sample_frac",
    }
    throughput = {
        "throughput/generated_tokens_per_second",
        "throughput/accepted_tokens_per_second",
        "throughput/optimizer_updates_per_second",
        "throughput/cohort_useful_efficiency",
        "queue/depth_time_mean",
        "queue/trainer_starvation_seconds",
        "queue/rollout_backpressure_seconds",
        "queue/rollout_idle_capacity_seconds",
        "queue/active_group_capacity_time_mean",
        "queue/consumption/wall_wait_seconds/p90",
        "perf/step_time",
    }
    checks = [
        _metric_set_check(
            "strict useful-rollout accounting",
            useful,
            metrics_seen,
            "generated-token efficiency after recycle, trimming, and the final loss mask",
            "run fully async with the additive telemetry build; custom sample converters intentionally mark it unavailable",
        ),
        _metric_set_check(
            "reason-coded waste vector",
            waste,
            metrics_seen,
            "decode/prefill token waste and tool/reward wall-time waste without summing unlike units",
        ),
        _metric_set_check(
            "recycle lifecycle reason codes",
            reasons,
            metrics_seen,
            "generation, reward, backpressure, queue, straggler, sync, and late-forward attribution",
        ),
        _metric_set_check(
            "selection populations",
            selection,
            metrics_seen,
            "P(feature | generated/consumed/recycled) and marginal selection bias",
        ),
        _metric_set_check(
            "pre-queue phase split",
            prequeue,
            metrics_seen,
            "active generation vs group straggler wait vs postprocessing attribution",
        ),
        _metric_set_check(
            "same-window throughput and queueing",
            throughput,
            metrics_seen,
            "whether node-ratio changes are generation-, queue-, trainer-, or sync-limited",
        ),
    ]

    logprob_timers = {
        key
        for key in metrics_seen
        if key
        in {
            "perf/log_probs_time",
            "perf/legacy_actor_log_probs_time",
            "perf/ref_log_probs_time",
        }
    }
    fused_logprobs = "train/fused_one_step_logprobs_enabled" in metrics_seen
    checks.append(
        Check(
            "log-probability forward timing",
            "PASS" if logprob_timers or fused_logprobs else "WARN",
            (
                ", ".join(sorted(logprob_timers))
                if logprob_timers
                else (
                    "fused one-step path: forward is included in perf/actor_train_time"
                    if fused_logprobs
                    else "no log-probability timer or fused-path marker found"
                )
            ),
            "separating an explicit log-probability forward when the algorithm actually runs one",
            "retain the standard perf stream" if not logprob_timers and not fused_logprobs else "",
        )
    )

    accounting_errors = [
        abs(float(record["metrics"]["rollout/fully_async/useful_rollout/accounting_error_tokens"]))
        for record in records
        if "rollout/fully_async/useful_rollout/accounting_error_tokens" in record["metrics"]
    ]
    if accounting_errors:
        maximum = max(accounting_errors)
        checks.append(
            Check(
                "useful-token partition invariant",
                "PASS" if maximum == 0 else "FAIL",
                f"max absolute accounting error {maximum:g} tokens",
                "a defensible denominator for useful rollout efficiency",
                "inspect recycle, postprocess-trim, and loss-mask partitions" if maximum else "",
            )
        )

    effective = any(key.startswith(("train/staleness_gradient/", "staleness_gradient/")) for key in metrics_seen)
    effective_suffixes = (
        "/consumed_sequence_mass",
        "/effective_contribution_mass",
        "/ppo_clip_fraction",
        "/final_mask_fraction",
        "/policy_rollout_ratio_token_ess",
        "/policy_rollout_ratio_sequence_ess",
    )
    missing_effective = [
        suffix for suffix in effective_suffixes if not any(key.endswith(suffix) for key in metrics_seen)
    ]
    checks.append(
        Check(
            "effective-contribution staleness bins",
            "PASS" if effective and not missing_effective else "WARN",
            (
                "consumed/effective mass, clipping, masks, and ESS are logged by lag bin"
                if effective and not missing_effective
                else "feature disabled" if not effective else f"missing suffixes: {', '.join(missing_effective)}"
            ),
            "which staleness region survives the actual objective for TIS/MIS/VCPO comparisons",
            "pass --log-staleness-gradient-metrics" if not effective else "",
        )
    )

    exact_segments = "staleness/token_lag/exact/covered_response_token_frac" in metrics_seen
    checks.append(
        Check(
            "exact response-token weight versions",
            "PASS" if exact_segments else "WARN",
            "exact token coverage is logged" if exact_segments else "only per-request forward-version summaries",
            "token-weighted lag when one response crosses a weight update",
            "pass --sglang-enable-response-weight-version-segments" if not exact_segments else "",
        )
    )

    sync_timers = {
        key
        for key in metrics_seen
        if key
        in {
            "perf/update_weights_implementation_time",
            "perf/finalize_and_resume_engines_time",
            "perf/ref_model_update_time",
        }
    }
    checks.append(
        Check(
            "weight synchronization timing",
            "PASS" if sync_timers else "WARN",
            ", ".join(sorted(sync_timers)) if sync_timers else "no weight-sync timer found",
            "separating actor synchronization delay from generation and queue delay",
            "retain the standard perf timer stream" if not sync_timers else "",
        )
    )

    rollout_dumps = sorted((dump_dir / "rollout_data").glob("[0-9]*.pt")) if dump_dir else []
    policy_dumps = sorted((dump_dir / "policy_loss_debug").glob("*.pt")) if dump_dir else []
    checks.append(
        Check(
            "joinable sample-level selection records",
            "PASS" if rollout_dumps and policy_dumps else "WARN",
            f"{len(rollout_dumps)} rollout dumps, {len(policy_dumps)} policy-loss dumps",
            "P(accepted | length, reward, difficulty) and clip/mask/log-ratio joins",
            (
                "pass --dump-details (policy-loss debug remains feature-gated by that debug mode)"
                if not rollout_dumps or not policy_dumps
                else ""
            ),
        )
    )
    return checks


def _metric_set_check(
    name: str,
    required: set[str],
    metrics_seen: set[str],
    unlocks: str,
    fix: str = "run the current telemetry build",
) -> Check:
    missing = sorted(required - metrics_seen)
    return Check(
        name,
        "PASS" if not missing else "WARN",
        "all required scalar series logged" if not missing else f"missing: {', '.join(missing)}",
        unlocks,
        "" if not missing else fix,
    )


def _factor_checks(dump_dir: Path | None) -> list[Check]:
    """The factorial design's own axes -- the thing meta.json does not record."""
    meta_path = (dump_dir / "dashboard" / "meta.json") if dump_dir else None
    recorded: set[str] = set()
    if meta_path and meta_path.is_file():
        recorded = set(json.loads(meta_path.read_text()).get("args", {}))
    wanted = {
        "max_weight_staleness",
        "num_steps_per_rollout",
        "pause_generation_mode",
        "advantage_estimator",
        "lr",
        "rollout_max_response_len",
        "use_tis",
        "seed",
    }
    missing = sorted(wanted - recorded)
    return [
        Check(
            "factor provenance",
            "WARN" if missing else "PASS",
            f"meta.json does not record: {', '.join(missing)}" if missing else "all study factors recorded",
            "attributing a run to a cell of the design without trusting its directory name",
            "pass --factor K=V (or --manifest) to extract_run.py; the durable fix is adding these "
            "keys to _SNAPSHOT_KEYS in miles/dashboard/args.py",
        )
    ]


def _engine_checks(dump_dir: Path | None) -> list[Check]:
    """sglang-side series, which only the dashboard scraper collects reliably."""
    if dump_dir is None:
        return [
            Check(
                "sglang engine series",
                "WARN",
                "no dump directory",
                "generation latency and queueing, the mechanism behind realized lag",
                "the job log's 'Decode batch, ...' lines are deduplicated by Ray across engines and "
                "are not a usable series; use --use-miles-dashboard",
            )
        ]
    engine_dir = dump_dir / "dashboard" / "engine_series"
    present = engine_dir.is_dir() or (dump_dir / "dashboard" / "engine_series.jsonl").is_file()
    return [
        Check(
            "sglang engine series",
            "PASS" if present else "WARN",
            "scraped" if present else "not collected",
            "TTFT / e2e latency / queue depth: why a given configured lag realizes as it does",
            "" if present else "pass --use-miles-dashboard (scrapes sglang /metrics every 2s)",
        )
    ]


def _resume_checks(records: list[dict[str, Any]]) -> list[Check]:
    """Whether this stream spans several Slurm allocations, and by how much.

    ``submit_training.sh`` documents runs as resumable across three 4 h jobs, so
    this is the normal case rather than the exception -- and it is the one place
    where the *primary* metric can silently go wrong, since queue wait between
    allocations is neither training time nor comparable across arms.
    """
    _, report = log_source.active_elapsed_hours(records)
    resumed = report["n_allocations"] > 1
    return [
        Check(
            "allocation continuity",
            "WARN" if resumed else "PASS",
            f"{report['n_allocations']} allocation(s); {report['active_h']:.2f} active h of "
            f"{report['span_h']:.2f} h span, {report['excluded_h']:.2f} h excluded as queue gaps",
            "the wall-clock axis, which is the study's primary metric",
            (
                ""
                if not resumed
                else "pass every allocation's log to extract_run.py with a repeated "
                "--slurm-log so the replayed steps dedupe correctly; never compute wall-clock as "
                "ts - meta.start_ts, which the dashboard rewrites on each resume (collector.py:153)"
            ),
        )
    ]


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for check in checks:
        lines.append(f"{check.status:<5} {check.name:<{width}}  {check.detail}")
        lines.append(f"{'':<5} {'':<{width}}  -> {check.unlocks}")
        if check.fix:
            lines.append(f"{'':<5} {'':<{width}}  fix: {check.fix}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-details", type=Path, default=None)
    p.add_argument("--slurm-log", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    assert args.dump_details or args.slurm_log, "pass --dump-details, --slurm-log, or both"
    checks = audit(args.dump_details, args.slurm_log)
    print(render(checks))
    failures = sum(1 for c in checks if c.status == "FAIL")
    warnings = sum(1 for c in checks if c.status == "WARN")
    print(f"\n{failures} FAIL, {warnings} WARN, {sum(1 for c in checks if c.status == 'PASS')} PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
