"""Preview eight Lightning node ratios at 32/32 and 64/64 serving limits (16 arms).

    python -m experiments.staleness.lightning_node_ratio --plan-file /tmp/lightning-plan.json

Defaults come from the async recipe's --print-config path. --set KEY=VALUE
changes a shared setting; --point S:T:R selects a node ratio. Repeat
--serving-profile REQUESTS:GRAPH to select serving limits (default both pairs).
Without --submit this never calls Slurm. Each arm has one 32-node, four-hour allocation.
"""

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from experiments.sweep import resolve_batch_shape

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "experiments/scripts/math/async/dapo-math-17k/nemotron-3.5-lightning"
LOG_ROOT = REPO / "experiments/outputs/lightning-node-ratio"
SLURM_LOG_ROOT = REPO / "experiments/outputs/nemotron-3.5-lightning/math/train-rollout-ratio"
POINTS = tuple((8, train, 32 - train) for train in range(4, 17, 2)) + ((1, 16, 16),)
SERVING_PROFILES = ((32, 32), (64, 64))
MANAGED = {
    "ACTOR_NUM_NODES",
    "ROLLOUT_NUM_GPUS",
    "MAX_WEIGHT_STALENESS",
    "CONFIG_TAG",
    "RUN_NAME",
    "CKPT_PATH",
    "SGLANG_MAX_RUNNING_REQUESTS",
    "SGLANG_CUDA_GRAPH_MAX_BS",
}
VARYING = MANAGED | {"ASYNC_MAX_CONCURRENT_SAMPLES"}
SOURCES = (
    RECIPE / "run.sbatch",
    RECIPE / "train.sh",
    Path(__file__).resolve(),
    REPO / "experiments/common/placement.sh",
    REPO / "experiments/common/staleness_buffer_policy.sh",
    REPO / "experiments/common/ray_cluster.sh",
    REPO / "experiments/env.sh",
    REPO / "experiments/src/math/lightning_sglang_compat.py",
    REPO / "experiments/container/apply_oci_sglang_provenance.sh",
    REPO / "experiments/container/patches/sglang-a8e5c63-prefill.patch",
    REPO / "miles/utils/arguments.py",
    REPO / "miles/backends/training_utils/loss.py",
    REPO / "miles/backends/training_utils/loss_hub/losses.py",
    REPO / "miles/backends/training_utils/loss_hub/sequence_filter.py",
    REPO / "miles/backends/training_utils/log_utils.py",
    REPO / "miles/backends/megatron_utils/actor.py",
    REPO / "miles/backends/training_utils/replay_data.py",
    REPO / "miles/utils/replay_base.py",
)


def resolve_config(overrides: dict[str, str]) -> dict[str, str]:
    # Do not inherit another experiment's learning, placement, or debug settings.
    env = {k: os.environ[k] for k in ("HOME", "USER", "PATH") if k in os.environ}
    env.update(overrides, MILES_REPO=str(REPO))
    result = subprocess.run(
        ["bash", str(RECIPE / "run.sbatch"), "--print-config"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"Invalid recipe settings: {result.stderr.strip() or 'recipe invariant failed'}")
    config = json.loads(result.stdout)
    if unknown := overrides.keys() - config.keys():
        raise ValueError(f"Unknown recipe settings: {sorted(unknown)}")
    if error := resolve_batch_shape(config, RECIPE):
        raise ValueError(error)
    if any("," in value or "\n" in value or "\0" in value for value in config.values()):
        raise ValueError("Recipe values cannot contain commas, newlines, or NUL in Slurm --export")
    return config


def source_hashes() -> dict[str, str]:
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}


def make_plan(
    namespace: str,
    points: list[tuple[int, int, int]],
    overrides: dict[str, str],
    *,
    serving_profiles: list[tuple[int, int]] | None = None,
) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", namespace):
        raise ValueError("Namespace must be 1-64 letters, digits, dots, underscores, or dashes")
    if not points or len(set(points)) != len(points) or any(point not in POINTS for point in points):
        raise ValueError("Select distinct points from s8 T:R=4:28..16:16, or s1 T:R=16:16")
    profiles = list(SERVING_PROFILES) if serving_profiles is None else serving_profiles
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in SERVING_PROFILES for p in profiles):
        raise ValueError("Select distinct serving profiles from 32:32 and 64:64")
    if forbidden := MANAGED.intersection(overrides):
        raise ValueError(f"Use --point/--serving-profile/--namespace for sweep-owned settings: {sorted(forbidden)}")
    arms = []
    for (requests, graph), (stale, train, rollout) in product(profiles, points):
        tag = f"{namespace}-s{stale}-t{train}r{rollout}-req{requests}-cg{graph}"
        config = resolve_config(
            dict(
                overrides,
                ACTOR_NUM_NODES=str(train),
                ROLLOUT_NUM_GPUS=str(4 * rollout),
                MAX_WEIGHT_STALENESS=str(stale),
                SGLANG_MAX_RUNNING_REQUESTS=str(requests),
                SGLANG_CUDA_GRAPH_MAX_BS=str(graph),
                CONFIG_TAG=tag,
                RUN_NAME=tag,
            )
        )
        checkpoint = Path(config["TRAIN_CKPT_DIR"]) / config["CKPT_PATH"].removeprefix("/ckpt/training/")
        arms.append(
            {
                "point": [stale, train, rollout],
                "serving_profile": [requests, graph],
                "env": config,
                "checkpoint": str(checkpoint),
                "dashboard": str(checkpoint / "dump/dashboard"),
            }
        )
    return {
        "schema": 2,
        "namespace": namespace,
        "nodes": 32,
        "walltime": "04:00:00",
        "allocations_per_arm": 1,
        "source_sha256": source_hashes(),
        "arms": arms,
    }


def command_for(arm: dict) -> list[str]:
    config = dict(arm["env"], MILES_REPO=str(REPO))
    return [
        "sbatch",
        "--parsable",
        "--account=nemotron_sw_post",
        "--partition=batch",
        "--qos=normal",
        "--nodes=32",
        "--time=04:00:00",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        f"--job-name={config['RUN_NAME']}",
        f"--output={SLURM_LOG_ROOT}/%x-%j.log",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in sorted(config.items())),
        str(RECIPE.relative_to(REPO) / "run.sbatch"),
    ]


def print_plan(plan: dict) -> None:
    print(f"{plan['namespace']}: {len(plan['arms'])} single-allocation jobs, 32 nodes each, 04:00:00, batch/normal")
    print("stale  train:rollout  train GPUs  rollout GPUs  DP  GBS/DP  engines  requests/graph  admission")
    for arm in plan["arms"]:
        stale, train, rollout = arm["point"]
        env = arm["env"]
        dp = train * 4 // (int(env["TENSOR_PARALLEL_SIZE"]) * int(env["CONTEXT_PARALLEL_SIZE"]))
        print(
            f"{stale:5}  {train:2}:{rollout:<2}          {train * 4:3}           {rollout * 4:3}  {dp:2}"
            f"  {int(env['GLOBAL_BATCH_SIZE']) // dp:6}  {rollout * 4 // int(env['ROLLOUT_NUM_GPUS_PER_ENGINE']):7}"
            f"  {env['SGLANG_MAX_RUNNING_REQUESTS']:>5}/{env['SGLANG_CUDA_GRAPH_MAX_BS']:<5}"
            f"  {env['ASYNC_MAX_CONCURRENT_SAMPLES']:>9}"
        )
    print("Shared resolved settings (placement, staleness, serving profile, and identity vary above):")
    print(json.dumps({k: v for k, v in plan["arms"][0]["env"].items() if k not in VARYING}, indent=2))
    print(
        "Megatron/HF save every "
        + plan["arms"][0]["env"]["SAVE_INTERVAL"]
        + "/"
        + plan["arms"][0]["env"]["HF_SAVE_INTERVAL"]
        + " updates; in-run eval OFF; local dashboard and W&B ON."
    )
    for arm in plan["arms"]:
        print(f"  {arm['env']['CONFIG_TAG']} -> {arm['checkpoint']}")


def submit_plan(plan: dict, *, max_jobs: int, plan_file: Path) -> None:
    if (plan["schema"], plan["nodes"], plan["walltime"], plan["allocations_per_arm"]) != (2, 32, "04:00:00", 1):
        raise ValueError("Expected a version-2 plan with serving profiles; regenerate and review the plan")
    if plan["source_sha256"] != source_hashes():
        raise ValueError("Launcher sources changed since this plan was saved; regenerate and review the plan")
    if len(plan["arms"]) > max_jobs:
        raise ValueError(f"{len(plan['arms'])} arms exceeds --max-jobs={max_jobs}")
    points = [tuple(arm["point"]) for arm in plan["arms"]]
    profiles = [tuple(arm["serving_profile"]) for arm in plan["arms"]]
    identities = list(zip(points, profiles, strict=True))
    if (
        not points
        or len(set(identities)) != len(identities)
        or any(p not in POINTS for p in points)
        or any(p not in SERVING_PROFILES for p in profiles)
    ):
        raise ValueError("Plan contains an invalid or duplicate arm")
    if len({arm["checkpoint"] for arm in plan["arms"]}) != len(points):
        raise ValueError("Every arm needs a different checkpoint path")
    shared = [{k: v for k, v in arm["env"].items() if k not in VARYING} for arm in plan["arms"]]
    if any(config != shared[0] for config in shared[1:]):
        raise ValueError("All arms must share settings except placement, staleness, serving profile, and run identity")
    for profile in set(profiles):
        admissions = {
            arm["env"]["ASYNC_MAX_CONCURRENT_SAMPLES"]
            for arm in plan["arms"]
            if tuple(arm["serving_profile"]) == profile
        }
        if len(admissions) != 1:
            raise ValueError("All node ratios within a serving profile must share the admission limit")
    # Validate every arm before submitting any. An existing path is not a fresh comparison.
    for arm in plan["arms"]:
        config = resolve_config(arm["env"])
        stale, train, rollout = arm["point"]
        if (
            int(config["MAX_WEIGHT_STALENESS"]),
            int(config["ACTOR_NUM_NODES"]),
            int(config["ROLLOUT_NUM_GPUS"]) // 4,
        ) != (stale, train, rollout):
            raise ValueError("Point and resolved placement disagree")
        if [int(config[k]) for k in ("SGLANG_MAX_RUNNING_REQUESTS", "SGLANG_CUDA_GRAPH_MAX_BS")] != arm[
            "serving_profile"
        ]:
            raise ValueError("Serving profile and resolved limits disagree")
        checkpoint = Path(config["TRAIN_CKPT_DIR"]) / config["CKPT_PATH"].removeprefix("/ckpt/training/")
        if str(checkpoint) != arm["checkpoint"] or checkpoint.exists():
            raise ValueError(f"Checkpoint must be a fresh path: {checkpoint}")
        parent = checkpoint.parent
        while not parent.exists():
            parent = parent.parent
        if not os.access(parent, os.W_OK):
            raise ValueError(f"Checkpoint parent is not writable: {parent}")
        for asset in (
            Path(config["SQSH_IMAGE"]),
            Path(config["HF_CKPT_DIR"]) / config["HF_MODEL_NAME"] / ".download_complete",
            Path(config["MEGATRON_CKPT_DIR"]) / f"{config['MODEL_NAME']}_torch_dist/latest_checkpointed_iteration.txt",
            Path(config["DATASET_DIR"]) / "dapo-math-17k/dapo-math-17k.jsonl",
        ):
            if not asset.is_file():
                raise ValueError(f"Missing asset: {asset}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    SLURM_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = LOG_ROOT / f"{plan['namespace']}-submitted.jsonl"
    # Exclusive creation prevents accidentally submitting the same plan twice.
    with manifest.open("x") as output:
        for arm in plan["arms"]:
            result = subprocess.run(command_for(arm), cwd=REPO, capture_output=True, text=True)
            record = dict(arm, plan_file=str(plan_file.resolve()), returncode=result.returncode)
            if result.returncode:
                record["error"] = result.stderr.strip()
            else:
                record["job_id"] = result.stdout.strip().split(";")[0]
            output.write(json.dumps(record) + "\n")
            output.flush()
            if result.returncode:
                raise RuntimeError(f"Submission stopped: {record['error']}; see {manifest}")
            print(f"Submitted {record['job_id']}: {arm['env']['CONFIG_TAG']}")
    print(f"Submission manifest: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", type=Path, help="write a review plan, or read it with --submit")
    parser.add_argument("--namespace")
    parser.add_argument("--point", action="append", help="select S:T:R, repeatable; default all eight")
    parser.add_argument(
        "--serving-profile",
        action="append",
        choices=["32:32", "64:64"],
        help="max running requests:CUDA graph max batch size; repeatable; default both",
    )
    parser.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--submit", action="store_true", help="submit the saved plan; never implied")
    parser.add_argument("--max-jobs", type=int, default=16)
    args = parser.parse_args()
    try:
        if args.submit:
            if args.plan_file is None or args.namespace or args.point or args.serving_profile or args.settings:
                raise ValueError(
                    "--submit requires --plan-file and takes no new point/profile/setting/namespace overrides"
                )
            plan = json.loads(args.plan_file.read_text())
            print_plan(plan)
            submit_plan(plan, max_jobs=args.max_jobs, plan_file=args.plan_file)
            return
        namespace = args.namespace or (
            "lightning-ratio-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        )
        overrides = dict(item.split("=", 1) for item in args.settings)
        points = [tuple(map(int, item.split(":"))) for item in args.point] if args.point else list(POINTS)
        profiles = (
            [tuple(map(int, item.split(":"))) for item in args.serving_profile] if args.serving_profile else None
        )
        plan = make_plan(namespace, points, overrides, serving_profiles=profiles)
        if len(plan["arms"]) > args.max_jobs:
            raise ValueError(f"{len(plan['arms'])} arms exceeds --max-jobs={args.max_jobs}")
        print_plan(plan)
        if args.plan_file:
            args.plan_file.parent.mkdir(parents=True, exist_ok=True)
            args.plan_file.write_text(json.dumps(plan, indent=2) + "\n")
            print(f"Review plan: {args.plan_file}")
        print("Preview only: no Slurm commands were executed. GPU qualification of this async recipe is pending.")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
