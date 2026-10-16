import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from experiments.src.difficulty_filtering.core.artifacts import write_json
from experiments.src.difficulty_filtering.utils.smoke_gate import verify_smokes

MODELS = ("qwen3-8b", "qwen3-30b-a3b", "qwen3-4b")
DATASETS = ("dapo-math", "nemotron-code", "instruction-following", "nemotron-mcqa")


class Slots:
    """Every PBS job, including CPU merge, continues exactly one of sixteen chains."""

    def __init__(self, anchors):
        if len(anchors) != 16:
            raise ValueError("Exactly sixteen slot anchors are required")
        self.jobs = [anchor["job_id"] for anchor in anchors]
        self.history = [[anchor["job_id"]] if anchor["job_id"] else [] for anchor in anchors]
        self.ready = [anchor["estimate"] for anchor in anchors]
        self.finished = {anchor["job_id"]: anchor["estimate"] for anchor in anchors if anchor["job_id"]}

    def assign(self, job_id, dependencies, cost, *, lane=None):
        if lane is None:
            lane = min(range(16), key=lambda index: (self.ready[index], index))
        success = []
        # A held job can be cancelled before its predecessors finish. Waiting
        # for every prior job prevents that cancellation from freeing this lane.
        # PBS deletes an afterok successor without running it when a shard fails.
        # That deletion also propagates through afterany successors on this site.
        # Merge validates complete journals itself; PBS only orders completion.
        completion = list(dict.fromkeys([*dependencies, *self.history[lane]]))
        finish = max([self.ready[lane], *(self.finished.get(job, 0) for job in [*success, *completion])]) + cost
        self.jobs[lane] = job_id
        self.history[lane].append(job_id)
        self.ready[lane] = finish
        self.finished[job_id] = finish
        return lane, success, completion


def math_anchors(run_dirs, data_root):
    anchors, gates = [], {}
    for model, directory in zip(MODELS[:2], run_dirs, strict=True):
        jobs = [line.split("\t") for line in (directory / "recovery-jobs.tsv").read_text().splitlines()]
        filters = {int(shard): job for action, shard, job in jobs if action == "filter"}
        merge = next(job for action, _, job in jobs if action == "merge")
        if set(filters) != set(range(8)):
            raise ValueError("Existing math run must have eight recovery lanes")
        gates[model] = merge
        config = json.loads((directory / "config.json").read_text())
        measurement = data_root / config["measurement_dir"].removeprefix("/data/")
        estimates = []
        for shard in range(8):
            part = measurement / f"part-{shard:03d}"
            manifest = json.loads((part / "manifest.json").read_text())
            target = manifest["stop"] - manifest["start"]
            journal = part / "pass_rates.jsonl"
            done = sum(line.endswith(b"\n") for line in journal.open("rb")) if journal.exists() else 0
            estimates.append(2 * (target - done) / target)
        # Reserve one old lane for each existing CPU merge. Other seven lanes
        # can start 4B immediately after their own shard, before the math barrier.
        anchors.extend(
            {
                "job_id": merge if shard == 0 else filters[shard],
                "estimate": max(estimates) + 0.01 if shard == 0 else estimates[shard],
            }
            for shard in range(8)
        )
    return anchors, gates


def prepare_recipe(root, model, dataset, prefix, *, submit, resume):
    name = f"{prefix}-{model}-{dataset}"
    directory = root / "experiments/outputs/difficulty_filtering" / name
    command = [
        "bash",
        str(root / "experiments/scripts/difficulty_filtering" / f"{model}-sft" / f"{dataset}.sh"),
        "filter",
    ]
    if submit and not (resume and (directory / "config.json").exists()):
        command.extend(["--submit", "--prepare-only"])
    if resume and (directory / "config.json").exists():
        config = json.loads((directory / "config.json").read_text())
    else:
        result = subprocess.run(
            command, env={**os.environ, "FILTER_RUN_NAME": name}, capture_output=True, text=True, check=True
        )
        config, _ = json.JSONDecoder().raw_decode(result.stdout)
    launch = json.loads((directory / "launch.json").read_text()) if submit else {}
    return {"name": name, "directory": directory, "config": config, "launch": launch}


def enqueue(root, prepared, record):
    launch = prepared["launch"]
    if record["success_dependencies"]:
        raise ValueError("Filtering suites must use completion-only PBS dependencies")
    dependency = "afterany:" + ":".join(record["completion_dependencies"]) if record["completion_dependencies"] else ""
    command = [
        "bash",
        str(root / "experiments/scripts/difficulty_filtering/submit-job.sh"),
        prepared["name"],
        record["action"],
        str(record["shard"]),
        dependency,
        "00:30:00" if record["action"] == "smoke" else launch["walltime"],
        str(launch["smoke_prompts"]),
    ]
    result = subprocess.run(
        command, env={**os.environ, "CONTAINER_IMAGE": launch["image"]}, capture_output=True, text=True, check=True
    )
    job = result.stdout.strip()
    if not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9._-]+)?", job):
        raise ValueError(f"Invalid PBS result: {job}")
    with (prepared["directory"] / "jobs.tsv").open("a") as stream:
        stream.write(f"{record['action']}\t{record['shard']}\t{job}\n")
    return job


def schedule(root, prepared, state, state_path, *, submit, stage="all"):
    if submit and stage == "all":
        raise ValueError("Submission requires separate smoke and full stages")
    slots = Slots(state["anchors"])
    gates = dict(state["math_gates"])
    previous = list(state["jobs"])
    state["jobs"] = []
    verified = set(state.get("verified_smokes", []))

    def add(model, dataset, action, shard, dependencies, cost, lane=None):
        ordinal = len(state["jobs"])
        key = f"planned-{ordinal}"
        lane, success, completion = slots.assign(key, dependencies, cost, lane=lane)
        success = [job for job in success if job not in verified]
        completion = [job for job in completion if job not in verified]
        record = dict(
            model=model,
            dataset=dataset,
            action=action,
            shard=shard,
            lane=lane,
            dependencies=[*success, *completion],
            success_dependencies=success,
            completion_dependencies=completion,
        )
        if ordinal < len(previous):
            saved = previous[ordinal]
            if record != {name: value for name, value in saved.items() if name != "job_id"}:
                raise ValueError("Saved submission plan differs; cannot resume")
            job = saved["job_id"]
        elif submit:
            job = enqueue(root, prepared[model, dataset], record)
        else:
            job = str(999000 + ordinal)
        slots.jobs[lane] = job
        slots.history[lane][-1] = job
        slots.finished[job] = slots.finished.pop(key)
        state["jobs"].append({**record, "job_id": job})
        if submit:
            write_json(state_path, state)
        print(f"slot={lane:02d} {model} {dataset} {action}/{shard}: {job}", flush=True)
        return job

    smokes = {}
    for dataset in DATASETS:
        for model in MODELS:
            if (model, dataset) in prepared:
                # Validate each independent model/dataset as soon as a slot is free.
                smokes[model, dataset] = add(model, dataset, "smoke", 0, [], 0.05)

    if stage == "smoke":
        return
    if stage == "full" and set(smokes.values()) != verified:
        raise ValueError("Full submission requires every smoke to be verified")

    for dataset in DATASETS:
        models = [model for model in MODELS if (model, dataset) in prepared]
        jobs = {model: [] for model in models}
        shard_counts = {model: prepared[model, dataset]["config"]["num_shards"] for model in models}
        for shard in range(max(shard_counts.values(), default=0)):
            for model in models:
                if shard < shard_counts[model]:
                    # Relative scheduling weights spread short 4B work into free lanes;
                    # only PBS dependencies enforce capacity, never these estimates.
                    cost = (0.5 if model == "qwen3-4b" else 1.0) * (4 if dataset == "nemotron-mcqa" else 1)
                    jobs[model].append(add(model, dataset, "filter", shard, [smokes[model, dataset]], cost))
        for model in models:
            last = max(jobs[model], key=slots.finished.__getitem__)
            lane = next(record["lane"] for record in state["jobs"] if record["job_id"] == last)
            gates[model] = add(model, dataset, "merge", 0, jobs[model], 0.01, lane=lane)
    state["final_merges"] = gates
    if submit:
        write_json(state_path, state)
    print("Final merges: " + json.dumps(gates), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Queue 4B/8B/30B filtering within sixteen shared node slots")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted submission from its saved plan")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument(
        "--slot-anchors", type=Path, help="JSON list of 16 existing slot predecessors; null job IDs are free"
    )
    parser.add_argument("--prefix", default="multi-filter-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--math-runs",
        nargs=2,
        type=Path,
        metavar=("RUN_8B", "RUN_30B"),
        help="Existing eight-shard math run directories with recovery-jobs.tsv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("RL_DATASET_DIR", "/groups/gai51740/kazuki_fujii/datasets/rl")),
    )
    args = parser.parse_args()
    args.datasets = list(args.datasets)
    if args.slot_anchors and args.math_runs:
        parser.error("--slot-anchors and --math-runs cannot be combined")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.prefix):
        parser.error("Invalid prefix")
    root = Path(__file__).resolve().parents[4]
    state_path = root / "experiments/outputs/difficulty_filtering" / (args.prefix + "-plan.json")
    if state_path.exists():
        if not args.resume:
            parser.error("Plan already exists; use --resume to continue the same submission")
        state = json.loads(state_path.read_text())
        if state.get("schema_version") != 4 or state.get("datasets") != args.datasets:
            parser.error("Use the same datasets and a current staged plan; legacy plans cannot be resumed")
        if state.get("repair"):
            parser.error("This plan was repaired; use repair_suite for subsequent recovery")
    else:
        anchors, gates = (
            math_anchors(args.math_runs, args.data_root)
            if args.math_runs
            else ([{"job_id": None, "estimate": 0}] * 16, {})
        )
        if args.slot_anchors:
            anchors = json.loads(args.slot_anchors.read_text())
        Slots(anchors)
        state = {"schema_version": 4, "datasets": args.datasets, "anchors": anchors, "math_gates": gates, "jobs": []}
    if args.stage == "full" and not state["jobs"]:
        parser.error("Submit and validate the smoke stage before full submission")
    if args.stage == "smoke" and any(record["action"] != "smoke" for record in state["jobs"]):
        parser.error("Full submission already started; resume with --stage full")
    prepared = {
        (model, dataset): prepare_recipe(root, model, dataset, args.prefix, submit=args.submit, resume=args.resume)
        for dataset in DATASETS
        if dataset in args.datasets
        for model in MODELS
        if not (dataset == "dapo-math" and model in state["math_gates"])
    }
    if args.stage == "full" and not state.get("verified_smokes"):
        receipt = verify_smokes(prepared, state["jobs"], args.data_root)
        state["verified_smokes"] = receipt["job_ids"]
        if args.submit:
            write_json(state_path.with_name(args.prefix + "-smoke-validation.json"), receipt)
            write_json(state_path, state)
    elif args.stage == "full":
        receipt = json.loads(state_path.with_name(args.prefix + "-smoke-validation.json").read_text())
        if set(receipt["job_ids"]) != set(state["verified_smokes"]) or set(receipt["outputs"]) != {
            "/".join(key) for key in prepared
        }:
            parser.error("Saved smoke validation does not match this suite")
    schedule(root, prepared, state, state_path, submit=args.submit, stage=args.stage)
    print(f"Plan: {state_path}", flush=True)


if __name__ == "__main__":
    main()
