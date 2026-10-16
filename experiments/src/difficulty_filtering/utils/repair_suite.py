import argparse
import json
import shutil
import subprocess
from pathlib import Path

from experiments.src.difficulty_filtering.core.artifacts import file_lock, write_json
from experiments.src.difficulty_filtering.utils.submit_suite import Slots, enqueue, prepare_recipe


def read_jobs(job_ids):
    jobs = {}
    fields = ("job_state", "Exit_status", "Hold_Types", "Resource_List", "exec_host", "depend", "stime")
    for start in range(0, len(job_ids), 24):
        batch = job_ids[start : start + 24]
        result = subprocess.run(
            ["qstat", "-x", "-f", "-F", "json", *batch], capture_output=True, text=True, check=True, timeout=60
        )
        for job_id, value in json.loads(result.stdout)["Jobs"].items():
            jobs[job_id] = {field: value.get(field) for field in fields}
    if set(job_ids) - jobs.keys():
        raise ValueError("PBS history is incomplete; refuse to guess which jobs are running")
    return jobs


def logical_key(record):
    return record["model"], record["dataset"], record["action"], record["shard"]


def build_repair(original, jobs, run_prefix):
    records = original["jobs"]
    if len({logical_key(row) for row in records}) != len(records):
        raise ValueError("Submission plan contains duplicate logical tasks")
    smokes = {row["job_id"] for row in records if row["action"] == "smoke"}
    if smokes != set(original.get("verified_smokes", [])):
        raise ValueError("Repair requires the previously validated smoke stage")
    anchors = [{"job_id": None, "estimate": 0} for _ in range(16)]
    retained, pending, retire = [], [], []
    for row in records:
        job = jobs[row["job_id"]]
        status = job["job_state"]
        if status in {"R", "E"}:
            if anchors[row["lane"]]["job_id"] or int(job["Resource_List"]["nodect"]) != 1:
                raise ValueError("Expected at most one running one-node job per lane")
            anchors[row["lane"]] = {"job_id": row["job_id"], "estimate": 1}
            retained.append(row)
        elif status == "F" and job.get("Exit_status") == 0:
            retained.append(row)
        elif status in {"H", "Q", "F"} and row["action"] != "smoke":
            pending.append(row)
            if status != "F":
                retire.append(row["job_id"])
        else:
            raise ValueError(f"Cannot repair job in state {status}: {row['job_id']}")
    slots = Slots(anchors)
    completed = {row["job_id"] for row in retained if jobs[row["job_id"]]["job_state"] == "F"}
    shards = {}
    for row in retained:
        if row["action"] == "filter":
            shards.setdefault((row["model"], row["dataset"]), []).append(row["job_id"])
    planned = []
    for index, old in enumerate(pending):
        key = old["model"], old["dataset"]
        dependencies = shards.get(key, []) if old["action"] == "merge" else []
        alias = f"repair-{index}"
        lane, success, completion = slots.assign(alias, dependencies, 1, lane=old["lane"])
        completion = [job for job in completion if job not in completed]
        row = {
            **{name: old[name] for name in ("model", "dataset", "action", "shard")},
            "lane": lane,
            "dependencies": completion,
            "success_dependencies": success,
            "completion_dependencies": completion,
            "alias": alias,
            "previous_job_id": old["job_id"],
        }
        planned.append(row)
        if old["action"] == "filter":
            shards.setdefault(key, []).append(alias)
    return {
        "schema_version": 4,
        "repair": True,
        "run_prefix": run_prefix,
        "datasets": original["datasets"],
        "verified_smokes": original["verified_smokes"],
        "anchors": anchors,
        "jobs": retained,
        "pending": planned,
        "retire": retire,
        "replacements": {},
        "phase": "prepared",
    }


def retire_waiting(state, path):
    if not state["retire"]:
        state["phase"] = "submitting"
        write_json(path, state)
        return
    current = read_jobs(state["retire"])
    waiting = []
    for job_id, job in current.items():
        if job["job_state"] == "F":
            continue
        if job["job_state"] != "H" or "u" not in (job["Hold_Types"] or ""):
            raise ValueError(f"Hold waiting jobs before repair; never delete a running job: {job_id}")
        waiting.append(job_id)
    # Deletion may itself cancel other held jobs in the legacy graph. Check the
    # final PBS states instead of treating those already-deleted IDs as failures.
    for start in range(0, len(waiting), 24):
        result = subprocess.run(["qdel", *waiting[start : start + 24]], capture_output=True, text=True, timeout=60)
        state.setdefault("delete_results", []).append(
            {"job_ids": waiting[start : start + 24], "returncode": result.returncode, "stderr": result.stderr}
        )
        write_json(path, state)
    after = read_jobs(state["retire"])
    if any(job["job_state"] != "F" or job.get("stime") for job in after.values()):
        raise ValueError("Waiting jobs were not safely retired; no replacements submitted")
    state["phase"] = "submitting"
    write_json(path, state)


def submit_repair(root, state, path):
    prepared = {}
    for row in state["pending"]:
        alias = row["alias"]
        if alias in state["replacements"]:
            continue
        key = row["model"], row["dataset"]
        if key not in prepared:
            prepared[key] = prepare_recipe(root, *key, state["run_prefix"], submit=True, resume=True)
        dependencies = [state["replacements"].get(job, job) for job in row["dependencies"]]
        if any(job.startswith("repair-") for job in dependencies):
            raise ValueError("Repair plan is not topologically ordered")
        record = {
            **{name: row[name] for name in ("model", "dataset", "action", "shard", "lane")},
            "dependencies": dependencies,
            "completion_dependencies": dependencies,
            "success_dependencies": [],
        }
        log = prepared[key]["directory"] / f"{row['action']}-{row['shard']}.ou"
        backup = log.with_name(f"{log.stem}.{row['previous_job_id']}.ou")
        if log.exists() and not backup.exists():
            shutil.copy2(log, backup)
        job_id = enqueue(root, prepared[key], record)
        state["replacements"][alias] = job_id
        state["jobs"].append({**record, "job_id": job_id, "previous_job_id": row["previous_job_id"]})
        write_json(path, state)
        print(f"Repaired {row['previous_job_id']} -> {job_id}: {key} {row['action']}/{row['shard']}", flush=True)
    state["phase"] = "submitted"
    write_json(path, state)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild held/failed filtering jobs behind the existing sixteen lanes"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    with file_lock(args.output.with_suffix(".lock")):
        if args.output.exists():
            if not args.resume:
                parser.error("Repair plan already exists; use --resume")
            state = json.loads(args.output.read_text())
            if state["source_plan"] != str(args.plan.resolve()):
                parser.error("Resume requires the original source plan")
        else:
            original = json.loads(args.plan.read_text())
            prefix = original.get("run_prefix", args.plan.name.removesuffix("-plan.json"))
            jobs = read_jobs([row["job_id"] for row in original["jobs"]])
            state = build_repair(original, jobs, prefix)
            state["source_plan"] = str(args.plan.resolve())
            state["observed_jobs"] = jobs
            write_json(args.output, state)
        print(
            f"Retained={len(state['jobs'])}; pending={len(state['pending'])}; retire={len(state['retire'])}",
            flush=True,
        )
        if not args.submit:
            return
        if state["phase"] == "prepared":
            retire_waiting(state, args.output)
        if state["phase"] == "submitting":
            submit_repair(root, state, args.output)


if __name__ == "__main__":
    main()
