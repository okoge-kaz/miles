import json
import subprocess
from itertools import islice

from experiments.src.difficulty_filtering.core.artifacts import sha256_file
from experiments.src.difficulty_filtering.core.config import load_config
from experiments.src.difficulty_filtering.core.records import validate_record


def verify_output(prepared, data_root):
    directory = prepared["directory"]
    config = load_config(directory / "config.json")
    launch = json.loads((directory / "launch.json").read_text())
    expected = launch["smoke_prompts"]
    part = data_root / config.measurement_dir.removeprefix("/data/") / "part-000"
    manifest = json.loads((part / "manifest.json").read_text())["measurement"]
    if manifest["config"] != config.measurement_settings():
        raise ValueError(f"Smoke configuration differs: {directory}")
    for relative, digest in manifest["source_sha256"].items():
        if sha256_file(directory / "source" / relative) != digest:
            raise ValueError(f"Smoke source changed: {directory}/{relative}")
    source = data_root / config.prompt_data.removeprefix("/data/")
    with source.open() as stream:
        rows = [json.loads(line) for line in islice(stream, expected)]
    with (part / "pass_rates.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    indices = [record["index"] for record in records]
    if len(rows) != expected or sorted(indices) != list(range(expected)):
        raise ValueError(f"Smoke must cover exactly the first {expected} prompts: {part}")
    for record in records:
        validate_record(record, rows[record["index"]], config)
    progress = json.loads((part / "progress.json").read_text())
    summary = json.loads((part / "summary.json").read_text())
    if summary["total"] != expected or progress["total"] != expected or not summary["scored_groups"]:
        raise ValueError(f"Smoke did not finish scoring: {part}")
    failures = summary.get("failure_kinds", {})
    if set(failures) - {"execution_time_limit"}:
        raise ValueError(f"Verifier infrastructure failure: {part}: {failures}")
    with (part / "audit.jsonl").open() as stream:
        audits = [json.loads(line) for line in stream]
    if sorted(row["index"] for row in audits) != list(range(min(expected, config.audit_prompts))):
        raise ValueError(f"Smoke audit coverage differs: {part}")
    unique = {}
    for row in audits:
        if len(row["responses"]) != config.n_samples:
            raise ValueError(f"Incomplete audit responses: {part}")
        unique[row["index"]] = len(set(row["responses"]))
    return {"measurement_dir": str(part), "summary": summary, "unique_responses": unique}


def verify_smokes(prepared, records, data_root):
    smokes = {(row["model"], row["dataset"]): row for row in records if row["action"] == "smoke"}
    if set(smokes) != set(prepared):
        raise ValueError("Every selected model/dataset needs a submitted smoke job")
    job_ids = [row["job_id"] for row in smokes.values()]
    result = subprocess.run(
        ["qstat", "-x", "-f", "-F", "json", *job_ids], capture_output=True, text=True, check=True, timeout=60
    )
    jobs = json.loads(result.stdout)["Jobs"]
    outputs = {}
    for key, record in smokes.items():
        job = jobs.get(record["job_id"], {})
        if job.get("job_state") != "F" or job.get("Exit_status") != 0:
            raise ValueError(f"Smoke has not completed successfully: {record['job_id']}")
        outputs["/".join(key)] = {
            "job_id": record["job_id"],
            "exit_status": job["Exit_status"],
            **verify_output(prepared[key], data_root),
        }
    return {"job_ids": job_ids, "outputs": outputs}
