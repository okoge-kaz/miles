"""Move a runtime SQSH after its Slurm consumers finish, preserving its inode.

Run with a JSON specification after an afterany dependency. Refuse to replace
an unrelated destination or move an image still referenced by an active job.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def active_consumers(spec: dict) -> list[str]:
    result = subprocess.run(
        ["squeue", "--noheader", "--user", spec["user"], "--format=%A"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    blockers = []
    for job_id in set(result.stdout.split()):
        if job_id == os.environ.get("SLURM_JOB_ID"):
            continue
        if job_id in spec["wait_for_jobs"]:
            blockers.append(job_id)
            continue
        job = subprocess.run(
            ["scontrol", "show", "job", job_id, "--oneliner"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if any(path in job.stdout for path in spec["source_aliases"]):
            blockers.append(job_id)
    return sorted(blockers)


def relocate(spec: dict) -> dict:
    if blockers := active_consumers(spec):
        raise RuntimeError(f"Image still referenced by active jobs: {blockers}")
    source, destination = Path(spec["source"]), Path(spec["destination"])
    if source.is_symlink():
        raise ValueError("Source must be the original regular image")
    stat = source.stat()
    identity = {key: getattr(stat, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")}
    if identity != spec["source_identity"]:
        raise ValueError("Source changed since the relocation was scheduled")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.stat().st_dev != stat.st_dev:
        raise ValueError("Expected a rename within the same filesystem")
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError("Destination is an unrelated symlink")
    elif destination.exists():
        raise FileExistsError(destination)
    # Replace the temporary new-path alias with the original inode atomically.
    os.replace(source, destination)
    assert destination.stat().st_ino == stat.st_ino
    return dict(spec, status="complete", completed_at=datetime.now(timezone.utc).isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.specification.read_text())
    result = relocate(spec)
    receipt = args.specification.with_suffix(".completed.json")
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Moved {spec['source']} -> {spec['destination']}; receipt: {receipt}", flush=True)


if __name__ == "__main__":
    main()
