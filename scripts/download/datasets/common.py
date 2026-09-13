"""Download and validate pinned prompt/label JSONL datasets."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def stage_dataset(root: Path, name: str, revision: str) -> dict:
    destination = root / name
    snapshot_download(repo_id=f"zhuzilin/{name}", repo_type="dataset", revision=revision, local_dir=destination)
    path = destination / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        assert row["prompt"] and row["label"] not in (None, ""), row
        assert all(message["content"].strip() for message in row["prompt"]), row
    report = {
        "repo_id": f"zhuzilin/{name}",
        "revision": revision,
        "rows": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "difficulty_filtering": False,
    }
    (destination / "staging_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report
