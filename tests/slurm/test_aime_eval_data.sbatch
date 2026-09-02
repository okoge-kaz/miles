#!/bin/bash
#PBS -N test-aime-eval
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:15:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

python3 -m pytest -q \
    tests/fast/experiments/test_aime_eval_data.py \
    tests/fast/experiments/test_reasoning_eval_data_marker.py
python3 - <<"PY"
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_root = Path("/data/evaluation/nemo-skills-26.03")
outputs = {
    "aime24": Path("/data/aime-2024/aime-2024.jsonl"),
    "aime25": Path("/data/aime-2025/aime-2025.jsonl"),
    "aime26": Path("/data/aime-2026/aime-2026.jsonl"),
}
for benchmark, data_path in outputs.items():
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 30, (benchmark, len(rows))
    record_ids = set()
    for index, row in enumerate(rows):
        assert set(row) == {"label", "metadata", "prompt"}, (benchmark, index)
        assert isinstance(row["label"], str) and row["label"], (benchmark, index)
        assert isinstance(row["prompt"], list) and len(row["prompt"]) == 1
        assert row["prompt"][0]["role"] == "user"
        assert "\\boxed" in row["prompt"][0]["content"]
        metadata = row["metadata"]
        assert metadata["benchmark"] == benchmark
        assert metadata["source"] == "nemo-skills-26.03"
        assert metadata["verifier"] == "math"
        assert metadata["rm_type"] == "math"
        assert metadata["eval_only"] is True
        record_id = metadata["record_id"]
        assert record_id and record_id not in record_ids, (benchmark, record_id)
        record_ids.add(record_id)

    provenance_path = data_path.with_name(f"{data_path.name}.provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_path = source_root / provenance["source"]
    assert provenance["adapter"] == "experiments.tools.reasoning_eval.export_miles_aime"
    assert provenance["benchmark"] == benchmark
    assert provenance["records"] == 30
    assert provenance["output_sha256"] == sha256(data_path)
    assert provenance["source_sha256"] == sha256(source_path)
print("AIME24/25/26 artifacts: 30 canonical eval-only math rows each with SHA256 provenance")
PY
'
