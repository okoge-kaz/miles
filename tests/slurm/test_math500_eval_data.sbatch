#!/bin/bash
#SBATCH --job-name=test-math500-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

python3 -m pytest -q tests/fast/experiments/test_math500_eval_data.py
python3 - <<"PY"
import json
from pathlib import Path

data_path = Path("/data/math-500/math-500.jsonl")
provenance_path = Path("/data/math-500/math-500.provenance.json")
rows = [
    json.loads(line)
    for line in data_path.read_text(encoding="utf-8").splitlines()
    if line
]
assert len(rows) == 500, len(rows)
record_ids = set()
for index, row in enumerate(rows):
    assert set(row) == {"label", "metadata", "prompt"}, (index, set(row))
    assert isinstance(row["label"], str) and row["label"], index
    assert isinstance(row["prompt"], list) and len(row["prompt"]) == 1, index
    assert row["prompt"][0]["role"] == "user", index
    assert isinstance(row["prompt"][0]["content"], str), index
    assert "\\boxed" in row["prompt"][0]["content"], index
    metadata = row["metadata"]
    assert metadata["source"] == "math-500", index
    assert metadata["verifier"] == "math", index
    assert metadata["rm_type"] == "math", index
    assert metadata["eval_only"] is True, index
    record_id = metadata["record_id"]
    assert record_id and record_id not in record_ids, (index, record_id)
    record_ids.add(record_id)

provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
assert provenance["dataset"] == "math500", provenance
assert provenance["output"] == str(data_path), provenance
assert provenance["kept"] == 500, provenance
assert provenance["skipped"] == 0, provenance
assert provenance["verifiers"] == {"math": 500}, provenance
assert provenance["inputs"], provenance
assert all(Path(path).is_file() for path in provenance["inputs"]), provenance
print("MATH-500 artifact: 500 canonical eval-only math rows with source provenance")
PY
'
