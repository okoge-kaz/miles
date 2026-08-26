#!/bin/bash
#SBATCH --job-name=test-multi-env
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
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
export WANDB_MODE=offline

python3 -m compileall -q experiments/src/reward_sets/math_code_stem.py
python3 -m pytest -q \
    tests/fast/experiments/test_multi_environment.py \
    tests/fast/experiments/test_domain_training_recipes.py \
    tests/fast/utils/test_arguments.py

python3 - <<"PY"
import collections
import json
from pathlib import Path


data_path = Path("/data/nemotron-performance-transfer/math-code-stem-balanced-train.jsonl")
summary_path = Path("/data/nemotron-performance-transfer/math-code-stem-balanced-summary.json")
audit_path = Path("/data/nemotron-performance-transfer/math-code-stem-balanced-audit.json")
counts = collections.Counter()
rows = 0
with data_path.open(encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        row = json.loads(line)
        assert {"label", "metadata", "prompt"}.issubset(row), line_number
        # Some Reasoning Gym tasks (for example BoxNet) derive their target
        # entirely from verifier_metadata and intentionally carry an empty
        # string label.  The training merger rejects only a missing/None label.
        assert row["prompt"] and row["label"] is not None, line_number
        verifier = row["metadata"].get("verifier")
        assert verifier in {"math", "python_code", "mcqa_regex", "reasoning_gym"}, (
            line_number,
            verifier,
        )
        counts[verifier] += 1
        rows += 1

assert rows == 32673, rows
assert set(counts) == {"math", "python_code", "mcqa_regex", "reasoning_gym"}, counts
assert all(counts[verifier] > 0 for verifier in counts), counts
summary = json.loads(summary_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
assert summary["rows"] == rows, summary
assert summary["verifiers"] == dict(sorted(counts.items())), summary
assert audit["rows"] == rows, audit
assert audit["verifiers"] == dict(sorted(counts.items())), audit
print(f"Math+Code+STEM artifact: rows={rows} verifiers={dict(sorted(counts.items()))}")
PY
'
