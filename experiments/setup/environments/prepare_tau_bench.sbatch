#!/bin/bash
#SBATCH --job-name=prepare-tau3
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Materialize only the official Tau three v1.0.1 held-out test split. The
# converter validates train/test/base disjointness in memory, but intentionally
# leaves no Tau training artifact on disk.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly TAU_DIR="${DATASET_DIR}/tau-bench"
TAU3_SQSH_IMAGE="${TAU3_SQSH_IMAGE:-${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0-tau3-1.0.1.sqsh}"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${TAU3_SQSH_IMAGE}" ]] || { echo "Tau three image not found: ${TAU3_SQSH_IMAGE}" >&2; exit 1; }
[[ -r "${DOTENV_MASK}" ]] || { echo "dotenv mask not found: ${DOTENV_MASK}" >&2; exit 1; }
mkdir -p "${TAU_DIR}"

srun \
    --container-image="${TAU3_SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro" \
    --container-writable \
    --no-container-mount-home \
    --export=USER \
    bash -lc '
set -euo pipefail
cd /root/miles
export PATH=/opt/tau3/.venv/bin:${PATH}
export PYTHONPATH=/root/miles
/opt/tau3/.venv/bin/python -m experiments.src.datasets.tau_bench.prepare \
    --output-dir /data/tau-bench

/opt/tau3/.venv/bin/python - <<"PY"
import json
from pathlib import Path

root = Path("/data/tau-bench")
expected = {
    "tau3-retail-test-miles.jsonl": 40,
    "tau3-airline-test-miles.jsonl": 20,
    "tau3-telecom-test-miles.jsonl": 40,
    "tau3-test-miles.jsonl": 100,
}
for name, expected_count in expected.items():
    rows = [json.loads(line) for line in (root / name).read_text().splitlines() if line]
    assert len(rows) == expected_count, (name, len(rows))
    assert all(row["metadata"]["eval_only"] is True for row in rows)
summary = json.loads((root / "prepare-summary.json").read_text())
assert summary["evaluation_policy"]["benchmark_split"] == "test"
assert summary["evaluation_policy"]["train_and_base_materialized"] is False
assert summary["evaluation_policy"]["training_dataset"] == "inclusionAI/AReaL-tau2-data"
assert summary["evaluation_policy"]["training_split"] == "tau2_rl_train.jsonl"
assert summary["evaluation_policy"]["training_environment"] == (
    "stateful_multi_turn_user_simulator_environment"
)
for domain in ("retail", "airline", "telecom"):
    for split in ("train", "base"):
        assert not (root / f"tau3-{domain}-{split}-miles.jsonl").exists()
for split in ("train", "base"):
    assert not (root / f"tau3-{split}-miles.jsonl").exists()
print("Tau three evaluation conversion: held-out-test=100; train/base not materialized")
PY
'

# These exact files are Tau v1 artifacts. Removal occurs only after every Tau
# three output above has passed its count and contamination checks.
rm -f -- \
    "${TAU_DIR}/airline_test_tasks.jsonl" \
    "${TAU_DIR}/retail_dev_tasks.jsonl" \
    "${TAU_DIR}/retail_test_tasks.jsonl" \
    "${TAU_DIR}/retail_train_tasks.jsonl" \
    "${TAU_DIR}/tau1_airline_test.jsonl" \
    "${TAU_DIR}/tau1_retail_test.jsonl" \
    "${TAU_DIR}/tau1_train.jsonl" \
    "${TAU_DIR}/tau1-retail-train-miles.jsonl" \
    "${TAU_DIR}/tau1-retail-train-reward-verified-miles.jsonl" \
    "${TAU_DIR}/tau1-retail-test-miles.jsonl" \
    "${TAU_DIR}/tau1-airline-test-miles.jsonl" \
    "${TAU_DIR}/nemotron3-agentic-tau-retail-train.jsonl"

if find "${TAU_DIR}" -maxdepth 1 -type f -name 'tau1*' -print -quit | grep -q .; then
    echo "Tau v1 artifact remains under ${TAU_DIR}" >&2
    exit 1
fi

echo "Tau evaluation artifacts prepared; Tau train/base outputs are absent."
