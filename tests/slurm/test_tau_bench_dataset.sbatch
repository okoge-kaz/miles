#!/bin/bash
#SBATCH --job-name=test-tau-data
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly TAU_COMMIT=09c26a85efd1d65168cfb57865ca2ca278c8153d
readonly TAU_SRC="${WS}/src/tau-bench"
[[ "$(git -C "${TAU_SRC}" rev-parse HEAD)" == "${TAU_COMMIT}" ]]

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${TAU_SRC}:/root/tau-bench" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/tau-bench:/root/miles
export WANDB_MODE=offline

TEST_ROOT="$(mktemp -d /tmp/miles-tau-data.XXXXXX)"
cleanup() {
    case "${TEST_ROOT}" in
        /tmp/miles-tau-data.*) rm -rf -- "${TEST_ROOT}" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT

pytest -q tests/fast/experiments/test_tau_bench_dataset.py
for run in first second; do
    python3 -m experiments.src.datasets.tau_bench.prepare \
        --output-dir "${TEST_ROOT}/${run}" \
        --conv-tooluse /data/nemotron-rl-conv-tooluse/miles-train.jsonl \
        --fncall-pivot /data/nemotron-rl-fncall-pivot/miles-train.jsonl \
        --expected-tau-rows 500 \
        --seed 42
done
diff -qr "${TEST_ROOT}/first" "${TEST_ROOT}/second"

python3 - "${TEST_ROOT}/first" <<"PY"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "tau1-retail-train-miles.jsonl": 500,
    "tau1-retail-train-reward-verified-miles.jsonl": 495,
    "tau1-retail-test-miles.jsonl": 115,
    "tau1-airline-test-miles.jsonl": 50,
    "nemotron3-agentic-tau-retail-train.jsonl": 1485,
}
for name, expected_rows in expected.items():
    path = root / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == expected_rows, (name, len(rows))
    assert all(row["metadata"].get("eval_only") is not True for row in rows) if "train" in name else True
summary = json.loads((root / "prepare-summary.json").read_text(encoding="utf-8"))
assert summary["tau_reward_audit"]["accepted"] == 495
assert summary["tau_reward_audit"]["rejected"] == 5
assert summary["rows_per_source"] == 495
verified = [
    json.loads(line)
    for line in (root / "tau1-retail-train-reward-verified-miles.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line
]
assert all(row["metadata"]["tau_reward_verified"] is True for row in verified)
print("Tau actual-data conversion: deterministic, pinned, official reward audit=495/500")
PY
'
