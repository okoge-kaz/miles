#!/bin/bash
#SBATCH --job-name=prep-tau-bench
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert the pinned Tau Bench tasks and create a balanced training file from
# reward-verified retail environment tasks plus the same number of rows from
# each of two Nemotron 3 agentic datasets. The raw 500 Tau tasks are retained
# for audit. No package installation mutates the training image.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly TAU_COMMIT=09c26a85efd1d65168cfb57865ca2ca278c8153d
readonly TAU_DIR="${DATASET_DIR}/tau-bench"
readonly TAU_SRC="${WS}/src/tau-bench"
[[ -d "${TAU_SRC}/.git" ]] || {
    echo "missing pinned Tau checkout: ${TAU_SRC}" >&2
    exit 1
}
[[ "$(git -C "${TAU_SRC}" rev-parse HEAD)" == "${TAU_COMMIT}" ]] || {
    echo "Tau checkout is not pinned to ${TAU_COMMIT}" >&2
    exit 1
}
mkdir -p "${TAU_DIR}"

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
python3 -m experiments.src.datasets.tau_bench.prepare \
    --output-dir /data/tau-bench \
    --conv-tooluse /data/nemotron-rl-conv-tooluse/miles-train.jsonl \
    --fncall-pivot /data/nemotron-rl-fncall-pivot/miles-train.jsonl \
    --expected-tau-rows 500 \
    --seed 42

python3 - <<"PY"
import json

expected = {
    "/data/tau-bench/tau1-retail-train-miles.jsonl": 500,
    "/data/tau-bench/tau1-retail-test-miles.jsonl": 115,
    "/data/tau-bench/tau1-airline-test-miles.jsonl": 50,
}
loaded = {}
for path, expected_count in expected.items():
    with open(path, encoding="utf-8") as handle:
        loaded[path] = [json.loads(line) for line in handle if line.strip()]
    assert len(loaded[path]) == expected_count, (path, len(loaded[path]))
with open("/data/tau-bench/prepare-summary.json", encoding="utf-8") as handle:
    summary = json.loads(next(line for line in handle if line.strip()))
verified_count = summary["tau_reward_audit"]["accepted"]
rejected_count = summary["tau_reward_audit"]["rejected"]
assert verified_count > 0
assert rejected_count > 0
assert verified_count + rejected_count == 500
verified_path = "/data/tau-bench/tau1-retail-train-reward-verified-miles.jsonl"
mixed_path = "/data/tau-bench/nemotron3-agentic-tau-retail-train.jsonl"
for path, expected_count in ((verified_path, verified_count), (mixed_path, 3 * verified_count)):
    with open(path, encoding="utf-8") as handle:
        loaded[path] = [json.loads(line) for line in handle if line.strip()]
    assert len(loaded[path]) == expected_count, (path, len(loaded[path]))
mixed = loaded[mixed_path]
assert not any(row["metadata"].get("eval_only") is True for row in mixed)
tau_rows = [row for row in mixed if row["metadata"].get("verifier") == "tau_bench_environment"]
assert len(tau_rows) == verified_count
assert all(row["metadata"].get("tau_reward_verified") is True for row in tau_rows)
print(
    f"Tau data assertions: raw=500 verified={verified_count} rejected={rejected_count} "
    f"retail-test=115 airline-test=50 mixed={len(mixed)}"
)
PY
'

for f in \
    tau1-retail-train-miles.jsonl \
    tau1-retail-train-reward-verified-miles.jsonl \
    tau1-retail-test-miles.jsonl \
    tau1-airline-test-miles.jsonl \
    nemotron3-agentic-tau-retail-train.jsonl; do
    [[ -s "${TAU_DIR}/${f}" ]] || { echo "MISSING ${TAU_DIR}/${f}"; exit 1; }
    printf "  %-26s %6s rows\n" "${f}" "$(wc -l < "${TAU_DIR}/${f}")"
done
