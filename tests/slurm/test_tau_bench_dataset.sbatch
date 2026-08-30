#!/bin/bash
#SBATCH --job-name=test-tau3-data
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
TAU3_SQSH_IMAGE="${TAU3_SQSH_IMAGE:-${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0-tau3-1.0.1.sqsh}"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${TAU3_SQSH_IMAGE}" ]]

srun \
    --container-image="${TAU3_SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro" \
    --container-workdir=/root/miles \
    --container-writable \
    --no-container-mount-home \
    --export=USER \
    bash -lc '
set -euo pipefail
export PATH=/opt/tau3/.venv/bin:${PATH}
export PYTHONPATH=/root/miles
export WANDB_MODE=offline

TEST_ROOT="$(mktemp -d /tmp/miles-tau3-data.XXXXXX)"
cleanup() {
    case "${TEST_ROOT}" in
        /tmp/miles-tau3-data.*) rm -rf -- "${TEST_ROOT}" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT
for run in first second; do
    /opt/tau3/.venv/bin/python -m experiments.src.datasets.tau_bench.prepare \
        --output-dir "${TEST_ROOT}/${run}" >/dev/null
done
diff -qr "${TEST_ROOT}/first" "${TEST_ROOT}/second"

/opt/tau3/.venv/bin/python - "${TEST_ROOT}/first" <<"PY"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "tau3-retail-test-miles.jsonl": 40,
    "tau3-airline-test-miles.jsonl": 20,
    "tau3-telecom-test-miles.jsonl": 40,
    "tau3-test-miles.jsonl": 100,
}
for name, count in expected.items():
    rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
    assert len(rows) == count
    assert all(row["metadata"]["eval_only"] is True for row in rows)
for domain in ("retail", "airline", "telecom"):
    for split in ("train", "base"):
        assert not (root / f"tau3-{domain}-{split}-miles.jsonl").exists()
for split in ("train", "base"):
    assert not (root / f"tau3-{split}-miles.jsonl").exists()
print("Tau three data: deterministic; held-out test=100; train/base absent")
PY
'
