#!/bin/bash
#PBS -N test-tau3-data
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${CONTAINER_IMAGE}" ]]

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro" \
    --cwd=/root/miles \
    --writable-tmpfs \
    --no-home \
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
