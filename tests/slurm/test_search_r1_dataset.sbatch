#!/bin/bash
#PBS -N test-search-r1-data
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
[[ -r "${CONTAINER_IMAGE}" ]] || { echo "missing Search-R1 image: ${CONTAINER_IMAGE}" >&2; exit 1; }

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
export WANDB_MODE=offline

TEST_ROOT="$(mktemp -d /tmp/miles-search-r1-data.XXXXXX)"
cleanup() {
    case "${TEST_ROOT}" in
        /tmp/miles-search-r1-data.*) rm -rf -- "${TEST_ROOT}" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT

pytest -q \
    tests/fast/experiments/test_search_r1_dataset.py \
    tests/fast/experiments/test_search_r1.py \
    tests/fast/rollout/test_search_r1.py
python3 -m experiments.src.datasets.search_r1.build_eval \
    --flashrag-dir /data/flashrag-datasets --out-dir "${TEST_ROOT}/first"
python3 -m experiments.src.datasets.search_r1.build_eval \
    --flashrag-dir /data/flashrag-datasets --out-dir "${TEST_ROOT}/second"
python3 -m experiments.src.datasets.search_r1.build_eval \
    --out-dir "${TEST_ROOT}/first" --validate-only
diff -qr "${TEST_ROOT}/first" "${TEST_ROOT}/second"

python3 - "${TEST_ROOT}/first" <<"PY"
import json
import sys
from pathlib import Path

expected = {
    "nq": 500,
    "hotpotqa": 500,
    "triviaqa": 500,
    "popqa": 500,
    "2wikimultihopqa": 500,
    "musique": 500,
    "bamboogle": 125,
}
root = Path(sys.argv[1])
for benchmark, expected_rows in expected.items():
    path = root / f"{benchmark}-miles.jsonl"
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            assert len(row["prompt"]) == 1, (path, line_number)
            assert row["prompt"][0]["role"] == "user", (path, line_number)
            targets = row["reward_model"]["ground_truth"]["target"]
            assert targets and all(isinstance(target, str) and target for target in targets)
            assert row["metadata"]["question"], (path, line_number)
            rows += 1
    assert rows == expected_rows, (path, rows)
print("Search-R1 actual-data conversion: seven deterministic schema-valid eval sets")
PY
'
