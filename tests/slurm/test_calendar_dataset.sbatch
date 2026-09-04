#!/bin/bash
#SBATCH --job-name=test-calendar-data
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

TEST_ROOT="$(mktemp -d /tmp/miles-calendar-data.XXXXXX)"
cleanup() {
    case "${TEST_ROOT}" in
        /tmp/miles-calendar-data.*) rm -rf -- "${TEST_ROOT}" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT

pytest -q tests/fast/experiments/test_calendar_dataset.py
for run in first second; do
    python3 -m experiments.src.datasets.calendar.prepare \
        --train-input /data/nemotron-rl-ifollow-calendar-v2/train.jsonl \
        --validation-input /data/nemotron-rl-ifollow-calendar-v2/validation.jsonl \
        --train-output "${TEST_ROOT}/${run}-train.jsonl" \
        --validation-output "${TEST_ROOT}/${run}-validation.jsonl"
done
cmp "${TEST_ROOT}/first-train.jsonl" "${TEST_ROOT}/second-train.jsonl"
cmp "${TEST_ROOT}/first-validation.jsonl" "${TEST_ROOT}/second-validation.jsonl"

python3 - "${TEST_ROOT}" <<"PY"
import json
import sys
from pathlib import Path

from experiments.src.environments.calendar.verifier import (
    build_calendar_solution,
    score_calendar_response,
)

root = Path(sys.argv[1])
for split, expected_rows, eval_only in (("train", 9659, False), ("validation", 256, True)):
    path = root / f"first-{split}.jsonl"
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            metadata = row["metadata"]
            assert metadata["verifier"] == "calendar_constraints", (path, line_number)
            assert metadata["eval_only"] is eval_only, (path, line_number)
            state = metadata["expected_calendar_state"]
            assert json.loads(row["label"]) == state, (path, line_number)
            assert score_calendar_response(build_calendar_solution(state), state) == 1.0
            rows += 1
    assert rows == expected_rows, (path, rows)
print("Calendar actual-data conversion: deterministic, schema-valid, locally verified")
PY
'
