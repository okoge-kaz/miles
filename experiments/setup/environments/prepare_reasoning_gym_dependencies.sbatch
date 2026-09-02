#!/bin/bash
#PBS -N prep-reasoning-gym-deps
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Pin the official task-specific scorers used by all 104 Reasoning Gym tasks.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail

TARGET=/data/reasoning-gym-deps
MARKER="${TARGET}/.miles-reasoning-gym-0.1.25"
if [[ -f "${MARKER}" ]]; then
    echo "already staged: ${TARGET}"
    exit 0
fi
[[ ! -e "${TARGET}" ]] || {
    echo "refusing to replace unmarked dependency directory: ${TARGET}" >&2
    exit 1
}

TEMP_TARGET="/data/.reasoning-gym-deps-${MILES_JOB_ID}"
cleanup() {
    case "${TEMP_TARGET}" in
        /data/.reasoning-gym-deps-*) rm -rf -- "${TEMP_TARGET}" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT
mkdir -p "${TEMP_TARGET}"
python3 -m pip install --no-input --target "${TEMP_TARGET}" reasoning-gym==0.1.25
PYTHONPATH="${TEMP_TARGET}" python3 - <<"PY"
import json

import reasoning_gym

path = "/data/nemotron-rl-reasoninggym/data/train.jsonl"
tasks = set()
with open(path, encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        tasks.add(row["metadata"]["source_dataset"])
for task in sorted(tasks):
    scorer = reasoning_gym.get_score_answer_fn(task)
    assert callable(scorer), task
assert len(tasks) == 104, len(tasks)
print(f"Reasoning Gym scorer coverage: {len(tasks)}/104 tasks")
PY
touch "${TEMP_TARGET}/.miles-reasoning-gym-0.1.25"
mv "${TEMP_TARGET}" "${TARGET}"
trap - EXIT
echo "staged: ${TARGET}"
'
