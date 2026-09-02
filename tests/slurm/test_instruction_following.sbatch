#!/bin/bash
#PBS -N test-ifevalg
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
export WANDB_MODE=offline
export OPEN_INSTRUCT_PATH=/data/open-instruct
export OPEN_INSTRUCT_DEPS_PATH=/data/open-instruct-deps
export NLTK_DATA=/data/open-instruct-deps/nltk_data

python3 -m compileall -q \
    experiments/src/environments/instruction_following \
    experiments/src/reward_sets/instruction_following.py
python3 -m pytest -q \
    tests/fast/experiments/test_instruction_following_environment.py \
    tests/fast/experiments/test_instruction_following_training_recipe.py \
    tests/fast/utils/test_arguments.py

python3 - <<"PY"
import json

from experiments.src.environments.instruction_following.verifier import (
    score_ifeval_sample,
    validate_ifeval_metadata,
)
from miles.utils.types import Sample

path = "/data/nemotron-rl-ifollow/miles-train.jsonl"
rows = 0
constraints = 0
fractional_row = None
with open(path, encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        row = json.loads(line)
        try:
            constraints += validate_ifeval_metadata(row["metadata"])
        except Exception as exc:
            raise AssertionError(f"{path}:{line_number}: incompatible IFEvalG metadata") from exc
        if row["metadata"].get("record_id") == 33964:
            fractional_row = row
        rows += 1

assert rows == 46391, rows
assert fractional_row is not None
metadata = fractional_row["metadata"]
partial = Sample(response="<think>reasoning, ignored</think>fact telephone", metadata=metadata)
complete = Sample(
    response="<think>reasoning, ignored</think>fact telephone\nP.P.S note",
    metadata=metadata,
)
assert score_ifeval_sample(partial) == 0.5
assert score_ifeval_sample(complete) == 1.0
print(f"IFEvalG actual data: rows={rows} constraints={constraints} partial=0.5 complete=1")
PY
'
