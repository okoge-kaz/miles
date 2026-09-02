#!/bin/bash
#PBS -N test-stem-env
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
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
export WANDB_MODE=offline
export REASONING_GYM_DEPS_PATH=/data/reasoning-gym-deps

python3 -m compileall -q \
    experiments/src/environments/reasoning_gym \
    experiments/src/reward_sets/_common.py \
    experiments/src/reward_sets/stem.py
python3 -m pytest -q \
    tests/fast/experiments/test_stem_environment.py \
    tests/fast/experiments/test_domain_training_recipes.py \
    tests/fast/utils/test_arguments.py

python3 - <<"PY"
import json

from experiments.src.environments.reasoning_gym.verifier import score_reasoning_gym_sample
from miles.utils.types import Sample

path = "/data/nemotron-rl-reasoninggym/miles-train.jsonl"
with open(path, encoding="utf-8") as stream:
    row = json.loads(next(stream))
label = row["label"]
correct = Sample(
    response=f"<answer>{label}</answer>",
    label=label,
    metadata=row["metadata"],
)
wrong = Sample(
    response="<answer>definitely-wrong</answer>",
    label=label,
    metadata=row["metadata"],
)
assert score_reasoning_gym_sample(correct) == 1.0
assert score_reasoning_gym_sample(wrong) == 0.0
print("Official Reasoning Gym scorer: correct=1 wrong=0")
PY
'
