#!/bin/bash
#SBATCH --job-name=test-stem-env
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
