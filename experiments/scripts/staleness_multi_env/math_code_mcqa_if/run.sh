#!/bin/bash
#PBS -N math-code-mcqa-if
#PBS -q R9920261300
#PBS -l select=8:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=24:00:00
#PBS -j oe
# In-run eval is OFF; evaluate saved checkpoints separately.
# Rollout dumps default OFF; W&B defaults online (settings in train.sh).

set -euo pipefail

REPO_ROOT="${MILES_REPO:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"

export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/containers/miles-v0.1.2-multi-env.sif}"
export MILES_JOB_NUM_NODES="${MILES_JOB_NUM_NODES:-8}"
exec bash "${REPO_ROOT}/experiments/common/run_nemo_gym_training.sh" \
    "${REPO_ROOT}/experiments/scripts/staleness_multi_env/math_code_mcqa_if/train.sh" \
    --services code_gen mcqa instruction_following
