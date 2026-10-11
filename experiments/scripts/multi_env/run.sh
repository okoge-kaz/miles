#!/bin/bash
#PBS -N multi-env-rl
#PBS -q R9920261300
#PBS -l select=2:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
REPO_ROOT="${MILES_REPO:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/containers/miles-v0.1.1-multi-env.sif}"
export MILES_JOB_NUM_NODES="${MILES_JOB_NUM_NODES:-2}"
if [[ -z "${NEMO_GYM_ENDPOINTS:-}" && "${MILES_LAUNCH_DRY_RUN:-0}" != 1 ]]; then
    : "${TAU_USER_MODEL_CONFIG:?set the frozen user simulator native JSON config (container path)}"
fi
exec bash "${REPO_ROOT}/experiments/common/run_nemo_gym_training.sh" \
    "${REPO_ROOT}/experiments/scripts/multi_env/train.sh" \
    --services instruction_following tau2 code_gen mcqa \
    --user-model-config "${TAU_USER_MODEL_CONFIG:-}"
