#!/bin/bash
#PBS -N multi-env-data
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${MILES_REPO:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/containers/miles-v0.1.2-multi-env.sif}"

source "${REPO_ROOT}/experiments/env.sh"
mkdir -p "${OUTPUT_DIR}/data"
exec > >(tee "${OUTPUT_DIR}/data/multi-env-${MILES_JOB_ID_SHORT}.log") 2>&1

miles_container_exec --image "${CONTAINER_IMAGE}" \
    --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles \
    --no-nv --cleanenv env PYTHONPATH=/root/miles:/root/Megatron-LM OMP_NUM_THREADS=1 \
    DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}" DATASET_CHUNK_SIZE="${DATASET_CHUNK_SIZE:-128}" \
    bash /root/miles/experiments/scripts/multi_env/prepare/prepare.sh
