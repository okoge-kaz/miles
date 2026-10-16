#!/bin/bash
#PBS -N miles-mcore-hf
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -j oe
# setup.sh supplies -o outputs/ckpt/mcore_hf/<model>_<variant>_<run-id>.ou.

set -euo pipefail

REPO_ROOT="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
model_preset="${1:-${CONVERT_MODEL_PRESET:-}}"
if [[ -n "${model_preset}" ]]; then
    [[ "${model_preset}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "invalid conversion model preset" >&2; exit 2; }
    source "${REPO_ROOT}/experiments/scripts/checkpoint_convert/models/${model_preset}.sh"
fi
[[ "${MILES_JOB_NUM_NODES}" == 1 ]] || { echo "Megatron -> HF conversion requires one node" >&2; exit 2; }
[[ -r "${CONTAINER_IMAGE}" ]] || { echo "missing container: ${CONTAINER_IMAGE}" >&2; exit 2; }

miles_container_exec --nv \
    --image "${CONTAINER_IMAGE}" \
    --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles \
    --export="MODEL_NAME,HF_MODEL_NAME,MODEL_PROFILE,GPUS_PER_NODE,CONVERT_INPUT_DIR,CONVERT_OUTPUT_DIR,CONVERT_ORIGIN_HF_DIR,CONVERT_CHUNK_SIZE,CONVERT_VOCAB_SIZE,CONVERT_BACKEND,CONVERT_TENSOR_PARALLEL_SIZE,CONVERT_PIPELINE_PARALLEL_SIZE,CONVERT_EXPERT_PARALLEL_SIZE" \
    bash --noprofile --norc /root/miles/experiments/scripts/checkpoint_convert/mcore_hf/convert.sh
