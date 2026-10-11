#!/bin/bash
set -euo pipefail

train_script="${1:?usage: run_training.sh <train.sh>}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export MILES_REPO="${REPO_ROOT}"
source "${REPO_ROOT}/experiments/env.sh"
_miles_load_python_runtime
source "${train_script}"
source "${REPO_ROOT}/experiments/common/run_identity.sh"
source "${REPO_ROOT}/experiments/common/placement.sh"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m experiments.common.training --dry-run
if [[ "${MILES_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi
[[ -r "${CONTAINER_IMAGE}" ]] || { echo "missing container: ${CONTAINER_IMAGE}" >&2; exit 2; }
[[ -f "${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json" ]] || {
    echo "missing HF config: ${HF_CKPT_DIR}/${HF_MODEL_NAME}/config.json" >&2; exit 2
}
[[ -f "${MEGATRON_CKPT_DIR}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt" ]] || {
    echo "missing converted checkpoint: ${MODEL_NAME}_torch_dist; see experiments/setup.sh convert" >&2; exit 2
}
python3 -m experiments.common.dataset_args --check --data-root "${DATASET_DIR}" --repo-root "${MILES_REPO}"

case "${MILES_NCCL_TRANSPORT}" in
    system) export NCCL_IB_DISABLE=0; unset NCCL_NET NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER ;;
    tcp) export NCCL_IB_DISABLE=1 NCCL_NET=Socket; unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER ;;
    *) echo "MILES_NCCL_TRANSPORT must be system or tcp" >&2; exit 2 ;;
esac

source "${REPO_ROOT}/experiments/common/clean_checkpoint.sh"
export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${MILES_JOB_ID}.done"
export RAY_HEAD_IP_FILE="${RAY_DONE_FLAG}.head-ip"
export RAY_TEMP_DIR="/tmp/miles-ray-${MILES_JOB_ID_SHORT}"

mkdir -p "${OUTPUT_DIR}/.ray"
rm -f -- "${OUTPUT_DIR}/.ray/${RUN_NAME}-${MILES_JOB_ID}.done" \
    "${OUTPUT_DIR}/.ray/${RUN_NAME}-${MILES_JOB_ID}.done.head-ip"
train_relative="${train_script#${REPO_ROOT}/}"
[[ "${train_relative}" != "${train_script}" ]] || { echo "recipe must be in this checkout" >&2; exit 2; }
echo "run=${RUN_NAME} checkpoints=${CKPT_PATH} image=${CONTAINER_IMAGE}"
export MILES_NODE_CPUS_PER_TASK="${PBS_GPU_CPUS_PER_NODE}"

miles_container_exec_all --image "${CONTAINER_IMAGE}" --nv --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles --export=ALL \
    bash /root/miles/experiments/common/train.sh "/root/miles/${train_relative}"
