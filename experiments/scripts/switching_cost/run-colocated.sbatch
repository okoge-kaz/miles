#!/bin/bash
#SBATCH --job-name=miles-colocated-switch
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/%x-%j.log

set -euo pipefail

usage() {
    cat <<'EOF'
Required submission variables:
  MODEL_VARIANT=qwen3-4b|qwen3-8b|qwen3-30b-a3b

Example:
  MODEL_VARIANT=qwen3-8b sbatch experiments/scripts/switching_cost/run-colocated.sbatch

The two SFT checkpoints supplied for this experiment are the defaults. Override
SFT_HF_RELATIVE or SFT_MEGATRON_RELATIVE only to use another matched pair.
EOF
}

if [[ "${1:-}" == --help ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT="${MILES_REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)}}"

# These SFT exports are being staged under checkpoints/huggingface rather than
# the older checkpoints/hf root used by the other experiments.
export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface}"
source "${REPO_ROOT}/experiments/env.sh"

: "${MODEL_VARIANT:?set MODEL_VARIANT=qwen3-4b, qwen3-8b, or qwen3-30b-a3b}"
: "${WANDB_MODE:=online}"
case "${WANDB_MODE}" in
    online)
        : "${WANDB_API_KEY:?set WANDB_API_KEY, put an api.wandb.ai entry in ~/.netrc, or use WANDB_MODE=offline}"
        ;;
    offline|disabled) ;;
    *)
        echo "WANDB_MODE must be online, offline, or disabled: ${WANDB_MODE}" >&2
        exit 1
        ;;
esac

case "${MODEL_VARIANT}" in
    qwen3-4b)
        MODEL_SCRIPT=qwen3-4B.sh
        : "${SFT_HF_RELATIVE:=Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000}"
        : "${SFT_MEGATRON_RELATIVE:=Qwen3-4B-Base-LR2e-5-Step4000_torch_dist}"
        : "${TENSOR_PARALLEL_SIZE:=2}"
        : "${EXPERT_PARALLEL_SIZE:=1}"
        : "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
        : "${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
        DEFAULT_PROMPT_DATA=/data/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl
        ;;
    qwen3-8b)
        MODEL_SCRIPT=qwen3-8B.sh
        : "${SFT_HF_RELATIVE:=Qwen3-8B-Base/LR1.5e-5-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}"
        : "${SFT_MEGATRON_RELATIVE:=Qwen3-8B-Base-LR1.5e-5-Step4000_torch_dist}"
        : "${TENSOR_PARALLEL_SIZE:=4}"
        : "${EXPERT_PARALLEL_SIZE:=1}"
        : "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
        : "${ROLLOUT_NUM_GPUS_PER_ENGINE:=4}"
        DEFAULT_PROMPT_DATA=/data/dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000/dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000.jsonl
        ;;
    qwen3-30b-a3b)
        MODEL_SCRIPT=qwen3-30B-A3B.sh
        : "${SFT_HF_RELATIVE:=Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000}"
        : "${SFT_MEGATRON_RELATIVE:=Qwen3-30B-A3B-Base-LR2.0e-5-Step4000_torch_dist}"
        : "${TENSOR_PARALLEL_SIZE:=4}"
        : "${EXPERT_PARALLEL_SIZE:=8}"
        : "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
        : "${ROLLOUT_NUM_GPUS_PER_ENGINE:=8}"
        DEFAULT_PROMPT_DATA=/data/dapo-math-p10-90-qwen3-30b-a3b-base-lr2.0e-5-step4000/dapo-math-p10-90-qwen3-30b-a3b-base-lr2.0e-5-step4000.jsonl
        ;;
    *)
        echo "unsupported MODEL_VARIANT=${MODEL_VARIANT}" >&2
        usage >&2
        exit 1
        ;;
esac

validate_relative_path() {
    local label=$1
    local value=$2
    if [[ -z "${value}" || "${value}" == /* ]]; then
        echo "${label} must be a non-empty relative path: ${value}" >&2
        return 1
    fi
    case "/${value}/" in
        *"/../"*|*"/./"*|*"//"*)
            echo "${label} contains an unsafe path component: ${value}" >&2
            return 1
            ;;
    esac
}

validate_relative_path SFT_HF_RELATIVE "${SFT_HF_RELATIVE}"
validate_relative_path SFT_MEGATRON_RELATIVE "${SFT_MEGATRON_RELATIVE}"

HF_CHECKPOINT_HOST="${HF_CKPT_DIR%/}/${SFT_HF_RELATIVE}"
MEGATRON_CHECKPOINT_HOST="${MEGATRON_CKPT_DIR%/}/${SFT_MEGATRON_RELATIVE}"
[[ -r "${HF_CHECKPOINT_HOST}/config.json" ]] || {
    echo "missing readable HuggingFace config: ${HF_CHECKPOINT_HOST}/config.json" >&2
    exit 1
}
if [[ ! -r "${HF_CHECKPOINT_HOST}/model.safetensors" \
      && ! -r "${HF_CHECKPOINT_HOST}/model.safetensors.index.json" ]]; then
    echo "missing HuggingFace safetensors or index below ${HF_CHECKPOINT_HOST}" >&2
    exit 1
fi
[[ -r "${MEGATRON_CHECKPOINT_HOST}/latest_checkpointed_iteration.txt" ]] || {
    echo "missing readable torch_dist tracker: ${MEGATRON_CHECKPOINT_HOST}/latest_checkpointed_iteration.txt" >&2
    echo "convert the matching SFT HuggingFace checkpoint before submitting RL; see experiments/scripts/switching_cost/README.md" >&2
    exit 1
}
[[ "$(<"${MEGATRON_CHECKPOINT_HOST}/latest_checkpointed_iteration.txt")" == release ]] || {
    echo "torch_dist checkpoint is incomplete: expected release in ${MEGATRON_CHECKPOINT_HOST}/latest_checkpointed_iteration.txt" >&2
    exit 1
}

: "${NUM_ROLLOUT:=12}"
: "${ROLLOUT_BATCH_SIZE:=192}"
: "${OVER_SAMPLING_BATCH_SIZE:=256}"
: "${N_SAMPLES_PER_PROMPT:=16}"
: "${GLOBAL_BATCH_SIZE:=3072}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${SGLANG_MEM_FRACTION:=0.65}"
: "${PROMPT_DATA:=${DEFAULT_PROMPT_DATA}}"
: "${ACTOR_NUM_NODES:=${SLURM_JOB_NUM_NODES}}"
: "${ACTOR_GPUS_PER_NODE:=${GPUS_PER_NODE}}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${BANDWIDTH_BENCHMARK:=1}"
: "${BANDWIDTH_BYTES_PER_GPU:=1073741824}"
: "${BANDWIDTH_REPEATS:=8}"
: "${CHECK_WEIGHT_UPDATE_EQUAL:=0}"
: "${LOG_COLOCATE_SWITCH_METRICS:=1}"
: "${LOG_COLOCATE_TRANSFER_BYTES:=1}"
: "${LOG_MEMORY_USAGE:=0}"
: "${ENABLE_MILES_DASHBOARD:=0}"
: "${ENABLE_DUMP_DETAILS:=0}"
: "${RAY_DEDUP_LOGS:=1}"
: "${RUN_NAME:=${MODEL_VARIANT}-colocated-switch-${SLURM_JOB_ID}}"

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash: ${RUN_NAME}" >&2
    exit 1
}
[[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "NUM_ROLLOUT must be a positive integer" >&2
    exit 1
}
[[ "${CHECK_WEIGHT_UPDATE_EQUAL}" =~ ^[01]$ ]] || {
    echo "CHECK_WEIGHT_UPDATE_EQUAL must be 0 or 1" >&2
    exit 1
}
for flag in LOG_COLOCATE_SWITCH_METRICS LOG_COLOCATE_TRANSFER_BYTES LOG_MEMORY_USAGE \
    ENABLE_MILES_DASHBOARD ENABLE_DUMP_DETAILS; do
    [[ "${!flag}" =~ ^[01]$ ]] || {
        echo "${flag} must be 0 or 1" >&2
        exit 1
    }
done
[[ "${LOG_COLOCATE_TRANSFER_BYTES}" == 0 || "${LOG_COLOCATE_SWITCH_METRICS}" == 1 ]] || {
    echo "LOG_COLOCATE_TRANSFER_BYTES=1 requires LOG_COLOCATE_SWITCH_METRICS=1" >&2
    exit 1
}
[[ "${OVER_SAMPLING_BATCH_SIZE}" -gt "${ROLLOUT_BATCH_SIZE}" ]] || {
    echo "partial rollout requires OVER_SAMPLING_BATCH_SIZE > ROLLOUT_BATCH_SIZE" >&2
    exit 1
}
[[ $(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT )) -eq \
   $(( GLOBAL_BATCH_SIZE )) ]] || {
    echo "rollout batch times samples must equal global batch for one optimizer step" >&2
    exit 1
}

PLACEMENT=colocated
ROLLOUT_NUM_GPUS=0
export MODEL_VARIANT MODEL_SCRIPT SFT_HF_RELATIVE SFT_MEGATRON_RELATIVE
export NUM_ROLLOUT ROLLOUT_BATCH_SIZE OVER_SAMPLING_BATCH_SIZE N_SAMPLES_PER_PROMPT
export GLOBAL_BATCH_SIZE MAX_RESPONSE_LEN MAX_TOKENS_PER_GPU SGLANG_MEM_FRACTION
export PROMPT_DATA
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE CONTEXT_PARALLEL_SIZE
export TENSOR_PARALLEL_SIZE EXPERT_PARALLEL_SIZE EXPERT_TENSOR_PARALLEL_SIZE
export ROLLOUT_NUM_GPUS ROLLOUT_NUM_GPUS_PER_ENGINE PLACEMENT RUN_NAME
export CHECK_WEIGHT_UPDATE_EQUAL
export LOG_COLOCATE_SWITCH_METRICS LOG_COLOCATE_TRANSFER_BYTES LOG_MEMORY_USAGE
export ENABLE_MILES_DASHBOARD ENABLE_DUMP_DETAILS WANDB_MODE
export WANDB_PROJECT="${WANDB_PROJECT:-switching-cost-colocated}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_NAME}}"
export RAY_DEDUP_LOGS

source "${REPO_ROOT}/experiments/common/placement.sh"

RAY_HEAD_IP=$(getent hosts "$(hostname)" | awk '{print $1; exit}')
if [[ -z "${RAY_HEAD_IP}" || "${RAY_HEAD_IP}" == 169.254.* ]]; then
    RAY_HEAD_IP=$(ip route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')
fi
if [[ -z "${RAY_HEAD_IP}" || "${RAY_HEAD_IP}" == 169.254.* ]]; then
    echo "no routable address for the Ray head on $(hostname): got '${RAY_HEAD_IP}'" >&2
    exit 1
fi
export RAY_HEAD_IP
export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}.done"
mkdir -p "${OUTPUT_DIR}/.ray" "${OUTPUT_DIR}/switching_cost/${RUN_NAME}"
rm -f "${OUTPUT_DIR}/.ray/${RUN_NAME}.done"

STEP_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE%%\(*}"
[[ "${STEP_CPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-unset}" >&2
    exit 1
}

echo "run=${RUN_NAME} model=${MODEL_VARIANT} nodes=${SLURM_JOB_NUM_NODES}"
echo "hf_checkpoint=${HF_CHECKPOINT_HOST}"
echo "megatron_checkpoint=${MEGATRON_CHECKPOINT_HOST}"
echo "parallelism=train-tp${TENSOR_PARALLEL_SIZE}-ep${EXPERT_PARALLEL_SIZE} rollout-tp${ROLLOUT_NUM_GPUS_PER_ENGINE}"
du -sb --apparent-size "${HF_CHECKPOINT_HOST}" "${MEGATRON_CHECKPOINT_HOST}"

if [[ "${BANDWIDTH_BENCHMARK}" != 0 ]]; then
    srun \
        --nodes="${SLURM_JOB_NUM_NODES}" \
        --ntasks-per-node=1 \
        --cpus-per-task="${STEP_CPUS_PER_NODE}" \
        --mem=0 \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        --container-writable \
        --no-container-mount-home \
        --export=ALL \
        python3 /root/miles/experiments/tools/switching_cost/benchmark_host_device_bandwidth.py \
            --bytes-per-gpu "${BANDWIDTH_BYTES_PER_GPU}" \
            --repeats "${BANDWIDTH_REPEATS}"
fi

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${STEP_CPUS_PER_NODE}" \
    --mem=0 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/scripts/switching_cost/train-colocated.sh
