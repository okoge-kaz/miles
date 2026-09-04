#!/bin/bash
#SBATCH --job-name=miles-tool-pivot-qwen3-4b
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/tool_call_pivot/nemotron-agentic-conv-tooluse-pivot/qwen3-4b/%x-%j.log

# Four-node async RL for exact, single-step actions from NVIDIA's conversational
# tool-use Pivot dataset. Held-out next-action evaluation runs separately.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." &>/dev/null && pwd)}"
: "${HF_CKPT_DIR:=${QWEN3_4B_BASE_HF_ROOT:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}}"
export HF_CKPT_DIR WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"

if [[ "${WANDB_MODE:-online}" != offline && "${WANDB_MODE:-online}" != disabled ]]; then
    : "${WANDB_API_KEY:?set WANDB_API_KEY, use WANDB_MODE=offline, or disable wandb}"
fi
[[ "${SLURM_JOB_NUM_NODES:?}" == 4 ]] || {
    echo "tool-call validation requires exactly four GPU nodes" >&2
    exit 1
}

TASK_FAMILY=tool_call_pivot
readonly ROLLOUT_SEMANTICS=static_single_turn_pivot
DATASET_TAG=nemotron-agentic-conv-tooluse-pivot
PROMPT_DATA=/data/nemotron-agentic-conv-tooluse-pivot/nemotron-agentic-conv-tooluse-pivot-train.jsonl
PROMPT_MANIFEST=/data/nemotron-agentic-conv-tooluse-pivot/manifest.json
TRAIN_SCRIPT=/root/miles/experiments/scripts/tool_call_pivot/async/nemotron-agentic-conv-tooluse-pivot/qwen3-4b/train.sh

: "${MODEL_NAME:=Qwen3-4B-Base-LR2e-5-Step4000}"
: "${HF_MODEL_NAME:=iter_0004000}"
: "${MODEL_PROFILE:=qwen3-4B}"
PLACEMENT=async
[[ -z "${CUSTOM_GENERATE_FUNCTION_PATH:-}" ]] || {
    echo "tool_call_pivot must use Miles single-turn generation" >&2
    exit 1
}

: "${ADVANTAGE_ESTIMATOR:=grpo}"
: "${ENTROPY_COEF:=0.00}"
: "${KL_LOSS_COEF:=0.00}"
: "${EPS_CLIP:=0.2}"
: "${EPS_CLIP_HIGH:=0.28}"
: "${EPS_CLIP_C:=}"
: "${RATIO_DENOMINATOR:=actor}"
: "${IS_CORRECTION:=tis}"
: "${TIS_CLIP:=2.0}"
: "${TIS_CLIP_LOW:=0}"
: "${MIS_PROFILE:=}"
: "${M2PO_BUDGET:=0.04}"
: "${USE_OPSM:=0}"
: "${OPSM_DELTA:=1e-4}"

: "${LR:=1e-6}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=32768}"
: "${NUM_ROLLOUT:=300}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"
: "${ROLLOUT_BATCH_SIZE:=192}"
: "${N_SAMPLES_PER_PROMPT:=16}"
: "${GLOBAL_BATCH_SIZE:=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"

: "${QUEUE_TYPE:=queue-recycle}"
: "${QUEUE_FACTOR:=1}"
: "${MAX_WEIGHT_STALENESS:=4}"
: "${STALENESS_REFERENCE:=prefill}"
: "${PAUSE_GENERATION_MODE:=in_place}"
: "${ASYNC_MAX_CONCURRENT_SAMPLES:=}"
: "${USE_REPLAY_BUFFER:=1}"
: "${REPLAY_BUFFER_TYPE:=inflight}"
: "${REPLAY_BUFFER_IDENTITY_TAG:=1}"

: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_GPUS_PER_NODE:=8}"
: "${ROLLOUT_NUM_GPUS:=24}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${RECOMPUTE_GRANULARITY:=full}"
: "${OVERLAP_COMM:=0}"
: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=1}"
: "${SGLANG_MEM_FRACTION:=0.70}"

RM_TYPE=
readonly VERIFIED_CUSTOM_RM_PATH=experiments.src.reward_sets.tool_call_pivot.reward
: "${CUSTOM_RM_PATH:=${VERIFIED_CUSTOM_RM_PATH}}"
[[ "${CUSTOM_RM_PATH}" == "${VERIFIED_CUSTOM_RM_PATH}" ]] || {
    echo "tool-call RL only permits ${VERIFIED_CUSTOM_RM_PATH}" >&2
    exit 1
}
: "${ZERO_REWARD_ON_TRUNCATED:=1}"

: "${EVAL_INTERVAL:=0}"
: "${SKIP_EVAL_BEFORE_TRAIN:=0}"
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=100}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=10}"
: "${DUMP_TRAIN_DATA:=0}"
: "${DUMP_POLICY_LOSS_DEBUG:=0}"
: "${FUSE_ONE_STEP_ACTOR_LOGPROBS:=1}"
: "${LOG_SAMPLE_STALENESS_METRICS:=1}"
: "${LOG_UPDATE_DIAGNOSTICS:=1}"
: "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:=1}"
: "${DEBUG_EXIT_AFTER_ROLLOUT:=}"

: "${MILES_NCCL_TRANSPORT:=efa}"
: "${ASYNC_SQSH_IMAGE_OVERRIDE:=/lustre/fsw/portfolios/coreai/users/kfujii/containers/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0.sqsh}"
case "${MILES_NCCL_TRANSPORT}" in
    efa)
        [[ "${NCCL_IB_DISABLE:-0}" != 1 ]] || {
            echo "EFA mode refuses NCCL_IB_DISABLE=1" >&2
            exit 1
        }
        [[ "${NCCL_NET_PLUGIN:-ofi}" == ofi ]] || {
            echo "EFA mode requires NCCL_NET_PLUGIN=ofi" >&2
            exit 1
        }
        NCCL_IB_DISABLE=0
        NCCL_NET_PLUGIN=ofi
        NCCL_TUNER_PLUGIN=ofi
        NCCL_NET="AWS Libfabric"
        FI_PROVIDER=efa
        LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        ;;
    tcp)
        NCCL_IB_DISABLE=1
        NCCL_NET=Socket
        unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
        ;;
    system)
        NCCL_IB_DISABLE=0
        unset NCCL_NET NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
        ;;
    *)
        echo "unknown MILES_NCCL_TRANSPORT=${MILES_NCCL_TRANSPORT}" >&2
        exit 1
        ;;
esac

: "${CONFIG_TAG:=4node-rollout-length-$((MAX_RESPONSE_LEN / 1024))k-lr${LR}-rbs${ROLLOUT_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-tseed${TRAIN_SEED}-rseed${ROLLOUT_SEED}-nr${NUM_ROLLOUT}}"

export MODEL_NAME HF_MODEL_NAME MODEL_PROFILE PLACEMENT TASK_FAMILY ROLLOUT_SEMANTICS
export DATASET_TAG PROMPT_DATA PROMPT_MANIFEST TRAIN_SCRIPT
export ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW MIS_PROFILE M2PO_BUDGET USE_OPSM OPSM_DELTA
export LR MAX_RESPONSE_LEN MAX_CONTEXT_LEN NUM_ROLLOUT TRAIN_SEED ROLLOUT_SEED
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
export QUEUE_TYPE QUEUE_FACTOR MAX_WEIGHT_STALENESS STALENESS_REFERENCE PAUSE_GENERATION_MODE
export ASYNC_MAX_CONCURRENT_SAMPLES USE_REPLAY_BUFFER REPLAY_BUFFER_TYPE REPLAY_BUFFER_IDENTITY_TAG
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
export MAX_TOKENS_PER_GPU RECOMPUTE_GRANULARITY OVERLAP_COMM ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export RM_TYPE CUSTOM_RM_PATH ZERO_REWARD_ON_TRUNCATED
export EVAL_INTERVAL SKIP_EVAL_BEFORE_TRAIN SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL
export DUMP_TRAIN_DATA DUMP_POLICY_LOSS_DEBUG FUSE_ONE_STEP_ACTOR_LOGPROBS
export LOG_SAMPLE_STALENESS_METRICS LOG_UPDATE_DIAGNOSTICS SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS
export DEBUG_EXIT_AFTER_ROLLOUT
export MILES_NCCL_TRANSPORT ASYNC_SQSH_IMAGE_OVERRIDE NCCL_IB_DISABLE LD_LIBRARY_PATH
export NCCL_NET NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER

source "${REPO_ROOT}/experiments/common/run_identity.sh"
export RUN_NAME CONFIG_TAG CKPT_PATH
source "${REPO_ROOT}/experiments/common/placement.sh"

export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
export RAY_HEAD_IP_FILE="${RAY_DONE_FLAG}.head-ip"
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f \
    "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done" \
    "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done.head-ip"
source "${REPO_ROOT}/experiments/common/clean_checkpoint.sh"

HOST_PROMPT_DATA="${DATASET_DIR}${PROMPT_DATA#/data}"
HOST_PROMPT_MANIFEST="${DATASET_DIR}${PROMPT_MANIFEST#/data}"
python3 - "${HOST_PROMPT_DATA}" "${HOST_PROMPT_MANIFEST}" <<'PY'
import hashlib
import json
import sys

prompt_path, manifest_path = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as stream:
    manifest = json.load(stream)
if manifest.get("policy") != "static-single-turn-pivot-exact-function-call-only":
    raise SystemExit("tool-call manifest has an unsupported policy")
if manifest.get("interaction_mode") != "static_single_turn_pivot":
    raise SystemExit("tool-call manifest is not static single-turn Pivot")
if manifest.get("stateful_environment") is not False:
    raise SystemExit("tool-call manifest unexpectedly requires environment state")
if manifest.get("train_counts") != {"conv-tooluse-pivot": 63559}:
    raise SystemExit("tool-call manifest has an unexpected training count")

digest = hashlib.sha256()
with open(prompt_path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != manifest.get("train_sha256"):
    raise SystemExit("tool-call training data differs from its manifest")

rows = 0
with open(prompt_path, encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        row = json.loads(line)
        metadata = row.get("metadata") or {}
        if (
            metadata.get("source") != "conv-tooluse-pivot"
            or metadata.get("verifier") != "expert_action"
            or metadata.get("expected_kind") != "function_call"
            or metadata.get("interaction_mode") != "static_single_turn_pivot"
            or metadata.get("stateful_environment") is not False
            or metadata.get("eval_only")
            or not row.get("tools")
        ):
            raise SystemExit(f"ineligible tool-call training row {line_number}")
        rows += 1
if rows != manifest["train_counts"]["conv-tooluse-pivot"]:
    raise SystemExit("tool-call row count differs from its manifest")
PY

STEP_CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE%%\(*}}"
[[ "${STEP_CPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-unset}" >&2
    exit 1
}
MOUNTS="${CONTAINER_MOUNTS}"

if [[ "${MILES_NCCL_TRANSPORT}" == efa ]]; then
    echo "running EFA fail-closed preflight on every node"
    srun \
        --nodes="${SLURM_JOB_NUM_NODES}" \
        --ntasks="${SLURM_JOB_NUM_NODES}" \
        --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --mem=0 \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${MOUNTS}" \
        --no-container-mount-home \
        --export=ALL \
        bash /root/miles/experiments/common/check_efa.sh
fi

container_command=(bash "${TRAIN_SCRIPT}")
if [[ "${MILES_NCCL_TRANSPORT}" == efa ]]; then
    container_command=(bash /root/miles/experiments/common/run_with_efa_env.sh "${container_command[@]}")
fi
srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${STEP_CPUS_PER_NODE}" \
    --mem=0 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    "${container_command[@]}"
