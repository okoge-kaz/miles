#!/bin/bash
#SBATCH --job-name=miles-areal-tau2-qwen3-4b-agentic-sft953
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/tau_bench/areal-tau2/qwen3-4b-agentic-sft-953/%x-%j.log

# Eight-node asynchronous GRPO on only the 1,982 AReaL Tau2 RL tasks. The
# trainer:rollout split is selectable from 1:7, 2:6, 3:5, and 4:4. Every
# trajectory uses an LLM user simulator, an isolated task DB, and terminal Tau
# state/action/communication reward. Tau three test evaluation remains offline.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." &>/dev/null && pwd)}"
export WANDB_MODE="${WANDB_MODE:-offline}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/load_job_env.sh"
load_job_env \
    "${REPO_ROOT}/.env" \
    GEMINI_API_KEY \
    NVIDIA_INFERENCE_API_KEY \
    NVIDIA_INFERENCE_BASE_URL \
    TAU_USER_MODEL \
    TAU_USER_PROVIDER

if [[ "${WANDB_MODE}" != offline && "${WANDB_MODE}" != disabled ]]; then
    : "${WANDB_API_KEY:?set WANDB_API_KEY, use WANDB_MODE=offline, or disable wandb}"
fi
: "${TOTAL_NODES:=8}"
readonly TOTAL_NODES
[[ "${TOTAL_NODES}" == 4 || "${TOTAL_NODES}" == 8 ]] || {
    echo "AReaL Tau2 supports TOTAL_NODES=4 for validation or TOTAL_NODES=8 for training studies" >&2
    exit 1
}
[[ "${SLURM_JOB_NUM_NODES:?}" == "${TOTAL_NODES}" ]] || {
    echo "AReaL Tau2 requested TOTAL_NODES=${TOTAL_NODES}, got ${SLURM_JOB_NUM_NODES}" >&2
    exit 1
}

TASK_FAMILY=tau_bench
DATASET_TAG=areal-tau2-user-simulator
ROLLOUT_SEMANTICS=stateful_multi_turn_user_simulator_environment
PROMPT_DATA=/data/areal-tau2-data/miles-tau2-rl-train.jsonl
PROMPT_SUMMARY=/data/areal-tau2-data/miles-tau2-rl-summary.json
TRAIN_SCRIPT=/root/miles/experiments/scripts/tau_bench/async/areal-tau2/qwen3-4b-agentic-sft-953/train.sh
: "${WANDB_PROJECT:=async-rl-tau}"

readonly AGENTIC_SFT_RUN="LR2.0e-5-SEQ40960-GBS96-MBS1-TP1-PP1-CP1-EP1-CHAT1-TOKpreserve-reasoning-content-v1-PACK1-rope-p2p-cp1-full-recompute-STEPS953"
: "${MODEL_NAME:=Qwen3-4B-Agentic-SFT-953}"
: "${HF_MODEL_NAME:=Qwen3-4B-Base/${AGENTIC_SFT_RUN}/iter_0000953}"
: "${MODEL_PROFILE:=qwen3-4B}"
: "${REF_LOAD:=/ckpt/training/reasoning-sft/nemotron-sft-agentic-v2-qwen3-preserve-thinking/7c804833427f633ccd53b582dbf02525fd680f78/Qwen3-4B-Base/${AGENTIC_SFT_RUN}}"
PLACEMENT=async

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
: "${M2PO_BUDGET:=0.04}"
: "${USE_OPSM:=0}"
: "${OPSM_DELTA:=1e-4}"

: "${LR:=1e-6}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=32768}"
: "${TRAIN_EPOCHS:=6}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"
: "${ROLLOUT_BATCH_SIZE:=63}"
: "${N_SAMPLES_PER_PROMPT:=16}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"
: "${ALLOW_NUM_ROLLOUT_OVERRIDE:=0}"
readonly AREAL_TAU2_ROWS=1982
[[ "${TRAIN_EPOCHS}" =~ ^[1-6]$ ]] || {
    echo "TRAIN_EPOCHS must be an integer from 1 through 6" >&2
    exit 1
}
[[ "${ROLLOUT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ROLLOUT_BATCH_SIZE must be positive" >&2
    exit 1
}
[[ "${N_SAMPLES_PER_PROMPT}" =~ ^[0-9]+$ ]] && ((N_SAMPLES_PER_PROMPT >= 2)) || {
    echo "N_SAMPLES_PER_PROMPT must be at least 2 for GRPO" >&2
    exit 1
}
[[ "${NUM_STEPS_PER_ROLLOUT}" == 1 ]] || {
    echo "AReaL Tau2 fixes NUM_STEPS_PER_ROLLOUT=1 for one weight update per batch" >&2
    exit 1
}
[[ "${ALLOW_NUM_ROLLOUT_OVERRIDE}" == 0 || "${ALLOW_NUM_ROLLOUT_OVERRIDE}" == 1 ]] || {
    echo "ALLOW_NUM_ROLLOUT_OVERRIDE must be 0 or 1" >&2
    exit 1
}
readonly EPOCH_PROMPT_GROUPS=$((AREAL_TAU2_ROWS * TRAIN_EPOCHS))
readonly EXPECTED_NUM_ROLLOUT=$(((EPOCH_PROMPT_GROUPS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))
: "${NUM_ROLLOUT:=${EXPECTED_NUM_ROLLOUT}}"
[[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "NUM_ROLLOUT must be positive" >&2
    exit 1
}
if [[ "${NUM_ROLLOUT}" != "${EXPECTED_NUM_ROLLOUT}" ]]; then
    [[ "${ALLOW_NUM_ROLLOUT_OVERRIDE}" == 1 ]] || {
        echo "NUM_ROLLOUT must equal ceil(1982 * TRAIN_EPOCHS / ROLLOUT_BATCH_SIZE)=${EXPECTED_NUM_ROLLOUT}; set ALLOW_NUM_ROLLOUT_OVERRIDE=1 for a bounded partial-epoch schedule" >&2
        exit 1
    }
    readonly SIX_EPOCH_NUM_ROLLOUT=$(((AREAL_TAU2_ROWS * 6 + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE))
    ((NUM_ROLLOUT <= SIX_EPOCH_NUM_ROLLOUT)) || {
        echo "explicit NUM_ROLLOUT=${NUM_ROLLOUT} exceeds the six-epoch cap ${SIX_EPOCH_NUM_ROLLOUT}" >&2
        exit 1
    }
    SCHEDULE_MODE=explicit-updates
else
    SCHEDULE_MODE=integer-epochs
fi
readonly GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
readonly PLANNED_PROMPT_GROUPS=$((NUM_ROLLOUT * ROLLOUT_BATCH_SIZE))
readonly PLANNED_TRAJECTORIES=$((PLANNED_PROMPT_GROUPS * N_SAMPLES_PER_PROMPT))
if [[ "${SCHEDULE_MODE}" == integer-epochs ]]; then
    REQUESTED_PROMPT_GROUPS=${EPOCH_PROMPT_GROUPS}
else
    REQUESTED_PROMPT_GROUPS=${PLANNED_PROMPT_GROUPS}
fi
readonly REQUESTED_PROMPT_GROUPS
readonly FINAL_BATCH_OVERSHOOT=$((PLANNED_PROMPT_GROUPS - REQUESTED_PROMPT_GROUPS))
EFFECTIVE_EPOCHS="$(awk -v groups="${PLANNED_PROMPT_GROUPS}" -v rows="${AREAL_TAU2_ROWS}" 'BEGIN { printf "%.6f", groups / rows }')"
readonly SCHEDULE_MODE EFFECTIVE_EPOCHS

: "${QUEUE_TYPE:=queue-recycle}"
: "${QUEUE_FACTOR:=1}"
: "${MAX_WEIGHT_STALENESS:=4}"
: "${STALENESS_REFERENCE:=prefill}"
: "${PAUSE_GENERATION_MODE:=in_place}"
: "${ASYNC_MAX_CONCURRENT_SAMPLES:=96}"
: "${USE_REPLAY_BUFFER:=1}"
: "${REPLAY_BUFFER_TYPE:=inflight}"
: "${REPLAY_BUFFER_IDENTITY_TAG:=1}"
: "${REPLAY_BUFFER_KEEP_LAST:=2}"
: "${ALLOW_TAU_REPLAY_ABLATION:=0}"
: "${LOG_REPLAY_RESUME_METRICS:=0}"
[[ "${ALLOW_TAU_REPLAY_ABLATION}" == 0 || "${ALLOW_TAU_REPLAY_ABLATION}" == 1 ]] || {
    echo "ALLOW_TAU_REPLAY_ABLATION must be 0 or 1" >&2
    exit 1
}
[[ "${USE_REPLAY_BUFFER}" == 0 || "${USE_REPLAY_BUFFER}" == 1 ]] || {
    echo "USE_REPLAY_BUFFER must be 0 or 1" >&2
    exit 1
}
if [[ "${ALLOW_TAU_REPLAY_ABLATION}" == 0 ]]; then
    [[ "${USE_REPLAY_BUFFER}" == 1 && "${REPLAY_BUFFER_TYPE}" == inflight ]] || {
        echo "AReaL Tau2 requires inflight replay outside an explicit replay ablation" >&2
        exit 1
    }
elif [[ "${USE_REPLAY_BUFFER}" == 1 ]]; then
    [[ "${REPLAY_BUFFER_TYPE}" == rollout || "${REPLAY_BUFFER_TYPE}" == inflight ]] || {
        echo "Tau replay ablation requires REPLAY_BUFFER_TYPE=rollout or inflight" >&2
        exit 1
    }
fi
[[ "${LOG_REPLAY_RESUME_METRICS}" == 0 || "${LOG_REPLAY_RESUME_METRICS}" == 1 ]] || {
    echo "LOG_REPLAY_RESUME_METRICS must be 0 or 1" >&2
    exit 1
}
[[ "${REPLAY_BUFFER_KEEP_LAST}" =~ ^[1-9][0-9]*$ ]] || {
    echo "REPLAY_BUFFER_KEEP_LAST must be positive" >&2
    exit 1
}

: "${TAU_MAX_TURNS:=30}"
: "${TAU_MAX_STEPS:=200}"
: "${TAU_USER_PROVIDER:=nvidia}"
: "${TAU_USER_MAX_TOKENS:=512}"
: "${TAU_USER_TEMPERATURE:=0.7}"
: "${TAU_USER_TOP_P:=0.95}"
: "${TAU_USER_REQUEST_TIMEOUT:=120}"
: "${TAU_USER_MAX_RETRIES:=4}"
: "${TAU_TOOL_CALL_PARSER:=qwen25}"
: "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL:=0}"
: "${TAU_LOG_OVERHEAD:=0}"
: "${TAU_LOG_LEVEL:=ERROR}"
case "${TAU_USER_PROVIDER}" in
    nvidia)
        : "${TAU_USER_MODEL:=us/gcp/google/eccn-gemini-2.5-flash-lite}"
        : "${NVIDIA_INFERENCE_BASE_URL:=https://inference-api.nvidia.com/v1}"
        : "${NVIDIA_INFERENCE_API_KEY:?set NVIDIA_INFERENCE_API_KEY in .env or the job environment}"
        ;;
    gemini)
        : "${TAU_USER_MODEL:=gemini-2.5-flash-lite}"
        : "${GEMINI_API_KEY:?set GEMINI_API_KEY in .env or the job environment}"
        ;;
    *)
        echo "TAU_USER_PROVIDER must be nvidia or gemini" >&2
        exit 1
        ;;
esac
[[ "${TAU_LOG_OVERHEAD}" == 0 || "${TAU_LOG_OVERHEAD}" == 1 ]] || {
    echo "TAU_LOG_OVERHEAD must be 0 or 1" >&2
    exit 1
}
[[ "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL}" == 0 \
    || "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL}" == 1 ]] || {
    echo "TAU_OVERLAP_DB_RESTORE_WITH_PREFILL must be 0 or 1" >&2
    exit 1
}
if [[ "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL}" == 1 ]]; then
    [[ "${USE_REPLAY_BUFFER}" == 1 && "${REPLAY_BUFFER_TYPE}" == inflight ]] || {
        echo "Tau DB/prefill resume overlap requires inflight replay" >&2
        exit 1
    }
fi
[[ "${TAU_LOG_LEVEL}" =~ ^(ERROR|WARNING|INFO|DEBUG)$ ]] || {
    echo "TAU_LOG_LEVEL must be ERROR, WARNING, INFO, or DEBUG" >&2
    exit 1
}

: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_GPUS_PER_NODE:=8}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"
[[ "${ACTOR_NUM_NODES}" =~ ^[1-4]$ ]] || {
    echo "ACTOR_NUM_NODES must be one of 1, 2, 3, or 4" >&2
    exit 1
}
((ACTOR_NUM_NODES < TOTAL_NODES)) || {
    echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES} leaves no rollout node within TOTAL_NODES=${TOTAL_NODES}" >&2
    exit 1
}
[[ "${ACTOR_GPUS_PER_NODE}" == 8 ]] || {
    echo "AReaL Tau2 node-ratio comparisons require eight trainer GPUs per node" >&2
    exit 1
}
readonly ROLLOUT_NUM_NODES=$((TOTAL_NODES - ACTOR_NUM_NODES))
readonly EXPECTED_ROLLOUT_NUM_GPUS=$((ROLLOUT_NUM_NODES * ACTOR_GPUS_PER_NODE))
: "${ROLLOUT_NUM_GPUS:=${EXPECTED_ROLLOUT_NUM_GPUS}}"
[[ "${ROLLOUT_NUM_GPUS}" == "${EXPECTED_ROLLOUT_NUM_GPUS}" ]] || {
    echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES} requires ROLLOUT_NUM_GPUS=${EXPECTED_ROLLOUT_NUM_GPUS}" >&2
    exit 1
}

readonly MODEL_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
readonly TRAIN_GPUS=$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))
((TRAIN_GPUS % MODEL_PARALLEL_SIZE == 0)) || {
    echo "tp*cp does not divide ${TRAIN_GPUS} trainer GPUs" >&2
    exit 1
}
readonly VALIDATED_DATA_PARALLEL_SIZE=$((TRAIN_GPUS / MODEL_PARALLEL_SIZE))
((GLOBAL_BATCH_SIZE % VALIDATED_DATA_PARALLEL_SIZE == 0)) || {
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} does not support ACTOR_NUM_NODES=${ACTOR_NUM_NODES} (dp=${VALIDATED_DATA_PARALLEL_SIZE})" >&2
    exit 1
}
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${RECOMPUTE_GRANULARITY:=full}"
: "${OVERLAP_COMM:=0}"
: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=1}"
: "${SGLANG_MEM_FRACTION:=0.70}"

RM_TYPE=
CUSTOM_RM_PATH=
readonly VERIFIED_GENERATOR=experiments.src.environments.areal_tau2.generator.generate
: "${CUSTOM_GENERATE_FUNCTION_PATH:=${VERIFIED_GENERATOR}}"
[[ "${CUSTOM_GENERATE_FUNCTION_PATH}" == "${VERIFIED_GENERATOR}" ]] || {
    echo "AReaL Tau2 requires ${VERIFIED_GENERATOR}" >&2
    exit 1
}
: "${ZERO_REWARD_ON_TRUNCATED:=1}"
: "${ZERO_LOSS_ON_TRUNCATED:=0}"
[[ "${ZERO_REWARD_ON_TRUNCATED}" == 0 || "${ZERO_REWARD_ON_TRUNCATED}" == 1 ]] || {
    echo "ZERO_REWARD_ON_TRUNCATED must be 0 or 1" >&2
    exit 1
}
[[ "${ZERO_LOSS_ON_TRUNCATED}" == 0 || "${ZERO_LOSS_ON_TRUNCATED}" == 1 ]] || {
    echo "ZERO_LOSS_ON_TRUNCATED must be 0 or 1" >&2
    exit 1
}

: "${EVAL_INTERVAL:=0}"
: "${SKIP_EVAL_BEFORE_TRAIN:=0}"
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=100}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=10}"
: "${SAVE_TRIGGER_SENTINEL:=}"
: "${CHECKPOINT_COMPLETION_PREFLIGHT:=0}"
: "${DUMP_TRAIN_DATA:=0}"
: "${DUMP_POLICY_LOSS_DEBUG:=0}"
: "${FUSE_ONE_STEP_ACTOR_LOGPROBS:=1}"
: "${LOG_SAMPLE_STALENESS_METRICS:=1}"
: "${LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM:=0}"
: "${SAMPLE_STALENESS_MAX_BIN:=16}"
: "${LOG_UPDATE_DIAGNOSTICS:=1}"
: "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS:=1}"
: "${DEBUG_EXIT_AFTER_ROLLOUT:=}"
: "${DEBUG_FAIL_AFTER_ROLLOUT:=}"
: "${DEBUG_FAILURE_MARKER:=}"
: "${DEBUG_FAILURE_MIN_OUTSTANDING_GROUPS:=0}"
: "${DEBUG_FAILURE_MIN_COMPLETED_GROUPS:=0}"
: "${DEBUG_FAILURE_MIN_INFLIGHT_GROUPS:=0}"
: "${DEBUG_FAILURE_MIN_INFLIGHT_TOKENS:=0}"
: "${DEBUG_FAILURE_MIN_REGENERATE_GROUPS:=0}"
: "${REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID:=}"
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "SAVE_INTERVAL must be a positive integer" >&2
    exit 1
}
[[ "${SAVE_RETAIN_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "SAVE_RETAIN_INTERVAL must be a positive integer" >&2
    exit 1
}
((SAVE_RETAIN_INTERVAL % SAVE_INTERVAL == 0)) || {
    echo "SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL} must be a multiple of SAVE_INTERVAL=${SAVE_INTERVAL}" >&2
    exit 1
}
[[ "${EVAL_INTERVAL}" == 0 ]] || {
    echo "Tau three evaluation is held out and offline; EVAL_INTERVAL must be 0" >&2
    exit 1
}
[[ "${CHECKPOINT_COMPLETION_PREFLIGHT}" == 0 || "${CHECKPOINT_COMPLETION_PREFLIGHT}" == 1 ]] || {
    echo "CHECKPOINT_COMPLETION_PREFLIGHT must be 0 or 1" >&2
    exit 1
}
if [[ "${CHECKPOINT_COMPLETION_PREFLIGHT}" == 1 ]]; then
    [[ "${SAVE_HF}" == 1 && "${HF_SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
        echo "CHECKPOINT_COMPLETION_PREFLIGHT requires SAVE_HF=1 and a positive HF_SAVE_INTERVAL" >&2
        exit 1
    }
fi
[[ "${SAMPLE_STALENESS_MAX_BIN}" =~ ^[0-9]+$ ]] || {
    echo "SAMPLE_STALENESS_MAX_BIN must be non-negative" >&2
    exit 1
}
[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT}" || "${DEBUG_EXIT_AFTER_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "DEBUG_EXIT_AFTER_ROLLOUT must be empty or positive" >&2
    exit 1
}
[[ -z "${DEBUG_FAIL_AFTER_ROLLOUT}" || "${DEBUG_FAIL_AFTER_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "DEBUG_FAIL_AFTER_ROLLOUT must be empty or positive" >&2
    exit 1
}
[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT}" || -z "${DEBUG_FAIL_AFTER_ROLLOUT}" ]] || {
    echo "DEBUG_EXIT_AFTER_ROLLOUT and DEBUG_FAIL_AFTER_ROLLOUT are mutually exclusive" >&2
    exit 1
}
for replay_failure_threshold in \
    DEBUG_FAILURE_MIN_OUTSTANDING_GROUPS \
    DEBUG_FAILURE_MIN_COMPLETED_GROUPS \
    DEBUG_FAILURE_MIN_INFLIGHT_GROUPS \
    DEBUG_FAILURE_MIN_INFLIGHT_TOKENS \
    DEBUG_FAILURE_MIN_REGENERATE_GROUPS; do
    [[ "${!replay_failure_threshold}" =~ ^[0-9]+$ ]] || {
        echo "${replay_failure_threshold} must be non-negative" >&2
        exit 1
    }
done
unset replay_failure_threshold
[[ -z "${REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID}" \
    || "${REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID}" =~ ^[0-9]+$ ]] || {
    echo "REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID must be empty or non-negative" >&2
    exit 1
}

: "${MILES_NCCL_TRANSPORT:=efa}"
: "${ASYNC_SQSH_IMAGE_OVERRIDE:=${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0-tau3-1.0.1.sqsh}"
[[ -r "${ASYNC_SQSH_IMAGE_OVERRIDE}" ]] || {
    echo "Tau runtime image is missing: ${ASYNC_SQSH_IMAGE_OVERRIDE}" >&2
    exit 1
}
case "${MILES_NCCL_TRANSPORT}" in
    efa)
        [[ "${NCCL_IB_DISABLE:-0}" != 1 ]]
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

: "${CONFIG_TAG:=${TOTAL_NODES}node-t${ACTOR_NUM_NODES}-r${ROLLOUT_NUM_NODES}-e${TRAIN_EPOCHS}-rbs${ROLLOUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-u${NUM_ROLLOUT}-s${MAX_WEIGHT_STALENESS}-${TAU_USER_PROVIDER}}"
if [[ "${TAU_OVERLAP_DB_RESTORE_WITH_PREFILL}" == 1 ]]; then
    CONFIG_TAG="${CONFIG_TAG}-db-prefill-overlap"
fi

export MODEL_NAME HF_MODEL_NAME MODEL_PROFILE REF_LOAD PLACEMENT TASK_FAMILY DATASET_TAG ROLLOUT_SEMANTICS
export PROMPT_DATA PROMPT_SUMMARY TRAIN_SCRIPT
export ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW M2PO_BUDGET USE_OPSM OPSM_DELTA
export LR MAX_RESPONSE_LEN MAX_CONTEXT_LEN TRAIN_EPOCHS NUM_ROLLOUT TRAIN_SEED ROLLOUT_SEED
export ALLOW_NUM_ROLLOUT_OVERRIDE SCHEDULE_MODE EFFECTIVE_EPOCHS
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
export REQUESTED_PROMPT_GROUPS PLANNED_PROMPT_GROUPS PLANNED_TRAJECTORIES FINAL_BATCH_OVERSHOOT
export QUEUE_TYPE QUEUE_FACTOR MAX_WEIGHT_STALENESS STALENESS_REFERENCE PAUSE_GENERATION_MODE
export ASYNC_MAX_CONCURRENT_SAMPLES USE_REPLAY_BUFFER REPLAY_BUFFER_TYPE REPLAY_BUFFER_IDENTITY_TAG
export REPLAY_BUFFER_KEEP_LAST ALLOW_TAU_REPLAY_ABLATION LOG_REPLAY_RESUME_METRICS
export TAU_MAX_TURNS TAU_MAX_STEPS TAU_USER_PROVIDER TAU_USER_MODEL TAU_USER_MAX_TOKENS
export TAU_USER_TEMPERATURE TAU_USER_TOP_P TAU_USER_REQUEST_TIMEOUT TAU_USER_MAX_RETRIES
export TAU_TOOL_CALL_PARSER TAU_OVERLAP_DB_RESTORE_WITH_PREFILL TAU_LOG_OVERHEAD TAU_LOG_LEVEL
export NVIDIA_INFERENCE_BASE_URL
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_NODES ROLLOUT_NUM_GPUS TENSOR_PARALLEL_SIZE
export CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE MAX_TOKENS_PER_GPU RECOMPUTE_GRANULARITY
export OVERLAP_COMM ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export RM_TYPE CUSTOM_RM_PATH CUSTOM_GENERATE_FUNCTION_PATH ZERO_REWARD_ON_TRUNCATED
export ZERO_LOSS_ON_TRUNCATED
export EVAL_INTERVAL SKIP_EVAL_BEFORE_TRAIN SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL
export SAVE_TRIGGER_SENTINEL CHECKPOINT_COMPLETION_PREFLIGHT
export DUMP_TRAIN_DATA DUMP_POLICY_LOSS_DEBUG FUSE_ONE_STEP_ACTOR_LOGPROBS
export LOG_SAMPLE_STALENESS_METRICS LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM LOG_UPDATE_DIAGNOSTICS
export SAMPLE_STALENESS_MAX_BIN
export SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS DEBUG_EXIT_AFTER_ROLLOUT DEBUG_FAIL_AFTER_ROLLOUT
export DEBUG_FAILURE_MARKER DEBUG_FAILURE_MIN_OUTSTANDING_GROUPS DEBUG_FAILURE_MIN_COMPLETED_GROUPS
export DEBUG_FAILURE_MIN_INFLIGHT_GROUPS DEBUG_FAILURE_MIN_INFLIGHT_TOKENS
export DEBUG_FAILURE_MIN_REGENERATE_GROUPS
export REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID
export TOTAL_NODES MILES_NCCL_TRANSPORT ASYNC_SQSH_IMAGE_OVERRIDE NCCL_IB_DISABLE LD_LIBRARY_PATH
export NCCL_NET NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER WANDB_MODE
export WANDB_PROJECT

source "${REPO_ROOT}/experiments/common/run_identity.sh"
readonly REPLAY_RESUME_FAILURE_MARKER_NAME=replay-resume-intentional-failure.json
if [[ -n "${DEBUG_FAIL_AFTER_ROLLOUT}" && -z "${DEBUG_FAILURE_MARKER}" ]]; then
    DEBUG_FAILURE_MARKER="${CKPT_PATH}/${REPLAY_RESUME_FAILURE_MARKER_NAME}"
fi
export DEBUG_FAILURE_MARKER
export RUN_NAME CONFIG_TAG CKPT_PATH
source "${REPO_ROOT}/experiments/common/placement.sh"

export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
export RAY_HEAD_IP_FILE="${RAY_DONE_FLAG}.head-ip"
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f \
    "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done" \
    "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done.head-ip"
source "${REPO_ROOT}/experiments/common/clean_checkpoint.sh"

replay_resume_failure_preflight() {
    local host_checkpoint_path="$1"
    local expected_rollout_id="$2"
    python3 - \
        "${host_checkpoint_path}" \
        "${expected_rollout_id}" \
        "${USE_REPLAY_BUFFER}" \
        "${REPLAY_RESUME_FAILURE_MARKER_NAME}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
use_replay = bool(int(sys.argv[3]))
marker = root / sys.argv[4]

try:
    payload = json.loads(marker.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    raise SystemExit(f"invalid intentional-failure marker {marker}: {error}") from error
if payload.get("event") != "intentional_whole_job_failure" or payload.get("rollout_id") != expected:
    raise SystemExit(f"unexpected intentional-failure marker payload: {payload}")

tracker = root / "latest_checkpointed_iteration.txt"
try:
    tracker_iteration = int(tracker.read_text(encoding="utf-8").strip())
except (OSError, ValueError) as error:
    raise SystemExit(f"invalid checkpoint tracker {tracker}: {error}") from error
if tracker_iteration != expected:
    raise SystemExit(f"checkpoint tracker={tracker_iteration}, expected={expected}")
if payload.get("checkpoint", {}).get("tracker_iteration") != expected:
    raise SystemExit("failure marker does not name the committed model checkpoint")

replay = root / "rollout" / f"replay_buffer_{expected}.pt"
manifest_path = Path(f"{replay}.sha256.json")
stored_manifest = payload.get("checkpoint", {}).get("replay_manifest")
if use_replay:
    try:
        current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid replay manifest {manifest_path}: {error}") from error
    if not replay.is_file() or replay.stat().st_size == 0:
        raise SystemExit(f"missing replay state {replay}")
    if stored_manifest != current_manifest:
        raise SystemExit("failure marker replay identity differs from the published manifest")
elif stored_manifest is not None or replay.exists() or manifest_path.exists():
    raise SystemExit("checkpoint-only recovery unexpectedly found replay state")

print(
    "Replay resume failure preflight: "
    f"checkpoint={expected} replay={int(use_replay)} marker={marker}"
)
PY
}

tau_target_artifacts_complete() {
    local host_checkpoint_path="$1"
    python3 - "${host_checkpoint_path}" "${NUM_ROLLOUT}" "${HF_SAVE_INTERVAL}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
num_rollout = int(sys.argv[2])
hf_interval = int(sys.argv[3])
target = num_rollout - 1

tracker = root / "latest_checkpointed_iteration.txt"
try:
    checkpoint_iteration = int(tracker.read_text(encoding="utf-8").strip())
except (OSError, ValueError) as error:
    raise SystemExit(f"invalid final checkpoint tracker {tracker}: {error}") from error
if checkpoint_iteration < target:
    raise SystemExit(f"checkpoint {checkpoint_iteration} has not reached target {target}")

expected_hf_ids = set(range(hf_interval - 1, num_rollout, hf_interval))
expected_hf_ids.add(target)
for rollout_id in sorted(expected_hf_ids):
    hf_path = root / "hf" / str(rollout_id)
    config = hf_path / "config.json"
    index = hf_path / "model.safetensors.index.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise SystemExit(f"missing HF config: {config}")
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"invalid HF weight index {index}: {error}") from error
    for filename in set(weight_map.values()):
        shard = hf_path / filename
        if not shard.is_file() or shard.stat().st_size == 0:
            raise SystemExit(f"missing HF weight shard: {shard}")

replay = root / "rollout" / f"replay_buffer_{target}.pt"
for path in (replay, Path(f"{replay}.sha256.json")):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing final replay artifact: {path}")
PY
}

if [[ -n "${REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID}" ]]; then
    checkpoint_relative="${CKPT_PATH#/ckpt/training/}"
    host_checkpoint_path="${TRAIN_CKPT_DIR%/}/${checkpoint_relative}"
    replay_resume_failure_preflight \
        "${host_checkpoint_path}" \
        "${REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID}"
    unset checkpoint_relative host_checkpoint_path
fi

if [[ "${CHECKPOINT_COMPLETION_PREFLIGHT}" == 1 ]]; then
    checkpoint_relative="${CKPT_PATH#/ckpt/training/}"
    host_checkpoint_path="${TRAIN_CKPT_DIR%/}/${checkpoint_relative}"
    checkpoint_tracker="${host_checkpoint_path}/latest_checkpointed_iteration.txt"
    if [[ -s "${checkpoint_tracker}" ]]; then
        checkpoint_iteration="$(< "${checkpoint_tracker}")"
        if [[ "${checkpoint_iteration}" =~ ^[0-9]+$ ]] \
            && ((checkpoint_iteration >= NUM_ROLLOUT - 1)); then
            if tau_target_artifacts_complete "${host_checkpoint_path}"; then
                printf 'AReaL Tau2 target already complete: checkpoint=%s target_last_iteration=%s\n' \
                    "${checkpoint_iteration}" "$((NUM_ROLLOUT - 1))"
                exit 0
            fi
            echo "final tracker exists but the expected HF/replay artifact set is incomplete: ${host_checkpoint_path}" >&2
            exit 1
        fi
    fi
    unset checkpoint_relative host_checkpoint_path checkpoint_tracker checkpoint_iteration
fi

HOST_PROMPT_DATA="${DATASET_DIR}${PROMPT_DATA#/data}"
HOST_PROMPT_SUMMARY="${DATASET_DIR}${PROMPT_SUMMARY#/data}"
python3 - "${HOST_PROMPT_DATA}" "${HOST_PROMPT_SUMMARY}" <<'PY'
import hashlib
import json
import sys

data_path, summary_path = sys.argv[1:]
with open(summary_path, encoding="utf-8") as stream:
    summary = json.load(stream)
if summary.get("rows") != 1982:
    raise SystemExit("AReaL Tau2 summary does not contain 1,982 rows")
if summary.get("domain_counts") != {"airline": 1148, "retail": 563, "telecom": 271}:
    raise SystemExit("AReaL Tau2 domain counts differ from the pinned split")
if summary.get("interaction_mode") != "stateful_multi_turn_user_simulator_environment":
    raise SystemExit("AReaL Tau2 data is not conversational multi-turn")
if summary.get("dataset_repo") != "inclusionAI/AReaL-tau2-data":
    raise SystemExit("AReaL Tau2 data has the wrong source repository")
if summary.get("dataset_revision") != "1322eae337f836fe0e19bae14dab1eefc26bc983":
    raise SystemExit("AReaL Tau2 data has the wrong source revision")
if summary.get("verifier") != "areal_tau2_environment":
    raise SystemExit("AReaL Tau2 data has the wrong verifier")
if summary.get("tau_package_version") != "1.0.1" or len(summary.get("db_sha256", {})) != 9:
    raise SystemExit("AReaL Tau2 data has the wrong Tau/DB contract")
digest = hashlib.sha256()
with open(data_path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != summary.get("output_sha256"):
    raise SystemExit("AReaL Tau2 prepared data digest differs from its summary")
PY

printf 'AReaL Tau2 plan: schedule=%s effective_epochs=%s updates=%s prompt_groups=%s trajectories=%s final_batch_overshoot=%s\n' \
    "${SCHEDULE_MODE}" "${EFFECTIVE_EPOCHS}" "${NUM_ROLLOUT}" "${PLANNED_PROMPT_GROUPS}" \
    "${PLANNED_TRAJECTORIES}" "${FINAL_BATCH_OVERSHOOT}"

STEP_CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE%%\(*}}"
[[ "${STEP_CPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

if [[ "${MILES_NCCL_TRANSPORT}" == efa ]]; then
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

if [[ "${CHECKPOINT_COMPLETION_PREFLIGHT}" == 1 ]]; then
    checkpoint_relative="${CKPT_PATH#/ckpt/training/}"
    host_checkpoint_path="${TRAIN_CKPT_DIR%/}/${checkpoint_relative}"
    tau_target_artifacts_complete "${host_checkpoint_path}" || {
        echo "training returned successfully without its complete HF/replay artifact set" >&2
        exit 1
    }
    printf 'verified all ten-update HF exports and final resumable state under %s\n' \
        "${host_checkpoint_path}"
fi
