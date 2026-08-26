#!/bin/bash
#SBATCH --job-name=miles-swe-e2b-qwen3-4b-instruct
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
#SBATCH --signal=B:USR1@900
#SBATCH --export=NIL
#SBATCH --output=experiments/outputs/training/swe/r2e-gym-swe-rebench-v2/qwen3-4b-instruct-2507/%x-%j.log

# Four-node asynchronous repository-level SWE RL.  Harbor owns task setup,
# verification, and the E2B provider; Miles sends only task IDs and serves the
# policy through its session server.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
USER="$(id -un)"
export USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_when_ready.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || {
    echo "MILES_SWE_FIXED_EXPORTS must be 1" >&2
    exit 2
}
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached the training allocation" >&2
    exit 2
}
export WANDB_MODE=offline
export PYTHON_DOTENV_DISABLED=1
# E2B credentials belong only to the separately launched Harbor service.  The
# training and rollout workers receive its URL, never provider credentials.
if [[ -n "${E2B_API_KEY+x}${HARBOR_E2B_API_KEY+x}${E2B_ACCESS_TOKEN+x}" ]]; then
    echo "E2B provider credentials are forbidden in the training allocation" >&2
    exit 2
fi
unset E2B_API_KEY HARBOR_E2B_API_KEY E2B_API_URL HARBOR_E2B_API_URL
source "${REPO_ROOT}/experiments/env.sh"

[[ "${SLURM_JOB_NUM_NODES:?}" == 4 ]] || {
    echo "SWE validation requires exactly four GPU nodes" >&2
    exit 1
}
: "${AGENT_SERVER_URL:?set the URL of a Harbor agent server using its E2B environment selector}"
: "${HARBOR_RUN_SECRET:?export the Harbor /run bearer secret before sbatch}"
for secret_name in HARBOR_RUN_SECRET; do
    secret_value="${!secret_name}"
    (( ${#secret_value} >= 32 && ${#secret_value} <= 4096 )) && \
        [[ "$secret_value" != *$'\r'* && "$secret_value" != *$'\n'* ]] || {
        echo "${secret_name} must be 32-4096 characters without CR/LF" >&2
        exit 2
    }
done
unset secret_value
SWE_TRIAL_REQUEST_TIMEOUT_SEC="${SWE_TRIAL_REQUEST_TIMEOUT_SEC:-13200}"
[[ "${SWE_TRIAL_REQUEST_TIMEOUT_SEC}" == 13200 ]] || {
    echo "SWE_TRIAL_REQUEST_TIMEOUT_SEC is fixed at 13200 for the four-hour pilot" >&2
    exit 2
}
SWE_AGENT_SERVER_READY_TIMEOUT_SEC="${SWE_AGENT_SERVER_READY_TIMEOUT_SEC:-60}"
[[ "${SWE_AGENT_SERVER_READY_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] && \
    (( SWE_AGENT_SERVER_READY_TIMEOUT_SEC <= 60 )) || {
    echo "SWE_AGENT_SERVER_READY_TIMEOUT_SEC must be in [1, 60]" >&2
    exit 2
}
HARBOR_CLIENT_ID="slurm-${SLURM_JOB_ID:?}"
export HARBOR_CLIENT_ID
SWE_AGENT_NAME="${SWE_AGENT_NAME:-terminus-2}"
[[ "${SWE_AGENT_NAME}" == terminus-2 ]] || {
    echo "this private-cluster recipe only admits Harbor's host-process terminus-2 agent" >&2
    exit 1
}
: "${SWE_DATASET:=swe-rebench-v2-filtered-verified}"
case "${SWE_DATASET}" in
    r2e-gym-v1)
        DATASET_TAG=r2e-gym-v1
        ;;
    nemotron3-super-swe2-r2e)
        DATASET_TAG=nemotron3-super-swe2-r2e
        ;;
    nemotron3-super-swe2-swe-gym)
        DATASET_TAG=nemotron3-super-swe2-swe-gym
        ;;
    swe-gym)
        DATASET_TAG=swe-gym
        ;;
    swe-rebench-v2)
        DATASET_TAG=swe-rebench-v2
        ;;
    swe-rebench-v2-filtered-verified)
        DATASET_TAG=swe-rebench-v2-filtered-verified
        ;;
    nemotron3-ultra-swe-rebench-v2)
        DATASET_TAG=nemotron3-ultra-swe-rebench-v2
        ;;
    nemotron3-ultra-swe-gym)
        DATASET_TAG=nemotron3-ultra-swe-gym
        ;;
    *)
        echo "unknown SWE_DATASET=${SWE_DATASET}" >&2
        exit 1
        ;;
esac
PROMPT_DATA="/data/miles-swe/admitted/${DATASET_TAG}-train.jsonl"
ADMISSION_SUMMARY="/data/miles-swe/admitted/${DATASET_TAG}-train.summary.json"
HOST_ADMISSION_SUMMARY="${DATASET_DIR}/miles-swe/admitted/${DATASET_TAG}-train.summary.json"

TASK_FAMILY=swe
TRAIN_SCRIPT=/root/miles/experiments/scripts/swe/async/r2e-gym-swe-rebench-v2/qwen3-4b-instruct-2507/train.sh
MODEL_NAME=Qwen3-4B-Instruct-2507
HF_MODEL_NAME=Qwen3-4B-Instruct-2507
MODEL_PROFILE=qwen3-4B-Instruct-2507
PLACEMENT=async

ADVANTAGE_ESTIMATOR=grpo
ENTROPY_COEF=0.0
KL_LOSS_COEF=0.0
EPS_CLIP=0.2
EPS_CLIP_HIGH=0.28
EPS_CLIP_C=
RATIO_DENOMINATOR=actor
IS_CORRECTION=tis
TIS_CLIP=2.0
TIS_CLIP_LOW=0
MIS_PROFILE=
M2PO_BUDGET=0.04
USE_OPSM=0
OPSM_DELTA=1e-4
LR=1e-6
MAX_RESPONSE_LEN=16384
MAX_SEQUENCE_LEN=32768
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
TRAIN_SEED=1234
ROLLOUT_SEED=42
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT=16
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
NUM_STEPS_PER_ROLLOUT=1

QUEUE_TYPE=queue-recycle
QUEUE_FACTOR=1
MAX_WEIGHT_STALENESS=4
STALENESS_REFERENCE=prefill
PAUSE_GENERATION_MODE=in_place
ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-64}"
USE_REPLAY_BUFFER=0
REPLAY_BUFFER_TYPE=rollout

ACTOR_NUM_NODES=1
ACTOR_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=24
TENSOR_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
EXPERT_PARALLEL_SIZE=1
MAX_TOKENS_PER_GPU=32768
RECOMPUTE_GRANULARITY=full
ROLLOUT_NUM_GPUS_PER_ENGINE=1
SGLANG_MEM_FRACTION=0.70
SGLANG_TOOL_CALL_PARSER=qwen25
SGLANG_REASONING_PARSER=qwen3
TITO_MODEL=qwen3

RM_TYPE=
CUSTOM_RM_PATH=experiments.src.reward_sets.swe.reward
CUSTOM_GENERATE_FUNCTION_PATH=miles.rollout.generate_hub.agentic_tool_call.generate
CUSTOM_AGENT_FUNCTION_PATH=swe_agent_function.run
ROLLOUT_FUNCTION_PATH=generate.RolloutFn
DYNAMIC_FILTER_PATH=miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-model}"
ZERO_REWARD_ON_TRUNCATED=1

EVAL_INTERVAL=0
SAVE_INTERVAL=10
SAVE_RETAIN_INTERVAL=100
SAVE_HF=1
HF_SAVE_INTERVAL=10
FUSE_ONE_STEP_ACTOR_LOGPROBS=1
LOG_SAMPLE_STALENESS_METRICS=1
LOG_UPDATE_DIAGNOSTICS=1
SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=1
DEBUG_EXIT_AFTER_ROLLOUT="${DEBUG_EXIT_AFTER_ROLLOUT:-}"

MILES_NCCL_TRANSPORT=efa
ASYNC_SQSH_IMAGE_OVERRIDE=/lustre/fsw/portfolios/coreai/users/kfujii/containers/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0.sqsh
[[ "${NCCL_IB_DISABLE:-0}" != 1 ]] || {
    echo "EFA mode refuses NCCL_IB_DISABLE=1" >&2
    exit 1
}
NCCL_IB_DISABLE=0
NCCL_NET_PLUGIN=ofi
NCCL_TUNER_PLUGIN=ofi
NCCL_NET="AWS Libfabric"
FI_PROVIDER=efa
LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

CONFIG_TAG="4node-e2b-16k-lr${LR}-rbs${ROLLOUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-s${MAX_WEIGHT_STALENESS}"

export TASK_FAMILY DATASET_TAG PROMPT_DATA ADMISSION_SUMMARY TRAIN_SCRIPT MODEL_NAME HF_MODEL_NAME MODEL_PROFILE PLACEMENT
export ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
export RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW MIS_PROFILE M2PO_BUDGET USE_OPSM OPSM_DELTA LR MAX_RESPONSE_LEN MAX_SEQUENCE_LEN
export NUM_ROLLOUT TRAIN_SEED ROLLOUT_SEED ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE
export NUM_STEPS_PER_ROLLOUT QUEUE_TYPE QUEUE_FACTOR MAX_WEIGHT_STALENESS STALENESS_REFERENCE PAUSE_GENERATION_MODE
export ASYNC_MAX_CONCURRENT_SAMPLES USE_REPLAY_BUFFER REPLAY_BUFFER_TYPE
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS TENSOR_PARALLEL_SIZE
export CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE MAX_TOKENS_PER_GPU RECOMPUTE_GRANULARITY
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION SGLANG_TOOL_CALL_PARSER SGLANG_REASONING_PARSER
export TITO_MODEL RM_TYPE CUSTOM_RM_PATH CUSTOM_GENERATE_FUNCTION_PATH CUSTOM_AGENT_FUNCTION_PATH
export ROLLOUT_FUNCTION_PATH DYNAMIC_FILTER_PATH AGENT_MODEL_NAME AGENT_SERVER_URL SWE_AGENT_NAME
export SWE_TRIAL_REQUEST_TIMEOUT_SEC HARBOR_RUN_SECRET
export ZERO_REWARD_ON_TRUNCATED EVAL_INTERVAL SAVE_INTERVAL SAVE_RETAIN_INTERVAL
export SAVE_HF HF_SAVE_INTERVAL FUSE_ONE_STEP_ACTOR_LOGPROBS LOG_SAMPLE_STALENESS_METRICS
export LOG_UPDATE_DIAGNOSTICS SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS DEBUG_EXIT_AFTER_ROLLOUT
export MILES_NCCL_TRANSPORT ASYNC_SQSH_IMAGE_OVERRIDE NCCL_IB_DISABLE NCCL_NET_PLUGIN
export NCCL_TUNER_PLUGIN NCCL_NET FI_PROVIDER LD_LIBRARY_PATH WANDB_MODE CONFIG_TAG
export PYTHON_DOTENV_DISABLED

source "${REPO_ROOT}/experiments/common/run_identity.sh"
export RUN_NAME CONFIG_TAG CKPT_PATH
source "${REPO_ROOT}/experiments/common/placement.sh"

export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
export RAY_HEAD_IP_FILE="${RAY_DONE_FLAG}.head-ip"
export RAY_DASHBOARD_HOST=127.0.0.1
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f \
    "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done" \
    "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done.head-ip"
source "${REPO_ROOT}/experiments/common/clean_checkpoint.sh"

STEP_CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-${SLURM_JOB_CPUS_PER_NODE%%\(*}}"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is missing or unsafe" >&2
    exit 2
}
MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

# Never forward the submission shell wholesale: it may contain unrelated cloud,
# registry, model-provider, or tracking credentials. Slurm always supplies its
# own SLURM_* variables in addition to these fixed names.
join_export_names() {
    local IFS=,
    printf '%s' "$*"
}
EFA_PREFLIGHT_EXPORT_NAMES=(
    FI_PROVIDER LD_LIBRARY_PATH MILES_NCCL_TRANSPORT NCCL_IB_DISABLE NCCL_NET
    NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN PYTHON_DOTENV_DISABLED WANDB_DISABLED
    WANDB_MODE
)
TRAINING_EXPORT_NAMES=(
    ACTOR_GPUS_PER_NODE ACTOR_NUM_NODES ADVANTAGE_ESTIMATOR
    ADMISSION_SUMMARY AGENT_MODEL_NAME AGENT_SERVER_URL ASYNC_MAX_CONCURRENT_SAMPLES
    CACHE_DIR CKPT_PATH CONFIG_TAG CONTEXT_PARALLEL_SIZE CUDA_CACHE_MAXSIZE
    CUDA_CACHE_PATH DATASET_TAG DEBUG_EXIT_AFTER_ROLLOUT DYNAMIC_FILTER_PATH
    ENTROPY_COEF EPS_CLIP EPS_CLIP_C EPS_CLIP_HIGH EVAL_INTERVAL
    EXPERT_PARALLEL_SIZE
    FI_PROVIDER FUSE_ONE_STEP_ACTOR_LOGPROBS GLOBAL_BATCH_SIZE GPUS_PER_NODE
    HARBOR_CLIENT_ID HARBOR_RUN_SECRET HF_MODEL_NAME HF_SAVE_INTERVAL
    KL_LOSS_COEF LD_LIBRARY_PATH LOG_SAMPLE_STALENESS_METRICS
    LOG_UPDATE_DIAGNOSTICS LR M2PO_BUDGET MAX_RESPONSE_LEN MAX_SEQUENCE_LEN
    MAX_TOKENS_PER_GPU MAX_WEIGHT_STALENESS MILES_NCCL_TRANSPORT MIS_PROFILE
    MODEL_NAME MODEL_PROFILE NCCL_IB_DISABLE NCCL_NET NCCL_NET_PLUGIN
    NCCL_TUNER_PLUGIN NODE_RAY_TEMP_DIR NUM_ROLLOUT NUM_STEPS_PER_ROLLOUT
    N_SAMPLES_PER_PROMPT
    OPSM_DELTA OUTPUT_DIR PAUSE_GENERATION_MODE PLACEMENT PROMPT_DATA
    PYTHON_DOTENV_DISABLED QUEUE_FACTOR QUEUE_TYPE RATIO_DENOMINATOR
    RAY_DASHBOARD_HOST RAY_DONE_FLAG RAY_HEAD_IP_FILE RECOMPUTE_GRANULARITY
    REPLAY_BUFFER_TYPE
    RM_TYPE ROLLOUT_BATCH_SIZE ROLLOUT_FUNCTION_PATH ROLLOUT_NUM_GPUS
    ROLLOUT_NUM_GPUS_PER_ENGINE ROLLOUT_SEED RUN_NAME SAVE_HF
    SAVE_INTERVAL SAVE_RETAIN_INTERVAL SGLANG_DG_CACHE_DIR
    SGLANG_DG_CACHE_DIR_PER_PROCESS SGLANG_MEM_FRACTION
    SGLANG_REASONING_PARSER SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS
    SGLANG_TOOL_CALL_PARSER STALENESS_REFERENCE SWE_AGENT_NAME
    SWE_TRIAL_REQUEST_TIMEOUT_SEC TASK_FAMILY TENSOR_PARALLEL_SIZE TIS_CLIP
    TIS_CLIP_LOW TITO_MODEL TORCHINDUCTOR_CACHE_DIR TORCH_HOME TRAINING_ATTENTION_BACKEND
    TRAIN_SCRIPT TRAIN_SEED TRITON_CACHE_DIR USE_OPSM USE_REPLAY_BUFFER
    VLLM_CACHE_ROOT WANDB_DISABLED WANDB_MODE ZERO_REWARD_ON_TRUNCATED
    CUSTOM_AGENT_FUNCTION_PATH CUSTOM_GENERATE_FUNCTION_PATH CUSTOM_RM_PATH
)
EFA_PREFLIGHT_EXPORT_CSV="$(join_export_names "${EFA_PREFLIGHT_EXPORT_NAMES[@]}")"
TRAINING_EXPORT_CSV="$(join_export_names "${TRAINING_EXPORT_NAMES[@]}")"

PYTHONPATH="${REPO_ROOT}" PYTHONDONTWRITEBYTECODE=1 \
    python3 \
    "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/wait_for_agent_server.py" \
    --server-url "${AGENT_SERVER_URL}" \
    --admission-summary "${HOST_ADMISSION_SUMMARY}" \
    --timeout-seconds "${SWE_AGENT_SERVER_READY_TIMEOUT_SEC}"

echo "running EFA fail-closed preflight on every node"
training_step_pid=""
drain_harbor_client() {
    PYTHONPATH="${REPO_ROOT}" PYTHONDONTWRITEBYTECODE=1 \
        python3 \
        "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/drain_harbor_client.py"
}
stop_training_step() {
    if [[ -n "${training_step_pid}" ]] && kill -0 "${training_step_pid}" 2>/dev/null; then
        kill -TERM "${training_step_pid}" 2>/dev/null || true
    fi
}
handle_pretimeout() {
    echo "Slurm pre-timeout signal received; draining Harbor trials" >&2
    drain_harbor_client || true
    stop_training_step
}
cleanup_training() {
    local status=$?
    trap - EXIT USR1 TERM
    drain_harbor_client || true
    stop_training_step
    if [[ -n "${training_step_pid}" ]]; then
        wait "${training_step_pid}" 2>/dev/null || true
    fi
    return "${status}"
}
trap handle_pretimeout USR1
trap 'handle_pretimeout; exit 143' TERM
trap cleanup_training EXIT

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=0 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --no-container-mount-home \
    --export="${EFA_PREFLIGHT_EXPORT_CSV}" \
    bash /root/miles/experiments/common/check_efa.sh

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${STEP_CPUS_PER_NODE}" \
    --mem=0 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export="${TRAINING_EXPORT_CSV}" \
    bash /root/miles/experiments/common/run_with_efa_env.sh bash "${TRAIN_SCRIPT}" &
training_step_pid=$!
set +e
wait "${training_step_pid}"
training_status=$?
set -e
training_step_pid=""
exit "${training_status}"
