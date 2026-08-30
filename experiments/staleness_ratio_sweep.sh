#!/bin/bash
# Sweep max weight staleness and the trainer:rollout node ratio. The math async
# recipe owns every learning default; this launcher overrides only the two sweep
# axes, resource placement, and run identity.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
RECIPE="experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch"
RECIPE_PATH="${REPO_ROOT}/${RECIPE}"
COLOCATED_RECIPE="experiments/scripts/math/sync/dapo-math-p10-90/qwen3-4b/run.sbatch"
COLOCATED_RECIPE_PATH="${REPO_ROOT}/${COLOCATED_RECIPE}"
source "${REPO_ROOT}/experiments/env.sh"

# The completed staleness-ratio cohort used this image. Matched mode resolves
# aliases before comparing, so the fsw/fs1 spelling may differ but the image
# itself may not. Keep this independent of SHARED_WS/CONTAINER_DIR overrides.
MATCHED_BASELINE_SQSH_IMAGE="/lustre/fsw/portfolios/coreai/users/kfujii/container/miles-staleness-weight-boundaries-f994b9aed.sqsh"
MATCHED_SQSH_IMAGE_RESOLVED=""
MATCHED_SQSH_IMAGE_STAT=""
MATCHED_REPO_ROOT_RESOLVED=""
MATCHED_PARTIAL_OVER_SAMPLING_BATCH_SIZE=256
MATCHED_SWITCH_METRIC_CONTRACT=colocate_switch_metrics_v1

: "${TOTAL_NODES:=8}"
: "${STALENESS_LEVELS:=1 2 4 8}"
: "${RATIOS:=1:7 2:6 3:5 4:4}"
: "${PARTITION:=batch}"
: "${WALL:=04:00:00}"
: "${CHAIN_JOBS:=10}"
WANDB_PROJECT=async-rl-dapo-math
if [[ -v RUN_NAMESPACE ]]; then
    RUN_NAMESPACE_WAS_EXPLICIT=1
else
    RUN_NAMESPACE_WAS_EXPLICIT=0
    RUN_NAMESPACE="sr-$(date +%Y%m%d-%H%M%S)-p$$"
fi
: "${PARTIAL_OVER_SAMPLING_BATCH_SIZE:=${MATCHED_PARTIAL_OVER_SAMPLING_BATCH_SIZE}}"

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b"
IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async RL waits for long math generations between optimizer steps"}}'

SUBMIT=0
INCLUDE_COLOCATED=0
DISABLE_ZERO_REWARD_ON_TRUNCATED=0
ENABLE_ZERO_LOSS_ON_TRUNCATED=0
TOTAL_LENGTH_32K=0
MATCH_PARTIAL_CONCURRENCY=0
CLEAN_CHECKPOINT=0
RESUME_CHAIN=0
RESUME_MATCHED_CHAIN=0
RERUN_CLEAN_MATCHED_ARMS=0
declare -a REQUESTED_POINTS=()

usage() {
    cat <<'EOF'
usage: experiments/staleness_ratio_sweep.sh [--submit] [--include-colocated]
                                            [--disable-zero-reward-on-truncated]
                                            [--zero-loss-on-truncated]
                                            [--total-length-32k]
                                            [--matched-partial-concurrency]
                                            [--rerun-clean-matched-arms]
                                            [--resume-chain]
                                            [--resume-matched-chain]
                                            [--clean-checkpoint]
                                            [--point M:T:R ...]

Without --submit, print the selected grid. M is max weight staleness, T is
trainer nodes, and R is rollout nodes. With no --point, the grid comes from
STALENESS_LEVELS and RATIOS. Learning settings come directly from:

  experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch

--include-colocated adds one on-policy arm using all TOTAL_NODES. The default
grid therefore contains 16 async arms plus one colocated arm. With
--clean-checkpoint, only the first allocation in each chain removes that arm's
exact derived checkpoint directory before starting.

--disable-zero-reward-on-truncated changes only the recipe default
ZERO_REWARD_ON_TRUNCATED from 1 to 0. It cannot be combined with matched
partial-concurrency modes.

--zero-loss-on-truncated sets ZERO_REWARD_ON_TRUNCATED=0 and
ZERO_LOSS_ON_TRUNCATED=1. Truncated samples retain their reward for group
baseline computation but contribute no training loss. It cannot be combined
with matched partial-concurrency modes.

--total-length-32k keeps ZERO_REWARD_ON_TRUNCATED=1 and sets the response
ceiling to 32768. The effective prompt+response ceiling is 32767 so SGLang can
reserve the final position in the model's 32768-token context window. The
inference request therefore receives at most 32767 - prompt_tokens new tokens.
This mode is async-only and cannot be combined with truncation ablations or
matched partial-concurrency modes.

--matched-partial-concurrency creates the five-arm follow-up: one colocated
partial-rollout arm at PARTIAL_OVER_SAMPLING_BATCH_SIZE=256, plus the
four node ratios at max weight staleness 4. It derives async concurrency as
over_sampling_batch_size * n_samples_per_prompt, requires fresh checkpoint
identities, and cannot be combined with --include-colocated or
--clean-checkpoint. To preserve the completed cohort's allocation and restart
protocol, this mode fixes TOTAL_NODES=8, PARTITION=batch, WALL=04:00:00, and
CHAIN_JOBS=10.

--rerun-clean-matched-arms creates a fresh two-arm replacement for the
colocated partial-rollout and async t1r7 arms from the matched cohort. These
are the two arms whose old terminal segments used contradictory truncation
flags. This mode retains the matched protocol, forces zero reward on truncated
on and zero loss on truncated off, and requires a fresh RUN_NAMESPACE.

--resume-chain appends CHAIN_JOBS new allocations per selected arm to an
existing namespace. Set RUN_NAMESPACE to the original namespace. Every
selected arm must already have a checkpoint, and this option never removes
checkpoints. CHAIN_JOBS counts only the newly submitted allocations.

--resume-matched-chain resumes an existing matched-partial-concurrency
namespace from its saved checkpoints and submits exactly nine chained jobs per
arm. Set RUN_NAMESPACE to the original namespace. This option never removes
checkpoints.

Useful environment overrides: TOTAL_NODES, STALENESS_LEVELS, RATIOS,
TRAINING_BUFFER_QUEUE_SIZE, CHAIN_JOBS, PARTITION, WALL, and RUN_NAMESPACE.
--resume-matched-chain ignores CHAIN_JOBS and always uses nine.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --submit)
            SUBMIT=1
            shift
            ;;
        --include-colocated)
            INCLUDE_COLOCATED=1
            shift
            ;;
        --disable-zero-reward-on-truncated)
            DISABLE_ZERO_REWARD_ON_TRUNCATED=1
            shift
            ;;
        --zero-loss-on-truncated)
            ENABLE_ZERO_LOSS_ON_TRUNCATED=1
            shift
            ;;
        --total-length-32k)
            TOTAL_LENGTH_32K=1
            shift
            ;;
        --matched-partial-concurrency)
            MATCH_PARTIAL_CONCURRENCY=1
            shift
            ;;
        --rerun-clean-matched-arms)
            MATCH_PARTIAL_CONCURRENCY=1
            RERUN_CLEAN_MATCHED_ARMS=1
            shift
            ;;
        --resume-chain)
            RESUME_CHAIN=1
            shift
            ;;
        --resume-matched-chain)
            MATCH_PARTIAL_CONCURRENCY=1
            RESUME_MATCHED_CHAIN=1
            shift
            ;;
        --clean-checkpoint)
            CLEAN_CHECKPOINT=1
            shift
            ;;
        --point)
            [[ $# -ge 2 ]] || { echo "--point needs M:T:R" >&2; exit 2; }
            REQUESTED_POINTS+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if (( INCLUDE_COLOCATED == 1 && MATCH_PARTIAL_CONCURRENCY == 1 )); then
    echo "--include-colocated and --matched-partial-concurrency are mutually exclusive" >&2
    exit 2
fi
if (( DISABLE_ZERO_REWARD_ON_TRUNCATED == 1 && MATCH_PARTIAL_CONCURRENCY == 1 )); then
    echo "--disable-zero-reward-on-truncated cannot be combined with matched partial-concurrency modes" >&2
    exit 2
fi
if (( ENABLE_ZERO_LOSS_ON_TRUNCATED == 1 && MATCH_PARTIAL_CONCURRENCY == 1 )); then
    echo "--zero-loss-on-truncated cannot be combined with matched partial-concurrency modes" >&2
    exit 2
fi
if (( DISABLE_ZERO_REWARD_ON_TRUNCATED == 1 && ENABLE_ZERO_LOSS_ON_TRUNCATED == 1 )); then
    echo "--disable-zero-reward-on-truncated and --zero-loss-on-truncated are mutually exclusive" >&2
    exit 2
fi
if (( TOTAL_LENGTH_32K == 1 \
      && (DISABLE_ZERO_REWARD_ON_TRUNCATED == 1 \
          || ENABLE_ZERO_LOSS_ON_TRUNCATED == 1 \
          || MATCH_PARTIAL_CONCURRENCY == 1 \
          || INCLUDE_COLOCATED == 1) )); then
    echo "--total-length-32k requires async zero-reward-on-truncated mode without colocated or matched arms" >&2
    exit 2
fi
if (( DISABLE_ZERO_REWARD_ON_TRUNCATED == 1 || ENABLE_ZERO_LOSS_ON_TRUNCATED == 1 )); then
    ZERO_REWARD_ON_TRUNCATED=0
fi
if (( ENABLE_ZERO_LOSS_ON_TRUNCATED == 1 )); then
    ZERO_LOSS_ON_TRUNCATED=1
fi
if (( MATCH_PARTIAL_CONCURRENCY == 1 && CLEAN_CHECKPOINT == 1 )); then
    echo "--matched-partial-concurrency requires fresh identities; do not use --clean-checkpoint" >&2
    exit 2
fi
if (( RESUME_CHAIN == 1 && MATCH_PARTIAL_CONCURRENCY == 1 )); then
    echo "--resume-chain cannot be combined with matched partial-concurrency modes" >&2
    exit 2
fi
if (( RESUME_CHAIN == 1 && RUN_NAMESPACE_WAS_EXPLICIT == 0 )); then
    echo "--resume-chain requires an explicit RUN_NAMESPACE" >&2
    exit 2
fi
if (( RESUME_CHAIN == 1 && CLEAN_CHECKPOINT == 1 )); then
    echo "--resume-chain never removes checkpoints; do not use --clean-checkpoint" >&2
    exit 2
fi
if (( RESUME_MATCHED_CHAIN == 1 && RUN_NAMESPACE_WAS_EXPLICIT == 0 )); then
    echo "--resume-matched-chain requires an explicit RUN_NAMESPACE" >&2
    exit 2
fi
if (( RERUN_CLEAN_MATCHED_ARMS == 1 && RESUME_MATCHED_CHAIN == 1 )); then
    echo "--rerun-clean-matched-arms requires a fresh namespace and cannot resume a matched chain" >&2
    exit 2
fi
if (( RESUME_MATCHED_CHAIN == 1 )); then
    CHAIN_JOBS=9
fi
if (( MATCH_PARTIAL_CONCURRENCY == 1 && ${#REQUESTED_POINTS[@]} > 0 )); then
    echo "--matched-partial-concurrency owns its exact five-arm grid; do not use --point" >&2
    exit 2
fi

if (( CLEAN_CHECKPOINT == 1 && SUBMIT == 0 )); then
    echo "--clean-checkpoint requires --submit" >&2
    exit 2
fi

recipe_default_from() {
    local recipe_path="$1"
    local key="$2"
    local value
    value="$(sed -n 's/^: "${'"${key}"':=\([^}]*\)}".*/\1/p' "${recipe_path}" | head -n 1)"
    [[ -n "${value}" ]] || {
        echo "could not read ${key} from ${recipe_path}" >&2
        return 1
    }
    printf '%s\n' "${value}"
}

recipe_default() {
    recipe_default_from "${RECIPE_PATH}" "$1"
}

recipe_or_environment() {
    local key="$1"
    if [[ -v "${key}" ]]; then
        printf '%s\n' "${!key}"
    else
        recipe_default "${key}"
    fi
}

require_setting() {
    local key="$1"
    local expected="$2"
    local actual
    actual="$(recipe_or_environment "${key}")"
    [[ "${actual}" == "${expected}" ]] || {
        echo "this sweep requires ${key}=${expected}, got ${actual}" >&2
        return 1
    }
}

[[ "${TOTAL_NODES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TOTAL_NODES must be a positive integer" >&2
    exit 1
}
[[ "${CHAIN_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "CHAIN_JOBS must be a positive integer" >&2
    exit 1
}
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "RUN_NAMESPACE contains unsupported characters: ${RUN_NAMESPACE}" >&2
    exit 1
}

GPUS_PER_NODE="$(recipe_or_environment ACTOR_GPUS_PER_NODE)"
GLOBAL_BATCH="$(recipe_or_environment GLOBAL_BATCH_SIZE)"
ROLLOUT_BATCH="$(recipe_or_environment ROLLOUT_BATCH_SIZE)"
SAMPLES_PER_PROMPT="$(recipe_or_environment N_SAMPLES_PER_PROMPT)"
TENSOR_PARALLEL="$(recipe_or_environment TENSOR_PARALLEL_SIZE)"
CONTEXT_PARALLEL="$(recipe_or_environment CONTEXT_PARALLEL_SIZE)"
MAX_TOKENS_PER_GPU_VALUE="$(recipe_or_environment MAX_TOKENS_PER_GPU)"
QUEUE_POLICY="$(recipe_or_environment QUEUE_TYPE)"
TRAINING_BUFFER_QUEUE_SIZE_VALUE="$(recipe_or_environment TRAINING_BUFFER_QUEUE_SIZE)"
STALENESS_REFERENCE_VALUE="$(recipe_or_environment STALENESS_REFERENCE)"
NUM_ROLLOUT_VALUE="$(recipe_or_environment NUM_ROLLOUT)"
NUM_STEPS_PER_ROLLOUT_VALUE="$(recipe_or_environment NUM_STEPS_PER_ROLLOUT)"
MAX_RESPONSE_LEN_VALUE="$(recipe_or_environment MAX_RESPONSE_LEN)"
ROLLOUT_MAX_CONTEXT_LEN_VALUE="$(recipe_or_environment ROLLOUT_MAX_CONTEXT_LEN)"
ZERO_REWARD_ON_TRUNCATED_VALUE="$(recipe_or_environment ZERO_REWARD_ON_TRUNCATED)"
ZERO_LOSS_ON_TRUNCATED_VALUE="$(recipe_or_environment ZERO_LOSS_ON_TRUNCATED)"
USE_REPLAY_BUFFER_VALUE="$(recipe_or_environment USE_REPLAY_BUFFER)"
REPLAY_BUFFER_TYPE_VALUE="$(recipe_or_environment REPLAY_BUFFER_TYPE)"
FUSE_ONE_STEP_ACTOR_LOGPROBS_VALUE="$(recipe_or_environment FUSE_ONE_STEP_ACTOR_LOGPROBS)"
SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS_VALUE="$(recipe_or_environment SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS)"
SAMPLE_STALENESS_MAX_BIN_VALUE="$(recipe_or_environment SAMPLE_STALENESS_MAX_BIN)"
SAVE_HF_VALUE="$(recipe_or_environment SAVE_HF)"
HF_SAVE_INTERVAL_VALUE="$(recipe_or_environment HF_SAVE_INTERVAL)"
SAVE_INTERVAL_VALUE="$(recipe_or_environment SAVE_INTERVAL)"
SAVE_RETAIN_INTERVAL_VALUE="$(recipe_or_environment SAVE_RETAIN_INTERVAL)"
ASYNC_GPUS_PER_ENGINE="$(recipe_or_environment ROLLOUT_NUM_GPUS_PER_ENGINE)"
ASYNC_MEM_FRACTION="$(recipe_or_environment SGLANG_MEM_FRACTION)"
COLOCATED_GPUS_PER_ENGINE="$(recipe_default_from "${COLOCATED_RECIPE_PATH}" ROLLOUT_NUM_GPUS_PER_ENGINE)"
COLOCATED_MEM_FRACTION="$(recipe_default_from "${COLOCATED_RECIPE_PATH}" SGLANG_MEM_FRACTION)"
ASYNC_MAX_CONCURRENT_SAMPLES_VALUE="${ASYNC_MAX_CONCURRENT_SAMPLES:-}"

[[ "${ROLLOUT_BATCH}" =~ ^[1-9][0-9]*$ && "${SAMPLES_PER_PROMPT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "rollout batch and samples per prompt must be positive integers" >&2
    exit 1
}
[[ "${MAX_TOKENS_PER_GPU_VALUE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_TOKENS_PER_GPU must be a positive integer" >&2
    exit 1
}
[[ "${TRAINING_BUFFER_QUEUE_SIZE_VALUE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAINING_BUFFER_QUEUE_SIZE must be a positive integer" >&2
    exit 1
}
[[ "${ROLLOUT_MAX_CONTEXT_LEN_VALUE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ROLLOUT_MAX_CONTEXT_LEN must be a positive integer" >&2
    exit 1
}
[[ "${SAMPLE_STALENESS_MAX_BIN_VALUE}" =~ ^[0-9]+$ ]] || {
    echo "SAMPLE_STALENESS_MAX_BIN must be a nonnegative integer" >&2
    exit 1
}
if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}" ]]; then
    [[ "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ASYNC_MAX_CONCURRENT_SAMPLES must be a positive integer" >&2
        exit 1
    }
    (( ASYNC_MAX_CONCURRENT_SAMPLES_VALUE % SAMPLES_PER_PROMPT == 0 )) || {
        echo "ASYNC_MAX_CONCURRENT_SAMPLES must be divisible by N_SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT}" >&2
        exit 1
    }
fi
if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
    [[ "${PARTIAL_OVER_SAMPLING_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
        echo "PARTIAL_OVER_SAMPLING_BATCH_SIZE must be a positive integer" >&2
        exit 1
    }
    [[ "${PARTIAL_OVER_SAMPLING_BATCH_SIZE}" == "${MATCHED_PARTIAL_OVER_SAMPLING_BATCH_SIZE}" ]] || {
        echo "matched mode requires PARTIAL_OVER_SAMPLING_BATCH_SIZE=${MATCHED_PARTIAL_OVER_SAMPLING_BATCH_SIZE}," \
             "got ${PARTIAL_OVER_SAMPLING_BATCH_SIZE}" >&2
        exit 1
    }
    (( PARTIAL_OVER_SAMPLING_BATCH_SIZE > ROLLOUT_BATCH )) || {
        echo "PARTIAL_OVER_SAMPLING_BATCH_SIZE must exceed ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH}" >&2
        exit 1
    }
    MATCHED_ASYNC_CONCURRENCY=$(( PARTIAL_OVER_SAMPLING_BATCH_SIZE * SAMPLES_PER_PROMPT ))
    if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}" \
          && "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}" -ne "${MATCHED_ASYNC_CONCURRENCY}" ]]; then
        echo "matched mode derives ASYNC_MAX_CONCURRENT_SAMPLES=${MATCHED_ASYNC_CONCURRENCY}," \
             "got ${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}" >&2
        exit 1
    fi
    ASYNC_MAX_CONCURRENT_SAMPLES_VALUE="${MATCHED_ASYNC_CONCURRENCY}"

    # This cohort is paired against the completed production sweep. Refuse
    # ambient training overrides instead of silently letting --export=ALL turn
    # the follow-up into a multi-variable experiment.
    matched_override_keys=(
        ADVANTAGE_ESTIMATOR ENTROPY_COEF KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C
        RATIO_DENOMINATOR IS_CORRECTION TIS_CLIP TIS_CLIP_LOW MIS_PROFILE M2PO_BUDGET
        USE_OPSM OPSM_DELTA LR MAX_RESPONSE_LEN ROLLOUT_MAX_CONTEXT_LEN NUM_ROLLOUT TRAIN_SEED ROLLOUT_SEED
        ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_STEPS_PER_ROLLOUT
        QUEUE_TYPE QUEUE_FACTOR TRAINING_BUFFER_QUEUE_SIZE MAX_WEIGHT_STALENESS
        STALENESS_REFERENCE PAUSE_GENERATION_MODE
        USE_REPLAY_BUFFER REPLAY_BUFFER_TYPE REPLAY_BUFFER_IDENTITY_TAG
        ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
        TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE
        MAX_TOKENS_PER_GPU LOG_PROBS_CHUNK_SIZE RECOMPUTE_GRANULARITY OVERLAP_COMM
        ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION RM_TYPE
        ZERO_REWARD_ON_TRUNCATED ZERO_LOSS_ON_TRUNCATED
        EVAL_INTERVAL N_SAMPLES_PER_EVAL_PROMPT EVAL_MAX_RESPONSE_LEN SKIP_EVAL_BEFORE_TRAIN
        SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL DUMP_TRAIN_DATA
        DUMP_POLICY_LOSS_DEBUG OBSERVE_TRAINING_ENTROPY FUSE_ONE_STEP_ACTOR_LOGPROBS
        VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS LOG_SAMPLE_STALENESS_METRICS SAMPLE_STALENESS_MAX_BIN
        LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM LOG_UPDATE_DIAGNOSTICS
        SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS PARTIAL_ROLLOUT OVER_SAMPLING_BATCH_SIZE
        MASK_OFFPOLICY_IN_PARTIAL_ROLLOUT MILES_EXPERIMENTAL_ROLLOUT_REFACTOR
        DYNAMIC_SAMPLING_FILTER_PATH SGLANG_MAX_RUNNING_REQUESTS SGLANG_CUDA_GRAPH_MAX_BS
        QWEN3_4B_BASE_HF_ROOT
    )
    for key in "${matched_override_keys[@]}"; do
        if [[ -v "${key}" ]]; then
            echo "matched mode refuses ambient training override ${key}=${!key}" >&2
            exit 1
        fi
    done
    unset matched_override_keys key

    [[ -d "${MILES_REPO}" ]] || {
        echo "matched cohort MILES_REPO is not a directory: ${MILES_REPO}" >&2
        exit 1
    }
    MATCHED_REPO_ROOT_RESOLVED="$(readlink -f -- "${REPO_ROOT}")"
    MATCHED_MILES_REPO_RESOLVED="$(readlink -f -- "${MILES_REPO}")"
    [[ "${MATCHED_MILES_REPO_RESOLVED}" == "${MATCHED_REPO_ROOT_RESOLVED}" ]] || {
        echo "matched mode must mount the checkout used for submission:" \
             "expected ${MATCHED_REPO_ROOT_RESOLVED}, got ${MATCHED_MILES_REPO_RESOLVED}" >&2
        exit 1
    }

    [[ -z "${ASYNC_SQSH_IMAGE_OVERRIDE:-}" ]] || {
        echo "matched mode refuses async-only container override" \
             "ASYNC_SQSH_IMAGE_OVERRIDE=${ASYNC_SQSH_IMAGE_OVERRIDE}" >&2
        exit 1
    }
    [[ -r "${MATCHED_BASELINE_SQSH_IMAGE}" ]] || {
        echo "matched cohort baseline container is not readable:" \
             "${MATCHED_BASELINE_SQSH_IMAGE}" >&2
        exit 1
    }
    [[ -r "${SQSH_IMAGE}" ]] || {
        echo "matched cohort container is not readable: ${SQSH_IMAGE}" >&2
        exit 1
    }
    MATCHED_BASELINE_SQSH_IMAGE_RESOLVED="$(readlink -f -- "${MATCHED_BASELINE_SQSH_IMAGE}")"
    MATCHED_SQSH_IMAGE_RESOLVED="$(readlink -f -- "${SQSH_IMAGE}")"
    [[ "${MATCHED_SQSH_IMAGE_RESOLVED}" == "${MATCHED_BASELINE_SQSH_IMAGE_RESOLVED}" ]] || {
        echo "matched mode requires the completed cohort container:" \
             "${MATCHED_BASELINE_SQSH_IMAGE_RESOLVED}; got ${MATCHED_SQSH_IMAGE_RESOLVED}" >&2
        exit 1
    }
    # A full sha256 would reread a 34 GB image at every submission. The
    # resolved immutable path plus device/inode/size/mtime is a cheap,
    # auditable identity; the image filename pins the SGLang build commit.
    MATCHED_SQSH_IMAGE_STAT="$(stat -Lc '%d:%i:%s:%Y' -- "${MATCHED_SQSH_IMAGE_RESOLVED}")"

    matched_chain_jobs=10
    (( RESUME_MATCHED_CHAIN == 1 )) && matched_chain_jobs=9
    [[ "${TOTAL_NODES}" == 8 && "${PARTITION}" == batch \
       && "${WALL}" == 04:00:00 && "${CHAIN_JOBS}" == "${matched_chain_jobs}" \
       && "${GPUS_PER_NODE}" == 8 \
       && "${ROLLOUT_BATCH}" == 192 && "${SAMPLES_PER_PROMPT}" == 16 \
       && "${TRAINING_BUFFER_QUEUE_SIZE_VALUE}" == 1000 \
       && "${GLOBAL_BATCH}" == 3072 && "${TENSOR_PARALLEL}" == 2 \
       && "${CONTEXT_PARALLEL}" == 1 \
       && "${ASYNC_GPUS_PER_ENGINE}" == 1 && "${ASYNC_MEM_FRACTION}" == 0.70 \
       && "${COLOCATED_GPUS_PER_ENGINE}" == 2 && "${COLOCATED_MEM_FRACTION}" == 0.65 ]] || {
        echo "matched mode requires the completed cohort protocol:" \
             "nodes=8, partition=batch, wall=04:00:00, chains=${matched_chain_jobs}," \
             "gpus/node=8, rbs=192, n=16, queue=1000, gbs=3072, tp=2, cp=1," \
             "async engine/mem=1/0.70, colocated engine/mem=2/0.65" >&2
        exit 1
    }
    unset matched_chain_jobs
fi

[[ "${QUEUE_POLICY}" == queue-recycle ]] || {
    echo "this sweep requires the recipe's QUEUE_TYPE=queue-recycle, got ${QUEUE_POLICY}" >&2
    exit 1
}
[[ "${STALENESS_REFERENCE_VALUE}" == prefill ]] || {
    echo "this sweep requires STALENESS_REFERENCE=prefill, got ${STALENESS_REFERENCE_VALUE}" >&2
    exit 1
}
require_setting NUM_ROLLOUT 300
require_setting NUM_STEPS_PER_ROLLOUT 1
if (( TOTAL_LENGTH_32K == 1 )); then
    require_setting MAX_RESPONSE_LEN 32768
    require_setting ROLLOUT_MAX_CONTEXT_LEN 32767
    (( MAX_TOKENS_PER_GPU_VALUE * CONTEXT_PARALLEL >= ROLLOUT_MAX_CONTEXT_LEN_VALUE )) || {
        echo "trainer token budget is too small for the 32K total sequence: " \
             "MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU_VALUE}, CP=${CONTEXT_PARALLEL}" >&2
        exit 1
    }
else
    require_setting MAX_RESPONSE_LEN 16384
    require_setting ROLLOUT_MAX_CONTEXT_LEN 32768
fi
require_setting ZERO_REWARD_ON_TRUNCATED \
    "$(( 1 - DISABLE_ZERO_REWARD_ON_TRUNCATED - ENABLE_ZERO_LOSS_ON_TRUNCATED ))"
require_setting ZERO_LOSS_ON_TRUNCATED "${ENABLE_ZERO_LOSS_ON_TRUNCATED}"
require_setting USE_REPLAY_BUFFER 1
require_setting REPLAY_BUFFER_TYPE inflight
require_setting FUSE_ONE_STEP_ACTOR_LOGPROBS 1
require_setting SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS 1
require_setting SAMPLE_STALENESS_MAX_BIN 32
require_setting SAVE_HF 1
require_setting HF_SAVE_INTERVAL 10
if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
    require_setting SAVE_INTERVAL 10
    require_setting SAVE_RETAIN_INTERVAL 100
fi
[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT:-}" ]] || {
    echo "this sweep requires DEBUG_EXIT_AFTER_ROLLOUT to be empty" >&2
    exit 1
}

declare -a RAW_POINTS=()
if (( ${#REQUESTED_POINTS[@]} > 0 )); then
    RAW_POINTS=("${REQUESTED_POINTS[@]}")
elif (( RERUN_CLEAN_MATCHED_ARMS == 1 )); then
    RAW_POINTS=("4:1:7")
elif (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
    # Literal rather than RATIOS: matched mode must stay a five-arm cohort even
    # when a caller has a RATIOS override in its shell.
    RAW_POINTS=("4:1:7" "4:2:6" "4:3:5" "4:4:4")
else
    for staleness in ${STALENESS_LEVELS}; do
        for ratio in ${RATIOS}; do
            RAW_POINTS+=("${staleness}:${ratio}")
        done
    done
fi

validate_point() {
    local point="$1"
    local staleness train_nodes rollout_nodes train_gpus model_parallel data_parallel
    [[ "${point}" =~ ^[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*$ ]] || {
        echo "invalid point '${point}'; expected M:T:R with positive integers" >&2
        return 1
    }
    IFS=: read -r staleness train_nodes rollout_nodes <<<"${point}"
    if (( MATCH_PARTIAL_CONCURRENCY == 1 && staleness != 4 )); then
        echo "matched partial/concurrency mode requires max weight staleness 4, got '${point}'" >&2
        return 1
    fi
    (( train_nodes + rollout_nodes == TOTAL_NODES )) || {
        echo "invalid point '${point}'; T+R must equal TOTAL_NODES=${TOTAL_NODES}" >&2
        return 1
    }
    train_gpus=$(( train_nodes * GPUS_PER_NODE ))
    model_parallel=$(( TENSOR_PARALLEL * CONTEXT_PARALLEL ))
    (( train_gpus % model_parallel == 0 )) || {
        echo "invalid point '${point}'; tp*cp does not divide trainer GPUs" >&2
        return 1
    }
    data_parallel=$(( train_gpus / model_parallel ))
    (( GLOBAL_BATCH % data_parallel == 0 )) || {
        echo "invalid point '${point}'; global batch ${GLOBAL_BATCH} is not divisible by dp=${data_parallel}" >&2
        return 1
    }
    printf '%s %s %s %s\n' "${staleness}" "${train_nodes}" "${rollout_nodes}" "${data_parallel}"
}

declare -a POINTS=()
declare -A SEEN_POINTS=()
for point in "${RAW_POINTS[@]}"; do
    validated="$(validate_point "${point}")" || exit 1
    [[ -n "${SEEN_POINTS[${validated}]:-}" ]] && continue
    SEEN_POINTS["${validated}"]=1
    POINTS+=("${validated}")
done
(( ${#POINTS[@]} > 0 )) || { echo "no sweep points selected" >&2; exit 1; }
if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
    if (( RERUN_CLEAN_MATCHED_ARMS == 1 )); then
        expected_matched_points=("4 1 7 4")
    else
        expected_matched_points=("4 1 7 4" "4 2 6 8" "4 3 5 12" "4 4 4 16")
    fi
    (( ${#POINTS[@]} == ${#expected_matched_points[@]} )) || {
        echo "matched mode selected an unexpected async arm set" >&2
        exit 1
    }
    for expected_point in "${expected_matched_points[@]}"; do
        [[ -n "${SEEN_POINTS[${expected_point}]:-}" ]] || {
            echo "matched mode is missing point '${expected_point}'" >&2
            exit 1
        }
    done
    unset expected_matched_points expected_point
fi

COLOCATED_DATA_PARALLEL=0
if (( INCLUDE_COLOCATED == 1 || MATCH_PARTIAL_CONCURRENCY == 1 )); then
    colocated_train_gpus=$(( TOTAL_NODES * GPUS_PER_NODE ))
    colocated_model_parallel=$(( TENSOR_PARALLEL * CONTEXT_PARALLEL ))
    (( colocated_train_gpus % colocated_model_parallel == 0 )) || {
        echo "invalid colocated arm; tp*cp does not divide trainer GPUs" >&2
        exit 1
    }
    COLOCATED_DATA_PARALLEL=$(( colocated_train_gpus / colocated_model_parallel ))
    (( GLOBAL_BATCH % COLOCATED_DATA_PARALLEL == 0 )) || {
        echo "invalid colocated arm; global batch ${GLOBAL_BATCH} is not divisible by dp=${COLOCATED_DATA_PARALLEL}" >&2
        exit 1
    }
fi

SETTING_COUNT=$(( ${#POINTS[@]} + INCLUDE_COLOCATED + MATCH_PARTIAL_CONCURRENCY ))
DEFAULT_ASYNC_CONCURRENCY=$(( ROLLOUT_BATCH * SAMPLES_PER_PROMPT ))

async_run_name() {
    local staleness="$1"
    local train_nodes="$2"
    local rollout_nodes="$3"
    local name="s${staleness}-t${train_nodes}r${rollout_nodes}"
    if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}" ]]; then
        name="${name}-c${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}"
    fi
    printf '%s-%s\n' "${name}" "${RUN_NAMESPACE}"
}

async_config_tag() {
    local staleness="$1"
    local train_nodes="$2"
    local rollout_nodes="$3"
    local config_tag="s${staleness}-t${train_nodes}r${rollout_nodes}-${RUN_NAMESPACE}"
    # common/run_identity.sh appends the non-default concurrency axis. Keep the
    # visible W&B group descriptive without duplicating that checkpoint suffix.
    if (( DISABLE_ZERO_REWARD_ON_TRUNCATED == 1 )); then
        config_tag="${config_tag}-zero-reward-trunc-off"
    fi
    if (( TOTAL_LENGTH_32K == 1 )); then
        config_tag="${config_tag}-total32k-cp${CONTEXT_PARALLEL}"
    fi
    printf '%s\n' "${config_tag}"
}

colocated_run_name() {
    if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
        printf 's0-colocated-partial-o%s-%s\n' "${PARTIAL_OVER_SAMPLING_BATCH_SIZE}" "${RUN_NAMESPACE}"
    else
        printf 's0-colocated-%s\n' "${RUN_NAMESPACE}"
    fi
}

colocated_config_tag() {
    local config_tag="s0-colocated-${RUN_NAMESPACE}"
    # common/run_identity.sh appends partial-oN and places this arm under
    # partial-rollout/unbounded.
    if (( DISABLE_ZERO_REWARD_ON_TRUNCATED == 1 )); then
        config_tag="${config_tag}-zero-reward-trunc-off"
    fi
    printf '%s\n' "${config_tag}"
}

printf 'recipe: %s\n' "${RECIPE}"
printf 'fixed by recipe: queue=%s, reference=%s, rollouts=%s, steps/rollout=%s, gbs=%s, tp=%s, cp=%s, max-tokens/gpu=%s\n' \
    "${QUEUE_POLICY}" "${STALENESS_REFERENCE_VALUE}" "${NUM_ROLLOUT_VALUE}" \
    "${NUM_STEPS_PER_ROLLOUT_VALUE}" \
    "${GLOBAL_BATCH}" "${TENSOR_PARALLEL}" "${CONTEXT_PARALLEL}" "${MAX_TOKENS_PER_GPU_VALUE}"
printf 'completed-group buffer: %s groups (%s training batches)\n' \
    "${TRAINING_BUFFER_QUEUE_SIZE_VALUE}" \
    "$(( TRAINING_BUFFER_QUEUE_SIZE_VALUE / ROLLOUT_BATCH ))"
printf 'fixed safety: response=%s, total-context=%s, zero-reward-trunc=%s, zero-loss-trunc=%s, replay=%s/%s, fused-logprobs=%s, exact-segments=%s, staleness-bin=%s\n' \
    "${MAX_RESPONSE_LEN_VALUE}" "${ROLLOUT_MAX_CONTEXT_LEN_VALUE}" \
    "${ZERO_REWARD_ON_TRUNCATED_VALUE}" "${ZERO_LOSS_ON_TRUNCATED_VALUE}" \
    "${USE_REPLAY_BUFFER_VALUE}" "${REPLAY_BUFFER_TYPE_VALUE}" \
    "${FUSE_ONE_STEP_ACTOR_LOGPROBS_VALUE}" "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS_VALUE}" \
    "${SAMPLE_STALENESS_MAX_BIN_VALUE}"
printf 'fixed checkpoints: save-hf=%s, hf interval=%s; settings=%s; clean=%s\n' \
    "${SAVE_HF_VALUE}" "${HF_SAVE_INTERVAL_VALUE}" "${SETTING_COUNT}" "${CLEAN_CHECKPOINT}"
printf 'submission: %s nodes, %s, %s, %s chained job(s), wandb=%s\n' \
    "${TOTAL_NODES}" "${PARTITION}" "${WALL}" "${CHAIN_JOBS}" "${WANDB_PROJECT}"
printf 'generation: rbs=%s groups, n=%s, async in-flight=%s trajectories\n' \
    "${ROLLOUT_BATCH}" "${SAMPLES_PER_PROMPT}" \
    "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE:-${DEFAULT_ASYNC_CONCURRENCY}}"
if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
    if (( RERUN_CLEAN_MATCHED_ARMS == 1 )); then
        printf 'clean matched rerun: colocated partial plus t1r7; O=%s groups; async C=%s trajectories; max staleness=4\n' \
            "${PARTIAL_OVER_SAMPLING_BATCH_SIZE}" "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}"
    else
        printf 'matched protocol: colocated partial O=%s groups; async C=%s trajectories; max staleness=4\n' \
            "${PARTIAL_OVER_SAMPLING_BATCH_SIZE}" "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}"
    fi
    printf 'causal warning: versus the old colocated baseline, both partial rollout and O change\n'
fi
if (( RUN_NAMESPACE_WAS_EXPLICIT == 1 )); then
    namespace_source=explicit
else
    namespace_source=automatic
fi
printf 'namespace: %s (%s)\n\n' "${RUN_NAMESPACE}" "${namespace_source}"
printf '  %-10s %-4s %-3s %-3s %-4s %-8s %-8s %-8s %-10s %s\n' \
    mode max T R dp gbs/dp inflight engines C/engine run
for point in "${POINTS[@]}"; do
    read -r staleness train_nodes rollout_nodes data_parallel <<<"${point}"
    run_name="$(async_run_name "${staleness}" "${train_nodes}" "${rollout_nodes}")"
    async_engines=$(( rollout_nodes * GPUS_PER_NODE / ASYNC_GPUS_PER_ENGINE ))
    printf '  %-10s %-4s %-3s %-3s %-4s %-8s %-8s %-8s %-10s %s\n' \
        async "${staleness}" "${train_nodes}" "${rollout_nodes}" "${data_parallel}" \
        "$(( GLOBAL_BATCH / data_parallel ))" \
        "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE:-${DEFAULT_ASYNC_CONCURRENCY}}" \
        "${async_engines}" \
        "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE:-${DEFAULT_ASYNC_CONCURRENCY}}/${async_engines}" \
        "${run_name}"
done
if (( INCLUDE_COLOCATED == 1 || MATCH_PARTIAL_CONCURRENCY == 1 )); then
    colocated_name="$(colocated_run_name)"
    if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
        colocated_mode=partial
        colocated_inflight=$(( PARTIAL_OVER_SAMPLING_BATCH_SIZE * SAMPLES_PER_PROMPT ))
    else
        colocated_mode=colocated
        colocated_inflight="${DEFAULT_ASYNC_CONCURRENCY}"
    fi
    colocated_engines=$(( TOTAL_NODES * GPUS_PER_NODE / COLOCATED_GPUS_PER_ENGINE ))
    printf '  %-10s %-4s %-3s %-3s %-4s %-8s %-8s %-8s %-10s %s\n' \
        "${colocated_mode}" none "${TOTAL_NODES}" 0 "${COLOCATED_DATA_PARALLEL}" \
        "$(( GLOBAL_BATCH / COLOCATED_DATA_PARALLEL ))" "${colocated_inflight}" \
        "${colocated_engines}" "${colocated_inflight}/${colocated_engines}" "${colocated_name}"
    if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
        if (( RERUN_CLEAN_MATCHED_ARMS == 1 )); then
            printf 'global C is matched; t1r7 retains the original async engine topology\n'
        else
            printf 'global C is matched; only t4r4 also matches colocated engine count and nominal C/engine\n'
        fi
    fi
fi

if (( RESUME_CHAIN == 1 )); then
    missing_resume_checkpoint=0
    for point in "${POINTS[@]}"; do
        read -r staleness train_nodes rollout_nodes _ <<<"${point}"
        config_tag="$(async_config_tag "${staleness}" "${train_nodes}" "${rollout_nodes}")"
        resume_checkpoint="$(find "${TRAIN_CKPT_DIR}" -type d -name "*${config_tag}*" -print -quit 2>/dev/null)"
        if [[ -z "${resume_checkpoint}" ]]; then
            echo "resume checkpoint not found for $(async_run_name "${staleness}" "${train_nodes}" "${rollout_nodes}")" >&2
            missing_resume_checkpoint=1
        fi
    done
    if (( INCLUDE_COLOCATED == 1 )); then
        config_tag="$(colocated_config_tag)"
        resume_checkpoint="$(find "${TRAIN_CKPT_DIR}" -type d -name "*${config_tag}*" -print -quit 2>/dev/null)"
        if [[ -z "${resume_checkpoint}" ]]; then
            echo "resume checkpoint not found for $(colocated_run_name)" >&2
            missing_resume_checkpoint=1
        fi
    fi
    (( missing_resume_checkpoint == 0 )) || exit 1
    unset missing_resume_checkpoint config_tag resume_checkpoint
fi

if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
    existing_checkpoint=""
    existing_log=""
    if [[ -d "${TRAIN_CKPT_DIR}" ]]; then
        existing_checkpoint="$(find "${TRAIN_CKPT_DIR}" -type d -name "*${RUN_NAMESPACE}*" -print -quit 2>/dev/null)"
    fi
    if [[ -d "${LOG_DIR}" ]]; then
        existing_log="$(find "${LOG_DIR}" -maxdepth 1 -type f -name "*${RUN_NAMESPACE}*" -print -quit 2>/dev/null)"
    fi
    if (( RESUME_MATCHED_CHAIN == 1 )); then
        [[ -n "${existing_checkpoint}" ]] || {
            echo "matched resume checkpoint namespace not found: ${RUN_NAMESPACE}" >&2
            exit 1
        }
        [[ -f "${LOG_DIR}/${RUN_NAMESPACE}.manifest.tsv" ]] || {
            echo "matched resume manifest not found: ${LOG_DIR}/${RUN_NAMESPACE}.manifest.tsv" >&2
            exit 1
        }
    elif [[ -n "${existing_checkpoint}" || -n "${existing_log}" ]]; then
        echo "matched mode refuses an existing namespace: ${RUN_NAMESPACE}" >&2
        [[ -z "${existing_checkpoint}" ]] || echo "checkpoint: ${existing_checkpoint}" >&2
        [[ -z "${existing_log}" ]] || echo "log: ${existing_log}" >&2
        exit 1
    fi
fi

if (( SUBMIT == 0 )); then
    printf '\ndry run; add --submit to enqueue these recipe invocations\n'
    exit 0
fi

mkdir -p "${LOG_DIR}"
MANIFEST_PATH=""
if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
    if (( RESUME_MATCHED_CHAIN == 1 )); then
        resume_id="$(date -u +%Y%m%dT%H%M%SZ)-p$$"
        MANIFEST_PATH="${LOG_DIR}/${RUN_NAMESPACE}.resume-${resume_id}.manifest.tsv"
    else
        MANIFEST_PATH="${LOG_DIR}/${RUN_NAMESPACE}.manifest.tsv"
    fi
    if ! (set -o noclobber; : > "${MANIFEST_PATH}") 2>/dev/null; then
        echo "matched mode could not reserve its namespace manifest: ${MANIFEST_PATH}" >&2
        exit 1
    fi
    prompt_data_container="$(sed -n 's/^PROMPT_DATA=//p' "${RECIPE_PATH}" | head -n 1)"
    prompt_data_host="${DATASET_DIR}/${prompt_data_container#/data/}"
    dataset_sha256=missing
    if [[ -f "${prompt_data_host}" ]]; then
        dataset_sha256="$(sha256sum "${prompt_data_host}" | awk '{print $1}')"
    fi
    tracked_diff_sha256="$(git -C "${REPO_ROOT}" diff --binary HEAD | sha256sum | awk '{print $1}')"
    {
        printf 'key\tvalue\n'
        printf 'namespace\t%s\n' "${RUN_NAMESPACE}"
        printf 'resume_matched_chain\t%s\n' "${RESUME_MATCHED_CHAIN}"
        printf 'rerun_clean_matched_arms\t%s\n' "${RERUN_CLEAN_MATCHED_ARMS}"
        if (( RESUME_MATCHED_CHAIN == 1 )); then
            printf 'resume_of_manifest\t%s\n' "${LOG_DIR}/${RUN_NAMESPACE}.manifest.tsv"
        fi
        printf 'tracked_diff_sha256\t%s\n' "${tracked_diff_sha256}"
        printf 'runtime_repo\t%s\n' "${MATCHED_REPO_ROOT_RESOLVED}"
        printf 'container\t%s\n' "${SQSH_IMAGE}"
        printf 'async_container_effective\t%s\n' "${MATCHED_SQSH_IMAGE_RESOLVED}"
        printf 'colocated_container_effective\t%s\n' "${MATCHED_SQSH_IMAGE_RESOLVED}"
        printf 'container_stat_device_inode_size_mtime\t%s\n' "${MATCHED_SQSH_IMAGE_STAT}"
        printf 'dataset\t%s\n' "${prompt_data_host}"
        printf 'dataset_sha256\t%s\n' "${dataset_sha256}"
        printf 'baseline_namespace\t%s\n' "${BASELINE_RUN_NAMESPACE:-sr-20260819-212906}"
        printf 'baseline_training_root\t%s\n' \
            "${BASELINE_TRAINING_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training}"
        printf 'account\t%s\n' "${ACCOUNT}"
        printf 'partition\t%s\n' "${PARTITION}"
        printf 'wall\t%s\n' "${WALL}"
        printf 'chain_jobs\t%s\n' "${CHAIN_JOBS}"
        printf 'total_nodes\t%s\n' "${TOTAL_NODES}"
        printf 'gpus_per_node\t%s\n' "${GPUS_PER_NODE}"
        printf 'wandb_project\t%s\n' "${WANDB_PROJECT}"
        printf 'save_interval\t%s\n' "${SAVE_INTERVAL_VALUE}"
        printf 'save_retain_interval\t%s\n' "${SAVE_RETAIN_INTERVAL_VALUE}"
        printf 'hf_save_interval\t%s\n' "${HF_SAVE_INTERVAL_VALUE}"
        printf 'async_gpus_per_engine\t%s\n' "${ASYNC_GPUS_PER_ENGINE}"
        printf 'async_mem_fraction\t%s\n' "${ASYNC_MEM_FRACTION}"
        printf 'colocated_gpus_per_engine\t%s\n' "${COLOCATED_GPUS_PER_ENGINE}"
        printf 'colocated_mem_fraction\t%s\n' "${COLOCATED_MEM_FRACTION}"
        printf 'rollout_batch_groups\t%s\n' "${ROLLOUT_BATCH}"
        printf 'samples_per_prompt\t%s\n' "${SAMPLES_PER_PROMPT}"
        printf 'partial_over_sampling_groups\t%s\n' "${PARTIAL_OVER_SAMPLING_BATCH_SIZE}"
        printf 'partial_over_sampling_groups_effective\t%s\n' "${PARTIAL_OVER_SAMPLING_BATCH_SIZE}"
        printf 'async_max_concurrent_samples\t%s\n' "${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}"
        printf 'training_buffer_queue_size\t%s\n' "${TRAINING_BUFFER_QUEUE_SIZE_VALUE}"
        printf 'sample_staleness_max_bin\t%s\n' "${SAMPLE_STALENESS_MAX_BIN_VALUE}"
        printf 'max_weight_staleness\t4\n'
        printf 'switch_metric_contract\t%s\n' "${MATCHED_SWITCH_METRIC_CONTRACT}"
        printf 'job\tarm\tchain_index\tjob_id\tdependency\trecipe\texports_csv\n'
    } >> "${MANIFEST_PATH}"
fi

submit_chain() {
    local staleness="$1"
    local train_nodes="$2"
    local rollout_nodes="$3"
    local run_name
    run_name="$(async_run_name "${staleness}" "${train_nodes}" "${rollout_nodes}")"
    local config_tag
    config_tag="$(async_config_tag "${staleness}" "${train_nodes}" "${rollout_nodes}")"
    local base_exports_csv exports_csv raw_job_id job_id chain_index clean_value
    local -a dependency=()
    local -a exports=(
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "RUN_NAME=${run_name}"
        "CONFIG_TAG=${config_tag}"
        "MAX_WEIGHT_STALENESS=${staleness}"
        "TRAINING_BUFFER_QUEUE_SIZE=${TRAINING_BUFFER_QUEUE_SIZE_VALUE}"
        "ACTOR_NUM_NODES=${train_nodes}"
        "ROLLOUT_NUM_GPUS=$(( rollout_nodes * GPUS_PER_NODE ))"
        "ASYNC_MAX_CONCURRENT_SAMPLES=${ASYNC_MAX_CONCURRENT_SAMPLES_VALUE}"
        "PARTIAL_ROLLOUT=0"
        "OVER_SAMPLING_BATCH_SIZE=${ROLLOUT_BATCH}"
        "MASK_OFFPOLICY_IN_PARTIAL_ROLLOUT=0"
        "DYNAMIC_SAMPLING_FILTER_PATH="
        "SGLANG_MAX_RUNNING_REQUESTS="
        "SGLANG_CUDA_GRAPH_MAX_BS="
        "NUM_ROLLOUT=${NUM_ROLLOUT_VALUE}"
        "NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT_VALUE}"
        "MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN_VALUE}"
        "ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN_VALUE}"
        "ZERO_REWARD_ON_TRUNCATED=${ZERO_REWARD_ON_TRUNCATED_VALUE}"
        "ZERO_LOSS_ON_TRUNCATED=${ZERO_LOSS_ON_TRUNCATED_VALUE}"
        "USE_REPLAY_BUFFER=${USE_REPLAY_BUFFER_VALUE}"
        "REPLAY_BUFFER_TYPE=${REPLAY_BUFFER_TYPE_VALUE}"
        "FUSE_ONE_STEP_ACTOR_LOGPROBS=${FUSE_ONE_STEP_ACTOR_LOGPROBS_VALUE}"
        "SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS_VALUE}"
        "SAMPLE_STALENESS_MAX_BIN=${SAMPLE_STALENESS_MAX_BIN_VALUE}"
        "SAVE_HF=${SAVE_HF_VALUE}"
        "HF_SAVE_INTERVAL=${HF_SAVE_INTERVAL_VALUE}"
        "DEBUG_EXIT_AFTER_ROLLOUT="
    )
    if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
        exports+=(
            "MILES_REPO=${MATCHED_REPO_ROOT_RESOLVED}"
            "SQSH_IMAGE=${MATCHED_SQSH_IMAGE_RESOLVED}"
            "ASYNC_SQSH_IMAGE_OVERRIDE="
        )
    fi
    base_exports_csv="$(IFS=,; printf '%s' "${exports[*]}")"

    for (( chain_index = 1; chain_index <= CHAIN_JOBS; chain_index++ )); do
        clean_value=0
        (( CLEAN_CHECKPOINT == 1 && chain_index == 1 )) && clean_value=1
        exports_csv="${base_exports_csv},CLEAN_CHECKPOINT=${clean_value}"
        raw_job_id="$(sbatch --parsable \
            "${dependency[@]}" \
            -A "${ACCOUNT}" \
            --partition="${PARTITION}" \
            --nodes="${TOTAL_NODES}" \
            --time="${WALL}" \
            --job-name="${run_name}" \
            --comment="${IDLE_EXEMPTION}" \
            --output="${LOG_DIR}/${run_name}-%j.log" \
            --export="ALL,${exports_csv}" \
            "${RECIPE_PATH}")"
        job_id="${raw_job_id%%;*}"
        printf 'submitted %-40s chain %d/%d job=%s dependency=%s\n' \
            "${run_name}" "${chain_index}" "${CHAIN_JOBS}" "${job_id}" \
            "${dependency[*]:-none}"
        if [[ -n "${MANIFEST_PATH}" ]]; then
            printf 'job\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${run_name}" "${chain_index}" "${job_id}" "${dependency[*]:-none}" \
                "${RECIPE}" "${exports_csv}" >> "${MANIFEST_PATH}"
        fi
        dependency=("--dependency=afterany:${job_id}")
    done
}

submit_colocated_chain() {
    local run_name
    run_name="$(colocated_run_name)"
    local config_tag
    config_tag="$(colocated_config_tag)"
    local colocated_over_sampling="${ROLLOUT_BATCH}"
    if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
        colocated_over_sampling="${PARTIAL_OVER_SAMPLING_BATCH_SIZE}"
    fi
    local base_exports_csv exports_csv raw_job_id job_id chain_index clean_value
    local -a dependency=()
    local -a exports=(
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "RUN_NAME=${run_name}"
        "CONFIG_TAG=${config_tag}"
        "ACTOR_NUM_NODES=${TOTAL_NODES}"
        "ROLLOUT_NUM_GPUS=0"
        "ASYNC_MAX_CONCURRENT_SAMPLES="
        "NUM_ROLLOUT=${NUM_ROLLOUT_VALUE}"
        "NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT_VALUE}"
        "MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN_VALUE}"
        "ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN_VALUE}"
        "ZERO_REWARD_ON_TRUNCATED=${ZERO_REWARD_ON_TRUNCATED_VALUE}"
        "ZERO_LOSS_ON_TRUNCATED=${ZERO_LOSS_ON_TRUNCATED_VALUE}"
        "SAVE_HF=${SAVE_HF_VALUE}"
        "HF_SAVE_INTERVAL=${HF_SAVE_INTERVAL_VALUE}"
        "MAX_WEIGHT_STALENESS="
        "TRAINING_BUFFER_QUEUE_SIZE=1000"
        "STALENESS_REFERENCE=completion"
        "PAUSE_GENERATION_MODE="
        "QUEUE_TYPE=none"
        "QUEUE_FACTOR=1"
        "USE_REPLAY_BUFFER=0"
        "REPLAY_BUFFER_TYPE=inflight"
        "REPLAY_BUFFER_IDENTITY_TAG=0"
        "FUSE_ONE_STEP_ACTOR_LOGPROBS=0"
        "SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=0"
        "SAMPLE_STALENESS_MAX_BIN=${SAMPLE_STALENESS_MAX_BIN_VALUE}"
        "OVER_SAMPLING_BATCH_SIZE=${colocated_over_sampling}"
        "PARTIAL_ROLLOUT=${MATCH_PARTIAL_CONCURRENCY}"
        "MASK_OFFPOLICY_IN_PARTIAL_ROLLOUT=0"
        "DYNAMIC_SAMPLING_FILTER_PATH="
        "SGLANG_MAX_RUNNING_REQUESTS="
        "SGLANG_CUDA_GRAPH_MAX_BS="
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0"
        "DEBUG_EXIT_AFTER_ROLLOUT="
    )
    if (( MATCH_PARTIAL_CONCURRENCY == 1 )); then
        exports+=(
            "MILES_REPO=${MATCHED_REPO_ROOT_RESOLVED}"
            "SQSH_IMAGE=${MATCHED_SQSH_IMAGE_RESOLVED}"
            "ASYNC_SQSH_IMAGE_OVERRIDE="
        )
    fi
    base_exports_csv="$(IFS=,; printf '%s' "${exports[*]}")"

    for (( chain_index = 1; chain_index <= CHAIN_JOBS; chain_index++ )); do
        clean_value=0
        (( CLEAN_CHECKPOINT == 1 && chain_index == 1 )) && clean_value=1
        exports_csv="${base_exports_csv},CLEAN_CHECKPOINT=${clean_value}"
        raw_job_id="$(sbatch --parsable \
            "${dependency[@]}" \
            -A "${ACCOUNT}" \
            --partition="${PARTITION}" \
            --nodes="${TOTAL_NODES}" \
            --time="${WALL}" \
            --job-name="${run_name}" \
            --comment="${IDLE_EXEMPTION}" \
            --output="${LOG_DIR}/${run_name}-%j.log" \
            --export="ALL,${exports_csv}" \
            "${COLOCATED_RECIPE_PATH}")"
        job_id="${raw_job_id%%;*}"
        printf 'submitted %-40s chain %d/%d job=%s dependency=%s\n' \
            "${run_name}" "${chain_index}" "${CHAIN_JOBS}" "${job_id}" \
            "${dependency[*]:-none}"
        if [[ -n "${MANIFEST_PATH}" ]]; then
            printf 'job\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${run_name}" "${chain_index}" "${job_id}" "${dependency[*]:-none}" \
                "${COLOCATED_RECIPE}" "${exports_csv}" >> "${MANIFEST_PATH}"
        fi
        dependency=("--dependency=afterany:${job_id}")
    done
}

printf '\n'
for point in "${POINTS[@]}"; do
    read -r staleness train_nodes rollout_nodes _ <<<"${point}"
    submit_chain "${staleness}" "${train_nodes}" "${rollout_nodes}"
done
if (( INCLUDE_COLOCATED == 1 || MATCH_PARTIAL_CONCURRENCY == 1 )); then
    submit_colocated_chain
fi
