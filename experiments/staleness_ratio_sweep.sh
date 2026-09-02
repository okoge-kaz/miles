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

: "${TOTAL_NODES:=8}"
: "${STALENESS_LEVELS:=1 2 4 8}"
: "${RATIOS:=1:7 2:6 3:5 4:4}"
: "${WALL:=${PBS_DEFAULT_WALLTIME}}"
: "${CHAIN_JOBS:=2}"
WANDB_PROJECT=async-rl-dapo-math
: "${RUN_NAMESPACE:=sr-$(date +%Y%m%d-%H%M%S)}"

LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b"

SUBMIT=0
INCLUDE_COLOCATED=0
CLEAN_CHECKPOINT=0
declare -a REQUESTED_POINTS=()

usage() {
    cat <<'EOF'
usage: experiments/staleness_ratio_sweep.sh [--submit] [--include-colocated]
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

Useful environment overrides: TOTAL_NODES, STALENESS_LEVELS, RATIOS,
CHAIN_JOBS, WALL, and RUN_NAMESPACE.
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

if (( CLEAN_CHECKPOINT == 1 && SUBMIT == 0 )); then
    echo "--clean-checkpoint requires --submit" >&2
    exit 2
fi

recipe_default() {
    local key="$1"
    local value
    value="$(sed -n 's/^: "${'"${key}"':=\([^}]*\)}".*/\1/p' "${RECIPE_PATH}" | head -n 1)"
    [[ -n "${value}" ]] || {
        echo "could not read ${key} from ${RECIPE}" >&2
        return 1
    }
    printf '%s\n' "${value}"
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
TENSOR_PARALLEL="$(recipe_or_environment TENSOR_PARALLEL_SIZE)"
CONTEXT_PARALLEL="$(recipe_or_environment CONTEXT_PARALLEL_SIZE)"
QUEUE_POLICY="$(recipe_or_environment QUEUE_TYPE)"
STALENESS_REFERENCE_VALUE="$(recipe_or_environment STALENESS_REFERENCE)"
NUM_ROLLOUT_VALUE="$(recipe_or_environment NUM_ROLLOUT)"
NUM_STEPS_PER_ROLLOUT_VALUE="$(recipe_or_environment NUM_STEPS_PER_ROLLOUT)"
MAX_RESPONSE_LEN_VALUE="$(recipe_or_environment MAX_RESPONSE_LEN)"
ZERO_REWARD_ON_TRUNCATED_VALUE="$(recipe_or_environment ZERO_REWARD_ON_TRUNCATED)"
USE_REPLAY_BUFFER_VALUE="$(recipe_or_environment USE_REPLAY_BUFFER)"
REPLAY_BUFFER_TYPE_VALUE="$(recipe_or_environment REPLAY_BUFFER_TYPE)"
FUSE_ONE_STEP_ACTOR_LOGPROBS_VALUE="$(recipe_or_environment FUSE_ONE_STEP_ACTOR_LOGPROBS)"
SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS_VALUE="$(recipe_or_environment SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS)"
SAVE_HF_VALUE="$(recipe_or_environment SAVE_HF)"
HF_SAVE_INTERVAL_VALUE="$(recipe_or_environment HF_SAVE_INTERVAL)"

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
require_setting MAX_RESPONSE_LEN 16384
require_setting ZERO_REWARD_ON_TRUNCATED 1
require_setting USE_REPLAY_BUFFER 1
require_setting REPLAY_BUFFER_TYPE inflight
require_setting FUSE_ONE_STEP_ACTOR_LOGPROBS 1
require_setting SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS 1
require_setting SAVE_HF 1
require_setting HF_SAVE_INTERVAL 10
[[ -z "${DEBUG_EXIT_AFTER_ROLLOUT:-}" ]] || {
    echo "this sweep requires DEBUG_EXIT_AFTER_ROLLOUT to be empty" >&2
    exit 1
}

declare -a RAW_POINTS=()
if (( ${#REQUESTED_POINTS[@]} > 0 )); then
    RAW_POINTS=("${REQUESTED_POINTS[@]}")
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

COLOCATED_DATA_PARALLEL=0
if (( INCLUDE_COLOCATED == 1 )); then
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

SETTING_COUNT=$(( ${#POINTS[@]} + INCLUDE_COLOCATED ))

printf 'recipe: %s\n' "${RECIPE}"
printf 'fixed by recipe: queue=%s, reference=%s, rollouts=%s, steps/rollout=%s, gbs=%s, tp=%s, cp=%s\n' \
    "${QUEUE_POLICY}" "${STALENESS_REFERENCE_VALUE}" "${NUM_ROLLOUT_VALUE}" \
    "${NUM_STEPS_PER_ROLLOUT_VALUE}" \
    "${GLOBAL_BATCH}" "${TENSOR_PARALLEL}" "${CONTEXT_PARALLEL}"
printf 'fixed safety: response=%s, zero-trunc=%s, replay=%s/%s, fused-logprobs=%s, exact-segments=%s\n' \
    "${MAX_RESPONSE_LEN_VALUE}" "${ZERO_REWARD_ON_TRUNCATED_VALUE}" \
    "${USE_REPLAY_BUFFER_VALUE}" "${REPLAY_BUFFER_TYPE_VALUE}" \
    "${FUSE_ONE_STEP_ACTOR_LOGPROBS_VALUE}" "${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS_VALUE}"
printf 'fixed checkpoints: save-hf=%s, hf interval=%s; settings=%s; clean=%s\n' \
    "${SAVE_HF_VALUE}" "${HF_SAVE_INTERVAL_VALUE}" "${SETTING_COUNT}" "${CLEAN_CHECKPOINT}"
printf 'submission: %s nodes, %s, %s chained job(s), wandb=%s\n' \
    "${TOTAL_NODES}" "${WALL}" "${CHAIN_JOBS}" "${WANDB_PROJECT}"
printf 'namespace: %s\n\n' "${RUN_NAMESPACE}"
printf '  %-4s %-3s %-3s %-4s %-8s %s\n' max T R dp gbs/dp run
for point in "${POINTS[@]}"; do
    read -r staleness train_nodes rollout_nodes data_parallel <<<"${point}"
    run_name="s${staleness}-t${train_nodes}r${rollout_nodes}-${RUN_NAMESPACE}"
    printf '  %-4s %-3s %-3s %-4s %-8s %s\n' \
        "${staleness}" "${train_nodes}" "${rollout_nodes}" "${data_parallel}" \
        "$(( GLOBAL_BATCH / data_parallel ))" "${run_name}"
done
if (( INCLUDE_COLOCATED == 1 )); then
    colocated_run_name="s0-colocated-${RUN_NAMESPACE}"
    printf '  %-4s %-3s %-3s %-4s %-8s %s\n' \
        0 "${TOTAL_NODES}" 0 "${COLOCATED_DATA_PARALLEL}" \
        "$(( GLOBAL_BATCH / COLOCATED_DATA_PARALLEL ))" "${colocated_run_name}"
fi

if (( SUBMIT == 0 )); then
    printf '\ndry run; add --submit to enqueue these recipe invocations\n'
    exit 0
fi

mkdir -p "${LOG_DIR}"

submit_chain() {
    local staleness="$1"
    local train_nodes="$2"
    local rollout_nodes="$3"
    local run_name="s${staleness}-t${train_nodes}r${rollout_nodes}-${RUN_NAMESPACE}"
    local base_exports_csv exports_csv raw_job_id job_id chain_index clean_value
    local -a dependency=()
    local -a exports=(
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "RUN_NAME=${run_name}"
        "CONFIG_TAG=${run_name}"
        "MAX_WEIGHT_STALENESS=${staleness}"
        "ACTOR_NUM_NODES=${train_nodes}"
        "ROLLOUT_NUM_GPUS=$(( rollout_nodes * GPUS_PER_NODE ))"
        "NUM_ROLLOUT=${NUM_ROLLOUT_VALUE}"
        "NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT_VALUE}"
        "MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN_VALUE}"
        "ZERO_REWARD_ON_TRUNCATED=${ZERO_REWARD_ON_TRUNCATED_VALUE}"
        "USE_REPLAY_BUFFER=${USE_REPLAY_BUFFER_VALUE}"
        "REPLAY_BUFFER_TYPE=${REPLAY_BUFFER_TYPE_VALUE}"
        "FUSE_ONE_STEP_ACTOR_LOGPROBS=${FUSE_ONE_STEP_ACTOR_LOGPROBS_VALUE}"
        "SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=${SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS_VALUE}"
        "SAVE_HF=${SAVE_HF_VALUE}"
        "HF_SAVE_INTERVAL=${HF_SAVE_INTERVAL_VALUE}"
        "DEBUG_EXIT_AFTER_ROLLOUT="
    )
    base_exports_csv="$(IFS=,; printf '%s' "${exports[*]}")"

    for (( chain_index = 1; chain_index <= CHAIN_JOBS; chain_index++ )); do
        clean_value=0
        (( CLEAN_CHECKPOINT == 1 && chain_index == 1 )) && clean_value=1
        exports_csv="${base_exports_csv},CLEAN_CHECKPOINT=${clean_value}"
        raw_job_id="$(pbs_submit --parsable --profile gpu \
            "${dependency[@]}" \
            --nodes="${TOTAL_NODES}" \
            --time="${WALL}" \
            --job-name="${run_name}" \
            --output="${LOG_DIR}/${run_name}-%j.log" \
            --export="ALL,${exports_csv}" \
            "${RECIPE_PATH}")"
        job_id="${raw_job_id}"
        printf 'submitted %-40s chain %d/%d job=%s dependency=%s\n' \
            "${run_name}" "${chain_index}" "${CHAIN_JOBS}" "${job_id}" \
            "${dependency[*]:-none}"
        dependency=("--dependency=afterany:${job_id}")
    done
}

submit_colocated_chain() {
    local run_name="s0-colocated-${RUN_NAMESPACE}"
    local base_exports_csv exports_csv raw_job_id job_id chain_index clean_value
    local -a dependency=()
    local -a exports=(
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "RUN_NAME=${run_name}"
        "CONFIG_TAG=${run_name}"
        "ACTOR_NUM_NODES=${TOTAL_NODES}"
        "ROLLOUT_NUM_GPUS=0"
        "NUM_ROLLOUT=${NUM_ROLLOUT_VALUE}"
        "NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT_VALUE}"
        "MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN_VALUE}"
        "ZERO_REWARD_ON_TRUNCATED=${ZERO_REWARD_ON_TRUNCATED_VALUE}"
        "SAVE_HF=${SAVE_HF_VALUE}"
        "HF_SAVE_INTERVAL=${HF_SAVE_INTERVAL_VALUE}"
        "MAX_WEIGHT_STALENESS="
        "STALENESS_REFERENCE=completion"
        "PAUSE_GENERATION_MODE="
        "QUEUE_TYPE=none"
        "QUEUE_FACTOR=1"
        "USE_REPLAY_BUFFER=0"
        "REPLAY_BUFFER_TYPE=inflight"
        "REPLAY_BUFFER_IDENTITY_TAG=0"
        "FUSE_ONE_STEP_ACTOR_LOGPROBS=0"
        "SGLANG_RESPONSE_WEIGHT_VERSION_SEGMENTS=0"
        "DEBUG_EXIT_AFTER_ROLLOUT="
    )
    base_exports_csv="$(IFS=,; printf '%s' "${exports[*]}")"

    for (( chain_index = 1; chain_index <= CHAIN_JOBS; chain_index++ )); do
        clean_value=0
        (( CLEAN_CHECKPOINT == 1 && chain_index == 1 )) && clean_value=1
        exports_csv="${base_exports_csv},CLEAN_CHECKPOINT=${clean_value}"
        raw_job_id="$(pbs_submit --parsable --profile gpu \
            "${dependency[@]}" \
            --nodes="${TOTAL_NODES}" \
            --time="${WALL}" \
            --job-name="${run_name}" \
            --output="${LOG_DIR}/${run_name}-%j.log" \
            --export="ALL,${exports_csv}" \
            "${COLOCATED_RECIPE_PATH}")"
        job_id="${raw_job_id}"
        printf 'submitted %-40s chain %d/%d job=%s dependency=%s\n' \
            "${run_name}" "${chain_index}" "${CHAIN_JOBS}" "${job_id}" \
            "${dependency[*]:-none}"
        dependency=("--dependency=afterany:${job_id}")
    done
}

printf '\n'
for point in "${POINTS[@]}"; do
    read -r staleness train_nodes rollout_nodes _ <<<"${point}"
    submit_chain "${staleness}" "${train_nodes}" "${rollout_nodes}"
done
if (( INCLUDE_COLOCATED == 1 )); then
    submit_colocated_chain
fi
