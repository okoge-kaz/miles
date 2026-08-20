#!/bin/bash
# Sweep max weight staleness and the trainer:rollout node ratio. The math async
# recipe owns every learning default; this launcher overrides only the two sweep
# axes, resource placement, and run identity.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
RECIPE="experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch"
RECIPE_PATH="${REPO_ROOT}/${RECIPE}"
source "${REPO_ROOT}/experiments/env.sh"

: "${TOTAL_NODES:=8}"
: "${STALENESS_LEVELS:=1 2 4 8}"
: "${RATIOS:=1:7 2:6 3:5 4:4}"
: "${PARTITION:=batch}"
: "${WALL:=04:00:00}"
: "${CHAIN_JOBS:=10}"
: "${WANDB_PROJECT:=async-rl-dapo-math-node-ratio}"
: "${RUN_NAMESPACE:=sr-$(date +%Y%m%d-%H%M%S)}"

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b"
IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async RL waits for long math generations between optimizer steps"}}'

SUBMIT=0
declare -a REQUESTED_POINTS=()

usage() {
    cat <<'EOF'
usage: experiments/staleness_ratio_sweep.sh [--submit] [--point M:T:R ...]

Without --submit, print the selected grid. M is max weight staleness, T is
trainer nodes, and R is rollout nodes. With no --point, the grid comes from
STALENESS_LEVELS and RATIOS. Learning settings come directly from:

  experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch

Useful environment overrides: TOTAL_NODES, STALENESS_LEVELS, RATIOS,
CHAIN_JOBS, PARTITION, WALL, WANDB_PROJECT, and RUN_NAMESPACE.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --submit)
            SUBMIT=1
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

[[ "${QUEUE_POLICY}" == queue-recycle ]] || {
    echo "this sweep requires the recipe's QUEUE_TYPE=queue-recycle, got ${QUEUE_POLICY}" >&2
    exit 1
}
[[ "${STALENESS_REFERENCE_VALUE}" == prefill ]] || {
    echo "this sweep requires STALENESS_REFERENCE=prefill, got ${STALENESS_REFERENCE_VALUE}" >&2
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

printf 'recipe: %s\n' "${RECIPE}"
printf 'fixed by recipe: queue=%s, reference=%s, rollouts=%s, gbs=%s, tp=%s, cp=%s\n' \
    "${QUEUE_POLICY}" "${STALENESS_REFERENCE_VALUE}" "${NUM_ROLLOUT_VALUE}" \
    "${GLOBAL_BATCH}" "${TENSOR_PARALLEL}" "${CONTEXT_PARALLEL}"
printf 'submission: %s nodes, %s, %s, %s chained job(s), wandb=%s\n' \
    "${TOTAL_NODES}" "${PARTITION}" "${WALL}" "${CHAIN_JOBS}" "${WANDB_PROJECT}"
printf 'namespace: %s\n\n' "${RUN_NAMESPACE}"
printf '  %-4s %-3s %-3s %-4s %-8s %s\n' max T R dp gbs/dp run
for point in "${POINTS[@]}"; do
    read -r staleness train_nodes rollout_nodes data_parallel <<<"${point}"
    run_name="s${staleness}-t${train_nodes}r${rollout_nodes}-${RUN_NAMESPACE}"
    printf '  %-4s %-3s %-3s %-4s %-8s %s\n' \
        "${staleness}" "${train_nodes}" "${rollout_nodes}" "${data_parallel}" \
        "$(( GLOBAL_BATCH / data_parallel ))" "${run_name}"
done

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
    local exports_csv raw_job_id job_id chain_index
    local -a dependency=()
    local -a exports=(
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "RUN_NAME=${run_name}"
        "CONFIG_TAG=${run_name}"
        "MAX_WEIGHT_STALENESS=${staleness}"
        "ACTOR_NUM_NODES=${train_nodes}"
        "ROLLOUT_NUM_GPUS=$(( rollout_nodes * GPUS_PER_NODE ))"
    )
    exports_csv="$(IFS=,; printf '%s' "${exports[*]}")"

    for (( chain_index = 1; chain_index <= CHAIN_JOBS; chain_index++ )); do
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
        dependency=("--dependency=afterany:${job_id}")
    done
}

printf '\n'
for point in "${POINTS[@]}"; do
    read -r staleness train_nodes rollout_nodes _ <<<"${point}"
    submit_chain "${staleness}" "${train_nodes}" "${rollout_nodes}"
done
