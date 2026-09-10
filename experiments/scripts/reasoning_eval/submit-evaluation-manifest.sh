#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RUNTIME_REPO_ROOT="$(realpath -- "${RUNTIME_REPO_ROOT:-${BUNDLE_REPO_ROOT}}")"
RUN_SCRIPT="${RUN_SCRIPT:-${RUNTIME_REPO_ROOT}/experiments/scripts/reasoning_eval/run-evaluation.sbatch}"
export MILES_REPO="${MILES_REPO:-${RUNTIME_REPO_ROOT}}"
source "${RUNTIME_REPO_ROOT}/experiments/env.sh"

SUBMIT=0
MANIFEST="${MANIFEST:-}"
EXPECTED_COUNT="${EXPECTED_COUNT:-0}"
MAX_SUBMISSIONS="${MAX_SUBMISSIONS:-0}"
TRAINING_ROOT="${TRAINING_ROOT:-${TRAIN_CKPT_DIR:-${WS:-/lustre/fsw/portfolios/coreai/users/${USER}}/checkpoints/training}}"
RESULT_STUDY_BASE="${RESULT_STUDY_BASE:-${WS:-/lustre/fsw/portfolios/coreai/users/${USER}}/evaluations/reasoning_eval/staleness-ratio-sweep}"
GRID_STUDY_BASE="${GRID_STUDY_BASE:-${RESULT_STUDY_BASE}}"
STUDY_RELATIVE_ROOT="math/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/Qwen3-4B-Base-LR2e-5-Step4000/grpo-clip0.2-0.28-tis2.0"
PROTOCOL_NAME="${PROTOCOL_NAME:-eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-thinking-t0.6-p0.95-k20-aime64-v1}"
EVAL_MODE="${EVAL_MODE:-full}"
TASKS="${TASKS:-aime24 aime25 aime26}"
ACCOUNT="${SLURM_ACCOUNT_NAME:-nemotron_sw_post}"
QOS="${QOS:-normal}"
ROUTES_TEXT="${ROUTES:-batch=04:00:00}"
SQUEUE_TIMEOUT_SECONDS="${SQUEUE_TIMEOUT_SECONDS:-30}"
PRINT_LIMIT="${PRINT_LIMIT:-30}"

usage() {
    cat <<'EOF'
usage: experiments/scripts/reasoning_eval/submit-evaluation-manifest.sh [options]

Submit exactly the checkpoint evaluations listed in a handoff manifest. The
default is a read-only dry run. Re-running the command skips complete results
and jobs that are still active.

Options:
  --manifest PATH          Four-column TSV: source job, namespace, arm, step.
  --expected-count N       Require exactly N unique manifest entries.
  --max-submissions N      Submit at most N pending entries (0 = all).
  --submit                 Submit pending entries after Slurm test-only checks.
  --help                   Show this help.

ROUTES is a space-separated round-robin list such as
"batch=04:00:00". TRAINING_ROOT and
RESULT_STUDY_BASE may be overridden. GRID_STUDY_BASE points at the read-only
source grid metadata when results are written under another user's workspace.
RUNTIME_REPO_ROOT selects the checkout used by the evaluation jobs; it defaults
to the checkout containing this script.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --manifest)
            [[ $# -ge 2 ]] || { echo "--manifest needs a path" >&2; exit 2; }
            MANIFEST="$2"
            shift 2
            ;;
        --expected-count)
            [[ $# -ge 2 ]] || { echo "--expected-count needs a value" >&2; exit 2; }
            EXPECTED_COUNT="$2"
            shift 2
            ;;
        --max-submissions)
            [[ $# -ge 2 ]] || { echo "--max-submissions needs a value" >&2; exit 2; }
            MAX_SUBMISSIONS="$2"
            shift 2
            ;;
        --submit)
            SUBMIT=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "${MANIFEST}" ]] || { echo "set MANIFEST or pass --manifest" >&2; exit 2; }
MANIFEST="$(realpath -- "${MANIFEST}")"
[[ -r "${MANIFEST}" ]] || { echo "manifest is not readable: ${MANIFEST}" >&2; exit 3; }
[[ -x "${RUN_SCRIPT}" ]] || { echo "evaluation runner is not executable: ${RUN_SCRIPT}" >&2; exit 3; }
for value in "${EXPECTED_COUNT}" "${MAX_SUBMISSIONS}" "${SQUEUE_TIMEOUT_SECONDS}" "${PRINT_LIMIT}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "integer controls must be nonnegative: ${value}" >&2; exit 3; }
done
(( SQUEUE_TIMEOUT_SECONDS > 0 )) || { echo "SQUEUE_TIMEOUT_SECONDS must be positive" >&2; exit 3; }
[[ "${EVAL_MODE}" == full || "${EVAL_MODE}" == smoke ]] || { echo "invalid EVAL_MODE: ${EVAL_MODE}" >&2; exit 3; }
[[ "${PROTOCOL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid PROTOCOL_NAME" >&2; exit 3; }

read -r -a ROUTES <<< "${ROUTES_TEXT}"
(( ${#ROUTES[@]} > 0 )) || { echo "ROUTES is empty" >&2; exit 3; }
for route in "${ROUTES[@]}"; do
    [[ "${route}" =~ ^([A-Za-z0-9_-]+)=([0-9]+)-?([0-9]*):([0-9]{2}):([0-9]{2})$ ]] || {
        echo "invalid route (expected partition=HH:MM:SS): ${route}" >&2
        exit 3
    }
done

declare -a SOURCE_JOB_IDS=()
declare -a NAMESPACES=()
declare -a ARMS=()
declare -a STEPS=()
declare -A SEEN_IDENTITIES=()
while IFS=$'\t' read -r source_job_id namespace arm step extra; do
    [[ -n "${source_job_id}" ]] || continue
    [[ "${source_job_id}" == \#* ]] && continue
    [[ -z "${extra}" ]] || { echo "manifest row has more than four columns" >&2; exit 4; }
    [[ "${source_job_id}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid source job id: ${source_job_id}" >&2; exit 4; }
    [[ "${namespace}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid namespace: ${namespace}" >&2; exit 4; }
    [[ "${arm}" =~ ^(s0-colocated|s[1-9][0-9]*-t[1-9][0-9]*r[1-9][0-9]*)$ ]] || {
        echo "invalid arm: ${arm}" >&2
        exit 4
    }
    [[ "${step}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid training step: ${step}" >&2; exit 4; }
    identity="${namespace}/${arm}/${step}"
    [[ -z "${SEEN_IDENTITIES[${identity}]:-}" ]] || { echo "duplicate manifest identity: ${identity}" >&2; exit 4; }
    SEEN_IDENTITIES["${identity}"]=1
    SOURCE_JOB_IDS+=("${source_job_id}")
    NAMESPACES+=("${namespace}")
    ARMS+=("${arm}")
    STEPS+=("${step}")
done < "${MANIFEST}"

manifest_count="${#SOURCE_JOB_IDS[@]}"
(( manifest_count > 0 )) || { echo "manifest has no evaluation entries" >&2; exit 4; }
if (( EXPECTED_COUNT > 0 && manifest_count != EXPECTED_COUNT )); then
    echo "manifest count mismatch: expected=${EXPECTED_COUNT} actual=${manifest_count}" >&2
    exit 4
fi

declare -A ASYNC_SUFFIX_BY_NAMESPACE=()
declare -A COLOCATED_SUFFIX_BY_NAMESPACE=()
for namespace in "${NAMESPACES[@]}"; do
    [[ -z "${ASYNC_SUFFIX_BY_NAMESPACE[${namespace}]:-}" ]] || continue
    grid_path="${GRID_STUDY_BASE}/${namespace}/grid.env"
    [[ -r "${grid_path}" ]] || { echo "grid configuration is not readable: ${grid_path}" >&2; exit 5; }
    suffixes="$(bash -eu -o pipefail -c '
        source "$1"
        training_identity_suffix=""
        if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES:-}" ]]; then
            training_identity_suffix="-concurrency-${ASYNC_MAX_CONCURRENT_SAMPLES}"
        fi
        if (( ${TRAINING_BUFFER_QUEUE_SIZE:-1000} != 1000 )); then
            training_identity_suffix+="-tbq${TRAINING_BUFFER_QUEUE_SIZE}"
        fi
        async_suffix="${ASYNC_RUN_SUFFIX:--zero-trunc-rb-inflight${training_identity_suffix}}"
        colocated_suffix="${COLOCATED_RUN_SUFFIX:--zero-trunc}"
        printf "%s|%s" "${async_suffix}" "${colocated_suffix}"
    ' bash "${grid_path}")"
    ASYNC_SUFFIX_BY_NAMESPACE["${namespace}"]="${suffixes%%|*}"
    COLOCATED_SUFFIX_BY_NAMESPACE["${namespace}"]="${suffixes#*|}"
done

STUDY_ROOT="${TRAINING_ROOT}/${STUDY_RELATIVE_ROOT}"
declare -a CHECKPOINT_PATHS=()
declare -a RESULT_ROOTS=()
for index in "${!SOURCE_JOB_IDS[@]}"; do
    namespace="${NAMESPACES[${index}]}"
    arm="${ARMS[${index}]}"
    step="${STEPS[${index}]}"
    checkpoint_directory=$((step - 1))
    if [[ "${arm}" == s0-colocated ]]; then
        checkpoint_path="${STUDY_ROOT}/colocated/on-policy/max-weight-staleness-0/${arm}-${namespace}${COLOCATED_SUFFIX_BY_NAMESPACE[${namespace}]}/hf/${checkpoint_directory}"
    else
        [[ "${arm}" =~ ^s([1-9][0-9]*)- ]] || { echo "cannot extract staleness from ${arm}" >&2; exit 5; }
        staleness="${BASH_REMATCH[1]}"
        checkpoint_path="${STUDY_ROOT}/async/off-policy/max-weight-staleness-${staleness}-from-prefill/${arm}-${namespace}${ASYNC_SUFFIX_BY_NAMESPACE[${namespace}]}/hf/${checkpoint_directory}"
    fi
    printf -v step_name 'step_%04d' "${step}"
    result_root="${RESULT_STUDY_BASE}/${namespace}/${arm}/${step_name}/${PROTOCOL_NAME}/${EVAL_MODE}"
    [[ -d "${checkpoint_path}" ]] || { echo "checkpoint not found: ${checkpoint_path}" >&2; exit 5; }
    CHECKPOINT_PATHS+=("${checkpoint_path}")
    RESULT_ROOTS+=("${result_root}")
done

active_job_output="$(timeout "${SQUEUE_TIMEOUT_SECONDS}" squeue --me --noheader --format='%i')" || {
    echo "could not query the current user's Slurm jobs; refusing to risk duplicate submissions" >&2
    exit 6
}
declare -A ACTIVE_JOB_IDS=()
while IFS= read -r active_job_id; do
    [[ "${active_job_id}" =~ ^[1-9][0-9]*$ ]] || continue
    ACTIVE_JOB_IDS["${active_job_id}"]=1
done <<< "${active_job_output}"

result_is_complete() {
    local result_root="$1"
    local task
    for task in ${TASKS}; do
        [[ -f "${result_root}/${task}/_SUCCESS" ]] || return 1
    done
}

result_has_active_job() {
    local result_root="$1"
    local marker job_id
    for marker in "${result_root}/.submitted-job" "${result_root}/.active-job"; do
        [[ -s "${marker}" ]] || continue
        read -r job_id _ < "${marker}"
        [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || continue
        [[ -n "${ACTIVE_JOB_IDS[${job_id}]:-}" ]] && return 0
    done
    return 1
}

declare -a PENDING_INDICES=()
complete_count=0
active_count=0
for index in "${!SOURCE_JOB_IDS[@]}"; do
    result_root="${RESULT_ROOTS[${index}]}"
    if result_is_complete "${result_root}"; then
        ((complete_count += 1))
    elif result_has_active_job "${result_root}"; then
        ((active_count += 1))
    else
        PENDING_INDICES+=("${index}")
    fi
done

printf 'manifest: %s\n' "${MANIFEST}"
printf 'entries: total=%d complete=%d active=%d pending=%d\n' \
    "${manifest_count}" "${complete_count}" "${active_count}" "${#PENDING_INDICES[@]}"
printf 'routes: %s\n' "${ROUTES_TEXT}"
printf 'checkpoint root: %s\n' "${TRAINING_ROOT}"
printf 'result study base: %s\n' "${RESULT_STUDY_BASE}"
printf 'grid study base: %s\n' "${GRID_STUDY_BASE}"
for preview_index in "${!PENDING_INDICES[@]}"; do
    (( preview_index < PRINT_LIMIT )) || break
    index="${PENDING_INDICES[${preview_index}]}"
    route="${ROUTES[$((index % ${#ROUTES[@]}))]}"
    printf '  pending source=%s route=%-28s %-14s step=%s namespace=%s\n' \
        "${SOURCE_JOB_IDS[${index}]}" "${route}" "${ARMS[${index}]}" \
        "${STEPS[${index}]}" "${NAMESPACES[${index}]}"
done
if (( ${#PENDING_INDICES[@]} > PRINT_LIMIT )); then
    printf '  ... %d more pending entries\n' "$((${#PENDING_INDICES[@]} - PRINT_LIMIT))"
fi

if (( SUBMIT == 0 )); then
    echo "dry run; add --submit to enqueue pending evaluations"
    exit 0
fi
(( ${#PENDING_INDICES[@]} > 0 )) || { echo "nothing to submit"; exit 0; }

submission_limit="${#PENDING_INDICES[@]}"
if (( MAX_SUBMISSIONS > 0 && MAX_SUBMISSIONS < submission_limit )); then
    submission_limit="${MAX_SUBMISSIONS}"
fi

for ((pending_position = 0; pending_position < submission_limit; pending_position++)); do
    index="${PENDING_INDICES[${pending_position}]}"
    result_root="${RESULT_ROOTS[${index}]}"
    mkdir -p "${result_root}"
    [[ -w "${result_root}" ]] || { echo "result root is not writable: ${result_root}" >&2; exit 7; }
done

declare -A TESTED_ROUTES=()
cd "${RUNTIME_REPO_ROOT}"
for ((pending_position = 0; pending_position < submission_limit; pending_position++)); do
    index="${PENDING_INDICES[${pending_position}]}"
    route="${ROUTES[$((index % ${#ROUTES[@]}))]}"
    [[ -z "${TESTED_ROUTES[${route}]:-}" ]] || continue
    partition="${route%%=*}"
    wall="${route#*=}"
    result_root="${RESULT_ROOTS[${index}]}"
    sbatch --test-only -A "${ACCOUNT}" --partition="${partition}" --qos="${QOS}" --time="${wall}" \
        --export="ALL,CHECKPOINT_PATH=${CHECKPOINT_PATHS[${index}]},RESULT_ROOT=${result_root},ARM_NAME=${ARMS[${index}]},TRAINING_STEP=${STEPS[${index}]},RUN_NAMESPACE=${NAMESPACES[${index}]},TASKS=${TASKS},EVAL_MODE=${EVAL_MODE},PROTOCOL_NAME=${PROTOCOL_NAME}" \
        "${RUN_SCRIPT}"
    TESTED_ROUTES["${route}"]=1
done

submitted_count=0
for ((pending_position = 0; pending_position < submission_limit; pending_position++)); do
    index="${PENDING_INDICES[${pending_position}]}"
    route="${ROUTES[$((index % ${#ROUTES[@]}))]}"
    partition="${route%%=*}"
    wall="${route#*=}"
    arm="${ARMS[${index}]}"
    step="${STEPS[${index}]}"
    namespace="${NAMESPACES[${index}]}"
    result_root="${RESULT_ROOTS[${index}]}"
    checkpoint_path="${CHECKPOINT_PATHS[${index}]}"
    [[ -w "${result_root}" ]] || { echo "result root is not writable: ${result_root}" >&2; exit 7; }
    printf -v step_label '%03d' "${step}"
    raw_job_id="$(sbatch --parsable \
        -A "${ACCOUNT}" \
        --partition="${partition}" \
        --qos="${QOS}" \
        --time="${wall}" \
        --job-name="q3e-${arm}-${step_label}" \
        --output="${result_root}/q3e-${arm}-${step_label}-%j.log" \
        --export="ALL,CHECKPOINT_PATH=${checkpoint_path},RESULT_ROOT=${result_root},ARM_NAME=${arm},TRAINING_STEP=${step},RUN_NAMESPACE=${namespace},TASKS=${TASKS},EVAL_MODE=${EVAL_MODE},PROTOCOL_NAME=${PROTOCOL_NAME}" \
        "${RUN_SCRIPT}")"
    job_id="${raw_job_id%%;*}"
    printf '%s %s handoff-source=%s\n' \
        "${job_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SOURCE_JOB_IDS[${index}]}" \
        > "${result_root}/.submitted-job"
    printf 'submitted source=%s job=%s partition=%s %-14s step=%s namespace=%s\n' \
        "${SOURCE_JOB_IDS[${index}]}" "${job_id}" "${partition}" "${arm}" "${step}" "${namespace}"
    ((submitted_count += 1))
done
printf 'submitted %d checkpoint evaluation job(s)\n' "${submitted_count}"
