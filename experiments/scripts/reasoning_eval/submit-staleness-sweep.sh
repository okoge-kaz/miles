#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run-evaluation.sbatch"
VALIDATE_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/validate_checkpoint.py"
source "${REPO_ROOT}/experiments/env.sh"

SUBMIT=0
MAX_SUBMISSIONS="${MAX_SUBMISSIONS:-8}"
RUN_NAMESPACE="${RUN_NAMESPACE:-sr-20260819-212906}"
START_STEP="${START_STEP:-10}"
END_STEP="${END_STEP:-300}"
STEP_INTERVAL="${STEP_INTERVAL:-10}"
EVAL_MODE="${EVAL_MODE:-full}"
TASKS="${TASKS:-aime24 aime25 aime26}"
PROTOCOL_NAME="${PROTOCOL_NAME:-eval-factory-26.03-vllm-0.20.2-cu130-qwen3-rl-thinking-t0.6-p0.95-k20-aime64-v1}"
TRAINING_ROOT="${TRAINING_ROOT:-${TRAIN_CKPT_DIR}}"
STUDY_RELATIVE_ROOT="math/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/Qwen3-4B-Base-LR2e-5-Step4000/grpo-clip0.2-0.28-tis2.0"
STUDY_ROOT="${STUDY_ROOT:-${TRAINING_ROOT}/${STUDY_RELATIVE_ROOT}}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${WS}/evaluations/reasoning_eval}"
RESULT_STUDY_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}"
CONTAINER_ROOT="${REASONING_EVAL_CONTAINER_ROOT:-${SHARED_WS}/containers}"
VLLM_IMAGE="${CONTAINER_ROOT}/vllm-openai-v0.20.2-70a098d9.sqsh"
NEMO_SKILLS_IMAGE="${CONTAINER_ROOT}/nemo-evaluator-nemo-skills-26.03-ac1b048e.sqsh"
VLLM_OCI_DIGEST="sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923"
NEMO_SKILLS_OCI_DIGEST="sha256:ac1b048e13fe7f2a59751b528fc23f5f471452197ad9ae40b715a77cda0a9612"
NEMO_SKILLS_DATA_ROOT="${REASONING_EVAL_DATA_ROOT:-${DATASET_DIR}/evaluation}/nemo-skills-26.03"
ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-normal}"
WALL="${WALL:-04:00:00}"
PRINT_LIMIT="${PRINT_LIMIT:-40}"
SQUEUE_TIMEOUT_SECONDS="${SQUEUE_TIMEOUT_SECONDS:-10}"
SNAPSHOT_ARM_MAX_STEPS="${SNAPSHOT_ARM_MAX_STEPS:-}"
TRUST_PINNED_SNAPSHOT="${TRUST_PINNED_SNAPSHOT:-0}"
LOG_DIR="${OUTPUT_DIR}/reasoning_eval/staleness-ratio-sweep/${RUN_NAMESPACE}"
STALENESS_LEVELS="${STALENESS_LEVELS:-1 2 4 8}"
RATIOS="${RATIOS:-1:7 2:6 3:5 4:4}"
INCLUDE_COLOCATED="${INCLUDE_COLOCATED:-1}"
TRAINING_BUFFER_QUEUE_SIZE="${TRAINING_BUFFER_QUEUE_SIZE:-1000}"
ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-}"
ASYNC_RUN_SUFFIX="${ASYNC_RUN_SUFFIX:-}"
COLOCATED_RUN_SUFFIX="${COLOCATED_RUN_SUFFIX:-}"

usage() {
    cat <<'EOF'
usage: experiments/scripts/reasoning_eval/submit-staleness-sweep.sh [options]

Scan every 10-step Hugging Face checkpoint from the configured
staleness/node-ratio arms. Completed AIME24/25/26 results and active jobs are
skipped. Without --submit this only reports what would be submitted.

Options:
  --submit                 Submit pending checkpoint jobs.
  --max-submissions N      Submit at most N jobs this invocation (0 = all).
  --namespace NAME         Training run namespace.
  --help                   Show this message.

Useful environment overrides: TRAINING_ROOT or STUDY_ROOT, EVALUATION_ROOT,
STALENESS_LEVELS, RATIOS, INCLUDE_COLOCATED, TRAINING_BUFFER_QUEUE_SIZE,
ASYNC_MAX_CONCURRENT_SAMPLES, ASYNC_RUN_SUFFIX, COLOCATED_RUN_SUFFIX,
START_STEP, END_STEP, EVAL_MODE, TASKS, PARTITION, QOS, WALL, and PRINT_LIMIT.
The run suffix overrides select training variants whose checkpoint identities
differ from the default zero-reward-on-truncation configuration.
SNAPSHOT_ARM_MAX_STEPS can pin an arm=max_step comma-separated snapshot; when
set, every configured arm must be present.
HF checkpoint directory N stores the model after learning step N+1, so the
default step 10,20,...,300 scan resolves directories 9,19,...,299.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --submit)
            SUBMIT=1
            shift
            ;;
        --max-submissions)
            [[ $# -ge 2 ]] || { echo "--max-submissions needs a value" >&2; exit 2; }
            MAX_SUBMISSIONS="$2"
            shift 2
            ;;
        --namespace)
            [[ $# -ge 2 ]] || { echo "--namespace needs a value" >&2; exit 2; }
            RUN_NAMESPACE="$2"
            RESULT_STUDY_ROOT="${EVALUATION_ROOT}/staleness-ratio-sweep/${RUN_NAMESPACE}"
            LOG_DIR="${OUTPUT_DIR}/reasoning_eval/staleness-ratio-sweep/${RUN_NAMESPACE}"
            shift 2
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

GRID_CONFIG_PATH="${RESULT_STUDY_ROOT}/grid.env"
if [[ -f "${GRID_CONFIG_PATH}" ]]; then
    # The first submitted evaluation freezes the training cohort contract so
    # refill and analysis commands only need its namespace thereafter.
    source "${GRID_CONFIG_PATH}"
fi

for value in \
    "${MAX_SUBMISSIONS}" "${START_STEP}" "${END_STEP}" "${STEP_INTERVAL}" \
    "${PRINT_LIMIT}" "${SQUEUE_TIMEOUT_SECONDS}" "${TRUST_PINNED_SNAPSHOT}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "integer controls must be nonnegative: ${value}" >&2; exit 3; }
done
(( START_STEP > 0 && END_STEP >= START_STEP && STEP_INTERVAL > 0 )) || {
    echo "invalid step range" >&2
    exit 4
}
(( SQUEUE_TIMEOUT_SECONDS > 0 )) || { echo "SQUEUE_TIMEOUT_SECONDS must be positive" >&2; exit 4; }
(( TRUST_PINNED_SNAPSHOT <= 1 )) || { echo "TRUST_PINNED_SNAPSHOT must be 0 or 1" >&2; exit 4; }
if (( TRUST_PINNED_SNAPSHOT == 1 )) && [[ -z "${SNAPSHOT_ARM_MAX_STEPS}" ]]; then
    echo "TRUST_PINNED_SNAPSHOT requires SNAPSHOT_ARM_MAX_STEPS" >&2
    exit 4
fi
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "RUN_NAMESPACE contains unsupported characters" >&2
    exit 5
}
[[ "${EVAL_MODE}" == full || "${EVAL_MODE}" == smoke ]] || {
    echo "EVAL_MODE must be full or smoke" >&2
    exit 6
}
[[ -f "${RUN_SCRIPT}" ]] || { echo "runner not found: ${RUN_SCRIPT}" >&2; exit 7; }
declare -A SEEN_TASKS=()
for task in ${TASKS}; do
    [[ "${task}" =~ ^aime(24|25|26)$ ]] || { echo "unsupported task: ${task}" >&2; exit 7; }
    [[ -z "${SEEN_TASKS[${task}]:-}" ]] || { echo "duplicate task: ${task}" >&2; exit 7; }
    SEEN_TASKS["${task}"]=1
done
(( ${#SEEN_TASKS[@]} > 0 )) || { echo "TASKS is empty" >&2; exit 7; }

[[ "${INCLUDE_COLOCATED}" =~ ^[01]$ ]] || {
    echo "INCLUDE_COLOCATED must be 0 or 1" >&2
    exit 7
}
[[ "${TRAINING_BUFFER_QUEUE_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "TRAINING_BUFFER_QUEUE_SIZE must be a positive integer" >&2
    exit 7
}
if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES}" ]]; then
    [[ "${ASYNC_MAX_CONCURRENT_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ASYNC_MAX_CONCURRENT_SAMPLES must be empty or a positive integer" >&2
        exit 7
    }
fi
for run_suffix in "${ASYNC_RUN_SUFFIX}" "${COLOCATED_RUN_SUFFIX}"; do
    if [[ -n "${run_suffix}" && ! "${run_suffix}" =~ ^-[A-Za-z0-9._-]+$ ]]; then
        echo "run suffixes must be empty or begin with '-' and contain only safe filename characters" >&2
        exit 7
    fi
done
declare -A SEEN_STALENESS=()
for staleness in ${STALENESS_LEVELS}; do
    [[ "${staleness}" =~ ^[1-9][0-9]*$ ]] || {
        echo "STALENESS_LEVELS must contain positive integers" >&2
        exit 7
    }
    [[ -z "${SEEN_STALENESS[${staleness}]:-}" ]] || {
        echo "duplicate staleness level: ${staleness}" >&2
        exit 7
    }
    SEEN_STALENESS["${staleness}"]=1
done
(( ${#SEEN_STALENESS[@]} > 0 )) || { echo "STALENESS_LEVELS is empty" >&2; exit 7; }
declare -A SEEN_RATIOS=()
for ratio in ${RATIOS}; do
    [[ "${ratio}" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] || {
        echo "RATIOS must contain positive T:R pairs" >&2
        exit 7
    }
    [[ -z "${SEEN_RATIOS[${ratio}]:-}" ]] || {
        echo "duplicate node ratio: ${ratio}" >&2
        exit 7
    }
    SEEN_RATIOS["${ratio}"]=1
done
(( ${#SEEN_RATIOS[@]} > 0 )) || { echo "RATIOS is empty" >&2; exit 7; }

if (( SUBMIT == 1 )); then
    for required_asset in "${VLLM_IMAGE}" "${NEMO_SKILLS_IMAGE}" "${NEMO_SKILLS_DATA_ROOT}/_PREPARED"; do
        [[ -f "${required_asset}" ]] || {
            echo "required evaluation asset not found: ${required_asset}" >&2
            echo "run the image import and AIME data preparation scripts first" >&2
            exit 8
        }
    done
    [[ -f "${VLLM_IMAGE}.oci-digest" && "$(< "${VLLM_IMAGE}.oci-digest")" == "${VLLM_OCI_DIGEST}" ]] || {
        echo "vLLM image OCI provenance is missing or mismatched" >&2
        exit 8
    }
    [[ -f "${NEMO_SKILLS_IMAGE}.oci-digest" \
        && "$(< "${NEMO_SKILLS_IMAGE}.oci-digest")" == "${NEMO_SKILLS_OCI_DIGEST}" ]] || {
        echo "NeMo Skills image OCI provenance is missing or mismatched" >&2
        exit 8
    }
    grep -Fx "nemo_skills_image=${NEMO_SKILLS_IMAGE}" "${NEMO_SKILLS_DATA_ROOT}/_PREPARED" >/dev/null || {
        echo "AIME data was prepared with a different NeMo Skills image" >&2
        exit 8
    }
    mkdir -p "${RESULT_STUDY_ROOT}"
    exec 8> "${RESULT_STUDY_ROOT}/.submission.lock"
    flock --nonblock 8 || { echo "another sweep submission process is active" >&2; exit 9; }
    grid_config_temporary="${GRID_CONFIG_PATH}.tmp.$$"
    {
        printf "STALENESS_LEVELS='%s'\n" "${STALENESS_LEVELS}"
        printf "RATIOS='%s'\n" "${RATIOS}"
        printf 'INCLUDE_COLOCATED=%s\n' "${INCLUDE_COLOCATED}"
        printf 'TRAINING_BUFFER_QUEUE_SIZE=%s\n' "${TRAINING_BUFFER_QUEUE_SIZE}"
        printf "ASYNC_MAX_CONCURRENT_SAMPLES='%s'\n" "${ASYNC_MAX_CONCURRENT_SAMPLES}"
        printf "ASYNC_RUN_SUFFIX='%s'\n" "${ASYNC_RUN_SUFFIX}"
        printf "COLOCATED_RUN_SUFFIX='%s'\n" "${COLOCATED_RUN_SUFFIX}"
    } > "${grid_config_temporary}"
    mv "${grid_config_temporary}" "${GRID_CONFIG_PATH}"
fi

declare -a ARM_NAMES=()
declare -a HF_ROOTS=()
training_identity_suffix=""
if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES}" ]]; then
    training_identity_suffix="-concurrency-${ASYNC_MAX_CONCURRENT_SAMPLES}"
fi
if (( TRAINING_BUFFER_QUEUE_SIZE != 1000 )); then
    training_identity_suffix+="-tbq${TRAINING_BUFFER_QUEUE_SIZE}"
fi
async_run_suffix="${ASYNC_RUN_SUFFIX:--zero-trunc-rb-inflight${training_identity_suffix}}"
colocated_run_suffix="${COLOCATED_RUN_SUFFIX:--zero-trunc}"
for staleness in ${STALENESS_LEVELS}; do
    for ratio in ${RATIOS}; do
        train_nodes="${ratio%%:*}"
        rollout_nodes="${ratio#*:}"
        arm_name="s${staleness}-t${train_nodes}r${rollout_nodes}"
        ARM_NAMES+=("${arm_name}")
        HF_ROOTS+=(
            "${STUDY_ROOT}/async/off-policy/max-weight-staleness-${staleness}-from-prefill/${arm_name}-${RUN_NAMESPACE}${async_run_suffix}/hf"
        )
    done
done
if (( INCLUDE_COLOCATED == 1 )); then
    ARM_NAMES+=(s0-colocated)
    HF_ROOTS+=(
        "${STUDY_ROOT}/colocated/on-policy/max-weight-staleness-0/s0-colocated-${RUN_NAMESPACE}${colocated_run_suffix}/hf"
    )
fi

declare -A KNOWN_ARMS=()
declare -A SNAPSHOT_MAX_STEP_BY_ARM=()
for arm_name in "${ARM_NAMES[@]}"; do
    KNOWN_ARMS["${arm_name}"]=1
done
if [[ -n "${SNAPSHOT_ARM_MAX_STEPS}" ]]; then
    IFS=',' read -r -a snapshot_entries <<< "${SNAPSHOT_ARM_MAX_STEPS}"
    for snapshot_entry in "${snapshot_entries[@]}"; do
        [[ "${snapshot_entry}" == *=* ]] || {
            echo "invalid SNAPSHOT_ARM_MAX_STEPS entry: ${snapshot_entry}" >&2
            exit 7
        }
        arm_name="${snapshot_entry%%=*}"
        max_step="${snapshot_entry#*=}"
        [[ -n "${KNOWN_ARMS[${arm_name}]:-}" ]] || {
            echo "unknown snapshot arm: ${arm_name}" >&2
            exit 7
        }
        [[ "${max_step}" =~ ^[1-9][0-9]*$ ]] \
            && (( max_step >= START_STEP && max_step <= END_STEP && max_step % STEP_INTERVAL == 0 )) || {
            echo "invalid snapshot max step for ${arm_name}: ${max_step}" >&2
            exit 7
        }
        [[ -z "${SNAPSHOT_MAX_STEP_BY_ARM[${arm_name}]:-}" ]] || {
            echo "duplicate snapshot arm: ${arm_name}" >&2
            exit 7
        }
        SNAPSHOT_MAX_STEP_BY_ARM["${arm_name}"]="${max_step}"
    done
    for arm_name in "${ARM_NAMES[@]}"; do
        [[ -n "${SNAPSHOT_MAX_STEP_BY_ARM[${arm_name}]:-}" ]] || {
            echo "snapshot max step is missing for arm: ${arm_name}" >&2
            exit 7
        }
    done
fi

checkpoint_is_complete() {
    local checkpoint_path="$1"
    if (( TRUST_PINNED_SNAPSHOT == 1 )); then
        return 0
    fi
    python3 "${VALIDATE_TOOL}" --quiet "${checkpoint_path}" 2>/dev/null
}

result_is_complete() {
    local result_root="$1"
    local task
    for task in ${TASKS}; do
        [[ -f "${result_root}/${task}/_SUCCESS" ]] || return 1
    done
}

declare -A ACTIVE_SLURM_JOBS=()
declare -A TRACKED_SLURM_JOBS=()
for ((step = START_STEP; step <= END_STEP; step += STEP_INTERVAL)); do
    printf -v step_name 'step_%04d' "${step}"
    for arm_name in "${ARM_NAMES[@]}"; do
        if [[ -n "${SNAPSHOT_ARM_MAX_STEPS}" ]] \
            && (( step > SNAPSHOT_MAX_STEP_BY_ARM[${arm_name}] )); then
            continue
        fi
        result_root="${RESULT_STUDY_ROOT}/${arm_name}/${step_name}/${PROTOCOL_NAME}/${EVAL_MODE}"
        for marker in "${result_root}/.submitted-job" "${result_root}/.active-job"; do
            [[ -s "${marker}" ]] || continue
            read -r tracked_job_id _ < "${marker}"
            [[ "${tracked_job_id}" =~ ^[1-9][0-9]*$ ]] || continue
            TRACKED_SLURM_JOBS["${tracked_job_id}"]=1
        done
    done
done
tracked_job_ids=""
for tracked_job_id in "${!TRACKED_SLURM_JOBS[@]}"; do
    [[ -z "${tracked_job_ids}" ]] || tracked_job_ids+=","
    tracked_job_ids+="${tracked_job_id}"
done
active_job_query_ok=1
if [[ -z "${tracked_job_ids}" ]]; then
    active_job_output=""
elif active_job_output="$(
    timeout "${SQUEUE_TIMEOUT_SECONDS}" \
        squeue --noheader --jobs="${tracked_job_ids}" --format='%i' 2>/dev/null
)"; then
    while IFS= read -r active_job_id; do
        [[ "${active_job_id}" =~ ^[1-9][0-9]*$ ]] || continue
        ACTIVE_SLURM_JOBS["${active_job_id}"]=1
    done <<< "${active_job_output}"
else
    active_job_query_ok=0
fi

job_is_active() {
    local result_root="$1"
    local marker job_id
    for marker in "${result_root}/.submitted-job" "${result_root}/.active-job"; do
        [[ -s "${marker}" ]] || continue
        read -r job_id _ < "${marker}"
        [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || continue
        if (( active_job_query_ok == 0 )); then
            echo "could not query Slurm job ${job_id}; treating it as active" >&2
            return 0
        fi
        [[ -n "${ACTIVE_SLURM_JOBS[${job_id}]:-}" ]] && return 0
    done
    return 1
}

declare -a PENDING_ARMS=()
declare -a PENDING_STEPS=()
declare -a PENDING_CHECKPOINTS=()
declare -a PENDING_RESULTS=()
available_count=0
complete_count=0
active_count=0
missing_count=0
incomplete_count=0
unreadable_count=0

for ((step = START_STEP; step <= END_STEP; step += STEP_INTERVAL)); do
    checkpoint_directory=$((step - 1))
    for arm_index in "${!ARM_NAMES[@]}"; do
        arm_name="${ARM_NAMES[${arm_index}]}"
        if [[ -n "${SNAPSHOT_ARM_MAX_STEPS}" ]] \
            && (( step > SNAPSHOT_MAX_STEP_BY_ARM[${arm_name}] )); then
            continue
        fi
        checkpoint_path="${HF_ROOTS[${arm_index}]}/${checkpoint_directory}"
        printf -v step_name 'step_%04d' "${step}"
        result_root="${RESULT_STUDY_ROOT}/${arm_name}/${step_name}/${PROTOCOL_NAME}/${EVAL_MODE}"
        if (( TRUST_PINNED_SNAPSHOT == 0 )); then
            if [[ ! -d "${checkpoint_path}" ]]; then
                ((missing_count += 1))
                continue
            fi
            checkpoint_validation_status=0
            checkpoint_is_complete "${checkpoint_path}" || checkpoint_validation_status=$?
            if (( checkpoint_validation_status != 0 )); then
                if (( checkpoint_validation_status == 2 )); then
                    ((unreadable_count += 1))
                else
                    ((incomplete_count += 1))
                fi
                continue
            fi
        fi
        ((available_count += 1))
        if result_is_complete "${result_root}"; then
            ((complete_count += 1))
            continue
        fi
        if job_is_active "${result_root}"; then
            ((active_count += 1))
            continue
        fi
        PENDING_ARMS+=("${arm_name}")
        PENDING_STEPS+=("${step}")
        PENDING_CHECKPOINTS+=("${checkpoint_path}")
        PENDING_RESULTS+=("${result_root}")
    done
done

printf 'namespace: %s\n' "${RUN_NAMESPACE}"
printf 'checkpoint study: %s\n' "${STUDY_ROOT}"
printf 'result study: %s\n' "${RESULT_STUDY_ROOT}"
printf 'protocol/mode: %s / %s\n' "${PROTOCOL_NAME}" "${EVAL_MODE}"
printf 'tasks: %s\n' "${TASKS}"
printf 'grid: %d arms x %d requested steps (staleness=%s; ratios=%s; colocated=%s; queue=%s; concurrency=%s)\n' \
    "${#ARM_NAMES[@]}" "$(((END_STEP - START_STEP) / STEP_INTERVAL + 1))" \
    "${STALENESS_LEVELS}" "${RATIOS}" "${INCLUDE_COLOCATED}" \
    "${TRAINING_BUFFER_QUEUE_SIZE}" "${ASYNC_MAX_CONCURRENT_SAMPLES:-recipe-default}"
printf 'training variant suffixes: async=%s; colocated=%s\n' \
    "${async_run_suffix}" "${colocated_run_suffix}"
if [[ -n "${SNAPSHOT_ARM_MAX_STEPS}" ]]; then
    printf 'snapshot arm max steps: %s\n' "${SNAPSHOT_ARM_MAX_STEPS}"
fi
if (( TRUST_PINNED_SNAPSHOT == 1 )); then
    echo "checkpoint validation: trusting the previously validated pinned snapshot"
fi
printf 'status: available=%d complete=%d active=%d pending=%d missing=%d incomplete=%d unreadable=%d\n' \
    "${available_count}" "${complete_count}" "${active_count}" "${#PENDING_ARMS[@]}" \
    "${missing_count}" "${incomplete_count}" "${unreadable_count}"
if (( active_job_query_ok == 1 )); then
    active_job_ids=""
    for active_job_id in "${!ACTIVE_SLURM_JOBS[@]}"; do
        [[ -z "${active_job_ids}" ]] || active_job_ids+=","
        active_job_ids+="${active_job_id}"
    done
    printf 'active job ids: %s\n' "${active_job_ids}"
else
    echo "active job ids: unavailable"
fi

for pending_index in "${!PENDING_ARMS[@]}"; do
    (( pending_index < PRINT_LIMIT )) || break
    printf '  pending %-14s step=%-3s checkpoint=%s\n' \
        "${PENDING_ARMS[${pending_index}]}" \
        "${PENDING_STEPS[${pending_index}]}" \
        "${PENDING_CHECKPOINTS[${pending_index}]}"
done
if (( ${#PENDING_ARMS[@]} > PRINT_LIMIT )); then
    printf '  ... %d additional pending checkpoints (set PRINT_LIMIT to change preview)\n' \
        "$((${#PENDING_ARMS[@]} - PRINT_LIMIT))"
fi

if (( SUBMIT == 0 )); then
    echo "dry run; add --submit to enqueue pending checkpoint evaluations"
    exit 0
fi
(( ${#PENDING_ARMS[@]} > 0 )) || { echo "nothing to submit"; exit 0; }

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"
submission_limit="${#PENDING_ARMS[@]}"
if (( MAX_SUBMISSIONS > 0 && MAX_SUBMISSIONS < submission_limit )); then
    submission_limit="${MAX_SUBMISSIONS}"
fi

first_arm="${PENDING_ARMS[0]}"
first_step="${PENDING_STEPS[0]}"
first_checkpoint="${PENDING_CHECKPOINTS[0]}"
first_result="${PENDING_RESULTS[0]}"
sbatch \
    --test-only \
    -A "${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --time="${WALL}" \
    --export="ALL,CHECKPOINT_PATH=${first_checkpoint},RESULT_ROOT=${first_result},ARM_NAME=${first_arm},TRAINING_STEP=${first_step},RUN_NAMESPACE=${RUN_NAMESPACE},TASKS=${TASKS},EVAL_MODE=${EVAL_MODE},PROTOCOL_NAME=${PROTOCOL_NAME}" \
    "${RUN_SCRIPT}"

for ((pending_index = 0; pending_index < submission_limit; pending_index++)); do
    arm_name="${PENDING_ARMS[${pending_index}]}"
    step="${PENDING_STEPS[${pending_index}]}"
    checkpoint_path="${PENDING_CHECKPOINTS[${pending_index}]}"
    result_root="${PENDING_RESULTS[${pending_index}]}"
    printf -v step_label '%03d' "${step}"
    mkdir -p "${result_root}"
    raw_job_id="$(sbatch \
        --parsable \
        -A "${ACCOUNT}" \
        --partition="${PARTITION}" \
        --qos="${QOS}" \
        --time="${WALL}" \
        --job-name="q3e-${arm_name}-${step_label}" \
        --output="${LOG_DIR}/q3e-${arm_name}-${step_label}-%j.log" \
        --export="ALL,CHECKPOINT_PATH=${checkpoint_path},RESULT_ROOT=${result_root},ARM_NAME=${arm_name},TRAINING_STEP=${step},RUN_NAMESPACE=${RUN_NAMESPACE},TASKS=${TASKS},EVAL_MODE=${EVAL_MODE},PROTOCOL_NAME=${PROTOCOL_NAME}" \
        "${RUN_SCRIPT}")"
    job_id="${raw_job_id%%;*}"
    printf '%s %s\n' "${job_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${result_root}/.submitted-job"
    printf 'submitted %-14s step=%-3s job=%s\n' "${arm_name}" "${step}" "${job_id}"
done

printf 'submitted %d checkpoint evaluation job(s); each job evaluates AIME24/25/26\n' \
    "${submission_limit}"
