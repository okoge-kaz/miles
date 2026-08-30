#!/bin/bash
# Compare Tau2 restart loss and warm-resume latency across four replay modes.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

: "${WANDB_MODE:=online}"
source experiments/env.sh

readonly RECIPE=experiments/scripts/tau_bench/async/areal-tau2/qwen3-4b-agentic-sft-953/run.sbatch
readonly SUMMARY_RECIPE=experiments/tools/replay_buffer_validation/tau2/summarize.sbatch
readonly ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
readonly WANDB_PROJECT=async-rl-tau
readonly FRESH_UPDATES=10
readonly RESUME_UPDATES=6
readonly ROLLOUT_BATCH_SIZE=8
readonly N_SAMPLES_PER_PROMPT=16
readonly GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
readonly ASYNC_MAX_CONCURRENT_SAMPLES=96
# Leave one unreachable rollout after the clean resume exit so the sixth
# measured resume update does not trigger an end-of-training checkpoint.
readonly NUM_ROLLOUT=$((FRESH_UPDATES + RESUME_UPDATES + 1))
readonly FRESH_WALL=04:00:00
readonly RESUME_WALL=02:00:00
readonly LOG_DIR="${OUTPUT_DIR}/training/tau_bench/areal-tau2/qwen3-4b-agentic-sft-953/replay-resume-ablation"
readonly MANIFEST_DIR="${OUTPUT_DIR}/replay_buffer_validation/tau2"
readonly IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Tau2 multi-turn replay validation waits for external user-simulator turns"}}'
readonly -a MODES=(no-replay rollout inflight inflight-overlap)

: "${VALIDATION_NAMESPACE:=tau2-rbresume-$(date +%Y%m%d-%H%M%S)}"
: "${SEEDS:=42}"
SUBMIT=0

usage() {
    cat <<'EOF'
usage: experiments/tools/replay_buffer_validation/tau2/submit_replay_resume_ablation.sh [--submit]

Dry-run by default. --submit enqueues a completely serial chain of four-node
interactive fresh/resume pairs and one CPU summary job. W&B remains async-rl-tau.

SEEDS is a whitespace-separated list (default: "42"). Each seed is used
for both training and rollout, and the same seed is paired across all modes.
EOF
}

while (($# > 0)); do
    case "$1" in
        --submit)
            SUBMIT=1
            shift
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

[[ "${VALIDATION_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "VALIDATION_NAMESPACE contains unsupported characters: ${VALIDATION_NAMESPACE}" >&2
    exit 1
}
[[ -r "${RECIPE}" && -r "${SUMMARY_RECIPE}" ]]
read -r -a SEED_VALUES <<<"${SEEDS}"
(( ${#SEED_VALUES[@]} > 0 )) || {
    echo "SEEDS must contain at least one non-negative integer" >&2
    exit 1
}
for seed in "${SEED_VALUES[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || {
        echo "invalid seed in SEEDS: ${seed}" >&2
        exit 1
    }
done

manifest="${MANIFEST_DIR}/${VALIDATION_NAMESPACE}.jobs"
summary="${MANIFEST_DIR}/${VALIDATION_NAMESPACE}.md"
printf 'Tau2 replay resume ablation: namespace=%s project=%s\n' \
    "${VALIDATION_NAMESPACE}" "${WANDB_PROJECT}"
printf 'fixed: nodes=4 train:rollout=1:3 context=response=40960 RBS=%s n=%s GBS=%s concurrency=%s\n' \
    "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT}" \
    "${GLOBAL_BATCH_SIZE}" "${ASYNC_MAX_CONCURRENT_SAMPLES}"
printf 'phases: fresh=%s updates (%s), resume=%s updates (%s)\n' \
    "${FRESH_UPDATES}" "${FRESH_WALL}" "${RESUME_UPDATES}" "${RESUME_WALL}"
printf 'paired seeds: %s\n' "${SEED_VALUES[*]}"
printf '  %-6s %-18s %-6s %-8s %-7s %s\n' seed mode replay overlap samples run

declare -a ARM_ROWS=()
for seed in "${SEED_VALUES[@]}"; do
    for mode in "${MODES[@]}"; do
        case "${mode}" in
            no-replay)
                use_replay=0
                replay_type=rollout
                overlap=0
                ;;
            rollout)
                use_replay=1
                replay_type=rollout
                overlap=0
                ;;
            inflight)
                use_replay=1
                replay_type=inflight
                overlap=0
                ;;
            inflight-overlap)
                use_replay=1
                replay_type=inflight
                overlap=1
                ;;
        esac
        run_stem="taurb-seed${seed}-${mode}-${VALIDATION_NAMESPACE}"
        config_tag="rbresume-40k-rbs${ROLLOUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-seed${seed}-${mode}-${VALIDATION_NAMESPACE}"
        printf '  %-6s %-18s %-6s %-8s %-7s %s\n' \
            "${seed}" "${mode}" "${use_replay}" "${overlap}" \
            "$((FRESH_UPDATES * GLOBAL_BATCH_SIZE))" "${run_stem}"
        ARM_ROWS+=(
            "${seed}:${mode}:${use_replay}:${replay_type}:${overlap}:${run_stem}:${config_tag}"
        )
    done
done

if ((SUBMIT == 0)); then
    gpu_jobs=$((${#ARM_ROWS[@]} * 2))
    printf '\ndry run; add --submit to enqueue %d GPU jobs and one CPU summary\n' "${gpu_jobs}"
    exit 0
fi

[[ ! -e "${manifest}" ]] || {
    echo "validation manifest already exists: ${manifest}" >&2
    exit 1
}
: "${WANDB_API_KEY:?WANDB_API_KEY is required for online validation logging}"
mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}"

COMMON_EXPORTS=(
    "TOTAL_NODES=4"
    "WANDB_PROJECT=${WANDB_PROJECT}"
    "WANDB_MODE=${WANDB_MODE}"
    "MAX_WEIGHT_STALENESS=4"
    "ACTOR_NUM_NODES=1"
    "ROLLOUT_NUM_GPUS=24"
    "MAX_CONTEXT_LEN=40960"
    "MAX_RESPONSE_LEN=40960"
    "MAX_TOKENS_PER_GPU=40960"
    "LR=1e-6"
    "IS_CORRECTION=tis"
    "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
    "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
    "TRAIN_EPOCHS=6"
    "NUM_ROLLOUT=${NUM_ROLLOUT}"
    "ALLOW_NUM_ROLLOUT_OVERRIDE=1"
    "ASYNC_MAX_CONCURRENT_SAMPLES=${ASYNC_MAX_CONCURRENT_SAMPLES}"
    "ALLOW_TAU_REPLAY_ABLATION=1"
    "LOG_REPLAY_RESUME_METRICS=1"
    "REPLAY_BUFFER_IDENTITY_TAG=1"
    "REPLAY_BUFFER_KEEP_LAST=2"
    "TAU_LOG_OVERHEAD=1"
    "TAU_LOG_LEVEL=ERROR"
    "ZERO_REWARD_ON_TRUNCATED=1"
    "ZERO_LOSS_ON_TRUNCATED=0"
    "SAVE_RETAIN_INTERVAL=100"
    "SAVE_HF=0"
    "HF_SAVE_INTERVAL="
    "EVAL_INTERVAL=0"
    "SKIP_EVAL_BEFORE_TRAIN=1"
    "CHECKPOINT_COMPLETION_PREFLIGHT=0"
    "LOG_SAMPLE_STALENESS_METRICS=0"
    "LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM=0"
    "LOG_UPDATE_DIAGNOSTICS=0"
    "CLEAN_CHECKPOINT=0"
)
COMMON_EXPORTS_CSV="$(IFS=,; printf '%s' "${COMMON_EXPORTS[*]}")"

submit_phase() {
    local seed="$1"
    local mode="$2"
    local phase="$3"
    local use_replay="$4"
    local replay_type="$5"
    local overlap="$6"
    local run_stem="$7"
    local config_tag="$8"
    local wall="$9"
    local dependency_kind="${10}"
    local dependency_job="${11}"
    local dependency=()
    local debug_exit=
    local debug_fail=
    local expected_failure_rollout=
    local save_interval
    local min_outstanding=0
    local min_completed=0
    local min_inflight=0
    local min_inflight_tokens=0
    local min_regenerate=0
    local train_seed=$((1234 + seed))

    [[ -z "${dependency_job}" ]] || dependency=(--dependency="${dependency_kind}:${dependency_job}")
    if [[ "${phase}" == fresh ]]; then
        debug_fail="${FRESH_UPDATES}"
        save_interval="${FRESH_UPDATES}"
        min_outstanding=1
        case "${mode}" in
            rollout)
                min_completed=1
                min_regenerate=1
                ;;
            inflight|inflight-overlap)
                min_completed=1
                min_inflight=1
                min_inflight_tokens=64
                ;;
        esac
    else
        debug_exit="${RESUME_UPDATES}"
        save_interval=1000
        expected_failure_rollout=$((FRESH_UPDATES - 1))
    fi
    sbatch --parsable \
        --account="${ACCOUNT}" \
        --partition=batch \
        --qos=interactive \
        --nodes=4 \
        --time="${wall}" \
        --job-name="taurb-s${seed}-${mode}-${phase}" \
        --comment="${IDLE_EXEMPTION}" \
        --output="${LOG_DIR}/${run_stem}-${phase}-%j.log" \
        "${dependency[@]}" \
        --export="ALL,${COMMON_EXPORTS_CSV},CONFIG_TAG=${config_tag},RUN_NAME=${run_stem}-${phase},TRAIN_SEED=${train_seed},ROLLOUT_SEED=${seed},USE_REPLAY_BUFFER=${use_replay},REPLAY_BUFFER_TYPE=${replay_type},TAU_OVERLAP_DB_RESTORE_WITH_PREFILL=${overlap},SAVE_INTERVAL=${save_interval},DEBUG_EXIT_AFTER_ROLLOUT=${debug_exit},DEBUG_FAIL_AFTER_ROLLOUT=${debug_fail},DEBUG_FAILURE_MIN_OUTSTANDING_GROUPS=${min_outstanding},DEBUG_FAILURE_MIN_COMPLETED_GROUPS=${min_completed},DEBUG_FAILURE_MIN_INFLIGHT_GROUPS=${min_inflight},DEBUG_FAILURE_MIN_INFLIGHT_TOKENS=${min_inflight_tokens},DEBUG_FAILURE_MIN_REGENERATE_GROUPS=${min_regenerate},REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID=${expected_failure_rollout}" \
        "${RECIPE}"
}

declare -A JOB_IDS=()
declare -a RESUME_JOB_IDS=()
previous_resume_job=
for arm_row in "${ARM_ROWS[@]}"; do
    IFS=: read -r seed mode use_replay replay_type overlap run_stem config_tag <<<"${arm_row}"
    fresh_job="$(submit_phase \
        "${seed}" "${mode}" fresh "${use_replay}" "${replay_type}" "${overlap}" \
        "${run_stem}" "${config_tag}" "${FRESH_WALL}" afterok "${previous_resume_job}")"
    resume_job="$(submit_phase \
        "${seed}" "${mode}" resume "${use_replay}" "${replay_type}" "${overlap}" \
        "${run_stem}" "${config_tag}" "${RESUME_WALL}" afterany "${fresh_job%%;*}")"
    JOB_IDS["${seed}_${mode}_fresh"]="${fresh_job%%;*}"
    JOB_IDS["${seed}_${mode}_resume"]="${resume_job%%;*}"
    RESUME_JOB_IDS+=("${resume_job%%;*}")
    previous_resume_job="${resume_job%%;*}"
    printf 'submitted seed=%-6s %-18s fresh=%s resume=%s\n' \
        "${seed}" "${mode}" "${fresh_job%%;*}" "${resume_job%%;*}"
done

summary_dependency="$(IFS=:; printf '%s' "${RESUME_JOB_IDS[*]}")"
summary_job="$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --partition=cpu \
    --qos=cpu-interactive \
    --dependency="afterany:${summary_dependency}" \
    --job-name=taurb-summary \
    --output="${MANIFEST_DIR}/summary-${VALIDATION_NAMESPACE}-%j.log" \
    --export="ALL,VALIDATION_MANIFEST=${manifest},VALIDATION_SUMMARY=${summary}" \
    "${SUMMARY_RECIPE}")"

{
    printf 'VALIDATION_NAMESPACE=%q\n' "${VALIDATION_NAMESPACE}"
    printf 'WANDB_PROJECT=%q\n' "${WANDB_PROJECT}"
    printf 'LOG_DIR=%q\n' "${LOG_DIR}"
    printf 'SEEDS=%q\n' "${SEED_VALUES[*]}"
    printf 'FRESH_UPDATES=%q\n' "${FRESH_UPDATES}"
    printf 'RESUME_UPDATES=%q\n' "${RESUME_UPDATES}"
    printf 'ROLLOUT_BATCH_SIZE=%q\n' "${ROLLOUT_BATCH_SIZE}"
    printf 'N_SAMPLES_PER_PROMPT=%q\n' "${N_SAMPLES_PER_PROMPT}"
    printf 'GLOBAL_BATCH_SIZE=%q\n' "${GLOBAL_BATCH_SIZE}"
    printf 'ASYNC_MAX_CONCURRENT_SAMPLES=%q\n' "${ASYNC_MAX_CONCURRENT_SAMPLES}"
    for seed in "${SEED_VALUES[@]}"; do
        for mode in "${MODES[@]}"; do
            key="${mode//-/_}"
            printf 'SEED_%s_%s_FRESH_JOB=%q\n' \
                "${seed}" "${key^^}" "${JOB_IDS["${seed}_${mode}_fresh"]}"
            printf 'SEED_%s_%s_RESUME_JOB=%q\n' \
                "${seed}" "${key^^}" "${JOB_IDS["${seed}_${mode}_resume"]}"
        done
    done
    printf 'SUMMARY_JOB=%q\n' "${summary_job%%;*}"
    printf 'SUMMARY_PATH=%q\n' "${summary}"
} >"${manifest}"

printf 'summary=%s output=%s\n' "${summary_job%%;*}" "${summary}"
