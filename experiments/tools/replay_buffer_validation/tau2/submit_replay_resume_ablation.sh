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
readonly FRESH_SAVE_INTERVAL="${FRESH_UPDATES}"
readonly FRESH_SAVE_RETAIN_INTERVAL=100
readonly RESUME_SAVE_INTERVAL=1000
readonly RESUME_SAVE_RETAIN_INTERVAL=1000
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
: "${RECOVERY_TAG:=}"
SUBMIT=0
REUSED_FIRST_FRESH_JOB=

usage() {
    cat <<'EOF'
usage: experiments/tools/replay_buffer_validation/tau2/submit_replay_resume_ablation.sh [--submit] [--reuse-first-fresh-job JOB_ID]

Dry-run by default. --submit enqueues a completely serial chain of four-node
interactive fresh/resume pairs and one CPU summary job. W&B remains async-rl-tau.

SEEDS is a whitespace-separated list (default: "42"). Each seed is used
for both training and rollout, and the same seed is paired across all modes.

--reuse-first-fresh-job resumes the first seed's no-replay arm from an already
completed intentional-failure job, then submits the remaining seven GPU phases.
VALIDATION_NAMESPACE must name the original run. RECOVERY_TAG optionally names
the new manifest and summary; its default contains the submission timestamp.
EOF
}

while (($# > 0)); do
    case "$1" in
        --submit)
            SUBMIT=1
            shift
            ;;
        --reuse-first-fresh-job)
            (($# >= 2)) || {
                echo "--reuse-first-fresh-job requires a Slurm job ID" >&2
                exit 2
            }
            REUSED_FIRST_FRESH_JOB="$2"
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

validate_checkpoint_schedule() {
    local phase="$1"
    local save_interval="$2"
    local save_retain_interval="$3"

    [[ "${save_interval}" =~ ^[1-9][0-9]*$ ]] || {
        echo "${phase} SAVE_INTERVAL must be positive, got ${save_interval}" >&2
        exit 1
    }
    [[ "${save_retain_interval}" =~ ^[1-9][0-9]*$ ]] || {
        echo "${phase} SAVE_RETAIN_INTERVAL must be positive, got ${save_retain_interval}" >&2
        exit 1
    }
    ((save_retain_interval % save_interval == 0)) || {
        echo "${phase} SAVE_RETAIN_INTERVAL=${save_retain_interval} must be a multiple of SAVE_INTERVAL=${save_interval}" >&2
        exit 1
    }
}

validate_checkpoint_schedule \
    fresh "${FRESH_SAVE_INTERVAL}" "${FRESH_SAVE_RETAIN_INTERVAL}"
validate_checkpoint_schedule \
    resume "${RESUME_SAVE_INTERVAL}" "${RESUME_SAVE_RETAIN_INTERVAL}"
((GLOBAL_BATCH_SIZE == ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
((ASYNC_MAX_CONCURRENT_SAMPLES % N_SAMPLES_PER_PROMPT == 0))
((NUM_ROLLOUT == FRESH_UPDATES + RESUME_UPDATES + 1))
((RESUME_UPDATES < RESUME_SAVE_INTERVAL)) || {
    echo "resume measurement would reach its disabled periodic save interval" >&2
    exit 1
}

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

recovery_source_manifest=
if [[ -n "${REUSED_FIRST_FRESH_JOB}" ]]; then
    [[ "${REUSED_FIRST_FRESH_JOB}" =~ ^[1-9][0-9]*$ ]] || {
        echo "invalid reused Slurm job ID: ${REUSED_FIRST_FRESH_JOB}" >&2
        exit 1
    }
    : "${RECOVERY_TAG:=recovery-$(date +%Y%m%d-%H%M%S)}"
    [[ "${RECOVERY_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
        echo "RECOVERY_TAG contains unsupported characters: ${RECOVERY_TAG}" >&2
        exit 1
    }
    recovery_source_manifest="${MANIFEST_DIR}/${VALIDATION_NAMESPACE}.jobs"
    [[ -r "${recovery_source_manifest}" ]] || {
        echo "original validation manifest is missing: ${recovery_source_manifest}" >&2
        exit 1
    }
    python3 - \
        "${recovery_source_manifest}" \
        "${VALIDATION_NAMESPACE}" \
        "${SEED_VALUES[*]}" \
        "${FRESH_UPDATES}" \
        "${RESUME_UPDATES}" \
        "${ROLLOUT_BATCH_SIZE}" \
        "${N_SAMPLES_PER_PROMPT}" \
        "${GLOBAL_BATCH_SIZE}" \
        "${ASYNC_MAX_CONCURRENT_SAMPLES}" \
        "${REUSED_FIRST_FRESH_JOB}" <<'PY'
import shlex
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected = {
    "VALIDATION_NAMESPACE": sys.argv[2],
    "SEEDS": sys.argv[3],
    "FRESH_UPDATES": sys.argv[4],
    "RESUME_UPDATES": sys.argv[5],
    "ROLLOUT_BATCH_SIZE": sys.argv[6],
    "N_SAMPLES_PER_PROMPT": sys.argv[7],
    "GLOBAL_BATCH_SIZE": sys.argv[8],
    "ASYNC_MAX_CONCURRENT_SAMPLES": sys.argv[9],
}
values = {}
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    key, raw_value = line.split("=", 1)
    parsed = shlex.split(raw_value)
    values[key] = parsed[0] if parsed else ""
for key, expected_value in expected.items():
    actual = values.get(key)
    if actual != expected_value:
        raise SystemExit(
            f"recovery manifest mismatch for {key}: {actual!r} != {expected_value!r}"
        )
first_seed = expected["SEEDS"].split()[0]
job_key = f"SEED_{first_seed}_NO_REPLAY_FRESH_JOB"
if values.get(job_key) != sys.argv[10]:
    raise SystemExit(
        f"reused job does not match {job_key}: {sys.argv[10]} != {values.get(job_key)}"
    )
PY
    mapfile -t reused_logs < <(
        find "${LOG_DIR}" -maxdepth 1 -type f \
            -name "*-${REUSED_FIRST_FRESH_JOB}.log" -print
    )
    ((${#reused_logs[@]} == 1)) || {
        echo "expected one log for reused job ${REUSED_FIRST_FRESH_JOB}, found ${#reused_logs[@]}" >&2
        exit 1
    }
    rg -q --fixed-strings \
        "debug_failure_after_rollout=${FRESH_UPDATES} reached at rollout_id=$((FRESH_UPDATES - 1))" \
        "${reused_logs[0]}" || {
        echo "reused fresh job did not reach the expected intentional failure" >&2
        exit 1
    }
    manifest="${MANIFEST_DIR}/${VALIDATION_NAMESPACE}.${RECOVERY_TAG}.jobs"
    summary="${MANIFEST_DIR}/${VALIDATION_NAMESPACE}.${RECOVERY_TAG}.md"
else
    [[ -z "${RECOVERY_TAG}" ]] || {
        echo "RECOVERY_TAG requires --reuse-first-fresh-job" >&2
        exit 1
    }
    manifest="${MANIFEST_DIR}/${VALIDATION_NAMESPACE}.jobs"
    summary="${MANIFEST_DIR}/${VALIDATION_NAMESPACE}.md"
fi
printf 'Tau2 replay resume ablation: namespace=%s project=%s\n' \
    "${VALIDATION_NAMESPACE}" "${WANDB_PROJECT}"
printf 'fixed: nodes=4 train:rollout=1:3 context=response=40960 RBS=%s n=%s GBS=%s concurrency=%s\n' \
    "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT}" \
    "${GLOBAL_BATCH_SIZE}" "${ASYNC_MAX_CONCURRENT_SAMPLES}"
printf 'phases: fresh=%s updates (%s), resume=%s updates (%s)\n' \
    "${FRESH_UPDATES}" "${FRESH_WALL}" "${RESUME_UPDATES}" "${RESUME_WALL}"
printf 'checkpoint: fresh save/retain=%s/%s, resume save/retain=%s/%s\n' \
    "${FRESH_SAVE_INTERVAL}" "${FRESH_SAVE_RETAIN_INTERVAL}" \
    "${RESUME_SAVE_INTERVAL}" "${RESUME_SAVE_RETAIN_INTERVAL}"
if [[ -n "${REUSED_FIRST_FRESH_JOB}" ]]; then
    printf 'recovery: reuse first no-replay fresh job=%s source=%s tag=%s\n' \
        "${REUSED_FIRST_FRESH_JOB}" "${recovery_source_manifest}" "${RECOVERY_TAG}"
fi
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
    [[ -z "${REUSED_FIRST_FRESH_JOB}" ]] || ((gpu_jobs -= 1))
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
    local save_retain_interval
    local min_outstanding=0
    local min_completed=0
    local min_inflight=0
    local min_inflight_tokens=0
    local min_regenerate=0
    local train_seed=$((1234 + seed))

    [[ -z "${dependency_job}" ]] || dependency=(--dependency="${dependency_kind}:${dependency_job}")
    if [[ "${phase}" == fresh ]]; then
        debug_fail="${FRESH_UPDATES}"
        save_interval="${FRESH_SAVE_INTERVAL}"
        save_retain_interval="${FRESH_SAVE_RETAIN_INTERVAL}"
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
        save_interval="${RESUME_SAVE_INTERVAL}"
        save_retain_interval="${RESUME_SAVE_RETAIN_INTERVAL}"
        expected_failure_rollout=$((FRESH_UPDATES - 1))
    fi
    validate_checkpoint_schedule "${phase}" "${save_interval}" "${save_retain_interval}"
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
        --export="ALL,${COMMON_EXPORTS_CSV},CONFIG_TAG=${config_tag},RUN_NAME=${run_stem}-${phase},TRAIN_SEED=${train_seed},ROLLOUT_SEED=${seed},USE_REPLAY_BUFFER=${use_replay},REPLAY_BUFFER_TYPE=${replay_type},TAU_OVERLAP_DB_RESTORE_WITH_PREFILL=${overlap},SAVE_INTERVAL=${save_interval},SAVE_RETAIN_INTERVAL=${save_retain_interval},DEBUG_EXIT_AFTER_ROLLOUT=${debug_exit},DEBUG_FAIL_AFTER_ROLLOUT=${debug_fail},DEBUG_FAILURE_MIN_OUTSTANDING_GROUPS=${min_outstanding},DEBUG_FAILURE_MIN_COMPLETED_GROUPS=${min_completed},DEBUG_FAILURE_MIN_INFLIGHT_GROUPS=${min_inflight},DEBUG_FAILURE_MIN_INFLIGHT_TOKENS=${min_inflight_tokens},DEBUG_FAILURE_MIN_REGENERATE_GROUPS=${min_regenerate},REPLAY_RESUME_EXPECT_FAILURE_ROLLOUT_ID=${expected_failure_rollout}" \
        "${RECIPE}"
}

declare -A JOB_IDS=()
declare -a RESUME_JOB_IDS=()
previous_resume_job=
arm_index=0
for arm_row in "${ARM_ROWS[@]}"; do
    IFS=: read -r seed mode use_replay replay_type overlap run_stem config_tag <<<"${arm_row}"
    if ((arm_index == 0)) && [[ -n "${REUSED_FIRST_FRESH_JOB}" ]]; then
        fresh_job="${REUSED_FIRST_FRESH_JOB}"
        printf 'reused    seed=%-6s %-18s fresh=%s\n' \
            "${seed}" "${mode}" "${fresh_job}"
    else
        fresh_job="$(submit_phase \
            "${seed}" "${mode}" fresh "${use_replay}" "${replay_type}" "${overlap}" \
            "${run_stem}" "${config_tag}" "${FRESH_WALL}" afterok "${previous_resume_job}")"
    fi
    resume_job="$(submit_phase \
        "${seed}" "${mode}" resume "${use_replay}" "${replay_type}" "${overlap}" \
        "${run_stem}" "${config_tag}" "${RESUME_WALL}" afterany "${fresh_job%%;*}")"
    JOB_IDS["${seed}_${mode}_fresh"]="${fresh_job%%;*}"
    JOB_IDS["${seed}_${mode}_resume"]="${resume_job%%;*}"
    RESUME_JOB_IDS+=("${resume_job%%;*}")
    previous_resume_job="${resume_job%%;*}"
    printf 'submitted seed=%-6s %-18s fresh=%s resume=%s\n' \
        "${seed}" "${mode}" "${fresh_job%%;*}" "${resume_job%%;*}"
    ((arm_index += 1))
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
    printf 'FRESH_SAVE_INTERVAL=%q\n' "${FRESH_SAVE_INTERVAL}"
    printf 'FRESH_SAVE_RETAIN_INTERVAL=%q\n' "${FRESH_SAVE_RETAIN_INTERVAL}"
    printf 'RESUME_SAVE_INTERVAL=%q\n' "${RESUME_SAVE_INTERVAL}"
    printf 'RESUME_SAVE_RETAIN_INTERVAL=%q\n' "${RESUME_SAVE_RETAIN_INTERVAL}"
    printf 'RECOVERY_SOURCE_MANIFEST=%q\n' "${recovery_source_manifest}"
    printf 'RECOVERY_TAG=%q\n' "${RECOVERY_TAG}"
    printf 'REUSED_FIRST_FRESH_JOB=%q\n' "${REUSED_FIRST_FRESH_JOB}"
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
