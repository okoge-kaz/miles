#!/bin/bash
# Submit the Tau2 40K staleness x placement x truncation-treatment study.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." &>/dev/null && pwd)"
SEGMENT_SCRIPT="${REPO_ROOT}/experiments/scripts/tau_bench/async/areal-tau2/internal/run_staleness_truncation_segment.sbatch"
: "${WANDB_MODE:=online}"
source "${REPO_ROOT}/experiments/env.sh"

readonly TOTAL_NODES=8
readonly GPUS_PER_NODE=8
readonly AREAL_TAU2_ROWS=1982
readonly ROLLOUT_BATCH_SIZE=63
readonly N_SAMPLES_PER_PROMPT=16
readonly GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
readonly TARGET_UPDATES=180
readonly PLANNED_PROMPT_GROUPS=$((TARGET_UPDATES * ROLLOUT_BATCH_SIZE))
readonly PLANNED_TRAJECTORIES=$((PLANNED_PROMPT_GROUPS * N_SAMPLES_PER_PROMPT))
readonly MAX_CONTEXT_LEN=40960
readonly MAX_RESPONSE_LEN=40960
readonly ASYNC_MAX_CONCURRENT_SAMPLES=96
readonly INFLIGHT_GROUPS=$((ASYNC_MAX_CONCURRENT_SAMPLES / N_SAMPLES_PER_PROMPT))
readonly OUTPUT_QUEUE_MAX_GROUPS=1000
readonly NATURAL_QUEUE_LAG_NUMERATOR=$((OUTPUT_QUEUE_MAX_GROUPS + INFLIGHT_GROUPS))
readonly -a STALENESS_LEVELS=(8 16 20)
readonly -a RATIOS=(1:7 2:6)
readonly -a TRUNCATION_MODES=(zero-reward zero-loss)

: "${RUN_NAMESPACE:=tau2-40k-stale-$(date +%Y%m%d-%H%M%S)}"
: "${CHAIN_MAX_SEGMENTS:=64}"
: "${CHAIN_PARTITION:=batch}"
: "${CHAIN_QOS:=normal}"
: "${WANDB_PROJECT:=async-rl-tau}"
CHAIN_ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
CHAIN_LOG_DIR="${OUTPUT_DIR}/training/tau_bench/areal-tau2/qwen3-4b-agentic-sft-953/staleness-truncation"
readonly RUN_NAMESPACE CHAIN_MAX_SEGMENTS CHAIN_PARTITION CHAIN_QOS WANDB_PROJECT
readonly CHAIN_ACCOUNT CHAIN_LOG_DIR
readonly IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Tau2 multi-turn RL waits for external user-simulator turns and long policy trajectories"}}'

SUBMIT=0
usage() {
    cat <<'EOF'
usage: experiments/scripts/tau_bench/async/areal-tau2/submit_staleness_truncation_sweep.sh [--submit]

Dry-run by default. --submit starts the 12-arm grid:

  max weight staleness: 8, 16, 20
  trainer:rollout nodes: 1:7, 2:6
  truncation treatment: zero reward, zero loss

Every arm targets 180 updates (5.721 effective epochs), uses 40,960-token
context/response limits, TIS, LR=1e-6, inflight replay, DB/prefill overlap,
and detailed Tau environment timing. Jobs run in dynamically extended four-hour
segments until the target tracker is present. RUN_NAMESPACE resumes the same
identity when reused. CHAIN_MAX_SEGMENTS is a fail-safe, not a planned length.
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

[[ -r "${SEGMENT_SCRIPT}" ]] || {
    echo "chain segment script is missing: ${SEGMENT_SCRIPT}" >&2
    exit 1
}
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "RUN_NAMESPACE contains unsupported characters: ${RUN_NAMESPACE}" >&2
    exit 1
}
[[ "${CHAIN_MAX_SEGMENTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "CHAIN_MAX_SEGMENTS must be positive" >&2
    exit 1
}

effective_epochs="$(awk -v groups="${PLANNED_PROMPT_GROUPS}" -v rows="${AREAL_TAU2_ROWS}" 'BEGIN { printf "%.6f", groups / rows }')"
natural_queue_lag="$(awk -v numerator="${NATURAL_QUEUE_LAG_NUMERATOR}" -v denominator="${ROLLOUT_BATCH_SIZE}" 'BEGIN { printf "%.3f", numerator / denominator }')"
printf 'Tau2 40K staleness study: 12 arms, W&B project=%s\n' "${WANDB_PROJECT}"
printf 'schedule: updates=%s, RBS=%s, n=%s, GBS=%s, prompt_groups=%s, trajectories=%s, effective_epochs=%s\n' \
    "${TARGET_UPDATES}" "${ROLLOUT_BATCH_SIZE}" "${N_SAMPLES_PER_PROMPT}" \
    "${GLOBAL_BATCH_SIZE}" "${PLANNED_PROMPT_GROUPS}" "${PLANNED_TRAJECTORIES}" \
    "${effective_epochs}"
printf 'sequence: max_context=%s, max_response=%s; optimizer: TIS, LR=1e-6\n' \
    "${MAX_CONTEXT_LEN}" "${MAX_RESPONSE_LEN}"
printf 'checkpoint: Megatron every 10 (latest only), HF every 10, replay latest 1; chain: 4h, max=%s segments\n' \
    "${CHAIN_MAX_SEGMENTS}"
printf 'timing: Tau user/tool/terminal/reset/close logging=on, DB restore/prefill overlap=on\n'
printf 'queue caveat: (capacity %s + inflight %s) / RBS %s = %s updates; M=20 is an effectively unbounded control and M=16 is near the natural ceiling.\n' \
    "${OUTPUT_QUEUE_MAX_GROUPS}" "${INFLIGHT_GROUPS}" "${ROLLOUT_BATCH_SIZE}" \
    "${natural_queue_lag}"
printf '\n  %-4s %-5s %-5s %-11s %-4s %-8s %s\n' M train roll truncation dp gbs/dp run

declare -a ARM_ROWS=()
for staleness in "${STALENESS_LEVELS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        IFS=: read -r train_nodes rollout_nodes <<<"${ratio}"
        train_gpus=$((train_nodes * GPUS_PER_NODE))
        rollout_gpus=$((rollout_nodes * GPUS_PER_NODE))
        data_parallel=$((train_gpus / 2))
        ((GLOBAL_BATCH_SIZE % data_parallel == 0)) || {
            echo "GBS=${GLOBAL_BATCH_SIZE} is not divisible by dp=${data_parallel} for ${ratio}" >&2
            exit 1
        }
        for truncation_mode in "${TRUNCATION_MODES[@]}"; do
            case "${truncation_mode}" in
                zero-reward)
                    mode_tag=zr
                    zero_reward=1
                    zero_loss=0
                    ;;
                zero-loss)
                    mode_tag=zl
                    zero_reward=0
                    zero_loss=1
                    ;;
                *)
                    echo "unsupported truncation mode: ${truncation_mode}" >&2
                    exit 1
                    ;;
            esac
            run_name="tau40k-s${staleness}-t${train_nodes}r${rollout_nodes}-${mode_tag}-${RUN_NAMESPACE}"
            config_tag="40k-u${TARGET_UPDATES}-rbs${ROLLOUT_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-t${train_nodes}r${rollout_nodes}-${mode_tag}-${RUN_NAMESPACE}"
            ((${#run_name} + 9 <= 128)) || {
                echo "run name is too long for W&B: ${run_name}" >&2
                exit 1
            }
            printf '  %-4s %-5s %-5s %-11s %-4s %-8s %s\n' \
                "${staleness}" "${train_nodes}" "${rollout_nodes}" "${truncation_mode}" \
                "${data_parallel}" "$((GLOBAL_BATCH_SIZE / data_parallel))" "${run_name}"
            ARM_ROWS+=("${staleness}:${train_nodes}:${rollout_nodes}:${rollout_gpus}:${mode_tag}:${zero_reward}:${zero_loss}:${run_name}:${config_tag}")
        done
    done
done

if ((SUBMIT == 0)); then
    printf '\ndry run; add --submit to enqueue the first four-hour segment of every arm\n'
    exit 0
fi

mkdir -p "${CHAIN_LOG_DIR}" "${OUTPUT_DIR}/.save-trigger"
if ! active_job_rows="$(squeue --noheader --user="${USER:?}" --format='%j|%A')"; then
    echo "could not query active jobs; refusing a potentially duplicate submission" >&2
    exit 1
fi
declare -A ACTIVE_JOB_IDS=()
while IFS='|' read -r active_name active_job_id; do
    [[ -n "${active_name}" && -n "${active_job_id}" ]] || continue
    ACTIVE_JOB_IDS["${active_name}"]="${ACTIVE_JOB_IDS["${active_name}"]:+${ACTIVE_JOB_IDS["${active_name}"]},}${active_job_id}"
done <<<"${active_job_rows}"

printf '\n'
for arm_row in "${ARM_ROWS[@]}"; do
    IFS=: read -r staleness train_nodes rollout_nodes rollout_gpus mode_tag zero_reward zero_loss run_name config_tag <<<"${arm_row}"
    if [[ -n "${ACTIVE_JOB_IDS["${run_name}"]:-}" ]]; then
        printf 'skipped   %-72s active_job=%s\n' \
            "${run_name}" "${ACTIVE_JOB_IDS["${run_name}"]}"
        continue
    fi
    save_trigger="/root/miles/experiments/outputs/.save-trigger/${run_name}.sentinel"
    exports=(
        "TOTAL_NODES=${TOTAL_NODES}"
        "RUN_NAME=${run_name}"
        "CONFIG_TAG=${config_tag}"
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "WANDB_MODE=${WANDB_MODE}"
        "MAX_WEIGHT_STALENESS=${staleness}"
        "ACTOR_NUM_NODES=${train_nodes}"
        "ROLLOUT_NUM_GPUS=${rollout_gpus}"
        "MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}"
        "MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN}"
        "MAX_TOKENS_PER_GPU=${MAX_CONTEXT_LEN}"
        "LR=1e-6"
        "IS_CORRECTION=tis"
        "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
        "N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
        "TRAIN_EPOCHS=6"
        "NUM_ROLLOUT=${TARGET_UPDATES}"
        "ALLOW_NUM_ROLLOUT_OVERRIDE=1"
        "ASYNC_MAX_CONCURRENT_SAMPLES=${ASYNC_MAX_CONCURRENT_SAMPLES}"
        "USE_REPLAY_BUFFER=1"
        "REPLAY_BUFFER_TYPE=inflight"
        "REPLAY_BUFFER_KEEP_LAST=1"
        "TAU_OVERLAP_DB_RESTORE_WITH_PREFILL=1"
        "TAU_LOG_OVERHEAD=1"
        "TAU_LOG_LEVEL=ERROR"
        "ZERO_REWARD_ON_TRUNCATED=${zero_reward}"
        "ZERO_LOSS_ON_TRUNCATED=${zero_loss}"
        "SAVE_INTERVAL=10"
        "SAVE_RETAIN_INTERVAL=$((TARGET_UPDATES + 1))"
        "SAVE_HF=1"
        "HF_SAVE_INTERVAL=10"
        "SAVE_TRIGGER_SENTINEL=${save_trigger}"
        "CHECKPOINT_COMPLETION_PREFLIGHT=1"
        "LOG_SAMPLE_STALENESS_METRICS=1"
        "LOG_SAMPLE_STALENESS_RATIO_HISTOGRAM=1"
        "SAMPLE_STALENESS_MAX_BIN=32"
        "CLEAN_CHECKPOINT=0"
        "DEBUG_EXIT_AFTER_ROLLOUT="
        "CHAIN_INDEX=1"
        "CHAIN_MAX_SEGMENTS=${CHAIN_MAX_SEGMENTS}"
        "CHAIN_ACCOUNT=${CHAIN_ACCOUNT}"
        "CHAIN_PARTITION=${CHAIN_PARTITION}"
        "CHAIN_QOS=${CHAIN_QOS}"
        "CHAIN_LOG_DIR=${CHAIN_LOG_DIR}"
    )
    exports_csv="$(IFS=,; printf '%s' "${exports[*]}")"
    raw_job_id="$(sbatch --parsable \
        --account="${CHAIN_ACCOUNT}" \
        --partition="${CHAIN_PARTITION}" \
        --qos="${CHAIN_QOS}" \
        --nodes="${TOTAL_NODES}" \
        --time=04:00:00 \
        --job-name="${run_name}" \
        --comment="${IDLE_EXEMPTION}" \
        --output="${CHAIN_LOG_DIR}/${run_name}-%j.log" \
        --export="ALL,${exports_csv}" \
        "${SEGMENT_SCRIPT}")"
    printf 'submitted %-72s job=%s chain=1/%s\n' \
        "${run_name}" "${raw_job_id%%;*}" "${CHAIN_MAX_SEGMENTS}"
done
