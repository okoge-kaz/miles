#!/bin/bash
# Compare rollout and inflight replay buffers with a clean fresh -> resume
# sequence after the policy-specific DAPO-Math dataset has been finalized.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

export HF_CKPT_DIR="${QWEN3_4B_BASE_HF_ROOT:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}"
source experiments/env.sh

MODE=dry-run
AFTEROK_JOB=
while (( $# > 0 )); do
    case "$1" in
        --submit)
            MODE=submit
            shift
            ;;
        --afterok)
            [[ $# -ge 2 ]] || { echo "--afterok needs a job id" >&2; exit 2; }
            AFTEROK_JOB="$2"
            shift 2
            ;;
        --help|-h)
            echo "usage: $0 [--submit] [--afterok JOB_ID]"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b/run.sbatch
ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
: "${VALIDATION_NAMESPACE:=rbtype-step4000-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
WANDB_PROJECT=async-rl-miles-replay-buffer
: "${FRESH_ROLLOUTS:=4}"
: "${RESUME_ROLLOUTS:=2}"
: "${NUM_ROLLOUT:=8}"
: "${FRESH_WALL:=02:00:00}"
: "${RESUME_WALL:=01:30:00}"

# This script normally runs inside the 32-GiB CPU finalizer.  Slurm preserves
# these exported variables across nested sbatch calls, so remove the parent
# step limits and let the two-node recipe establish its own resources.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK

[[ "${VALIDATION_NAMESPACE}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "VALIDATION_NAMESPACE contains unsupported characters: ${VALIDATION_NAMESPACE}" >&2
    exit 1
}
[[ "${FRESH_ROLLOUTS}" =~ ^[1-9][0-9]*$ && "${RESUME_ROLLOUTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FRESH_ROLLOUTS and RESUME_ROLLOUTS must be positive integers" >&2
    exit 1
}
(( FRESH_ROLLOUTS + RESUME_ROLLOUTS <= NUM_ROLLOUT )) || {
    echo "fresh + resume rollouts exceed NUM_ROLLOUT=${NUM_ROLLOUT}" >&2
    exit 1
}

FILTERED_DATASET="${DATASET_DIR}/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000.jsonl"
TRAIN_LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b"
if [[ "${MODE}" == submit ]]; then
    [[ -s "${FILTERED_DATASET}" ]] || {
        echo "filtered dataset is missing or empty: ${FILTERED_DATASET}" >&2
        exit 1
    }
    HF_CHECKPOINT="${HF_CKPT_DIR}/iter_0004000"
    MCORE_CHECKPOINT="${MEGATRON_CKPT_DIR}/Qwen3-4B-Base-LR2e-5-Step4000_torch_dist"
    [[ -s "${HF_CHECKPOINT}/config.json" && -s "${HF_CHECKPOINT}/chat_template.jinja" ]] || {
        echo "HF checkpoint or its chat template is incomplete: ${HF_CHECKPOINT}" >&2
        exit 1
    }
    [[ -s "${MCORE_CHECKPOINT}/latest_checkpointed_iteration.txt" ]] || {
        echo "MCore checkpoint tracker is missing: ${MCORE_CHECKPOINT}" >&2
        exit 1
    }
    [[ "$(< "${MCORE_CHECKPOINT}/latest_checkpointed_iteration.txt")" == release ]] || {
        echo "MCore checkpoint tracker does not select release: ${MCORE_CHECKPOINT}" >&2
        exit 1
    }
    : "${WANDB_API_KEY:?WANDB_API_KEY is required for validation logging}"
    mkdir -p "${TRAIN_LOG_DIR}"
fi

COMMON_EXPORTS=(
    "CONFIG_TAG=${VALIDATION_NAMESPACE}"
    "NUM_ROLLOUT=${NUM_ROLLOUT}"
    "MAX_RESPONSE_LEN=16384"
    "ROLLOUT_BATCH_SIZE=192"
    "N_SAMPLES_PER_PROMPT=16"
    "GLOBAL_BATCH_SIZE=3072"
    "NUM_STEPS_PER_ROLLOUT=1"
    "QUEUE_POLICY=queue-recycle"
    "USE_REPLAY_BUFFER=1"
    "REPLAY_BUFFER_IDENTITY_TAG=1"
    "MAX_WEIGHT_STALENESS=8"
    "STALENESS_REFERENCE=prefill"
    "ZERO_REWARD_ON_TRUNCATED=1"
    "RM_TYPE=deepscaler"
    "SAVE_INTERVAL=1"
    "SAVE_RETAIN_INTERVAL=100"
    "SAVE_HF=0"
    "EVAL_INTERVAL=0"
    "SKIP_EVAL_BEFORE_TRAIN=1"
    "WANDB_PROJECT=${WANDB_PROJECT}"
)
COMMON_EXPORTS_CSV="$(IFS=,; echo "${COMMON_EXPORTS[*]}")"

submit_phase() {
    local buffer_type="$1"
    local phase="$2"
    local rollout_count="$3"
    local wall="$4"
    local dependency_job="$5"
    local dependency=()
    local job_name="rbv-${buffer_type}-${phase}"
    local run_name="rbv-step4000-${buffer_type}-${VALIDATION_NAMESPACE}"

    [[ -n "${dependency_job}" ]] && dependency=(--dependency="afterok:${dependency_job}")
    sbatch --parsable \
        -A "${ACCOUNT}" \
        -p interactive \
        --nodes=2 \
        --time="${wall}" \
        --job-name="${job_name}" \
        "${dependency[@]}" \
        --export="ALL,${COMMON_EXPORTS_CSV},REPLAY_BUFFER_TYPE=${buffer_type},RUN_NAME=${run_name},DEBUG_EXIT_AFTER_ROLLOUT=${rollout_count}" \
        "${RECIPE}"
}

printf 'validation namespace: %s\n' "${VALIDATION_NAMESPACE}"
printf 'wandb project: %s\n' "${WANDB_PROJECT}"
printf 'configuration: 16k response, n=16, max weight staleness=8, zero truncated reward\n'
printf 'fresh/resume rollouts: %s/%s (NUM_ROLLOUT=%s)\n' \
    "${FRESH_ROLLOUTS}" "${RESUME_ROLLOUTS}" "${NUM_ROLLOUT}"
printf 'initial dependency: %s\n' "${AFTEROK_JOB:-none}"

if [[ "${MODE}" != submit ]]; then
    echo "dry-run only; pass --submit to enqueue rollout fresh/resume then inflight fresh/resume"
    exit 0
fi

rollout_fresh_job="$(submit_phase rollout fresh "${FRESH_ROLLOUTS}" "${FRESH_WALL}" "${AFTEROK_JOB}")"
rollout_resume_job="$(submit_phase rollout resume "${RESUME_ROLLOUTS}" "${RESUME_WALL}" "${rollout_fresh_job}")"
inflight_fresh_job="$(submit_phase inflight fresh "${FRESH_ROLLOUTS}" "${FRESH_WALL}" "${rollout_resume_job}")"
inflight_resume_job="$(submit_phase inflight resume "${RESUME_ROLLOUTS}" "${RESUME_WALL}" "${inflight_fresh_job}")"

manifest_dir="${OUTPUT_DIR}/replay_buffer_validation"
mkdir -p "${manifest_dir}"
manifest="${manifest_dir}/${VALIDATION_NAMESPACE}.jobs"
summary="${manifest_dir}/${VALIDATION_NAMESPACE}.md"
summary_job="$(sbatch --parsable \
    -A "${ACCOUNT}" \
    -p cpu_interactive \
    --dependency="afterany:${inflight_resume_job}" \
    --job-name=rbv-summary \
    --export="ALL,VALIDATION_MANIFEST=${manifest},VALIDATION_SUMMARY=${summary}" \
    experiments/math_async/dapo-math-p10-90/qwen3-4b/summarize_replay_buffer_validation.sbatch)"
{
    printf 'VALIDATION_NAMESPACE=%q\n' "${VALIDATION_NAMESPACE}"
    printf 'WANDB_PROJECT=%q\n' "${WANDB_PROJECT}"
    printf 'FRESH_ROLLOUTS=%q\n' "${FRESH_ROLLOUTS}"
    printf 'RESUME_ROLLOUTS=%q\n' "${RESUME_ROLLOUTS}"
    printf 'NUM_ROLLOUT=%q\n' "${NUM_ROLLOUT}"
    printf 'ROLLOUT_FRESH_JOB=%q\n' "${rollout_fresh_job}"
    printf 'ROLLOUT_RESUME_JOB=%q\n' "${rollout_resume_job}"
    printf 'INFLIGHT_FRESH_JOB=%q\n' "${inflight_fresh_job}"
    printf 'INFLIGHT_RESUME_JOB=%q\n' "${inflight_resume_job}"
    printf 'SUMMARY_JOB=%q\n' "${summary_job}"
    printf 'SUMMARY_PATH=%q\n' "${summary}"
} > "${manifest}"

printf 'rollout fresh=%s resume=%s\n' "${rollout_fresh_job}" "${rollout_resume_job}"
printf 'inflight fresh=%s resume=%s\n' "${inflight_fresh_job}" "${inflight_resume_job}"
printf 'summary=%s (after %s)\n' "${summary_job}" "${inflight_resume_job}"
printf 'manifest: %s\n' "${manifest}"
printf 'analyze: python3 %s %s\n' \
    "experiments/math_async/dapo-math-p10-90/qwen3-4b/analyze_replay_buffer_validation.py" \
    "${manifest}"
