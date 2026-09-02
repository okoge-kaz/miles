#!/bin/bash
# Submit the PBS cluster validation ladder:
#   1. host/container/CUDA preflight,
#   2. four-prompt SGLang + DAPO verifier smoke,
#   3. one colocated Miles optimizer update with W&B offline.
#
# The final job waits for the 4B SFT conversion when its release checkpoint is
# not ready yet. The conversion job ID is read from the SFT staging manifest.

set -euo pipefail

[[ "${1:-}" == --submit && $# -eq 1 ]] || {
    echo "usage: $0 --submit" >&2
    exit 2
}

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh

namespace="pbs-$(date -u +%Y%m%dT%H%M%SZ)"
validation_dir="${OUTPUT_DIR}/validation"
mkdir -p "${validation_dir}" "${DATASET_DIR}/difficulty/smoke"

four_b_name=Qwen3-4B-Base-LR2e-5-Step4000
four_b_hf_model="${QWEN3_4B_BASE_HF_MODEL}"
four_b_tracker="${MEGATRON_CKPT_DIR}/${four_b_name}_torch_dist/latest_checkpointed_iteration.txt"
conversion_manifest="${OUTPUT_DIR}/convert/sft-checkpoints.jobs"
conversion_job=""

if [[ ! -f "${four_b_tracker}" || "$(cat "${four_b_tracker}" 2>/dev/null)" != release ]]; then
    [[ -f "${conversion_manifest}" ]] || {
        echo "4B conversion is not ready and ${conversion_manifest} is missing" >&2
        echo "run experiments/setup/models/stage_sft_checkpoints.sh --submit first" >&2
        exit 1
    }
    conversion_job="$(awk -F= '$1 == "QWEN3_4B_BASE_LR2E_5_STEP4000_JOB" {print $2}' "${conversion_manifest}")"
    [[ "${conversion_job}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || {
        echo "invalid 4B conversion job in ${conversion_manifest}: ${conversion_job}" >&2
        exit 1
    }
fi

preflight_job="$(pbs_submit --parsable --profile gpu \
    --time=01:00:00 \
    --job-name="preflight-${namespace}" \
    --output="${validation_dir}/%x-%j.log" \
    experiments/validate_cluster.sbatch)"
smoke_pass_rates="/data/difficulty/smoke/dapo-math-17k.${four_b_name}.${namespace}.passrate.jsonl"
smoke_audit="/data/difficulty/smoke/dapo-math-17k.${four_b_name}.${namespace}.audit.jsonl"
inference_job="$(pbs_submit --parsable --profile gpu \
    --time=01:00:00 \
    --dependency="afterok:${preflight_job}" \
    --job-name="sglang-${namespace}" \
    --output="${validation_dir}/%x-%j.log" \
    --export="ALL,WANDB_MODE=offline,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_NAME=${four_b_name},HF_MODEL_NAME=${four_b_hf_model},PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl,OUTPUT=${smoke_pass_rates},N_SAMPLES=2,SAMPLES_PER_REQUEST=2,MAX_NEW_TOKENS=512,MAX_CONTEXT_LENGTH=2048,CONCURRENCY=4,MAX_RUNNING_REQUESTS=4,RM_TYPE=deepscaler,ZERO_REWARD_ON_TRUNCATED=0,PREFLIGHT=strict,LIMIT=4,DUMP_RESPONSES=${smoke_audit},DUMP_LIMIT=4" \
    experiments/tools/difficulty_filter/run_measure.sbatch)"

training_dependencies="afterok:${inference_job}"
[[ -n "${conversion_job}" ]] && training_dependencies="${training_dependencies}:${conversion_job}"
training_job="$(pbs_submit --parsable --profile gpu \
    --nodes=1 --time=01:00:00 \
    --dependency="${training_dependencies}" \
    --job-name="miles-${namespace}" \
    --output="${validation_dir}/%x-%j.log" \
    --export="ALL,WANDB_MODE=offline,WANDB_API_KEY=,WANDB_PROJECT=miles-cluster-smoke,RUN_NAME=${namespace},TRAINING_ATTENTION_BACKEND=${TRAINING_ATTENTION_BACKEND},DATASET_TAG=dapo-math-17k-cluster-smoke,PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl,ACTOR_NUM_NODES=1,ACTOR_GPUS_PER_NODE=8,ROLLOUT_NUM_GPUS=0,TENSOR_PARALLEL_SIZE=2,CONTEXT_PARALLEL_SIZE=1,ROLLOUT_BATCH_SIZE=4,N_SAMPLES_PER_PROMPT=2,GLOBAL_BATCH_SIZE=8,NUM_STEPS_PER_ROLLOUT=1,NUM_ROLLOUT=1,MAX_RESPONSE_LEN=512,MAX_TOKENS_PER_GPU=2048,EVAL_INTERVAL=0,SAVE_HF=0,SAVE_INTERVAL=100" \
    experiments/scripts/math/sync/dapo-math-p10-90/qwen3-4b/run.sbatch)"

manifest="${validation_dir}/${namespace}.jobs"
{
    printf 'NAMESPACE=%q\n' "${namespace}"
    printf 'WANDB_MODE=offline\n'
    printf 'TRAINING_ATTENTION_BACKEND=%q\n' "${TRAINING_ATTENTION_BACKEND}"
    printf 'CONVERSION_JOB=%q\n' "${conversion_job}"
    printf 'PREFLIGHT_JOB=%q\n' "${preflight_job}"
    printf 'INFERENCE_JOB=%q\n' "${inference_job}"
    printf 'TRAINING_JOB=%q\n' "${training_job}"
} > "${manifest}"

echo "preflight=${preflight_job} inference=${inference_job} training=${training_job}"
echo "wandb=offline manifest=${manifest}"
