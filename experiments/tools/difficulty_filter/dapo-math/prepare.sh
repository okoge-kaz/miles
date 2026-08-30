#!/bin/bash
# Convert the prepared HF checkpoint and rebuild its policy-specific DAPO-Math window.

set -euo pipefail

MODE="${1:-all}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

export HF_CKPT_DIR="${QWEN3_4B_BASE_HF_ROOT:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}"
source experiments/env.sh

MODEL_NAME=Qwen3-4B-Base-LR2e-5-Step4000
HF_MODEL_NAME=iter_0004000
DATASET_NAME=dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000
PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
PASS_RATES="/data/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.jsonl"
FILTERED_OUTPUT="/data/${DATASET_NAME}/${DATASET_NAME}.jsonl"
AUDIT_OUTPUT="/data/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.audit.jsonl"
ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
TOTAL_PROMPTS=17398
HALF_PROMPTS=$(( TOTAL_PROMPTS / 2 ))

submit_convert() {
    sbatch --parsable \
        -A "${ACCOUNT}" -p batch --qos=interactive --time=04:00:00 \
        --job-name=cv-qwen3-4b-base-step4000 \
        --export="ALL,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_NAME=${MODEL_NAME},HF_MODEL_NAME=${HF_MODEL_NAME},MEGATRON_MODEL_TYPE=qwen3-4B" \
        experiments/setup/models/convert_checkpoint.sbatch
}

filter_exports() {
    local output="$1"
    local extra="$2"
    # Keep four waves queued client-side so long n=8 request tails do not drain
    # the GPUs between prompt groups. SGLang still admits at most 80 sequences
    # per DP replica; the smoke run reached 48% KV usage at 64 per replica.
    printf '%s' "ALL,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_NAME=${MODEL_NAME},HF_MODEL_NAME=${HF_MODEL_NAME},PROMPT_DATA=${PROMPT_DATA},OUTPUT=${output},N_SAMPLES=16,SAMPLES_PER_REQUEST=8,MAX_NEW_TOKENS=16384,MAX_CONTEXT_LENGTH=32768,CONCURRENCY=160,MAX_RUNNING_REQUESTS=80,RM_TYPE=deepscaler,ZERO_REWARD_ON_TRUNCATED=1,PASS_RATE_MIN=0.1,PASS_RATE_MAX=0.9,${extra}"
}

submit_smoke() {
    local smoke_output="/data/difficulty/smoke/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.jsonl"
    local smoke_audit="/data/difficulty/smoke/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.audit.jsonl"
    sbatch --parsable \
        -A "${ACCOUNT}" -p batch --qos=interactive --time=01:00:00 \
        --job-name=smoke-pr-qwen3-4b-base-step4000 \
        --export="$(filter_exports "${smoke_output}" "LIMIT=32,DUMP_RESPONSES=${smoke_audit},DUMP_LIMIT=8")" \
        experiments/tools/difficulty_filter/run_measure.sbatch
}

submit_shard_round() {
    local shard="$1"
    local start_index="$2"
    local end_index="$3"
    local previous_job="$4"
    local output="${PASS_RATES%.jsonl}.shard-${shard}-of-2.jsonl"
    local audit="${AUDIT_OUTPUT%.jsonl}.shard-${shard}-of-2.jsonl"
    local dependency=()
    [[ -n "${previous_job}" ]] && dependency=(--dependency="afterany:${previous_job}")

    sbatch --parsable \
        -A "${ACCOUNT}" -p batch --qos=interactive --time=04:00:00 \
        --job-name="pr${shard}-qwen3-4b-step4000" \
        "${dependency[@]}" \
        --export="$(filter_exports "${output}" "START_INDEX=${start_index},END_INDEX=${end_index},DUMP_RESPONSES=${audit},DUMP_LIMIT=8")" \
        experiments/tools/difficulty_filter/run_measure.sbatch
}

submit_filter() {
    local shard_0_job shard_1_job continuation_job
    shard_0_job="$(submit_shard_round 0 0 "${HALF_PROMPTS}" "")"
    shard_1_job="$(submit_shard_round 1 "${HALF_PROMPTS}" "${TOTAL_PROMPTS}" "")"
    continuation_job="$(sbatch --parsable \
        -A "${ACCOUNT}" \
        --dependency="afterany:${shard_0_job}:${shard_1_job}" \
        --export="ALL,FILTER_ROUND=1,FILTER_MAX_ROUNDS=${FILTER_MAX_ROUNDS:-8},HF_CKPT_DIR=${HF_CKPT_DIR}" \
        experiments/tools/difficulty_filter/dapo-math/continue_filter.sbatch)"
    echo "round 1: shard0=${shard_0_job} shard1=${shard_1_job} continuation=${continuation_job}" >&2
}

case "${MODE}" in
    convert)
        echo "convert $(submit_convert)"
        ;;
    smoke)
        echo "smoke $(submit_smoke)"
        ;;
    filter)
        submit_filter
        ;;
    all)
        echo "convert $(submit_convert)"
        submit_filter
        ;;
    *)
        echo "usage: $0 {convert|smoke|filter|all}" >&2
        exit 2
        ;;
esac
