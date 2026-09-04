#!/bin/bash
#SBATCH --job-name=continue-pr-qwen3-4b-step4000
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=experiments/outputs/difficulty/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)}"
export HF_CKPT_DIR="${QWEN3_4B_BASE_HF_ROOT:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}"
source "${REPO_ROOT}/experiments/env.sh"
cd "${REPO_ROOT}"

MODEL_NAME=Qwen3-4B-Base-LR2e-5-Step4000
HF_MODEL_NAME=iter_0004000
PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
PASS_RATES="/data/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.jsonl"
AUDIT_OUTPUT="/data/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.audit.jsonl"
ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
TOTAL_PROMPTS=17398
HALF_PROMPTS=$(( TOTAL_PROMPTS / 2 ))
FILTER_ROUND="${FILTER_ROUND:-1}"
FILTER_MAX_ROUNDS="${FILTER_MAX_ROUNDS:-8}"
TAIL_START_INDEX="${TAIL_START_INDEX:-}"
TAIL_OUTPUT="${PASS_RATES%.jsonl}.shard-0-tail-${TAIL_START_INDEX}-of-${HALF_PROMPTS}.jsonl"

# Do not export this 2-GiB coordinator's step limits into GPU child jobs.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK

filter_exports() {
    local output="$1"
    local extra="$2"
    printf '%s' "ALL,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_NAME=${MODEL_NAME},HF_MODEL_NAME=${HF_MODEL_NAME},PROMPT_DATA=${PROMPT_DATA},OUTPUT=${output},N_SAMPLES=16,SAMPLES_PER_REQUEST=8,MAX_NEW_TOKENS=16384,MAX_CONTEXT_LENGTH=32768,CONCURRENCY=160,MAX_RUNNING_REQUESTS=80,RM_TYPE=deepscaler,ZERO_REWARD_ON_TRUNCATED=1,PASS_RATE_MIN=0.1,PASS_RATE_MAX=0.9,${extra}"
}

shard_complete() {
    local shard="$1"
    local output="${PASS_RATES%.jsonl}.shard-${shard}-of-2.jsonl"
    local host_output="${DATASET_DIR}/${output#/data/}"
    [[ -f "${host_output}.complete" ]]
}

tail_complete() {
    [[ -n "${TAIL_START_INDEX}" ]] || return 1
    local host_output="${DATASET_DIR}/${TAIL_OUTPUT#/data/}"
    [[ -f "${host_output}.complete" ]]
}

reconcile_tail() {
    [[ -n "${TAIL_START_INDEX}" ]] || return 0
    shard_complete 0 && return 0
    tail_complete || return 0

    local primary="${PASS_RATES%.jsonl}.shard-0-of-2.jsonl"
    python3 experiments/tools/difficulty_filter/reconcile_pass_rate_fragments.py \
        --primary "${DATASET_DIR}/${primary#/data/}" \
        --fragments "${DATASET_DIR}/${TAIL_OUTPUT#/data/}" \
        --mark-complete
}

submit_shard() {
    local shard="$1"
    local start_index="$2"
    local end_index="$3"
    local output="${PASS_RATES%.jsonl}.shard-${shard}-of-2.jsonl"
    local audit="${AUDIT_OUTPUT%.jsonl}.shard-${shard}-of-2.jsonl"

    sbatch --parsable \
        -A "${ACCOUNT}" -p batch --qos=interactive --time=04:00:00 \
        --job-name="pr${shard}-qwen3-4b-step4000" \
        --export="$(filter_exports "${output}" "START_INDEX=${start_index},END_INDEX=${end_index},DUMP_RESPONSES=${audit},DUMP_LIMIT=8")" \
        experiments/tools/difficulty_filter/run_measure.sbatch
}

submit_tail() {
    local audit="${AUDIT_OUTPUT%.jsonl}.shard-0-tail-${TAIL_START_INDEX}-of-${HALF_PROMPTS}.jsonl"

    sbatch --parsable \
        -A "${ACCOUNT}" -p batch --qos=interactive --time=04:00:00 \
        --job-name="pr0tail-qwen3-4b-step4000" \
        --export="$(filter_exports "${TAIL_OUTPUT}" "START_INDEX=${TAIL_START_INDEX},END_INDEX=${HALF_PROMPTS},DUMP_RESPONSES=${audit},DUMP_LIMIT=8")" \
        experiments/tools/difficulty_filter/run_measure.sbatch
}

reconcile_tail

if shard_complete 0 && shard_complete 1; then
    finalize_job="$(sbatch --parsable -A "${ACCOUNT}" \
        experiments/tools/difficulty_filter/dapo-math/finalize_filter.sbatch)"
    echo "both shards complete after round ${FILTER_ROUND}; finalize=${finalize_job}"
    exit 0
fi

if (( FILTER_ROUND >= FILTER_MAX_ROUNDS )); then
    echo "filtering incomplete after ${FILTER_MAX_ROUNDS} rounds; refusing an unbounded retry loop" >&2
    exit 1
fi

next_round=$(( FILTER_ROUND + 1 ))
jobs=()
if ! shard_complete 0; then
    jobs+=("$(submit_shard 0 0 "${HALF_PROMPTS}")")
fi
if ! shard_complete 1; then
    jobs+=("$(submit_shard 1 "${HALF_PROMPTS}" "${TOTAL_PROMPTS}")")
fi
if [[ -n "${TAIL_START_INDEX}" ]] && ! tail_complete; then
    jobs+=("$(submit_tail)")
fi

dependency="$(IFS=:; echo "${jobs[*]}")"
continuation_job="$(sbatch --parsable -A "${ACCOUNT}" \
    --dependency="afterany:${dependency}" \
    --export="ALL,FILTER_ROUND=${next_round},FILTER_MAX_ROUNDS=${FILTER_MAX_ROUNDS},TAIL_START_INDEX=${TAIL_START_INDEX},HF_CKPT_DIR=${HF_CKPT_DIR}" \
    experiments/tools/difficulty_filter/dapo-math/continue_filter.sbatch)"
echo "round ${next_round}: shards=${jobs[*]} continuation=${continuation_job}"
