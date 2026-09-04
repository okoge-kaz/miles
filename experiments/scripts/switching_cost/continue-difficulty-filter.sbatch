#!/bin/bash
#SBATCH --job-name=continue-switch-difficulty
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/difficulty/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/experiments/scripts/switching_cost"
cd "${REPO_ROOT}"

export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface}"
source experiments/env.sh
source "${SCRIPT_DIR}/difficulty-filter-config.sh"

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
FILTER_ROUND="${FILTER_ROUND:-1}"
FILTER_MAX_ROUNDS="${FILTER_MAX_ROUNDS:-8}"

# Do not export this coordinator's small CPU allocation limits into GPU jobs.
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_CPUS_PER_TASK

measurement_exports() {
    printf '%s' "ALL,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_VARIANT=${MODEL_VARIANT},MODEL_NAME=${MODEL_NAME},HF_MODEL_NAME=${HF_MODEL_NAME},PROMPT_DATA=${PROMPT_DATA},N_SAMPLES=16,SAMPLES_PER_REQUEST=8,MAX_NEW_TOKENS=16384,MAX_CONTEXT_LENGTH=32768,CONCURRENCY=${CONCURRENCY},MAX_RUNNING_REQUESTS=80,SGLANG_TP_SIZE=${SGLANG_TP_SIZE},SGLANG_DP_SIZE=${SGLANG_DP_SIZE},MEM_FRACTION_STATIC=0.85,RM_TYPE=deepscaler,ZERO_REWARD_ON_TRUNCATED=1,PASS_RATE_MIN=0.1,PASS_RATE_MAX=0.9,FILTERED_OUTPUT=,ANNOTATED_OUTPUT="
}

shard_output() {
    local shard="$1"
    echo "${PASS_RATES%.jsonl}.shard-${shard}-of-2.jsonl"
}

shard_complete() {
    local output
    output=$(shard_output "$1")
    [[ -f "${DATASET_DIR}/${output#/data/}.complete" ]]
}

submit_shard() {
    local shard="$1"
    local start end output audit
    if [[ "${shard}" == 0 ]]; then
        start=0
        end="${HALF_PROMPTS}"
    else
        start="${HALF_PROMPTS}"
        end="${TOTAL_PROMPTS}"
    fi
    output=$(shard_output "${shard}")
    audit="${AUDIT_OUTPUT%.jsonl}.shard-${shard}-of-2.jsonl"

    sbatch --parsable \
        -A "${ACCOUNT}" -p batch --qos=interactive --time=04:00:00 \
        --job-name="df${shard}-${FILTER_JOB_SUFFIX}" \
        --export="$(measurement_exports),OUTPUT=${output},START_INDEX=${start},END_INDEX=${end},DUMP_RESPONSES=${audit},DUMP_LIMIT=8" \
        experiments/tools/difficulty_filter/run_measure.sbatch
}

if shard_complete 0 && shard_complete 1; then
    SHARD_0=$(shard_output 0)
    SHARD_1=$(shard_output 1)
    export SHARD_0 SHARD_1
    srun \
        --cpus-per-task=8 \
        --mem=0 \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        --container-writable \
        --no-container-mount-home \
        --export=ALL \
        bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/Megatron-LM:/root/miles
python3 -m experiments.tools.difficulty_filter.merge_pass_rate_shards \
    --inputs "${SHARD_0}" "${SHARD_1}" \
    --output "${PASS_RATES}" \
    --prompt-data "${PROMPT_DATA}"
python3 -m experiments.tools.difficulty_filter.apply_filter \
    --prompt-data "${PROMPT_DATA}" \
    --pass-rates "${PASS_RATES}" \
    --output "${FILTERED_OUTPUT}" \
    --pass-rate-min 0.1 \
    --pass-rate-max 0.9
touch "${PASS_RATES}.measured" "${PASS_RATES}.complete" "${FILTERED_OUTPUT}.complete"
'
    echo "completed ${MODEL_VARIANT}: ${FILTERED_OUTPUT}"
    exit 0
fi

if (( FILTER_ROUND >= FILTER_MAX_ROUNDS )); then
    echo "${MODEL_VARIANT} filtering incomplete after ${FILTER_MAX_ROUNDS} rounds" >&2
    exit 1
fi

next_round=$(( FILTER_ROUND + 1 ))
jobs=()
for shard in 0 1; do
    if ! shard_complete "${shard}"; then
        jobs+=("$(submit_shard "${shard}")")
    fi
done

dependency=$(IFS=:; echo "${jobs[*]}")
continuation_job=$(sbatch --parsable \
    -A "${ACCOUNT}" -p cpu --qos=cpu-interactive --time=00:20:00 \
    --job-name="dfcont-${FILTER_JOB_SUFFIX}" \
    --dependency="afterany:${dependency}" \
    --export="ALL,HF_CKPT_DIR=${HF_CKPT_DIR},MODEL_VARIANT=${MODEL_VARIANT},FILTER_ROUND=${next_round},FILTER_MAX_ROUNDS=${FILTER_MAX_ROUNDS}" \
    experiments/scripts/switching_cost/continue-difficulty-filter.sbatch)
echo "${MODEL_VARIANT} round ${next_round}: shards=${jobs[*]} continuation=${continuation_job}"
