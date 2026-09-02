#!/bin/bash
#PBS -N merge-pr-qwen3-4b-step4000
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

MODEL_NAME=Qwen3-4B-Base-LR2e-5-Step4000
DATASET_NAME=dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000
PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
PASS_RATE_BASE="/data/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate"
SHARD_0="${PASS_RATE_BASE}.shard-0-of-2.jsonl"
SHARD_1="${PASS_RATE_BASE}.shard-1-of-2.jsonl"
MERGED_OUTPUT="${PASS_RATE_BASE}.jsonl"
FILTERED_OUTPUT="/data/${DATASET_NAME}/${DATASET_NAME}.jsonl"

for shard in "${SHARD_0}" "${SHARD_1}"; do
    host_path="${DATASET_DIR}/${shard#/data/}"
    [[ -f "${host_path}.complete" ]] || {
        echo "incomplete shard: ${host_path}" >&2
        exit 1
    }
done
unset shard host_path

miles_srun \
    --cpus-per-task="${PBS_NCPUS:-8}" \
    --mem=0 \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc "
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/Megatron-LM:/root/miles

python3 -m experiments.tools.difficulty_filter.merge_pass_rate_shards \
    --inputs ${SHARD_0} ${SHARD_1} \
    --output ${MERGED_OUTPUT} \
    --prompt-data ${PROMPT_DATA}

python3 -m experiments.tools.difficulty_filter.apply_filter \
    --prompt-data ${PROMPT_DATA} \
    --pass-rates ${MERGED_OUTPUT} \
    --output ${FILTERED_OUTPUT} \
    --pass-rate-min 0.1 \
    --pass-rate-max 0.9

touch ${MERGED_OUTPUT}.measured ${MERGED_OUTPUT}.complete
"

if [[ "${SUBMIT_REPLAY_BUFFER_VALIDATION:-1}" != "0" ]]; then
    [[ -n "${MILES_JOB_ID:-}" ]] || {
        echo "SUBMIT_REPLAY_BUFFER_VALIDATION requires a PBS job" >&2
        exit 1
    }
    "${REPO_ROOT}/experiments/tools/replay_buffer_validation/dapo-math/submit_replay_buffer_validation.sh" \
        --submit \
        --afterok "${MILES_JOB_ID}"
fi
