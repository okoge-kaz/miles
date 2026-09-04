#!/bin/bash
#SBATCH --job-name=gate-pr-qwen3-4b-step4000
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:15:00
#SBATCH --output=experiments/outputs/difficulty/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
cd "${REPO_ROOT}"

: "${MAIN_FILTER_JOB_ID:?export MAIN_FILTER_JOB_ID=<job id>}"

MODEL_NAME=Qwen3-4B-Base-LR2e-5-Step4000
MAIN_OUTPUT="${DATASET_DIR}/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.shard-0-of-2.jsonl"
TAIL_OUTPUT="${DATASET_DIR}/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.shard-0-tail-8000-of-8699.jsonl"

while true; do
    main_state="$(squeue -h -j "${MAIN_FILTER_JOB_ID}" -o '%T' || true)"
    if [[ -z "${main_state}" ]]; then
        echo "main job ${MAIN_FILTER_JOB_ID} already ended; gate exits"
        exit 0
    fi

    if [[ -f "${TAIL_OUTPUT}.complete" ]]; then
        main_coverage="$(jq -s '[.[].index | select(. >= 0 and . < 8000)] | unique | length' "${MAIN_OUTPUT}" 2>/dev/null || true)"
        tail_coverage="$(jq -s '[.[].index | select(. >= 8000 and . < 8699)] | unique | length' "${TAIL_OUTPUT}" 2>/dev/null || true)"
        if [[ "${main_coverage}" == 8000 && "${tail_coverage}" == 699 ]]; then
            echo "exact split coverage ready: main=${main_coverage} tail=${tail_coverage}; cancelling ${MAIN_FILTER_JOB_ID}"
            scancel "${MAIN_FILTER_JOB_ID}"
            exit 0
        fi
    fi

    sleep 30
done
