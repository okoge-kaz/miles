#!/bin/bash
#PBS -N gate-pr-qwen3-4b-step4000
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:15:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
cd "${REPO_ROOT}"

: "${MAIN_FILTER_JOB_ID:?export MAIN_FILTER_JOB_ID=<job id>}"

MODEL_NAME=Qwen3-4B-Base-LR2e-5-Step4000
MAIN_OUTPUT="${DATASET_DIR}/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.shard-0-of-2.jsonl"
TAIL_OUTPUT="${DATASET_DIR}/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.shard-0-tail-8000-of-8699.jsonl"

while true; do
    main_state="$(
        qstat -f -F json "${MAIN_FILTER_JOB_ID}" 2>/dev/null \
            | jq -r '.Jobs[]?.job_state // empty' || true
    )"
    if [[ -z "${main_state}" ]]; then
        echo "main job ${MAIN_FILTER_JOB_ID} already ended; gate exits"
        exit 0
    fi

    if [[ -f "${TAIL_OUTPUT}.complete" ]]; then
        main_coverage="$(jq -s '[.[].index | select(. >= 0 and . < 8000)] | unique | length' "${MAIN_OUTPUT}" 2>/dev/null || true)"
        tail_coverage="$(jq -s '[.[].index | select(. >= 8000 and . < 8699)] | unique | length' "${TAIL_OUTPUT}" 2>/dev/null || true)"
        if [[ "${main_coverage}" == 8000 && "${tail_coverage}" == 699 ]]; then
            echo "exact split coverage ready: main=${main_coverage} tail=${tail_coverage}; deleting ${MAIN_FILTER_JOB_ID}"
            qdel "${MAIN_FILTER_JOB_ID}"
            exit 0
        fi
    fi

    sleep 30
done
