#!/bin/bash
#PBS -N miles-judge
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
[[ "${MILES_JOB_NUM_NODES}" == 1 ]] || { echo "judge server requires exactly one node" >&2; exit 2; }
export JUDGE_RUN_NAME="${JUDGE_RUN_NAME:-judge-${MILES_JOB_ID}}"
[[ "${JUDGE_RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid JUDGE_RUN_NAME" >&2; exit 2; }
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${MILES_LAUNCH_DRY_RUN:-0}" == 1 ]]; then
    exec bash "${REPO_ROOT}/experiments/scripts/nemo_gym/llm_judge/serve.sh" --dry-run
fi
[[ -n "${PBS_JOBID:-}" ]] || { echo "start the GPU judge inside a PBS allocation" >&2; exit 2; }
[[ -r "${CONTAINER_IMAGE}" ]] || { echo "missing container: ${CONTAINER_IMAGE}" >&2; exit 2; }

miles_container_exec \
    --nv --image "${CONTAINER_IMAGE}" \
    --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles \
    --export=JUDGE_MODEL_PATH,JUDGE_MODEL_REVISION,JUDGE_MODEL_NAME,JUDGE_PORT,JUDGE_TP_SIZE,JUDGE_CONTEXT_LENGTH,JUDGE_MEM_FRACTION,JUDGE_MAX_RUNNING_REQUESTS \
    bash --noprofile --norc /root/miles/experiments/scripts/nemo_gym/llm_judge/serve.sh
