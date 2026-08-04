#!/bin/bash
# Submit a training run with its log under experiments/outputs/training/<RUN_NAME>/.
#
#   experiments/submit_training.sh <task>/<dataset>/<model> <run-name> [sbatch args...]
#
#   recipe: <task>/<dataset>/<model>, e.g. math_sync/dapo-math/qwen3-8b
#   (an unknown recipe lists what actually exists on disk)
#
# Example — a real 24k-response colocated run, resumable across three 4 h jobs:
#   experiments/submit_training.sh math_sync/dapo-math/qwen3-4b real-math-24k \
#       -p batch --time=04:00:00 --export=ALL,MAX_RESPONSE_LEN=24576,SAVE_INTERVAL=5
#
# RUN_NAME is exported for the recipe (it names the checkpoint directory) and
# also used verbatim for the log directory, so a resumed run appends its logs
# next to the original.

set -euo pipefail

RECIPE="${1:?usage: submit_training.sh <task>/<dataset>/<model> <run-name> [sbatch args...]}"
RUN_NAME="${2:?usage: submit_training.sh <task>/<dataset>/<model> <run-name> [sbatch args...]}"
shift 2

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "experiments/${RECIPE}/run.sbatch" ]]; then
    echo "no such recipe: ${RECIPE}"
    echo "available:"
    find experiments -mindepth 4 -maxdepth 4 -name run.sbatch -printf '  %h\n' | sed 's|^  experiments/|  |' | sort
    exit 1
fi

LOG_DIR="experiments/outputs/training/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

jid=$(sbatch --parsable \
      -A "${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}" \
      --job-name="${RUN_NAME}" \
      --output="${LOG_DIR}/%x-%j.log" \
      --export="ALL,RUN_NAME=${RUN_NAME}" \
      "$@" \
      "experiments/${RECIPE}/run.sbatch")

echo "${jid}  ${RECIPE}  ${RUN_NAME}  -> ${LOG_DIR}/${RUN_NAME}-${jid}.log"
