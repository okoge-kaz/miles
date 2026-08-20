#!/bin/bash
# Submit a training run with its log under experiments/outputs/training/<RUN_NAME>/.
#
#   experiments/submit_training.sh <task>/<mode>/<dataset>/<model> <run-name> [sbatch args...]
#
#   recipe: <task>/<mode>/<dataset>/<model>, e.g. math/sync/dapo-math-p10-90/qwen3-4b
#   (an unknown recipe lists what actually exists on disk)
#
# Example — a real 24k-response colocated run, resumable across three 4 h jobs:
#   experiments/submit_training.sh math/sync/dapo-math-p10-90/qwen3-4b real-math-24k \
#       -p batch --time=04:00:00 --export=ALL,MAX_RESPONSE_LEN=24576,SAVE_INTERVAL=5
#
# RUN_NAME is exported for the recipe (it names the checkpoint directory) and
# also used verbatim for the log directory, so a resumed run appends its logs
# next to the original.

set -euo pipefail

RECIPE="${1:?usage: submit_training.sh <task>/<mode>/<dataset>/<model> <run-name> [sbatch args...]}"
RUN_NAME="${2:?usage: submit_training.sh <task>/<mode>/<dataset>/<model> <run-name> [sbatch args...]}"
shift 2

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "experiments/scripts/${RECIPE}/run.sbatch" ]]; then
    echo "no such recipe: ${RECIPE}"
    echo "available:"
    find experiments/scripts -name run.sbatch -printf '  %h\n' | sed 's|^  experiments/scripts/|  |' | sort
    exit 1
fi

# Mirror the recipe path, like the checkpoints do: <task>/<dataset>/<model>/.
# Sync/async are placement axes, not separate task families.
TASK_FAMILY="${RECIPE%%/*}"
DATASET_AND_MODEL="${RECIPE#*/*/}"
LOG_DIR="experiments/outputs/training/${TASK_FAMILY}/$(dirname "${DATASET_AND_MODEL}")/$(basename "${DATASET_AND_MODEL}")"
mkdir -p "${LOG_DIR}"

jid=$(sbatch --parsable \
      -A "${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}" \
      -p "${GPU_PARTITION:-batch}" \
      --job-name="${RUN_NAME}" \
      --output="${LOG_DIR}/%x-%j.log" \
      --export="ALL,RUN_NAME=${RUN_NAME}" \
      "$@" \
      "experiments/scripts/${RECIPE}/run.sbatch")

echo "${jid}  ${RECIPE}  ${RUN_NAME}  -> ${LOG_DIR}/${RUN_NAME}-${jid}.log"
