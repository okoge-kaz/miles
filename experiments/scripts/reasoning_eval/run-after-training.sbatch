#!/bin/bash
#SBATCH --job-name=q3-4b-post-rl-aime-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/%x-%j.log

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
: "${TRAINING_HF_ROOT:?set TRAINING_HF_ROOT to the training run hf directory}"
: "${RESULT_ROOT:?set RESULT_ROOT}"

VALIDATE_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/validate_checkpoint.py"
TRAINING_HF_ROOT="$(realpath -- "${TRAINING_HF_ROOT}")"
CHECKPOINT_PATH="$(python3 "${VALIDATE_TOOL}" --latest-under "${TRAINING_HF_ROOT}")"

export CHECKPOINT_PATH RESULT_ROOT
checkpoint_index="${CHECKPOINT_PATH##*/}"
TRAINING_STEP=$((10#${checkpoint_index} + 1))
export TRAINING_STEP
mkdir -p "${RESULT_ROOT}"
selected_checkpoint_file="${RESULT_ROOT}/selected-checkpoint.txt"
selected_checkpoint_partial="${selected_checkpoint_file}.partial-${SLURM_JOB_ID}"
printf '%s\n' "${CHECKPOINT_PATH}" > "${selected_checkpoint_partial}"
if [[ -f "${selected_checkpoint_file}" ]]; then
    cmp --silent "${selected_checkpoint_partial}" "${selected_checkpoint_file}" || {
        rm -f -- "${selected_checkpoint_partial}"
        echo "RESULT_ROOT already records a different selected checkpoint" >&2
        exit 2
    }
    rm -f -- "${selected_checkpoint_partial}"
else
    mv -- "${selected_checkpoint_partial}" "${selected_checkpoint_file}"
fi
exec bash "${REPO_ROOT}/experiments/scripts/reasoning_eval/run-evaluation.sbatch"
