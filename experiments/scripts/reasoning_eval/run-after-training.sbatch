#!/bin/bash
#PBS -N q3-4b-post-rl-aime-eval
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=04:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="$(realpath -- "${PBS_O_WORKDIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"
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
selected_checkpoint_partial="${selected_checkpoint_file}.partial-${MILES_JOB_ID}"
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
