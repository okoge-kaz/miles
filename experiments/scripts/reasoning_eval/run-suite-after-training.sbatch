#!/bin/bash
#SBATCH --job-name=q3-4b-post-rl-reasoning-suite
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/%x-%j.log

# Resolve the newest complete periodic Hugging Face export after a training job,
# then run the ordinary held-out evaluator in this allocation. Submit this job
# with an afterany dependency: a four-hour training allocation is expected to
# reach its wall-clock limit, while its earlier atomic exports remain valid.

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
: "${TRAINING_HF_ROOT:?set TRAINING_HF_ROOT to the training run hf directory}"
: "${RESULT_ROOT:?set RESULT_ROOT}"

VALIDATE_TOOL="${REPO_ROOT}/experiments/tools/reasoning_eval/validate_checkpoint.py"
TRAINING_HF_ROOT="$(realpath -- "${TRAINING_HF_ROOT}")"

mapfile -t CANDIDATES < <(
    find "${TRAINING_HF_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' \
        | awk -F '\t' '$1 ~ /^[0-9]+$/ { print }' \
        | sort -t $'\t' -k1,1nr \
        | cut -f2-
)
(( ${#CANDIDATES[@]} > 0 )) || {
    echo "no numeric Hugging Face exports under ${TRAINING_HF_ROOT}" >&2
    exit 1
}

CHECKPOINT_PATH=""
for candidate in "${CANDIDATES[@]}"; do
    if python3 "${VALIDATE_TOOL}" --quiet "${candidate}"; then
        CHECKPOINT_PATH="${candidate}"
        break
    fi
done
[[ -n "${CHECKPOINT_PATH}" ]] || {
    echo "no complete Hugging Face export under ${TRAINING_HF_ROOT}" >&2
    exit 1
}

export CHECKPOINT_PATH RESULT_ROOT
mkdir -p "${RESULT_ROOT}"
printf '%s\n' "${CHECKPOINT_PATH}" > "${RESULT_ROOT}/selected-checkpoint.txt"
exec bash "${REPO_ROOT}/experiments/scripts/reasoning_eval/run-suite.sbatch"
