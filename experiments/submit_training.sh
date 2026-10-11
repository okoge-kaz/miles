#!/bin/bash

set -euo pipefail

# Read the recipe and run name; forward the remaining arguments as PBS options.
recipe="${1:?usage: submit_training.sh <project>[/<experiment>] <run-name> [PBS options]}"
export RUN_NAME="${2:?usage: submit_training.sh <project>[/<experiment>] <run-name> [PBS options]}"
shift 2
# Restrict names to the expected directory layout and safe path components.
[[ "${recipe}" =~ ^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)?$ && "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "recipe must be project or project/experiment and run-name must be path-safe" >&2; exit 2
}

# Resolve the checkout from this file so submission works from any directory.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export MILES_REPO="${REPO_ROOT}"

# Load PBS helpers and the job-variable allowlist; reject unsupported legacy settings.
source "${REPO_ROOT}/experiments/common/pbs.sh"
source "${REPO_ROOT}/experiments/common/job_env.sh"
miles_check_pending_features

# Select the recipe's scheduler entry point and prepare its log directory.
recipe_path="${REPO_ROOT}/experiments/scripts/${recipe}/run.sh"
[[ -f "${recipe_path}" ]] || { echo "unknown experiment: ${recipe}" >&2; exit 2; }
log_dir="${REPO_ROOT}/experiments/outputs/train/${recipe}"
mkdir -p "${log_dir}"

# Build a distinct log filename from the model, run, timestamp, and submitting PID.
model_prefix="${MODEL_PRESET:-qwen3-4b}"
[[ "${model_prefix}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid MODEL_PRESET" >&2; exit 2; }
model_prefix="${model_prefix//[-.]/_}"
log_file="${log_dir}/${model_prefix}_${RUN_NAME}_$(date +%Y%m%d-%H%M%S)-p$$.ou"

# Submit with the GPU profile and allowlisted variables. --test-only previews qsub;
# --parsable returns the PBS job ID for callers that build dependent job chains.
pbs_submit --parsable --profile=gpu --job-kind=training --chdir="${REPO_ROOT}" \
    --job-name="${RUN_NAME}" --output="${log_file}" \
    --export="$(miles_job_export_spec)" "$@" "${recipe_path}"
