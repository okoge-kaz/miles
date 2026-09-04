#!/bin/bash
#SBATCH --job-name=taurb-summary
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=experiments/outputs/replay_buffer_validation/tau2/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)}"
cd "${REPO_ROOT}"

: "${VALIDATION_MANIFEST:?}"
: "${VALIDATION_SUMMARY:?}"

python3 experiments/tools/replay_buffer_validation/tau2/analyze.py \
    "${VALIDATION_MANIFEST}" \
    --output "${VALIDATION_SUMMARY}"
