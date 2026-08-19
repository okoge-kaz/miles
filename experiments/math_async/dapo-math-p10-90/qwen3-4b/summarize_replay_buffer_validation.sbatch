#!/bin/bash
#SBATCH --job-name=rbv-summary
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=experiments/outputs/replay_buffer_validation/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)}"
cd "${REPO_ROOT}"

: "${VALIDATION_MANIFEST:?}"
: "${VALIDATION_SUMMARY:?}"

python3 experiments/math_async/dapo-math-p10-90/qwen3-4b/analyze_replay_buffer_validation.py \
    "${VALIDATION_MANIFEST}" \
    --output "${VALIDATION_SUMMARY}"
