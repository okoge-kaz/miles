#!/bin/bash

# Run this from Hiso's checkout of the same Miles revision. The dry run verifies
# all 400 checkpoint paths. Results are written below Hiso's own evaluation root.
# For automatic refills under a per-user submit cap, submit
# refill-hiso-handoff-400.sbatch instead of invoking this with --submit directly.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MANIFEST="${MANIFEST:-${SCRIPT_DIR}/manifests/hiso-handoff-400-20260908.tsv}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-400}"
export TRAINING_ROOT="${TRAINING_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training}"
export RESULT_STUDY_BASE="${RESULT_STUDY_BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/evaluations/reasoning_eval/staleness-ratio-sweep}"
export GRID_STUDY_BASE="${GRID_STUDY_BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/evaluations/reasoning_eval/staleness-ratio-sweep}"
export ROUTES="${ROUTES:-batch=04:00:00}"

exec "${SCRIPT_DIR}/submit-evaluation-manifest.sh" "$@"
