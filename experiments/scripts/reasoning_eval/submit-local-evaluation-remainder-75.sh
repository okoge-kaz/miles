#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MANIFEST="${MANIFEST:-${SCRIPT_DIR}/manifests/kfujii-local-75-20260908.tsv}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-75}"
export TRAINING_ROOT="${TRAINING_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training}"
export RESULT_STUDY_BASE="${RESULT_STUDY_BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/evaluations/reasoning_eval/staleness-ratio-sweep}"
export GRID_STUDY_BASE="${GRID_STUDY_BASE:-${RESULT_STUDY_BASE}}"
export ROUTES="${ROUTES:-batch_short=02:00:00 interactive=04:00:00}"

exec "${SCRIPT_DIR}/submit-evaluation-manifest.sh" "$@"
