#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MANIFEST="${MANIFEST:-${SCRIPT_DIR}/manifests/kfujii-local-75-20260908.tsv}"
export EXPECTED_COUNT="${EXPECTED_COUNT:-75}"
export TRAINING_ROOT="${TRAINING_ROOT:-${TRAIN_CKPT_DIR:-${WS:-/lustre/fsw/portfolios/coreai/users/${USER}}/checkpoints/training}}"
export RESULT_STUDY_BASE="${RESULT_STUDY_BASE:-${WS:-/lustre/fsw/portfolios/coreai/users/${USER}}/evaluations/reasoning_eval/staleness-ratio-sweep}"
export GRID_STUDY_BASE="${GRID_STUDY_BASE:-${RESULT_STUDY_BASE}}"
export ROUTES="${ROUTES:-batch=04:00:00}"

exec "${SCRIPT_DIR}/submit-evaluation-manifest.sh" "$@"
