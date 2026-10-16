#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export STALENESS_LEVELS="${STALENESS_LEVELS:-28}"
export RATIOS="${RATIOS:-1:1}"
export TOTAL_NODES="${TOTAL_NODES:-2}"
export CHAIN_JOBS="${CHAIN_JOBS:-1}"

exec bash "${REPO_ROOT}/experiments/common/sweep_training.sh" multi_env "$@"
