#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"

export STALENESS_LEVELS="${STALENESS_LEVELS:-12 16 20 24 28}"
export RATIOS="${RATIOS:-1:7}"
export TOTAL_NODES="${TOTAL_NODES:-8}"

exec bash "${REPO_ROOT}/experiments/common/sweep_training.sh" \
    staleness_math/zero_reward_truncation "$@"
