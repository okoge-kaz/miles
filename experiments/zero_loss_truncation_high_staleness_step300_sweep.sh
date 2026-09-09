#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SWEEP_PATH="${REPO_ROOT}/experiments/staleness_ratio_sweep.sh"

usage() {
    cat <<'EOF'
usage: experiments/zero_loss_truncation_high_staleness_step300_sweep.sh [--submit]
                                                                         [--resume-chain]
                                                                         [--clean-checkpoint]
                                                                         [--point M:T:R ...]

Without --submit, print the exact two-arm grid. --point can select one existing
arm for a partial resume. This launcher fixes:

  optimizer updates:              300
  max weight staleness:           24, 28
  trainer:rollout node ratio:     1:7
  response-only ceiling:          16384 tokens
  total prompt+response limit:    32768 tokens
  completed-group buffer:         6000 groups
  async in-flight samples:        4096
  zero reward on truncated:       off
  zero loss on truncated:         on
  staleness-aware loss:           off
  importance-sampling correction: token TIS clipped to [0, 2]
  policy-lag diagnostics: on (Delta and pre/post loss-sensitivity-weighted)

The two arms isolate whether fully removing the direct policy-gradient loss
from truncated samples extends the stable boundary beyond the completed S=20
run. SAMPLE_STALENESS_MAX_BIN=40 preserves the expanded logging contract.

Useful environment overrides: CHAIN_JOBS, PARTITION, WALL, and RUN_NAMESPACE.
With --resume-chain, RUN_NAMESPACE must name the existing study and CHAIN_JOBS
defaults to nine new allocations per arm. Existing checkpoints are preserved.
EOF
}

declare -a FORWARD_ARGS=()
RESUME_CHAIN=0
while (( $# > 0 )); do
    case "$1" in
        --submit|--clean-checkpoint)
            FORWARD_ARGS+=("$1")
            shift
            ;;
        --resume-chain)
            RESUME_CHAIN=1
            FORWARD_ARGS+=("$1")
            [[ -v CHAIN_JOBS ]] || export CHAIN_JOBS=9
            shift
            ;;
        --point)
            [[ $# -ge 2 ]] || { echo "--point needs M:T:R" >&2; exit 2; }
            [[ "$2" == 24:1:7 || "$2" == 28:1:7 ]] || {
                echo "this launcher accepts only --point 24:1:7 or 28:1:7" >&2
                exit 2
            }
            FORWARD_ARGS+=("$1" "$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if (( RESUME_CHAIN == 1 )) && [[ ! -v RUN_NAMESPACE ]]; then
    echo "--resume-chain requires the original RUN_NAMESPACE" >&2
    exit 2
fi

export MILES_REPO="${REPO_ROOT}"
export TOTAL_NODES=8
export STALENESS_LEVELS="24 28"
export RATIOS="1:7"
export NUM_ROLLOUT=300
export MAX_RESPONSE_LEN=16384
export ROLLOUT_MAX_CONTEXT_LEN=32768
export CONTEXT_PARALLEL_SIZE=1
export MAX_TOKENS_PER_GPU=32768
export TRAINING_BUFFER_QUEUE_SIZE=6000
export ASYNC_MAX_CONCURRENT_SAMPLES=4096
export ZERO_REWARD_ON_TRUNCATED=0
export ZERO_LOSS_ON_TRUNCATED=1
export IS_CORRECTION=tis
export TIS_CLIP=2.0
export TIS_CLIP_LOW=0
export RATIO_DENOMINATOR=actor
export USE_STALENESS_AWARE_LOSS=0
export LOG_STALENESS_AWARE_LOSS_DETAILS=0
export LOG_POLICY_LAG_METRICS=1
export SAMPLE_STALENESS_MAX_BIN=40
if [[ ! -v RUN_NAMESPACE ]]; then
    export RUN_NAMESPACE="zero-loss-trunc-s24-28-t1r7-step300-$(date +%Y%m%d-%H%M%S)-p$$"
fi

exec "${SWEEP_PATH}" --zero-loss-on-truncated "${FORWARD_ARGS[@]}"
