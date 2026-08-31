#!/bin/bash
# Run the five-arm zero-reward-on-truncation ablation against the staleness
# ratio sweep: one colocated on-policy arm, plus max weight staleness 8 and 16
# crossed with trainer:rollout node ratios 1:7 and 2:6.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SWEEP_PATH="${REPO_ROOT}/experiments/staleness_ratio_sweep.sh"

usage() {
    cat <<'EOF'
usage: experiments/zero_truncation_ablation_sweep.sh [--submit]
                                                     [--resume-chain]
                                                     [--async-only]
                                                     [--clean-checkpoint]

Without --submit, print the exact five-arm grid. This launcher uses the same
recipes and defaults as experiments/staleness_ratio_sweep.sh, except it sets
ZERO_REWARD_ON_TRUNCATED=0 and fixes:

  colocated on-policy
  max weight staleness 8:  trainer:rollout nodes 1:7, 2:6
  max weight staleness 16: trainer:rollout nodes 1:7, 2:6

The shared staleness policy fixes TRAINING_BUFFER_QUEUE_SIZE=6000 for all four
async arms. The colocated arm keeps its async-inapplicable queue value of 1000.
--async-only starts only the four async arms under a fresh namespace; use it
when replacing the queue-mismatched async checkpoints while retaining the
unaffected colocated checkpoint.

Useful environment overrides: CHAIN_JOBS, PARTITION, WALL, and RUN_NAMESPACE.
With --resume-chain, only the four async arms are resumed; the colocated arm is
not submitted. RUN_NAMESPACE must name the existing study and CHAIN_JOBS
defaults to nine new allocations per async arm. Existing checkpoints are
preserved.
EOF
}

declare -a FORWARD_ARGS=()
RESUME_CHAIN=0
ASYNC_ONLY=0
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
        --async-only)
            ASYNC_ONLY=1
            shift
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

export TOTAL_NODES=8
export STALENESS_LEVELS="8 16"
export RATIOS="1:7 2:6"
if (( RESUME_CHAIN == 1 )) && [[ ! -v RUN_NAMESPACE ]]; then
    echo "--resume-chain requires the original RUN_NAMESPACE" >&2
    exit 2
fi
if [[ ! -v RUN_NAMESPACE ]]; then
    export RUN_NAMESPACE="zero-trunc-off-$(date +%Y%m%d-%H%M%S)-p$$"
fi

declare -a SWEEP_ARGS=(--disable-zero-reward-on-truncated)
if (( RESUME_CHAIN == 0 && ASYNC_ONLY == 0 )); then
    SWEEP_ARGS+=(--include-colocated)
fi

exec "${SWEEP_PATH}" "${SWEEP_ARGS[@]}" "${FORWARD_ARGS[@]}"
