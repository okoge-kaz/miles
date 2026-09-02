#!/bin/bash
# Compare staleness-aware truncation feedback at high max-weight-staleness
# bounds using the established 16K, trainer:rollout 1:7 protocol.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SWEEP_PATH="${REPO_ROOT}/experiments/staleness_ratio_sweep.sh"

usage() {
    cat <<'EOF'
usage: experiments/zero_reward_truncation_staleness_aware_loss_sweep.sh [--submit]
                                                                         [--resume-chain]
                                                                         [--clean-checkpoint]

Without --submit, print the exact five-arm grid. This launcher fixes:

  max weight staleness:          12, 16, 20, 24, 28
  trainer:rollout node ratio:    1:7
  response-only ceiling:         16384 tokens
  total prompt+response limit:   32768 tokens
  completed-group buffer:        6000 groups
  async in-flight samples:       4096
  zero reward on truncated:      on
  zero loss on truncated:        off
  staleness-aware loss:          on
  safe training staleness:       4
  post-TIS objective diagnostics: on

The diagnostics report absolute post-TIS policy-gradient objective mass before
and after staleness-aware scaling. The logging flag is opt-in and remains off
in the base recipe. To launch the unscaled S=12 control from the existing
launcher, use:

  experiments/staleness_ratio_sweep.sh --s12-t1r7-baseline

Useful environment overrides: CHAIN_JOBS, PARTITION, WALL, and RUN_NAMESPACE.
With --resume-chain, RUN_NAMESPACE must name the existing study.
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

# Pin runtime source to the checkout containing this launcher. A clone made for
# this experiment therefore mounts itself and cannot accidentally run another
# checkout through an inherited MILES_REPO value.
export MILES_REPO="${REPO_ROOT}"
export TOTAL_NODES=8
export STALENESS_LEVELS="12 16 20 24 28"
export RATIOS="1:7"
export MAX_RESPONSE_LEN=16384
export ROLLOUT_MAX_CONTEXT_LEN=32768
export CONTEXT_PARALLEL_SIZE=1
export MAX_TOKENS_PER_GPU=32768
export TRAINING_BUFFER_QUEUE_SIZE=6000
export ASYNC_MAX_CONCURRENT_SAMPLES=4096
export ZERO_REWARD_ON_TRUNCATED=1
export ZERO_LOSS_ON_TRUNCATED=0
export IS_CORRECTION=tis
export TIS_CLIP=2.0
export TIS_CLIP_LOW=0
export RATIO_DENOMINATOR=actor
export USE_STALENESS_AWARE_LOSS=1
export SAFE_TRAINING_STALENESS=4
export LOG_STALENESS_AWARE_LOSS_DETAILS=1
if [[ ! -v RUN_NAMESPACE ]]; then
    export RUN_NAMESPACE="staleness-aware-loss-safe4-s12-16-20-24-28-t1r7-$(date +%Y%m%d-%H%M%S)-p$$"
fi

exec "${SWEEP_PATH}" "${FORWARD_ARGS[@]}"
