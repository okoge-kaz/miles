#!/bin/bash
# Run three async zero-reward-on-truncation arms whose authoritative generation
# limit is 32K total tokens (prompt + response), at max weight staleness 16/20/24.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SWEEP_PATH="${REPO_ROOT}/experiments/staleness_ratio_sweep.sh"

usage() {
    cat <<'EOF'
usage: experiments/zero_reward_truncation_total32k_staleness_sweep.sh [--submit]
                                                                      [--clean-checkpoint]

Without --submit, print the exact three-arm grid. The launcher fixes:

  max weight staleness:        16, 20, 24
  trainer:rollout node ratio:  1:7
  total prompt+response limit: 32768 tokens
  response-only ceiling:       32768 tokens
  context parallel size:       1
  trainer token budget/GPU:    32768 tokens
  completed-group buffer:      6000 groups
  async in-flight samples:     4096
  zero reward on truncated:    on
  zero loss on truncated:      off

The actual max_new_tokens for a prompt is min(32768, 32768 - prompt_tokens),
so samples without EOS at the total 32K boundary are marked truncated and
receive reward 0.

Useful environment overrides: CHAIN_JOBS, PARTITION, WALL, and RUN_NAMESPACE.
EOF
}

declare -a FORWARD_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --submit|--clean-checkpoint)
            FORWARD_ARGS+=("$1")
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
export STALENESS_LEVELS="16 20 24"
export RATIOS="1:7"
export MAX_RESPONSE_LEN=32768
export ROLLOUT_MAX_CONTEXT_LEN=32768
export CONTEXT_PARALLEL_SIZE=1
export MAX_TOKENS_PER_GPU=32768
export TRAINING_BUFFER_QUEUE_SIZE=6000
export ASYNC_MAX_CONCURRENT_SAMPLES=4096
export ZERO_REWARD_ON_TRUNCATED=1
export ZERO_LOSS_ON_TRUNCATED=0
if [[ ! -v RUN_NAMESPACE ]]; then
    export RUN_NAMESPACE="zero-reward-trunc-total32k-s16-20-24-t1r7-$(date +%Y%m%d-%H%M%S)-p$$"
fi

exec "${SWEEP_PATH}" --total-length-32k "${FORWARD_ARGS[@]}"
