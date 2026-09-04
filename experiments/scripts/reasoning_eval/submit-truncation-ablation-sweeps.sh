#!/bin/bash
# Evaluate the five zero-reward-off and six zero-loss-on-truncation training
# arms launched for the 2026-08-27 truncation ablation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/submit-staleness-sweep.sh"
DEFAULT_HISO_TRAINING_ROOT="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training"

REWARD_OFF_RUN_NAMESPACE="${REWARD_OFF_RUN_NAMESPACE:-hiso-reward-off-trunc-coloc-s8-16-r12-20260827-v1}"
ZERO_LOSS_RUN_NAMESPACE="${ZERO_LOSS_RUN_NAMESPACE:-hiso-zero-loss-trunc-s8-16-20-r12-20260827-v1}"
TRAINING_ROOT="${TRAINING_ROOT:-${DEFAULT_HISO_TRAINING_ROOT}}"
COHORT=all
declare -a FORWARD_ARGS=()

usage() {
    cat <<'EOF'
usage: experiments/scripts/reasoning_eval/submit-truncation-ablation-sweeps.sh [options]

Scan the 11 truncation-ablation arms and report checkpoints that are complete,
active, pending evaluation, missing, incomplete, or unreadable. Without
--submit this is read-only.

Options:
  --submit                 Submit pending checkpoint evaluations.
  --max-submissions N      Maximum submissions per selected cohort (0 = all).
  --cohort NAME            all, reward-off, or zero-loss (default: all).
  --help                   Show this message.

The reward-off cohort contains one colocated arm and staleness 8/16 crossed
with trainer:rollout ratios 1:7/2:6. The zero-loss cohort contains staleness
8/16/20 crossed with the same two ratios. TRAINING_ROOT, PARTITION, QOS, WALL,
and the two *_RUN_NAMESPACE variables may be overridden through the environment.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --submit)
            FORWARD_ARGS+=(--submit)
            shift
            ;;
        --max-submissions)
            [[ $# -ge 2 ]] || { echo "--max-submissions needs a value" >&2; exit 2; }
            FORWARD_ARGS+=(--max-submissions "$2")
            shift 2
            ;;
        --cohort)
            [[ $# -ge 2 ]] || { echo "--cohort needs a value" >&2; exit 2; }
            COHORT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${COHORT}" in
    all|reward-off|zero-loss) ;;
    *) echo "--cohort must be all, reward-off, or zero-loss" >&2; exit 2 ;;
esac

run_reward_off() {
    TRAINING_ROOT="${TRAINING_ROOT}" \
    STALENESS_LEVELS="8 16" \
    RATIOS="1:7 2:6" \
    INCLUDE_COLOCATED=1 \
    TRAINING_BUFFER_QUEUE_SIZE=1000 \
    ASYNC_MAX_CONCURRENT_SAMPLES= \
    ASYNC_RUN_SUFFIX=-zero-reward-trunc-off-rb-inflight \
    COLOCATED_RUN_SUFFIX=-zero-reward-trunc-off \
        "${BASE_LAUNCHER}" --namespace "${REWARD_OFF_RUN_NAMESPACE}" "${FORWARD_ARGS[@]}"
}

run_zero_loss() {
    TRAINING_ROOT="${TRAINING_ROOT}" \
    STALENESS_LEVELS="8 16 20" \
    RATIOS="1:7 2:6" \
    INCLUDE_COLOCATED=0 \
    TRAINING_BUFFER_QUEUE_SIZE=1000 \
    ASYNC_MAX_CONCURRENT_SAMPLES= \
    ASYNC_RUN_SUFFIX=-zero-loss-trunc-rb-inflight \
    COLOCATED_RUN_SUFFIX=-zero-trunc \
        "${BASE_LAUNCHER}" --namespace "${ZERO_LOSS_RUN_NAMESPACE}" "${FORWARD_ARGS[@]}"
}

if [[ "${COHORT}" == all || "${COHORT}" == reward-off ]]; then
    run_reward_off
fi
if [[ "${COHORT}" == all || "${COHORT}" == zero-loss ]]; then
    run_zero_loss
fi
