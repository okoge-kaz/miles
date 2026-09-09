#!/bin/bash
# Compare no truncation treatment with zero loss on truncated responses while
# collecting the policy_lag schema from fresh, treatment-isolated checkpoints.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SWEEP_PATH="${REPO_ROOT}/experiments/staleness_ratio_sweep.sh"

usage() {
    cat <<'EOF'
usage: experiments/policy_lag_truncation_treatment_s8_12_16_sweep.sh [--submit]
                                                                      [--resume-chain]
                                                                      [--clean-checkpoint]
                                                                      [--treatment NAME]
                                                                      [--point M:T:R ...]

Without --submit, print both three-arm grids. NAME is either no-treatment,
zero-loss, or both (the default). --point selects S=8, 12, or 16 at T:R=1:7.
This launcher fixes:

  optimizer updates:              300
  max weight staleness:           8, 12, 16
  trainer:rollout node ratio:     1:7
  response-only ceiling:          16384 tokens
  total prompt+response limit:    32768 tokens
  completed-group buffer:         6000 groups
  async in-flight samples:        4096
  importance-sampling correction: token TIS clipped to [0, 2]
  policy_lag diagnostics:         always on

The two treatments use different derived namespaces and therefore different
checkpoint directories:

  ${RUN_NAMESPACE}-no-treatment
  ${RUN_NAMESPACE}-zero-loss

The default RUN_NAMESPACE includes policy-lag-v1, a timestamp, and the launcher
PID, so it cannot overlap the existing truncation sweeps. To resume, set
RUN_NAMESPACE to the original shared root (without either treatment suffix),
then add --resume-chain. CHAIN_JOBS defaults to nine new allocations on resume.
Existing checkpoints are never removed unless --clean-checkpoint is explicit.

Useful environment overrides: CHAIN_JOBS, PARTITION, WALL, and RUN_NAMESPACE.
EOF
}

declare -a FORWARD_ARGS=()
TREATMENT=both
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
        --treatment)
            [[ $# -ge 2 ]] || { echo "--treatment needs a name" >&2; exit 2; }
            [[ "$2" =~ ^(both|no-treatment|zero-loss)$ ]] || {
                echo "--treatment must be both, no-treatment, or zero-loss" >&2
                exit 2
            }
            TREATMENT="$2"
            shift 2
            ;;
        --point)
            [[ $# -ge 2 ]] || { echo "--point needs M:T:R" >&2; exit 2; }
            [[ "$2" =~ ^(8|12|16):1:7$ ]] || {
                echo "this launcher accepts only S=8/12/16 with T:R=1:7" >&2
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
    echo "--resume-chain requires the original shared RUN_NAMESPACE root" >&2
    exit 2
fi
if [[ ! -v RUN_NAMESPACE ]]; then
    RUN_NAMESPACE="policy-lag-v1-s8-12-16-t1r7-step300-$(date +%Y%m%d-%H%M%S)-p$$"
fi
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "RUN_NAMESPACE contains unsupported characters: ${RUN_NAMESPACE}" >&2
    exit 2
}

export MILES_REPO="${REPO_ROOT}"
export TOTAL_NODES=8
export STALENESS_LEVELS="8 12 16"
export RATIOS="1:7"
export NUM_ROLLOUT=300
export MAX_RESPONSE_LEN=16384
export ROLLOUT_MAX_CONTEXT_LEN=32768
export CONTEXT_PARALLEL_SIZE=1
export MAX_TOKENS_PER_GPU=32768
export TRAINING_BUFFER_QUEUE_SIZE=6000
export ASYNC_MAX_CONCURRENT_SAMPLES=4096
export IS_CORRECTION=tis
export TIS_CLIP=2.0
export TIS_CLIP_LOW=0
export RATIO_DENOMINATOR=actor
export USE_STALENESS_AWARE_LOSS=0
export LOG_STALENESS_AWARE_LOSS_DETAILS=0
export LOG_POLICY_LAG_METRICS=1
export SAMPLE_STALENESS_MAX_BIN=40

run_treatment() {
    local treatment="$1"
    local treatment_namespace="${RUN_NAMESPACE}-${treatment}"
    printf '\n=== policy_lag treatment: %s; namespace: %s ===\n' \
        "${treatment}" "${treatment_namespace}"

    if [[ "${treatment}" == no-treatment ]]; then
        RUN_NAMESPACE="${treatment_namespace}" \
            ZERO_REWARD_ON_TRUNCATED=0 \
            ZERO_LOSS_ON_TRUNCATED=0 \
            "${SWEEP_PATH}" --disable-zero-reward-on-truncated "${FORWARD_ARGS[@]}"
    else
        RUN_NAMESPACE="${treatment_namespace}" \
            ZERO_REWARD_ON_TRUNCATED=0 \
            ZERO_LOSS_ON_TRUNCATED=1 \
            "${SWEEP_PATH}" --zero-loss-on-truncated "${FORWARD_ARGS[@]}"
    fi
}

if [[ "${TREATMENT}" == both || "${TREATMENT}" == no-treatment ]]; then
    run_treatment no-treatment
fi
if [[ "${TREATMENT}" == both || "${TREATMENT}" == zero-loss ]]; then
    run_treatment zero-loss
fi
