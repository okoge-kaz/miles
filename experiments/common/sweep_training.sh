#!/bin/bash
# Expand a study's staleness/node-ratio grid; preview unless --submit is given.
set -euo pipefail
recipe="${1:?usage: sweep_training.sh <project>[/<experiment>] [--submit] [--resume-chain]}"
shift
submit=0 resume=0 clean=0
while (( $# > 0 )); do
    case "$1" in
        --submit) submit=1 ;;
        --resume-chain) resume=1 ;;
        --clean-checkpoint) clean=1 ;;
        --help|-h)
            echo "usage: sweep.sh [--submit] [--resume-chain] [--clean-checkpoint]"
            echo "overrides: STALENESS_LEVELS, RATIOS, TOTAL_NODES, CHAIN_JOBS, WALL, RUN_NAMESPACE"
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done
if (( resume == 1 )); then
    : "${RUN_NAMESPACE:?--resume-chain requires the original RUN_NAMESPACE}"
    (( clean == 0 )) || { echo "resume cannot clean checkpoints" >&2; exit 2; }
    : "${CHAIN_JOBS:=9}"
fi
: "${RUN_NAMESPACE:=$(basename -- "${recipe}")-$(date +%Y%m%d-%H%M%S)-p$$}"
: "${TOTAL_NODES:=8}"
: "${CHAIN_JOBS:=2}"
: "${WALL:=24:00:00}"
[[ "${RUN_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && "${TOTAL_NODES}" =~ ^[1-9][0-9]*$ && "${CHAIN_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid namespace/node/chain setting" >&2; exit 2
}
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "project/experiment=${recipe} namespace=${RUN_NAMESPACE} nodes=${TOTAL_NODES} chain_jobs=${CHAIN_JOBS}"
for staleness in ${STALENESS_LEVELS:?}; do
    [[ "${staleness}" =~ ^(none|0|[1-9][0-9]*)$ ]] || { echo "invalid staleness: ${staleness}" >&2; exit 2; }
    for ratio in ${RATIOS:?}; do
        [[ "${ratio}" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] || { echo "invalid ratio: ${ratio}" >&2; exit 2; }
        train_nodes="${BASH_REMATCH[1]}"
        rollout_nodes="${BASH_REMATCH[2]}"
        (( train_nodes + rollout_nodes == TOTAL_NODES )) || { echo "ratio ${ratio} must sum to TOTAL_NODES" >&2; exit 2; }
        export ACTOR_NUM_NODES="${train_nodes}" ROLLOUT_NUM_GPUS="$(( rollout_nodes * ${GPUS_PER_NODE:-8} ))"
        export MAX_WEIGHT_STALENESS="${staleness}"
        run_name="s${staleness}-t${train_nodes}r${rollout_nodes}-${RUN_NAMESPACE}"
        previous_job=""
        for (( allocation = 0; allocation < CHAIN_JOBS; allocation++ )); do
            export CLEAN_CHECKPOINT=0
            (( allocation != 0 || clean == 0 )) || export CLEAN_CHECKPOINT=1
            options=(--nodes="${TOTAL_NODES}" --time="${WALL}")
            if (( submit == 0 )); then
                options+=(--test-only)
                (( allocation == 0 )) || previous_job=PREVIOUS_JOB_ID
            fi
            [[ -z "${previous_job}" ]] || options+=(--dependency="afterany:${previous_job}")
            result="$(bash "${REPO_ROOT}/experiments/submit_training.sh" "${recipe}" "${run_name}" "${options[@]}")"
            printf '%s\n' "${result}"
            (( submit == 0 )) || previous_job="${result}"
        done
    done
done
