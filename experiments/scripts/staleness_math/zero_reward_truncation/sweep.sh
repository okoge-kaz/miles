#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"

for option in "$@"; do
    case "${option}" in
        --help|-h)
            echo "usage: sweep.sh [--submit] [--resume-chain] [--clean-checkpoint]"
            echo "Default: 3 SFT models x 5 staleness levels x 4 node ratios = 60 TIS runs."
            echo "Preview only unless --submit is given. Each run requests 8 nodes for 168 hours."
            echo "Select: MODELS (or MODEL_PRESET), STALENESS_LEVELS, RATIOS. Loss is TIS only."
            echo "Common overrides: LR, GLOBAL_BATCH_SIZE, ROLLOUT_BATCH_SIZE, WALL, RUN_NAMESPACE"
            echo "30B-A3B always uses R3, disables ReplayDataBuffer, and must use one allocation."
            exit 0 ;;
        --submit|--resume-chain|--clean-checkpoint) ;;
        *) echo "unknown option: ${option}" >&2; exit 2 ;;
    esac
done

: "${MODELS:=${MODEL_PRESET:-qwen3-4b-sft qwen3-8b-sft qwen3-30b-a3b-sft}}"
: "${LOSS_VARIANTS:=${LOSS_VARIANT:-tis}}"
[[ "${LOSS_VARIANTS}" == tis && "${LOSS_VARIANT:-tis}" == tis ]] || {
    echo "This study supports TIS only; score centering has been excluded." >&2
    exit 2
}
if [[ " $* " == *" --resume-chain "* ]]; then
    : "${RUN_NAMESPACE:?--resume-chain requires the original RUN_NAMESPACE}"
fi
: "${RUN_NAMESPACE:=zr-$(date +%Y%m%d-%H%M%S)-p$$}"
export LR="${LR:-1e-6}" GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-3072}" ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-192}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}" NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-300}" LOSS_VARIANT=tis
export IS_CORRECTION=tis TRUNCATION_MODE=zero_reward PLACEMENT=disaggregate
export WALL="${WALL:-168:00:00}" CHAIN_JOBS="${CHAIN_JOBS:-1}"
export STALENESS_LEVELS="${STALENESS_LEVELS:-12 16 20 24 28}"
export RATIOS="${RATIOS:-1:7 2:6 3:5 4:4}"
export TOTAL_NODES="${TOTAL_NODES:-8}"

# Validate all choices before any submission can occur.
[[ "${TOTAL_NODES}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid TOTAL_NODES" >&2; exit 2; }
for staleness in ${STALENESS_LEVELS}; do
    [[ "${staleness}" =~ ^(none|0|[1-9][0-9]*)$ ]] || { echo "invalid staleness: ${staleness}" >&2; exit 2; }
done
for ratio in ${RATIOS}; do
    [[ "${ratio}" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] || { echo "invalid ratio: ${ratio}" >&2; exit 2; }
    (( BASH_REMATCH[1] + BASH_REMATCH[2] == TOTAL_NODES )) || { echo "ratio ${ratio} must sum to TOTAL_NODES" >&2; exit 2; }
done
for model in ${MODELS}; do
    case "${model}" in
        qwen3-4b-sft|qwen3-8b-sft) ;;
        qwen3-30b-a3b-sft)
            [[ "${CHAIN_JOBS}" == 1 && " $* " != *" --resume-chain "* ]] || {
                echo "30B-A3B has ReplayDataBuffer disabled; use one uninterrupted allocation, not a resume chain" >&2
                exit 2
            } ;;
        *) echo "unsupported model: ${model}" >&2; exit 2 ;;
    esac
done
for model in ${MODELS}; do
    (
        export MODEL_PRESET="${model}"
        export RUN_NAMESPACE="${RUN_NAMESPACE}-${model}-tis"
        # This study's fixed R3/replay/forward contracts win over caller overrides.
        if [[ "${model}" == qwen3-30b-a3b-sft ]]; then
            export USE_ROLLOUT_ROUTING_REPLAY=1 USE_REPLAY_DATA_BUFFER=0 REPLAY_DATA_BUFFER_ASYNC_RESTORE=0
        else
            export USE_ROLLOUT_ROUTING_REPLAY=0 USE_REPLAY_DATA_BUFFER=1 REPLAY_DATA_BUFFER_ASYNC_RESTORE=1
        fi
        export REPLAY_DATA_BUFFER_MODE=rollout SKIP_ACTOR_FORWARD_ONLY=1
        echo "model=${model} loss=tis lr=${LR} gbs=${GLOBAL_BATCH_SIZE} rbs=${ROLLOUT_BATCH_SIZE} steps=${NUM_ROLLOUT} wall=${WALL} r3=${USE_ROLLOUT_ROUTING_REPLAY} replay=${USE_REPLAY_DATA_BUFFER}"
        bash "${REPO_ROOT}/experiments/common/sweep_training.sh" staleness_math/zero_reward_truncation "$@"
    )
done
