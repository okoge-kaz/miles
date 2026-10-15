#!/bin/bash

set -euo pipefail

: "${FILTER_INPUT:=/data/multi_env/if-training.jsonl}"
: "${FILTER_REWARD_BACKEND:=instruction_following}"
: "${FILTER_SELECTION_RULE:=reward_variance}"
: "${FILTER_CORRECT_THRESHOLD:=1.0}"
: "${FILTER_NUM_SHARDS:=16}"
: "${FILTER_MAX_PARALLEL:=8}"
: "${FILTER_WALLTIME:=24:00:00}"
: "${FILTER_DATASET_TAG:=instruction-following-nonzero-std-qwen3-30b-a3b-sft-iter4000}"
: "${FILTER_MEASUREMENT_DIR:=/data/difficulty/instruction-following-qwen3-30b-a3b-sft-iter4000-n${FILTER_N_SAMPLES:-16}-len${FILTER_MAX_NEW_TOKENS:-16384}-${FILTER_NUM_SHARDS}parts}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/containers/miles-v0.1.1-multi-env.sif}"

# Reuse this model's generation settings and checkpoint; native Gym owns verification.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/dapo-math.sh"
