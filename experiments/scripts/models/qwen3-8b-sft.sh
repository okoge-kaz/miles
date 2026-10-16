#!/bin/bash

# Qwen3-8B SFT iter4000. Environment overrides take precedence.
: "${QWEN3_8B_BASE_HF_RELATIVE_DIR:=Qwen3-8B-Base/LR1.5e-5-MINLR1.5e-6-SEQ32768-GBS128-MBS1-TP2-PP1-CP2-EP1-PACK1-standard-cp-STEPS4000}"
: "${MODEL_NAME:=Qwen3-8B-SFT-iter4000}"

: "${LR:=1e-6}"
: "${RM_NUM_WORKERS:=4}"
: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"
: "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
: "${USE_ROLLOUT_ROUTING_REPLAY:=0}"
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=32768}"
: "${TRAINING_ATTENTION_BACKEND:=fused}"
: "${SKIP_ACTOR_FORWARD_ONLY:=1}"

if [[ "${PLACEMENT:-disaggregate}" == disaggregate ]]; then
    : "${ROLLOUT_NUM_GPUS_PER_ENGINE:=1}"
    : "${SGLANG_MEM_FRACTION:=0.70}"
else
    : "${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
    : "${SGLANG_MEM_FRACTION:=0.65}"
fi

# Resolve HF_MODEL_NAME (or its explicit override) and remaining defaults.
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen3-8b.sh"
