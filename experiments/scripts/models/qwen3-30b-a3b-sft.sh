#!/bin/bash

# Qwen3-30B-A3B SFT iter4000. Environment overrides take precedence.
: "${QWEN3_30B_A3B_BASE_HF_RELATIVE_DIR:=Qwen3-30B-A3B-Base/LR2.0e-5-MINLR2.0e-6-SEQ32768-GBS128-MBS1-TP4-PP2-CP2-EP4-PACK1-standard-cp-STEPS4000}"
: "${MODEL_NAME:=Qwen3-30B-A3B-SFT-iter4000}"

: "${LR:=1e-6}"
: "${RM_NUM_WORKERS:=4}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=8}"
: "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
: "${USE_ROLLOUT_ROUTING_REPLAY:=1}"
: "${MAX_TOKENS_PER_GPU:=32768}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=32768}"
: "${TRAINING_ATTENTION_BACKEND:=fused}"
: "${SKIP_ACTOR_FORWARD_ONLY:=1}"

: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
if [[ "${PLACEMENT:-disaggregate}" == disaggregate ]]; then
    : "${SGLANG_MEM_FRACTION:=0.70}"
else
    : "${SGLANG_MEM_FRACTION:=0.65}"
fi

# Resolve HF_MODEL_NAME (or its explicit override) and remaining defaults.
source "$(dirname -- "${BASH_SOURCE[0]}")/qwen3-30b-a3b.sh"
