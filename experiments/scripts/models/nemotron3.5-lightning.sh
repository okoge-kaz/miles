#!/bin/bash

# Nemotron3.5 Lightning 30B-A3B
# hf: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/config.json
# hf base: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16
# Math settings follow the Qwen3-4B study; GPU validation is still required.

: "${MODEL_NAME:=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
: "${HF_MODEL_NAME:=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
: "${MODEL_PROFILE:=nemotron-3-nano-30b-a3b}"
: "${MEGATRON_TO_HF_MODE:=bridge}"

: "${LR:=1e-6}"
: "${NUM_ROLLOUT:=300}"
: "${ROLLOUT_BATCH_SIZE:=192}"
: "${N_SAMPLES_PER_PROMPT:=16}"
: "${GLOBAL_BATCH_SIZE:=3072}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"
: "${MAX_RESPONSE_LEN:=16384}"
: "${MAX_CONTEXT_LEN:=32768}"

: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${PIPELINE_PARALLEL_SIZE:=2}"
: "${EXPERT_PARALLEL_SIZE:=2}"
: "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
: "${TRAINING_ATTENTION_BACKEND:=auto}"
: "${MAX_TOKENS_PER_GPU:=32768}"

: "${SKIP_ACTOR_FORWARD_ONLY:=1}"

: "${IS_CORRECTION:=tis}"
: "${TIS_CLIP:=2.0}"
: "${TIS_CLIP_LOW:=0}"
: "${ASYNC_DATA_BUFFER_CAPACITY_FACTOR:=30}"
: "${ASYNC_UNUSED_SAMPLES_HANDLER:=retry}"

: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=100}"
: "${SAVE_HF:=1}"
: "${EVAL_INTERVAL:=0}"
