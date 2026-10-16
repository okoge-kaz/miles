#!/bin/bash

# The official Workplace example pairs Lightning with this architecture profile.
# HF Base: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16
# HF RL: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
# SFT: set HF_SFT_REF to the absolute host path; no source-branch checkpoint is recorded.

set -euo pipefail

export HF_BASE_REF="${HF_BASE_REF:-https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16}"
export HF_RL_REF="${HF_RL_REF:-https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"

export MODEL_PROFILE="${MODEL_PROFILE:-nemotron-3-nano-30b-a3b}"
export CONVERT_BACKEND="${CONVERT_BACKEND:-bridge}"

# Eight ranks: PP=2 divides 52 layers; EP=4 shards the 128 experts.
export CONVERT_TENSOR_PARALLEL_SIZE="${CONVERT_TENSOR_PARALLEL_SIZE:-1}"
export CONVERT_PIPELINE_PARALLEL_SIZE="${CONVERT_PIPELINE_PARALLEL_SIZE:-2}"
export CONVERT_EXPERT_PARALLEL_SIZE="${CONVERT_EXPERT_PARALLEL_SIZE:-4}"

source "$(dirname -- "${BASH_SOURCE[0]}")/_select_variant.sh"
