#!/bin/bash

# HF Base: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
# HF RL: https://huggingface.co/Qwen/Qwen3-30B-A3B
# SFT: /groups/gai51740/kazuki_fujii/checkpoints/hf/Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000/iter_0004000

set -euo pipefail

export HF_BASE_REF="${HF_BASE_REF:-https://huggingface.co/Qwen/Qwen3-30B-A3B-Base}"
export HF_RL_REF="${HF_RL_REF:-https://huggingface.co/Qwen/Qwen3-30B-A3B}"
export HF_SFT_REF="${HF_SFT_REF:-${HF_CKPT_DIR:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/checkpoints/hf}/Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000/iter_0004000}"
export MODEL_SFT_NAME="${MODEL_SFT_NAME:-Qwen3-30B-A3B-Base-LR2e-5-Step4000}"

export MODEL_PROFILE="${MODEL_PROFILE:-qwen3-30B-A3B}"
export CONVERT_BACKEND="${CONVERT_BACKEND:-raw}"

source "$(dirname -- "${BASH_SOURCE[0]}")/_select_variant.sh"
