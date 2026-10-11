#!/bin/bash
# Source before submit.sh; the fixed simulator is independent of policy updates.
export TAU_SERVER_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export TAU_SERVER_MODEL_PATH="/ckpt/hf/Qwen/Qwen3.8-27B/${TAU_SERVER_MODEL_REVISION}/model"
export TAU_SERVER_MODEL_NAME=qwen3.8-27b-user
export TAU_SERVER_TOOL_CALL_PARSER=qwen3_coder
export TAU_SERVER_REASONING_PARSER=qwen3
export TAU_SERVER_TP_SIZE=8
export TAU_SERVER_CONTEXT_LENGTH=32768
export TAU_SERVER_MEM_FRACTION=0.70
export TAU_SERVER_MAX_RUNNING_REQUESTS=64
