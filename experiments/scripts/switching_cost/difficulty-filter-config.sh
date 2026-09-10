#!/bin/bash

# Shared policy-specific configuration for the SFT difficulty measurements.
# The caller must set MODEL_VARIANT before sourcing this file.

case "${MODEL_VARIANT:?set MODEL_VARIANT=qwen3-8b or qwen3-30b-a3b}" in
    qwen3-8b)
        MODEL_NAME=Qwen3-8B-Base-LR1.5e-5-Step4000
        HF_MODEL_NAME=Qwen3-8B-Base/LR1.5e-5-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000
        DATASET_NAME=dapo-math-p10-90-qwen3-8b-base-lr1.5e-5-step4000
        FILTER_JOB_SUFFIX=q3-8b-sft
        SGLANG_TP_SIZE=1
        SGLANG_DP_SIZE=8
        CONCURRENCY=160
        ;;
    qwen3-30b-a3b)
        MODEL_NAME=Qwen3-30B-A3B-Base-LR2.0e-5-Step4000
        HF_MODEL_NAME=Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000
        DATASET_NAME=dapo-math-p10-90-qwen3-30b-a3b-base-lr2.0e-5-step4000
        FILTER_JOB_SUFFIX=q3-30b-sft
        SGLANG_TP_SIZE=2
        SGLANG_DP_SIZE=4
        CONCURRENCY=80
        ;;
    *)
        echo "unsupported MODEL_VARIANT=${MODEL_VARIANT}" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
PASS_RATES="/data/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.passrate.jsonl"
AUDIT_OUTPUT="/data/difficulty/dapo-math-17k.${MODEL_NAME}.n16-len16384-zero-trunc.audit.jsonl"
FILTERED_OUTPUT="/data/${DATASET_NAME}/${DATASET_NAME}.jsonl"
TOTAL_PROMPTS=17398
HALF_PROMPTS=$(( TOTAL_PROMPTS / 2 ))

export MODEL_NAME HF_MODEL_NAME DATASET_NAME FILTER_JOB_SUFFIX
export SGLANG_TP_SIZE SGLANG_DP_SIZE CONCURRENCY
export PROMPT_DATA PASS_RATES AUDIT_OUTPUT FILTERED_OUTPUT TOTAL_PROMPTS HALF_PROMPTS
