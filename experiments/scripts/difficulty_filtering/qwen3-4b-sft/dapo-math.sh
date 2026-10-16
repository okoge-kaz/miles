#!/bin/bash

set -euo pipefail

# Target policy and original (unfiltered) dataset. HF_MODEL is relative to /ckpt/hf.
: "${FILTER_POLICY:=Qwen3-4B-SFT-iter4000}"
: "${FILTER_HF_MODEL:=training/reasoing-sft/Qwen3-4B-Base/LR2.0e-5-MINLR2.0e-6-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1/iter_0004000}"
: "${FILTER_INPUT:=/data/dapo-math-17k/dapo-math-17k.jsonl}"
: "${FILTER_DATASET_TAG:=dapo-math-nonzero-std-qwen3-4b-sft-iter4000}"

# Keep 0 < n_correct < n_samples: with n=16, retain 1..15 correct answers.
# Like RL, generate one response per request with independently seeded engines.
: "${FILTER_N_SAMPLES:=16}"
: "${FILTER_MAX_NEW_TOKENS:=16384}"
: "${FILTER_MAX_CONTEXT_LENGTH:=32768}"
: "${FILTER_TEMPERATURE:=1.0}"
: "${FILTER_TOP_P:=1.0}"
: "${FILTER_TOP_K:=-1}"
: "${FILTER_SEED:=1234}"
: "${FILTER_CORRECT_THRESHOLD:=0.5}"
: "${FILTER_ZERO_REWARD_ON_TRUNCATED:=1}"

# One PBS job per shard, one 8-GPU node per job. Validate 4B memory use with smoke.
: "${FILTER_NUM_SHARDS:=16}"
: "${FILTER_TP_SIZE:=1}"
: "${FILTER_NUM_ENGINES:=8}"
: "${FILTER_CONCURRENCY:=160}"
: "${FILTER_MAX_RUNNING_REQUESTS:=80}"
: "${FILTER_MEM_FRACTION_STATIC:=0.85}"
: "${FILTER_SERVER_PORT:=30000}"
: "${FILTER_STARTUP_TIMEOUT:=900}"
: "${FILTER_REQUEST_TIMEOUT:=3600}"
: "${FILTER_AUDIT_PROMPTS:=4}"
: "${FILTER_SMOKE_PROMPTS:=32}"
: "${FILTER_WALLTIME:=12:00:00}"

: "${FILTER_MEASUREMENT_DIR:=/data/difficulty/dapo-math-qwen3-4b-sft-iter4000-n${FILTER_N_SAMPLES}-len${FILTER_MAX_NEW_TOKENS}-${FILTER_NUM_SHARDS}parts}"
: "${FILTER_OUTPUT:=/data/${FILTER_DATASET_TAG}/${FILTER_DATASET_TAG}.jsonl}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../submit.sh"
