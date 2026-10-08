#!/bin/bash

: "${DATASET_TAG:=dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000}"
: "${PROMPT_DATA:=/data/${DATASET_TAG}/${DATASET_TAG}.jsonl}"
: "${INPUT_KEY:=prompt}"
: "${LABEL_KEY:=label}"
: "${RM_TYPE:=deepscaler}"
