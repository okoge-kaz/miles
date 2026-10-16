#!/bin/bash

: "${DATASET_TAG:=dapo-math-nonzero-std-qwen3-8b-sft-iter4000}"
: "${PROMPT_DATA:=/data/${DATASET_TAG}/${DATASET_TAG}.jsonl}"
: "${INPUT_KEY:=prompt}"
: "${LABEL_KEY:=label}"
: "${RM_TYPE:=deepscaler}"
