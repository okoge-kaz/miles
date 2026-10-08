#!/bin/bash

: "${DATASET_TAG:=dapo-math-17k}"
: "${PROMPT_DATA:=/data/${DATASET_TAG}/${DATASET_TAG}.jsonl}"
: "${INPUT_KEY:=prompt}"
: "${LABEL_KEY:=label}"
: "${RM_TYPE:=deepscaler}"
