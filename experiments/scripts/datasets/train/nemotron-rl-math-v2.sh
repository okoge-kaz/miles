#!/bin/bash

: "${DATASET_TAG:=nemotron-rl-math-v2}"
: "${PROMPT_DATA:=/data/${DATASET_TAG}/nemotron-rl-math-v2-miles.jsonl}"
: "${INPUT_KEY:=prompt}"
: "${LABEL_KEY:=label}"
: "${RM_TYPE:=deepscaler}"
