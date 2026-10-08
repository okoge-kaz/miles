#!/bin/bash

: "${DATASET_TAG:=multi-env}"
: "${PROMPT_DATA:=}"
: "${PROMPT_DATA_CONFIG:=}"
: "${PROMPT_DATA_FILES:=}"
: "${PROMPT_DATA_RATIOS:=}"
: "${PROMPT_DATA_EPOCH_SIZE:=}"
: "${INPUT_KEY:=prompt}"
: "${LABEL_KEY:=label}"
: "${RM_TYPE:=deepscaler}"
[[ -n "${PROMPT_DATA_CONFIG}${PROMPT_DATA_FILES}" ]] || {
    echo "multi-env requires PROMPT_DATA_CONFIG or PROMPT_DATA_FILES (JSON array)" >&2
    return 2
}
