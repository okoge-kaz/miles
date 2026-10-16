#!/bin/bash

: "${MODEL_NAME:?}"
: "${DATASET_TAG:?}"
: "${PLACEMENT:?}"
: "${PROJECT_NAME:?}"
: "${EXPERIMENT_NAME:?}"
: "${RUN_NAME:=${EXPERIMENT_NAME}-${PLACEMENT}-${MILES_JOB_ID_SHORT}}"
for _miles_identity in PROJECT_NAME EXPERIMENT_NAME MODEL_NAME DATASET_TAG RUN_NAME; do
    [[ "${!_miles_identity}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
        echo "${_miles_identity} must be a path-safe name: ${!_miles_identity}" >&2
        return 2
    }
done
unset _miles_identity
: "${CKPT_PATH:=/ckpt/training/v0.1.2/${PROJECT_NAME}/${EXPERIMENT_NAME}/${DATASET_TAG}/${MODEL_NAME}/${PLACEMENT}/${RUN_NAME}}"
[[ "${CKPT_PATH}" == /ckpt/training/* ]] || {
    echo "CKPT_PATH must be under /ckpt/training: ${CKPT_PATH}" >&2
    return 2
}
case "${CKPT_PATH}/" in
    *'/../'*|*'/./'*|*'//'*) echo "unsafe CKPT_PATH: ${CKPT_PATH}" >&2; return 2 ;;
esac
export RUN_NAME CKPT_PATH
