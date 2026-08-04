#!/bin/bash
# Needs MODEL_NAME, ROLLOUT_MODE, DATASET_TAG, LR, MAX_RESPONSE_LEN,
# NUM_STEPS_PER_ROLLOUT. Sets CONFIG_TAG, RUN_NAME, CKPT_PATH.
# Sourced by both run.sbatch and train.sh so the two cannot disagree.

: "${MODEL_NAME:?}"
: "${ROLLOUT_MODE:?}"
: "${DATASET_TAG:?}"
: "${LR:?}"
: "${MAX_RESPONSE_LEN:?}"
: "${NUM_STEPS_PER_ROLLOUT:?}"

TASK_FAMILY=math

CONFIG_TAG="${CONFIG_TAG:-${ROLLOUT_MODE}-${NUM_STEPS_PER_ROLLOUT}step-rollout-length-$(( MAX_RESPONSE_LEN / 1024 ))k-lr${LR}}"
RUN_NAME="${RUN_NAME:-${TASK_FAMILY}-${DATASET_TAG}-${MODEL_NAME}-${CONFIG_TAG}}"
CKPT_PATH="/ckpt/training/${TASK_FAMILY}/${DATASET_TAG}/${MODEL_NAME}/${CONFIG_TAG}"
