#!/bin/bash
# Needs MODEL_NAME, DATASET_TAG, PLACEMENT, ADVANTAGE_ESTIMATOR, EPS_CLIP,
# EPS_CLIP_HIGH, EPS_CLIP_C, RATIO_DENOMINATOR, IS_CORRECTION, TIS_CLIP,
# TIS_CLIP_LOW, MIS_PROFILE, USE_OPSM, OPSM_DELTA, KL_LOSS_COEF, LR,
# MAX_RESPONSE_LEN, NUM_STEPS_PER_ROLLOUT, ROLLOUT_BATCH_SIZE,
# GLOBAL_BATCH_SIZE, N_SAMPLES_PER_PROMPT, TRAIN_SEED, ROLLOUT_SEED, and for PLACEMENT=async also
# MAX_WEIGHT_STALENESS.
# Sets RL_ALGORITHM, POLICY_REGIME, CONFIG_TAG, RUN_NAME, CKPT_PATH.
# Sourced by both run.sbatch and train.sh so the two cannot disagree.

: "${MODEL_NAME:?}"
: "${DATASET_TAG:?}"
: "${PLACEMENT:?}"
: "${ADVANTAGE_ESTIMATOR:?}"
: "${EPS_CLIP:?}"
: "${EPS_CLIP_HIGH:?}"
: "${RATIO_DENOMINATOR:?}"
: "${IS_CORRECTION:?}"
: "${TIS_CLIP:?}"
: "${TIS_CLIP_LOW:?}"
: "${USE_OPSM:?}"
: "${M2PO_BUDGET:=0.04}"
: "${OPSM_DELTA:?}"
: "${KL_LOSS_COEF:?}"
: "${LR:?}"
: "${MAX_RESPONSE_LEN:?}"
: "${NUM_STEPS_PER_ROLLOUT:?}"
: "${ROLLOUT_BATCH_SIZE:?}"
: "${GLOBAL_BATCH_SIZE:?}"
: "${N_SAMPLES_PER_PROMPT:?}"
: "${TRAIN_SEED:?}"
: "${ROLLOUT_SEED:?}"

case "${PLACEMENT}" in
    colocated)
        # None of these flags exist on this path; a non-default value would be silently dropped.
        [[ "${MAX_WEIGHT_STALENESS:-0}" == 0 && "${PAUSE_GENERATION_MODE:-none}" == none \
           && "${STALENESS_REFERENCE:-completion}" == completion ]] ||
            { echo "PLACEMENT=colocated cannot carry MAX_WEIGHT_STALENESS/PAUSE_GENERATION_MODE/STALENESS_REFERENCE" >&2; exit 1; }
        MAX_WEIGHT_STALENESS=0
        PAUSE_GENERATION_MODE=none
        STALENESS_REFERENCE=completion
        ;;
    async)
        : "${MAX_WEIGHT_STALENESS:?}"
        : "${STALENESS_REFERENCE:=completion}"
        [[ "${STALENESS_REFERENCE}" == completion || "${STALENESS_REFERENCE}" == submission ]] ||
            { echo "STALENESS_REFERENCE must be completion or submission, got '${STALENESS_REFERENCE}'" >&2; exit 1; }
        ;;
    *)
        echo "PLACEMENT must be colocated or async, got '${PLACEMENT}'" >&2
        exit 1
        ;;
esac

TASK_FAMILY=math

# The two ways a sample can be off-policy: generated under older weights, or
# reused across more than one optimizer step.
if [[ "${MAX_WEIGHT_STALENESS}" -eq 0 && "${NUM_STEPS_PER_ROLLOUT}" -eq 1 ]]; then
    POLICY_REGIME=on-policy
else
    POLICY_REGIME=off-policy
fi

# RL_ALGORITHM names the whole loss configuration, so every correction that can
# be swept is a different directory. Each part is omitted at its default, which
# keeps the common case short and makes any deviation visible in the name.
if [[ "${TIS_CLIP_LOW}" == "0" ]]; then
    BOUNDS="${TIS_CLIP}"
else
    BOUNDS="${TIS_CLIP_LOW}-${TIS_CLIP}"
fi
case "${IS_CORRECTION}" in
    none)   IS_TAG=nois ;;
    tis)    IS_TAG="tis${BOUNDS}" ;;
    icepop) IS_TAG="icepop${BOUNDS}" ;;
    mis)    : "${MIS_PROFILE:?IS_CORRECTION=mis needs MIS_PROFILE}"; IS_TAG="mis-${MIS_PROFILE}" ;;
    m2po)   IS_TAG="m2po${M2PO_BUDGET}" ;;
    *)      echo "IS_CORRECTION must be none|tis|icepop|mis|m2po, got '${IS_CORRECTION}'" >&2; exit 1 ;;
esac

case "${RATIO_DENOMINATOR}" in
    actor)            DENOM_TAG="" ;;
    rollout-logprobs) DENOM_TAG="-rolloutlp" ;;
    old-actor)        DENOM_TAG="-oldactor" ;;
    *) echo "RATIO_DENOMINATOR must be actor|rollout-logprobs|old-actor, got '${RATIO_DENOMINATOR}'" >&2; exit 1 ;;
esac
# arguments.py:2851 rejects the combination outright.
[[ "${RATIO_DENOMINATOR}" != rollout-logprobs || "${IS_CORRECTION}" == none || "${IS_CORRECTION}" == m2po ]] ||
    { echo "RATIO_DENOMINATOR=rollout-logprobs cannot be combined with IS_CORRECTION=${IS_CORRECTION}" >&2; exit 1; }

RL_ALGORITHM="${ADVANTAGE_ESTIMATOR}-clip${EPS_CLIP}-${EPS_CLIP_HIGH}"
[[ -z "${EPS_CLIP_C}" ]] || RL_ALGORITHM="${RL_ALGORITHM}-dualclip${EPS_CLIP_C}"
RL_ALGORITHM="${RL_ALGORITHM}${DENOM_TAG}-${IS_TAG}"
[[ "${USE_OPSM}" == "0" ]] || RL_ALGORITHM="${RL_ALGORITHM}-opsm${OPSM_DELTA}"
if awk "BEGIN{exit !(${KL_LOSS_COEF} != 0)}"; then
    RL_ALGORITHM="${RL_ALGORITHM}-kl${KL_LOSS_COEF}"
fi

CONFIG_TAG="${CONFIG_TAG:-rollout-length-$(( MAX_RESPONSE_LEN / 1024 ))k-lr${LR}-rbs${ROLLOUT_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}-n${N_SAMPLES_PER_PROMPT}-tseed${TRAIN_SEED}-rseed${ROLLOUT_SEED}}"
# Suffixed only away from the default, so paths written before the option existed
# keep their spelling. The reference decides which groups are recycled, so two runs
# that differ only here are different runs and must not share a directory.
STALENESS_TAG="max-weight-staleness-${MAX_WEIGHT_STALENESS}"
[[ "${STALENESS_REFERENCE:-completion}" == completion ]] || STALENESS_TAG="${STALENESS_TAG}-from-${STALENESS_REFERENCE}"

# RUN_NAME is the wandb group and the log directory, not a path. It shows the
# axes this study varies and closes over the rest with a hash; the full identity
# is in CKPT_PATH and in the config wandb logs from the arguments anyway, so
# spelling it out here only bought length. The hash is deterministic, so a
# resumed job rejoins its group, and two configurations that share the visible
# part still differ.
_regime=$([[ "${POLICY_REGIME}" == on-policy ]] && echo onp || echo offp)
_identity="${MODEL_NAME}-${PLACEMENT}-${_regime}-s${MAX_WEIGHT_STALENESS}-${CONFIG_TAG}-${RL_ALGORITHM}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${PLACEMENT}-${_regime}-s${MAX_WEIGHT_STALENESS}-$(( MAX_RESPONSE_LEN / 1024 ))k-lr${LR}-$(printf '%s' "${_identity}" | md5sum | cut -c1-8)}"

# wandb_utils.py:52 appends "_" and an 8-character id to the group whenever
# --wandb-random-suffix is on, which is its default, so the group runs nine
# characters longer than this. Fail here, at submission, rather than 100 seconds
# into an allocation where the whole chain is already queued behind it.
WANDB_GROUP_LIMIT=128
WANDB_GROUP_SUFFIX=9
if (( ${#RUN_NAME} + WANDB_GROUP_SUFFIX > WANDB_GROUP_LIMIT )); then
    echo "RUN_NAME is ${#RUN_NAME} chars, so the wandb group would be" \
         "$(( ${#RUN_NAME} + WANDB_GROUP_SUFFIX )) > ${WANDB_GROUP_LIMIT}: ${RUN_NAME}" >&2
    exit 1
fi
CKPT_PATH="/ckpt/training/${TASK_FAMILY}/${DATASET_TAG}/${MODEL_NAME}/${RL_ALGORITHM}/${PLACEMENT}/${POLICY_REGIME}/${STALENESS_TAG}/${CONFIG_TAG}"
