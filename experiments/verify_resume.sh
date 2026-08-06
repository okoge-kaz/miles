#!/bin/bash
# Verify that a run survives being cut at a partition wall clock and picks up
# where it left off, and that --save-retain-interval prunes without ever
# removing the checkpoint the next job needs.
#
#     experiments/verify_resume.sh [--submit]
#
# This is not optional polish. `batch` caps a job at 4 hours and one sweep point
# is ~10 hours of training, so every real run is 3-4 chained jobs and every one
# of them resumes. The 40-minute throughput jobs never reach a save, so this
# path has never executed.
#
# Two jobs, chained with afterany so the second runs whether or not the first
# exits clean:
#
#   phase A  25 min of wall clock, then Slurm kills it mid-step. save-interval 2
#            and save-retain-interval 4, so it saves at 2, 4, 6, ... and prunes
#            every iteration that is not a multiple of 4 as the next one lands.
#   phase B  same CKPT_PATH, 50 min. Must load A's last surviving iteration and
#            continue from it rather than starting over.
#
# What that proves, in order of how badly it would hurt to be wrong:
#   1. --load finds the tracker and start_rollout_id resumes rather than resets
#      (arguments.py:2752, actor.py:196). A reset would silently restart every
#      chained job from step 0 and the sweep would never finish a single point.
#   2. rollout_manager.load restores the data source position, so the second job
#      does not replay the prompts the first one already trained on
#      (placement_group.py:182).
#   3. the retain sweep never deletes the iteration the tracker points at
#      (Megatron checkpointing.py:865: it reads prev_iteration, writes the new
#      tracker, and only then deletes prev).
#   4. optimizer and RNG state come back. On the fallback path
#      (arguments.py:2758) miles sets no_load_optim/no_load_rng/finetune, which
#      is right for a fresh run and wrong for a resume -- so the log must NOT
#      show those on phase B.
#
# CONFIG_TAG is pinned to resume-test so this writes to its own checkpoint
# directory and cannot collide with a real run.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
CONFIG_TAG=resume-test
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507"

# CKPT_PATH now has RL_ALGORITHM / PLACEMENT / POLICY_REGIME / staleness levels
# above CONFIG_TAG, so derive it the way the recipe does instead of spelling it
# out here. The `: "${VAR:=default}"` lines are the recipe's own defaults.
export ACTOR_NUM_NODES=1 ROLLOUT_NUM_GPUS=16
eval "$(sed -n 's/[[:space:]]*#.*$//; /^: "\${[A-Z_]*:=/p; /^\(MODEL_NAME\|PLACEMENT\|DATASET_TAG\)=/p' "${REPO_ROOT}/${RECIPE}")"
source "${REPO_ROOT}/experiments/common/run_identity.sh"
CKPT_DIR="${TRAIN_CKPT_DIR}${CKPT_PATH#/ckpt/training}"

if [[ "${1:-}" == "--check" ]]; then
    a=$(ls -t "${LOG_DIR}"/resume-a-*.log 2>/dev/null | head -1)
    b=$(ls -t "${LOG_DIR}"/resume-b-*.log 2>/dev/null | head -1)
    [[ -n "${a}" ]] || { echo "no phase A log under ${LOG_DIR}"; exit 1; }

    echo "=== tracker and retained iterations ==="
    if [[ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
        echo "  tracker -> $(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt")"
    else
        echo "  NO TRACKER: nothing was saved"
    fi
    echo "  dist  : $(ls -d "${CKPT_DIR}"/iter_* 2>/dev/null | xargs -rn1 basename | tr '\n' ' ')"
    echo "  hf    : $(ls "${CKPT_DIR}/hf" 2>/dev/null | tr '\n' ' ')"
    echo "  du    : $(du -sh "${CKPT_DIR}" 2>/dev/null | cut -f1) total,"\
         "$(du -sh "${CKPT_DIR}/hf" 2>/dev/null | cut -f1) of it hf"

    for phase in A B; do
        log=$([[ ${phase} == A ]] && echo "${a}" || echo "${b}")
        [[ -n "${log}" ]] || { echo; echo "=== phase B: not run yet ==="; continue; }
        echo
        echo "=== phase ${phase}: $(basename "${log}") ==="
        # start_rollout_id is the whole question for B: 0 means the resume was
        # silently dropped and the job retrained from scratch.
        grep -oE "start_rollout_id[ =:]+[0-9]+" "${log}" | head -3 | sed 's/^/  /'
        grep -oE "will not load any checkpoints|could not find arguments|loading checkpoint from [^ ]*|successfully loaded checkpoint from [^ ]* \[ *t [0-9]+" \
            "${log}" | sort -u | head -5 | sed 's/^/  /'
        # On a resume these must be absent; on a fresh start they are correct.
        grep -oE "no_load_optim|no_load_rng|finetune" "${log}" | sort -u | tr '\n' ' ' | sed 's/^/  fallback flags seen: /'
        echo
        grep -oE "successfully saved checkpoint from iteration +[0-9]+" "${log}" | sed 's/^/  /'
        grep -oE "(skipping )?deleting checkpoint from iteration +[0-9]+" "${log}" | sed 's/^/  /'
    done
    exit 0
fi

# Everything except num-rollout is identical between the phases, and identical
# to the production recipe except for the save cadence -- the point is to
# exercise the real checkpoint shape, not a small one.
#
# NUM_ROLLOUT must be IDENTICAL in both phases. Megatron derives the LR
# schedule's total iteration count from it and asserts it against the value in
# the checkpoint (`OptimizerParamScheduler: class input value ... and checkpoint
# value ... do not match`). A first attempt used 5 then 9 and phase B died on
# that assert -- a flaw in the test, not in resume, since a real run keeps
# --num-rollout fixed across its chained jobs. It is still worth knowing that
# --num-rollout is frozen for the lifetime of a run: a run cannot be extended
# later by raising it.
COMMON="CONFIG_TAG=${CONFIG_TAG},NUM_ROLLOUT=30,SAVE_INTERVAL=2,SAVE_RETAIN_INTERVAL=4,SKIP_EVAL_BEFORE_TRAIN=1,EVAL_INTERVAL=1000"

if [[ "${1:-}" != "--submit" ]]; then
    echo "would submit, 3 nodes each:"
    echo "  A  25min wall clock -- killed mid-run, which is the case that matters"
    echo "  B  50min, afterany:A -- must resume from A's last save"
    echo "  ${COMMON}"
    echo
    echo "checkpoints land in ${CKPT_DIR}"
    echo "re-run with --submit"
    exit 0
fi

if [[ -d "${CKPT_DIR}" ]]; then
    echo "removing the previous resume-test checkpoints at ${CKPT_DIR}"
    echo "  ($(du -sh "${CKPT_DIR}" 2>/dev/null | cut -f1)) -- phase A must start from"
    echo "  the reference weights, and a stale tracker would resume into the old"
    echo "  LR schedule and fail the same assert again."
    rm -rf "${CKPT_DIR}"
fi

# Phase A is cut by the wall clock rather than by finishing, because that is what
# happens to every real run: batch stops at 4h and a sweep point needs ~10.
JOB_A=$(sbatch --parsable \
    -A "${SLURM_ACCOUNT_NAME}" \
    --partition=batch,batch_short \
    --job-name=resume-a \
    --nodes=3 --time=00:25:00 \
    --export="ALL,${COMMON},ACTOR_NUM_NODES=1,ROLLOUT_NUM_GPUS=16" \
    "${RECIPE}")
echo "phase A: ${JOB_A}"

JOB_B=$(sbatch --parsable \
    -A "${SLURM_ACCOUNT_NAME}" \
    --partition=batch,batch_short \
    --job-name=resume-b \
    --nodes=3 --time=00:50:00 \
    --dependency=afterany:"${JOB_A}" \
    --export="ALL,${COMMON},ACTOR_NUM_NODES=1,ROLLOUT_NUM_GPUS=16" \
    "${RECIPE}")
echo "phase B: ${JOB_B}  (after ${JOB_A})"

echo
echo "check with:  experiments/verify_resume.sh --check"
