#!/bin/bash
# Submit the full tool-call pre/train/post chain only when this user's Slurm
# queue is completely empty. This guard implements the requested scheduling
# policy; it does not wait or poll.

set -euo pipefail

echo "tool-call automatic submission is disabled until its recipe is migrated to Qwen3-4B-Base-LR2e-5-Step4000" >&2
exit 2

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh

if [[ -n "$(squeue -h -u "${USER}" -o '%i')" ]]; then
    echo "refusing submission: Slurm queue for ${USER} is not empty" >&2
    squeue -u "${USER}" -o '%.18i|%.36j|%.10P|%.12T|%.12M|%.12l|%.6D|%R' >&2
    exit 2
fi

RESULT_BASE="${REPO_ROOT}/experiments/outputs/downstream_effectiveness/tool_call"
CONFIG_TAG=4h-effectiveness-20260824-16k-rbs192-gbs3072-n16
TRAINING_HF_ROOT="${TRAIN_CKPT_DIR}/tool_call/nemotron-agentic-static-exact-function-call/Qwen3-4B-Instruct-2507/grpo-clip0.2-0.28-tis2.0/async/off-policy/max-weight-staleness-4-from-prefill/${CONFIG_TAG}-zero-trunc-rb-inflight/hf"
BASE_CHECKPOINT="${HF_CKPT_DIR}/Qwen3-4B-Instruct-2507"

prep_job="$(sbatch --parsable experiments/setup/datasets/prepare_tool_call_env.sbatch)"
pre_job="$(sbatch --parsable \
    --dependency="afterok:${prep_job}" \
    --job-name=tool-call-pre-eval \
    --export="ALL,CHECKPOINT_PATH=${BASE_CHECKPOINT},RESULT_ROOT=${RESULT_BASE}/pre" \
    experiments/scripts/tool_call/evaluate.sbatch)"
train_job="$(sbatch --parsable \
    --dependency="afterok:${pre_job}" \
    --job-name=miles-tool-call-4h \
    --export="ALL,CONFIG_TAG=${CONFIG_TAG},CLEAN_CHECKPOINT=1,USE_REPLAY_BUFFER=1,SAVE_INTERVAL=5,HF_SAVE_INTERVAL=5" \
    experiments/scripts/tool_call/async/nemotron3-agentic-static/qwen3-4b-instruct-2507/run.sbatch)"
post_job="$(sbatch --parsable \
    --dependency="afterany:${train_job}" \
    --job-name=tool-call-post-eval \
    --export="ALL,TRAINING_HF_ROOT=${TRAINING_HF_ROOT},RESULT_ROOT=${RESULT_BASE}/post" \
    experiments/scripts/tool_call/evaluate.sbatch)"

printf 'prepare=%s\npre_eval=%s\ntrain=%s\npost_eval=%s\n' \
    "${prep_job}" "${pre_job}" "${train_job}" "${post_job}"
