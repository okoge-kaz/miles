#!/bin/bash
#SBATCH --job-name=miles-tau-bench-tau1-qwen3-4b-instruct-2507
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

# tau-bench (retail / airline) via examples/experimental/tau-bench.
#
#   experiments/submit_training.sh tau_bench/tau1/qwen3-4b-instruct-2507 <run-name>
#
# The only recipe here whose environment answers with *another model*: a user
# simulator drives the conversation through litellm while tau-bench's own mock
# tools return database rows. Both are non-LLM-policy observations that get
# prefilled between decode phases, and the episode genuinely branches on what the
# policy did -- unlike conv-tooluse, where the prefix is fixed.
#
# Requires:
#   * tau-bench installed in the image and tau1_mock.py run once (see
#     experiments/setup/prepare_tau_bench.sbatch)
#   * a litellm-reachable user-simulator model. Pick the provider and supply its
#     key; the example reads these env vars:
#       TAU_USER_MODEL_PROVIDER=gemini   TAU_USER_MODEL=gemini-2.5-flash-lite  GEMINI_API_KEY=...
#       TAU_USER_MODEL_PROVIDER=deepseek TAU_USER_MODEL=deepseek-chat          DEEPSEEK_API_KEY=...
#
# No database server: tau-bench ships its domain data in-repo.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${WANDB_API_KEY:?set WANDB_API_KEY, or put an api.wandb.ai entry in ~/.netrc}"
: "${TAU_USER_MODEL_PROVIDER:=gemini}"
: "${TAU_USER_MODEL:=gemini-2.5-flash-lite}"

# Fail here rather than three hours in: the user simulator is called once per
# turn, so a missing key turns every episode into an immediate zero and the run
# looks like a policy that cannot converse.
_key_var="$(echo "${TAU_USER_MODEL_PROVIDER}" | tr '[:lower:]' '[:upper:]')_API_KEY"
[[ -n "${!_key_var:-}" ]] || { echo "set ${_key_var} for the tau-bench user simulator"; exit 1; }

[[ -f "${DATASET_DIR}/tau-bench/tau1_train.jsonl" ]] || {
    echo "missing ${DATASET_DIR}/tau-bench/tau1_train.jsonl; run experiments/setup/prepare_tau_bench.sbatch"; exit 1; }

MODEL_NAME=Qwen3-4B-Instruct-2507
ROLLOUT_MODE=async-off
DATASET_TAG=tau-bench-tau1
PROMPT_DATA=/data/tau-bench/tau1_train.jsonl

: "${LR:=1e-6}"
# Per turn. Episodes are long in turns, not in tokens per turn.
: "${MAX_RESPONSE_LEN:=2048}"
# Whole episode: system policy document + every user turn + every tool result.
: "${ROLLOUT_MAX_CONTEXT_LEN:=32768}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"
: "${TAU_MAX_TURNS:=30}"

: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_GPUS_PER_NODE:=8}"
: "${ROLLOUT_NUM_GPUS:=0}"

: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"
: "${MAX_TOKENS_PER_GPU:=16384}"

: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
: "${SGLANG_MEM_FRACTION:=0.7}"

# Smaller than the math recipes: every sample is a full multi-turn episode, and
# each turn waits on an external API call, so wall-clock per rollout is bounded
# by the user simulator rather than by GPU throughput.
: "${ROLLOUT_BATCH_SIZE:=16}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${GLOBAL_BATCH_SIZE:=128}"
: "${NUM_ROLLOUT:=3000}"
: "${DYNAMIC_SAMPLING:=1}"
: "${DUMP_TRAIN_DATA:=0}"

export MODEL_NAME ROLLOUT_MODE DATASET_TAG PROMPT_DATA LR MAX_RESPONSE_LEN NUM_STEPS_PER_ROLLOUT
export ROLLOUT_MAX_CONTEXT_LEN TAU_MAX_TURNS TAU_USER_MODEL_PROVIDER TAU_USER_MODEL
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE MAX_TOKENS_PER_GPU
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_ROLLOUT
export DYNAMIC_SAMPLING DUMP_TRAIN_DATA

source "${REPO_ROOT}/experiments/common/run_identity.sh"
export RUN_NAME CONFIG_TAG CKPT_PATH

source "${REPO_ROOT}/experiments/common/placement.sh"

export RAY_HEAD_IP="$(hostname -I | awk '{print $1}')"
export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"

echo "run ${RUN_NAME}"
echo "checkpoints ${CKPT_PATH}"
echo "user simulator ${TAU_USER_MODEL_PROVIDER}/${TAU_USER_MODEL}"

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/tau_bench/tau1/qwen3-4b-instruct-2507/train.sh
