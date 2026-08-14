#!/bin/bash
#SBATCH --job-name=miles-search-r1-nq-hotpotqa-qwen3-4b-instruct-2507
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/training/%x-%j.log

# Search-R1 on NQ + HotpotQA. Multi-turn with a *retriever* as the environment:
# the retrieved passages are the non-LLM observation that gets prefilled between
# decode phases.
#
#   experiments/submit_training.sh search_r1/nq-hotpotqa/qwen3-4b-instruct-2507 <run-name>
#
# Requires experiments/setup/prepare_search_r1.sbatch to have run: it
# concatenates the split FAISS index, decompresses the corpus and fetches the
# e5-base-v2 query encoder.
#
# Single node on purpose. The retriever holds a 65 GB index in host RAM and the
# nodes have 2 TB, but one retriever per job keeps the search path off the
# inter-node network while we are measuring rollout latency.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${WANDB_API_KEY:?set WANDB_API_KEY, or put an api.wandb.ai entry in ~/.netrc}"

for asset in "${DATASET_DIR}/searchr1-nq-hotpotqa/train.parquet" \
             "${DATASET_DIR}/search-r1/e5_Flat.index" \
             "${DATASET_DIR}/search-r1/wiki-18.jsonl" \
             "${DATASET_DIR}/search-r1/eval/nq-miles.jsonl" \
             "${DATASET_DIR}/search-r1/eval/hotpotqa-miles.jsonl" \
             "${DATASET_DIR}/search-r1/eval/triviaqa-miles.jsonl" \
             "${DATASET_DIR}/search-r1/eval/popqa-miles.jsonl" \
             "${DATASET_DIR}/search-r1/eval/2wikimultihopqa-miles.jsonl" \
             "${DATASET_DIR}/search-r1/eval/musique-miles.jsonl" \
             "${DATASET_DIR}/search-r1/eval/bamboogle-miles.jsonl" \
             "${HF_CKPT_DIR}/e5-base-v2/.download_complete"; do
    [[ -e "${asset}" ]] || {
        echo "missing ${asset}; stage datasets.txt assets, then run experiments/setup/prepare_search_r1.sbatch"
        exit 1
    }
done

MODEL_NAME=Qwen3-4B-Instruct-2507
PLACEMENT=colocated
TASK_FAMILY=search_r1
DATASET_TAG=search-r1-nq-hotpotqa
PROMPT_DATA=/data/searchr1-nq-hotpotqa/train.parquet

: "${ADVANTAGE_ESTIMATOR:=grpo}"
: "${ENTROPY_COEF:=0.00}"
: "${EPS_CLIP:=0.2}"
: "${EPS_CLIP_HIGH:=0.28}"
: "${EPS_CLIP_C:=}"
: "${RATIO_DENOMINATOR:=actor}"
: "${IS_CORRECTION:=none}"
: "${KL_LOSS_COEF:=0.00}"
: "${TIS_CLIP:=2.0}"
: "${TIS_CLIP_LOW:=0}"
: "${MIS_PROFILE:=}"
: "${USE_OPSM:=0}"
: "${OPSM_DELTA:=1e-4}"
: "${LR:=1e-6}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"
# Per turn. The policy writes a query or a short answer; the passages it reads
# back are far longer than anything it writes.
: "${MAX_RESPONSE_LEN:=512}"
# Question + SEARCH_MAX_TURNS x SEARCH_TOPK passages has to fit here.
: "${ROLLOUT_MAX_CONTEXT_LEN:=16384}"
: "${NUM_STEPS_PER_ROLLOUT:=1}"

: "${SEARCH_MAX_TURNS:=3}"
: "${SEARCH_TOPK:=3}"
# Searches in flight against the retriever. The server coalesces nearby requests
# into one native batch; this bound limits how many trajectories can wait behind
# a 21M-vector flat-index scan and keeps timeout behavior predictable.
: "${SEARCH_CONCURRENCY:=16}"
: "${SEARCH_TIMEOUT:=60}"
: "${SEARCH_MAX_ATTEMPTS:=3}"
# The Search-R1 paper uses outcome exact match. Set 0.2 to reproduce the
# optional format shaping used by the legacy Miles example.
: "${SEARCH_FORMAT_SCORE:=0.0}"
: "${RETRIEVER_PORT:=8000}"
# Native PyTorch/FAISS calls are serialized through one dynamic batch worker;
# nearby rollout requests share one encoder pass and one 65 GB index scan.
: "${RETRIEVER_BATCH_MAX_REQUESTS:=64}"
: "${RETRIEVER_BATCH_WAIT_MS:=50}"
: "${RETRIEVER_FAISS_THREADS:=32}"

: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_GPUS_PER_NODE:=8}"
: "${ROLLOUT_NUM_GPUS:=0}"

: "${TENSOR_PARALLEL_SIZE:=2}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=1}"
: "${MAX_TOKENS_PER_GPU:=16384}"

: "${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
# Lower than the math recipes: the retriever's encoder shares the node, and
# trajectories are short so a big KV pool buys little.
: "${SGLANG_MEM_FRACTION:=0.6}"

: "${ROLLOUT_BATCH_SIZE:=32}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${GLOBAL_BATCH_SIZE:=256}"
: "${NUM_ROLLOUT:=3000}"
: "${DYNAMIC_SAMPLING:=1}"
: "${DUMP_TRAIN_DATA:=0}"
# Reported evaluation is run from HF exports, off the training critical path.
: "${EVAL_INTERVAL:=0}"
: "${SKIP_EVAL_BEFORE_TRAIN:=1}"

# torch_dist is for resume; HF snapshots are the lightweight inference series
# consumed by experiments/src/offline_eval/run_search_r1_eval.sbatch.
: "${SAVE_INTERVAL:=20}"
: "${SAVE_RETAIN_INTERVAL:=100}"
: "${SAVE_HF:=1}"
: "${HF_SAVE_INTERVAL:=20}"

export ADVANTAGE_ESTIMATOR ENTROPY_COEF EPS_CLIP EPS_CLIP_HIGH EPS_CLIP_C RATIO_DENOMINATOR IS_CORRECTION KL_LOSS_COEF TIS_CLIP TIS_CLIP_LOW MIS_PROFILE USE_OPSM OPSM_DELTA TRAIN_SEED ROLLOUT_SEED MODEL_NAME PLACEMENT TASK_FAMILY DATASET_TAG PROMPT_DATA LR MAX_RESPONSE_LEN NUM_STEPS_PER_ROLLOUT
export ROLLOUT_MAX_CONTEXT_LEN SEARCH_MAX_TURNS SEARCH_TOPK SEARCH_CONCURRENCY SEARCH_TIMEOUT SEARCH_MAX_ATTEMPTS SEARCH_FORMAT_SCORE RETRIEVER_PORT
export RETRIEVER_BATCH_MAX_REQUESTS RETRIEVER_BATCH_WAIT_MS RETRIEVER_FAISS_THREADS
export ACTOR_NUM_NODES ACTOR_GPUS_PER_NODE ROLLOUT_NUM_GPUS
export TENSOR_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE EXPERT_PARALLEL_SIZE MAX_TOKENS_PER_GPU
export ROLLOUT_NUM_GPUS_PER_ENGINE SGLANG_MEM_FRACTION
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE NUM_ROLLOUT
export DYNAMIC_SAMPLING DUMP_TRAIN_DATA EVAL_INTERVAL SKIP_EVAL_BEFORE_TRAIN
export SAVE_INTERVAL SAVE_RETAIN_INTERVAL SAVE_HF HF_SAVE_INTERVAL

source "${REPO_ROOT}/experiments/common/run_identity.sh"
export RUN_NAME CONFIG_TAG CKPT_PATH

source "${REPO_ROOT}/experiments/common/placement.sh"

export RAY_HEAD_IP="$(hostname -I | awk '{print $1}')"
export RAY_DONE_FLAG="/root/miles/experiments/outputs/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"
mkdir -p "${OUTPUT_DIR}/.ray"
rm -f "${OUTPUT_DIR}/.ray/${RUN_NAME}-${SLURM_JOB_ID}.done"

echo "run ${RUN_NAME}"
echo "checkpoints ${CKPT_PATH}"

srun \
    --nodes="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash /root/miles/experiments/search_r1/nq-hotpotqa/qwen3-4b-instruct-2507/train.sh
