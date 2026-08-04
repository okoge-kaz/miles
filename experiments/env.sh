#!/bin/bash
# Shared configuration for every job under experiments/.
# Source this from an sbatch script; it defines paths only, no side effects.

# --- Slurm ------------------------------------------------------------------
export SLURM_ACCOUNT_NAME="coreai_horizon_dilations"
export GPU_PARTITION="${GPU_PARTITION:-batch}"       # batch 4h / batch_long 8h / batch_large_long 14d
export CPU_PARTITION="${CPU_PARTITION:-cpu}"         # cpu 1d / cpu_long 7d
export GPUS_PER_NODE=8                               # every pool0-* node is H100 x8, 128 CPUs

# --- Workspace on lustre ----------------------------------------------------
export WS="/lustre/fsw/portfolios/coreai/users/kfujii"
export DATASET_DIR="${WS}/datasets"
export CKPT_ROOT="${WS}/checkpoints"
export HF_CKPT_DIR="${CKPT_ROOT}/hf"                 # HuggingFace-format weights
export MEGATRON_CKPT_DIR="${CKPT_ROOT}/megatron"     # torch_dist (Megatron) weights
export TRAIN_CKPT_DIR="${CKPT_ROOT}/training"        # checkpoints written during training
export CONTAINER_DIR="${WS}/container"
export CACHE_DIR="${WS}/cache"

# The miles checkout that gets mounted over /root/miles inside the container.
export MILES_REPO="${MILES_REPO:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/src/miles}"

# Where sbatch logs land (stdout and stderr combined, one file per job).
# The #SBATCH --output directives use the relative path experiments/outputs/,
# so submit from the repo root. Git-ignored.
export OUTPUT_DIR="${MILES_REPO}/experiments/outputs"

# --- Container --------------------------------------------------------------
export DOCKER_IMAGE="${DOCKER_IMAGE:-radixark/miles:latest}"
# Every import writes a dated file (miles-latest-YYYYMMDD.sqsh) and repoints the
# `miles-latest.sqsh` symlink at it. Runs follow the symlink by default; pin a
# dated file explicitly (SQSH_IMAGE=.../miles-latest-20260803.sqsh) when a
# result has to stay reproducible against one image.
export SQSH_IMAGE="${SQSH_IMAGE:-${CONTAINER_DIR}/miles-latest.sqsh}"

# In-container paths. Keep these stable: every train.sh references them.
#   /root/miles       miles checkout (over the image's copy)
#   /root/Megatron-LM stays as shipped by the image
#   /data             datasets
#   /ckpt/{hf,megatron,training}
export CONTAINER_MOUNTS="\
${MILES_REPO}:/root/miles,\
${DATASET_DIR}:/data,\
${HF_CKPT_DIR}:/ckpt/hf,\
${MEGATRON_CKPT_DIR}:/ckpt/megatron,\
${TRAIN_CKPT_DIR}:/ckpt/training,\
${CACHE_DIR}:/root/.cache"

mkdir -p "${DATASET_DIR}" "${HF_CKPT_DIR}" "${MEGATRON_CKPT_DIR}" "${TRAIN_CKPT_DIR}" \
         "${CONTAINER_DIR}" "${CACHE_DIR}" "${OUTPUT_DIR}"

# --- Compile / JIT caches ---------------------------------------------------
# $HOME is /root in the container and is not mounted, so anything writing to
# ~/.cache already lands on CACHE_DIR and survives the job — that is how
# huggingface/, tvm-ffi/ (sgl_kernel) and deep_gemm/ got there. These are the
# ones that do NOT, because they default outside ~/.cache:
#
#   torch inductor  /tmp/torchinductor_$USER   <- /tmp is RAM-backed here
#   triton          ~/.triton/cache            <- /root/.triton, not mounted
#   CUDA PTX JIT    ~/.nv/ComputeCache         <- /root/.nv, not mounted
#
# Left alone, every job recompiles from scratch and the inductor cache eats
# node RAM while doing it. The values are container paths; --export=ALL carries
# them in. A corrupt entry after a killed job shows up as a JIT/JSONDecodeError
# (docs/faq.md:112) — delete the directory under $CACHE_DIR and rerun.
export TRITON_CACHE_DIR=/root/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
export TORCH_HOME=/root/.cache/torch
export CUDA_CACHE_PATH=/root/.cache/nv_compute
export CUDA_CACHE_MAXSIZE=4294967296
export VLLM_CACHE_ROOT=/root/.cache/vllm

# SGLang's DeepGEMM JIT cache. miles otherwise pins this to
# /tmp/sglang_deep_gemm/<worker>_rank_<n> for per-rank isolation
# (ray/rollout/server_group.py:107) — on RAM-backed /tmp, discarded every job.
# It reads the env var first, so setting it here wins; PER_PROCESS=1 is the
# supported way to keep the per-rank isolation under a shared directory
# (the TODO at server_group.py:105, and scripts/run_deepseek_v4.py:575).
export SGLANG_DG_CACHE_DIR=/root/.cache/deep_gemm
export SGLANG_DG_CACHE_DIR_PER_PROCESS=1

mkdir -p "${CACHE_DIR}"/{triton,torchinductor,torch,nv_compute,vllm,deep_gemm}

# --- Weights & Biases -------------------------------------------------------
# $HOME is not mounted into the container (--no-container-mount-home), so the
# key is resolved here on the host and carried in by --export=ALL. Set
# WANDB_API_KEY yourself to override; leave everything unset to disable wandb.
# WANDB_PROJECT is deliberately NOT defaulted here. Each recipe derives it from
# the experiment axis and the dataset (off-policy-<dataset>), which a default set
# at this level would silently shadow. Export it yourself to override.
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    _wandb_key=$(awk '
        /^[[:space:]]*machine[[:space:]]+api\.wandb\.ai/ { f = 1; next }
        f && /^[[:space:]]*machine[[:space:]]/           { f = 0 }
        f && /^[[:space:]]*password[[:space:]]/          { print $2; exit }
    ' "${HOME}/.netrc")
    if [[ -n "${_wandb_key}" ]]; then
        export WANDB_API_KEY="${_wandb_key}"
    fi
    unset _wandb_key
fi
