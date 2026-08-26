#!/bin/bash
# Shared configuration for every job under experiments/.
# Source this from an sbatch script; it defines paths only, no side effects.

# --- Slurm ------------------------------------------------------------------
export SLURM_ACCOUNT_NAME="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
export GPU_PARTITION="${GPU_PARTITION:-batch}"       # batch 4h / batch_long 7d
export GPU_QOS="${GPU_QOS:-normal}"
export INTERACTIVE_GPU_QOS="${INTERACTIVE_GPU_QOS:-interactive}"
export CPU_PARTITION="${CPU_PARTITION:-cpu}"         # cpu 7d
export CPU_QOS="${CPU_QOS:-cpu-normal}"
export INTERACTIVE_CPU_QOS="${INTERACTIVE_CPU_QOS:-cpu-interactive}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"           # aws-pdx pool0: B300 x8, 192 CPUs
export TRAINING_ATTENTION_BACKEND="${TRAINING_ATTENTION_BACKEND:-fused}"  # TE fused attention on B300

# --- Workspace on lustre ----------------------------------------------------
# Split by whether a job READS or WRITES the directory.
#
# Read-only assets are shared from one workspace so that a second person does not
# re-download 8 GB of weights or re-run the difficulty filter to get a
# byte-identical prompt file. They are world-readable and stay put.
export SHARED_WS="${SHARED_WS:-/lustre/fsw/portfolios/coreai/users/kfujii}"
export DATASET_DIR="${SHARED_WS}/datasets"           # prompt files, eval benchmarks
export HF_CKPT_DIR="${HF_CKPT_DIR:-${SHARED_WS}/checkpoints/huggingface}"  # HuggingFace-format weights
export MEGATRON_CKPT_DIR="${MEGATRON_CKPT_DIR:-${SHARED_WS}/checkpoints/megatron}"  # torch_dist weights
export CONTAINER_DIR="${CONTAINER_DIR:-${SHARED_WS}/containers}"

# Written directories are per-user. Sharing them would be worse than a
# permissions problem: CKPT_PATH is derived from the configuration, so two people
# running the same config would land on the same directory, and since --load and
# --save are the same path the second run would silently resume the first
# (see notes/off-policy-variables.md, "Run identity").
# Two people run arms of the same study and read each other's checkpoints for
# offline evaluation, so a private umask makes another user's run unreadable --
# and it fails late, inside the inference engine, as a FileNotFoundError on a
# safetensors shard whose config.json loaded fine.
umask 0022

export WS="${WS:-/lustre/fsw/portfolios/coreai/users/${USER}}"
export CKPT_ROOT="${WS}/checkpoints"
# Overridable so offline evaluation can read another user's run: point it at
# their checkpoints/training and the container mount follows.
export TRAIN_CKPT_DIR="${TRAIN_CKPT_DIR:-${CKPT_ROOT}/training}"
export CACHE_DIR="${WS}/cache"                       # compile / JIT caches

# The miles checkout that gets mounted over /root/miles inside the container.
# Derived from this file's own location, so a second checkout mounts itself
# rather than whichever path happened to be baked in. A hardcoded default here is
# the worst kind of wrong: the job runs, but against someone else's code.
export MILES_REPO="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"

# --- Secrets and local overrides --------------------------------------------
# This script deliberately does not read `.env` (or any other repository file
# containing credentials). Supply secrets through the process environment or a
# scheduler-supported secret mechanism. Canonical submit wrappers pass only
# explicit fixed-name allowlists; do not use `sbatch --export=ALL` for jobs that
# can reach credentials. `.env.example` documents recognized names only.

# Inference Hub is OpenAI-compatible, so litellm reaches it as provider "openai"
# with the base URL overridden rather than as a gemini/anthropic provider.
export NVIDIA_INFERENCE_BASE_URL="${NVIDIA_INFERENCE_BASE_URL:-https://inference-api.nvidia.com/v1}"

# Where sbatch logs land (stdout and stderr combined, one file per job).
# The #SBATCH --output directives use the relative path experiments/outputs/,
# so submit from the repo root. Git-ignored.
export OUTPUT_DIR="${MILES_REPO}/experiments/outputs"

# --- Container --------------------------------------------------------------
export DOCKER_IMAGE="${DOCKER_IMAGE:-radixark/miles:latest}"
# The prefill-staleness runs require SGLang's scheduler-authoritative policy
# provenance fields, which are present in this derived image. An explicitly set
# SQSH_IMAGE still takes precedence for historical reproduction.
export SQSH_IMAGE="${SQSH_IMAGE:-${CONTAINER_DIR}/miles-search-r1-b300-20260815.sqsh}"

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

# Only the directories this user owns. The shared ones are read-only to everyone
# else, and creating them here would mask a missing asset as a silent empty
# directory rather than failing where it is staged.
mkdir -p "${TRAIN_CKPT_DIR}" "${CACHE_DIR}" "${OUTPUT_DIR}"
for _shared in "${DATASET_DIR}" "${HF_CKPT_DIR}" "${MEGATRON_CKPT_DIR}" "${CONTAINER_DIR}"; do
    [[ -d "${_shared}" ]] || echo "env.sh: missing shared asset ${_shared} (see experiments/setup/)" >&2
done
unset _shared

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
# node RAM while doing it. Each recipe's fixed srun export list carries only the
# cache variables it uses. A corrupt entry after a killed job shows up as a JIT/JSONDecodeError
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
# key may be resolved here on the host, but only recipes that explicitly opt in
# should include WANDB_API_KEY in their fixed export list. Set it yourself to
# override; leave everything unset to disable wandb.
# WANDB_PROJECT is deliberately NOT defaulted here. Each recipe derives it from
# the experiment axis and the dataset (off-policy-<dataset>), which a default set
# at this level would silently shadow. Export it yourself to override.
if [[ "${WANDB_MODE:-online}" == offline || "${WANDB_MODE:-online}" == disabled ]]; then
    unset WANDB_API_KEY
elif [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
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
