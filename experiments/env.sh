#!/bin/bash

umask 0022

# --- PBS --------------------------------------------------------------------
export PBS_PROJECT="${PBS_PROJECT:-}"
export PBS_GPU_QUEUE="${PBS_GPU_QUEUE:-R9920261300}"
export PBS_GPU_RESOURCE_TYPE="${PBS_GPU_RESOURCE_TYPE:-rt_HF}"

export PBS_CPU_QUEUE="${PBS_CPU_QUEUE:-${PBS_GPU_QUEUE}}"
export PBS_CPU_RESOURCE_TYPE="${PBS_CPU_RESOURCE_TYPE:-rt_HC}"
export PBS_GPU_CPUS_PER_NODE="${PBS_GPU_CPUS_PER_NODE:-192}"
export PBS_CPU_CPUS_PER_NODE="${PBS_CPU_CPUS_PER_NODE:-32}"

export ABCI_HPCX_MODULE="${ABCI_HPCX_MODULE:-hpcx/2.20}"
export ABCI_PYTHON_MODULE="${ABCI_PYTHON_MODULE:-python/3.12/3.12.9}"

export PBS_DEFAULT_WALLTIME="${PBS_DEFAULT_WALLTIME:-24:00:00}"
export PBS_CONTAINER_WALLTIME="${PBS_CONTAINER_WALLTIME:-00:30:00}"
export PBS_PREP_WALLTIME="${PBS_PREP_WALLTIME:-08:00:00}"
export PBS_DOWNLOAD_WALLTIME="${PBS_DOWNLOAD_WALLTIME:-24:00:00}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# --- Workspace --------------------------------------------------------------
export MILES_WORKSPACE_ROOT="${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}"

export SHARED_WS="${SHARED_WS:-${MILES_WORKSPACE_ROOT}}"
export WS="${WS:-${MILES_WORKSPACE_ROOT}}"

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CKPT_ROOT:-${MILES_WORKSPACE_ROOT}/checkpoints}}"
export CKPT_ROOT="${CHECKPOINT_ROOT}"
export HF_CKPT_DIR="${HF_CKPT_DIR:-${CHECKPOINT_ROOT}/hf}"
export MEGATRON_CKPT_DIR="${MEGATRON_CKPT_DIR:-${CHECKPOINT_ROOT}/megatron}"
export TRAIN_CKPT_DIR="${TRAIN_CKPT_DIR:-${CHECKPOINT_ROOT}/training}"

export DATASET_ROOT="${DATASET_ROOT:-${MILES_WORKSPACE_ROOT}/datasets}"
export PRETRAIN_DATASET_DIR="${PRETRAIN_DATASET_DIR:-${DATASET_ROOT}/pre-train}"
export RL_DATASET_DIR="${RL_DATASET_DIR:-${DATASET_DIR:-${DATASET_ROOT}/rl}}"
export SFT_DATASET_DIR="${SFT_DATASET_DIR:-${DATASET_ROOT}/sft}"
export DATASET_DIR="${RL_DATASET_DIR}"

export CONTAINER_DIR="${CONTAINER_DIR:-${MILES_WORKSPACE_ROOT}/containers}"
export CACHE_DIR="${CACHE_DIR:-${MILES_WORKSPACE_ROOT}/cache}"

# The miles checkout that gets mounted over /root/miles inside the container.
# Derived from this file's own location, so a second checkout mounts itself
# rather than whichever path happened to be baked in. A hardcoded default here is
# the worst kind of wrong: the job runs, but against someone else's code.
export MILES_REPO="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"

# --- Secrets and local overrides --------------------------------------------
# This script deliberately does not read `.env` (or any other repository file
# containing credentials). Supply secrets through the process environment or a
# scheduler-supported secret mechanism. Canonical submit wrappers pass only
# explicit fixed-name allowlists; do not use `qsub -V` for jobs that
# can reach credentials. `.env.example` documents recognized names only.

# Inference Hub is OpenAI-compatible, so litellm reaches it as provider "openai"
# with the base URL overridden rather than as a gemini/anthropic provider.
export NVIDIA_INFERENCE_BASE_URL="${NVIDIA_INFERENCE_BASE_URL:-https://inference-api.nvidia.com/v1}"

# Scheduler logs and local run manifests. Git-ignored.
export OUTPUT_DIR="${MILES_REPO}/experiments/outputs"

# --- Container --------------------------------------------------------------
# Dependency revisions come from this checkout's official release lock.
# Use the versioned release image: the CI image is a base whose dependencies
# CI checks out at runtime, so its baked commits do not necessarily match.
_miles_release_settings="$(python3 - "${MILES_REPO}/release-lock.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    release = json.load(handle)
print(release["release_branch"].removeprefix("release/"))
for name in ("sglang_commit", "megatron_commit"):
    print(release[name])
PY
)" || return 1
mapfile -t _miles_release_values <<< "${_miles_release_settings}"
export DOCKER_IMAGE="${DOCKER_IMAGE:-radixark/miles:${_miles_release_values[0]}}"
export SGLANG_COMMIT="${_miles_release_values[1]}"
export MEGATRON_COMMIT="${_miles_release_values[2]}"
unset _miles_release_settings _miles_release_values
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${CONTAINER_DIR}/miles-v0.1.1.sif}"

# Use the cluster-provided NCCL and InfiniBand stack. TCP remains available only
# as an explicit diagnostic control in validate_nccl_collective.sbatch.
export MILES_NCCL_TRANSPORT="${MILES_NCCL_TRANSPORT:-system}"

# Singularity's host-side image/layer cache. The build job deliberately
# overrides both cache and temp paths to its node-local scratch directory; this
# persistent default remains useful for non-build pulls and inspection.
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${CACHE_DIR}/singularity}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SINGULARITY_CACHEDIR}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"

# In-container paths shared with the ABCI v0.1 recipes.
#   /root/miles       miles checkout (over the image's copy)
#   /root/Megatron-LM stays as shipped by the image
#   /data             RL and evaluation datasets
#   /data/pre-train   pre-training datasets
#   /data/sft         supervised fine-tuning datasets
#   /ckpt/{hf,megatron,training}
#   /cache            persistent framework and compiler caches
export CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/cache}"
export CONTAINER_HOME="${CONTAINER_HOME:-${CONTAINER_CACHE_DIR}/home}"
export CONTAINER_MOUNTS="\
${MILES_REPO}:/root/miles,\
${MILES_REPO}/experiments/common/dotenv.disabled:/root/miles/.env:ro,\
${DATASET_DIR}:/data,\
${PRETRAIN_DATASET_DIR}:/data/pre-train,\
${SFT_DATASET_DIR}:/data/sft,\
${HF_CKPT_DIR}:/ckpt/hf,\
${MEGATRON_CKPT_DIR}:/ckpt/megatron,\
${TRAIN_CKPT_DIR}:/ckpt/training,\
${CACHE_DIR}:${CONTAINER_CACHE_DIR},\
${CACHE_DIR}:/root/.cache"

# Creating the fixed layout is idempotent; completion markers still distinguish
# staged assets from empty directories and partial downloads.
mkdir -p \
    "${HF_CKPT_DIR}" \
    "${MEGATRON_CKPT_DIR}" \
    "${TRAIN_CKPT_DIR}" \
    "${CONTAINER_DIR}" \
    "${PRETRAIN_DATASET_DIR}" \
    "${RL_DATASET_DIR}" \
    "${RL_DATASET_DIR}/pre-train" \
    "${RL_DATASET_DIR}/sft" \
    "${SFT_DATASET_DIR}" \
    "${CACHE_DIR}" \
    "${OUTPUT_DIR}"

# --- Compile / JIT caches ---------------------------------------------------
# Keep every large or expensive cache on shared storage. The duplicate
# /root/.cache bind above preserves older train.sh files while these variables
# direct current tools to the format-neutral /cache mount.
export XDG_CACHE_HOME="${CONTAINER_CACHE_DIR}/xdg"
export HF_HOME="${CONTAINER_CACHE_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TRITON_CACHE_DIR="${CONTAINER_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CONTAINER_CACHE_DIR}/torchinductor"
export TORCH_HOME="${CONTAINER_CACHE_DIR}/torch"
export TORCH_EXTENSIONS_DIR="${CONTAINER_CACHE_DIR}/torch_extensions"
export CUDA_CACHE_PATH="${CONTAINER_CACHE_DIR}/nv_compute"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
export VLLM_CACHE_ROOT="${CONTAINER_CACHE_DIR}/vllm"
export PIP_CACHE_DIR="${CONTAINER_CACHE_DIR}/pip"
export UV_CACHE_DIR="${CONTAINER_CACHE_DIR}/uv"
export NUMBA_CACHE_DIR="${CONTAINER_CACHE_DIR}/numba"

# SGLang's DeepGEMM JIT cache. miles otherwise pins this to
# /tmp/sglang_deep_gemm/<worker>_rank_<n> for per-rank isolation
# (ray/rollout/server_group.py:107) — on RAM-backed /tmp, discarded every job.
# It reads the env var first, so setting it here wins; PER_PROCESS=1 is the
# supported way to keep the per-rank isolation under a shared directory
# (the TODO at server_group.py:105, and scripts/run_deepseek_v4.py:575).
export SGLANG_CACHE_DIR="${CONTAINER_CACHE_DIR}/sglang"
export SGLANG_DG_CACHE_DIR="${CONTAINER_CACHE_DIR}/sglang/deep_gemm"
export SGLANG_DG_CACHE_DIR_PER_PROCESS="${SGLANG_DG_CACHE_DIR_PER_PROCESS:-1}"

mkdir -p \
    "${SINGULARITY_CACHEDIR}" \
    "${CACHE_DIR}/xdg" \
    "${CACHE_DIR}/home" \
    "${CACHE_DIR}/huggingface/hub" \
    "${CACHE_DIR}/huggingface/datasets" \
    "${CACHE_DIR}/huggingface/transformers" \
    "${CACHE_DIR}/triton" \
    "${CACHE_DIR}/torchinductor" \
    "${CACHE_DIR}/torch" \
    "${CACHE_DIR}/torch_extensions" \
    "${CACHE_DIR}/nv_compute" \
    "${CACHE_DIR}/vllm" \
    "${CACHE_DIR}/pip" \
    "${CACHE_DIR}/uv" \
    "${CACHE_DIR}/numba" \
    "${CACHE_DIR}/sglang/deep_gemm"

# These survive singularity exec --cleanenv and become the unprefixed names in
# the container. Explicit assignments also avoid depending on host shell policy.
for _cache_name in \
    XDG_CACHE_HOME HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE \
    HF_DATASETS_CACHE TRANSFORMERS_CACHE TRITON_CACHE_DIR \
    TORCHINDUCTOR_CACHE_DIR TORCH_HOME TORCH_EXTENSIONS_DIR CUDA_CACHE_PATH \
    CUDA_CACHE_MAXSIZE VLLM_CACHE_ROOT PIP_CACHE_DIR UV_CACHE_DIR \
    NUMBA_CACHE_DIR SGLANG_CACHE_DIR SGLANG_DG_CACHE_DIR SGLANG_DG_CACHE_DIR_PER_PROCESS; do
    printf -v "SINGULARITYENV_${_cache_name}" '%s' "${!_cache_name}"
    export "SINGULARITYENV_${_cache_name}"
done
unset _cache_name

# --no-home prevents accidental access to the submitter's host home. HOME is a
# protected variable in SingularityCE 4, so configure its supported home mount
# instead of trying to pass HOME through SINGULARITYENV_HOME.
export SINGULARITY_HOME="${CACHE_DIR}/home:${CONTAINER_HOME}"
export APPTAINER_HOME="${SINGULARITY_HOME}"

# --- Weights & Biases -------------------------------------------------------
# $HOME is not mounted into the container (--no-home), so the
# key is resolved on the compute host and passed through the private container
# environment. Training recipes default to online; select offline or disabled
# explicitly to opt out. An existing WANDB_API_KEY overrides the netrc lookup.
# WANDB_PROJECT is deliberately NOT defaulted here. Each recipe derives it from
# its experiment settings, which a default set
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

# --- Scheduler and container helpers ---------------------------------------
_miles_experiments_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=experiments/common/pbs.sh
source "${_miles_experiments_dir}/common/pbs.sh"
# shellcheck source=experiments/common/singularity.sh
source "${_miles_experiments_dir}/common/singularity.sh"
unset _miles_experiments_dir
