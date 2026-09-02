#!/bin/bash
# Shared configuration for every job under experiments/.
# Source this from a scheduler wrapper before resolving assets or mounts.

# --- PBS --------------------------------------------------------------------
# Keep reusable PBS headers project-neutral. Operators may pass a project to a
# direct qsub command (for example, -P gai51740) without baking it into recipes.
# Queue and resource defaults remain overridable without repetition.
export PBS_GPU_QUEUE="${PBS_GPU_QUEUE:-R9920261300}"
export PBS_GPU_RESOURCE_TYPE="${PBS_GPU_RESOURCE_TYPE:-rt_HF}"
# CPU-only payloads also default to the project reservation. They request no
# GPUs and select its HC resource class. Override PBS_CPU_QUEUE for an
# intentional normal-queue submission; both defaults remain configurable.
export PBS_CPU_QUEUE="${PBS_CPU_QUEUE:-${PBS_GPU_QUEUE}}"
export PBS_CPU_RESOURCE_TYPE="${PBS_CPU_RESOURCE_TYPE:-rt_HC}"
export PBS_GPU_CPUS_PER_NODE="${PBS_GPU_CPUS_PER_NODE:-192}"
export PBS_CPU_CPUS_PER_NODE="${PBS_CPU_CPUS_PER_NODE:-32}"
export ABCI_HPCX_MODULE="${ABCI_HPCX_MODULE:-hpcx/2.20}"
export PBS_DEFAULT_WALLTIME="${PBS_DEFAULT_WALLTIME:-24:00:00}"
export PBS_CONTAINER_WALLTIME="${PBS_CONTAINER_WALLTIME:-00:30:00}"
export PBS_PREP_WALLTIME="${PBS_PREP_WALLTIME:-08:00:00}"
export PBS_DOWNLOAD_WALLTIME="${PBS_DOWNLOAD_WALLTIME:-24:00:00}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export TRAINING_ATTENTION_BACKEND="${TRAINING_ATTENTION_BACKEND:-fused}"

# --- Workspace --------------------------------------------------------------
# Change this one value when moving the experiments to another filesystem. The
# individual paths remain overridable for one-off reads of external checkpoints.
export MILES_WORKSPACE_ROOT="${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}"

# Compatibility names used by the older evaluation scripts. New code should use
# MILES_WORKSPACE_ROOT and the purpose-specific paths below.
export SHARED_WS="${SHARED_WS:-${MILES_WORKSPACE_ROOT}}"
export WS="${WS:-${MILES_WORKSPACE_ROOT}}"

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CKPT_ROOT:-${MILES_WORKSPACE_ROOT}/checkpoints}}"
export CKPT_ROOT="${CHECKPOINT_ROOT}"
export HF_CKPT_DIR="${HF_CKPT_DIR:-${CHECKPOINT_ROOT}/hf}"
export MEGATRON_CKPT_DIR="${MEGATRON_CKPT_DIR:-${CHECKPOINT_ROOT}/megatron}"
# Overridable so offline evaluation can read another user's training run.
export TRAIN_CKPT_DIR="${TRAIN_CKPT_DIR:-${CHECKPOINT_ROOT}/training}"

export DATASET_ROOT="${DATASET_ROOT:-${MILES_WORKSPACE_ROOT}/datasets}"
export PRETRAIN_DATASET_DIR="${PRETRAIN_DATASET_DIR:-${DATASET_ROOT}/pre-train}"
export RL_DATASET_DIR="${RL_DATASET_DIR:-${DATASET_DIR:-${DATASET_ROOT}/rl}}"
export SFT_DATASET_DIR="${SFT_DATASET_DIR:-${DATASET_ROOT}/sft}"
# Existing recipes use DATASET_DIR and the /data container path for RL/eval data.
export DATASET_DIR="${RL_DATASET_DIR}"

export CONTAINER_DIR="${CONTAINER_DIR:-${MILES_WORKSPACE_ROOT}/containers}"
export CACHE_DIR="${CACHE_DIR:-${MILES_WORKSPACE_ROOT}/cache}"

# Centralized defaults for the SFT policies referenced by the maintained recipes.
# *_HF_MODEL is relative to HF_CKPT_DIR and is therefore also the path seen
# below /ckpt/hf inside Singularity.
export QWEN3_4B_BASE_HF_RELATIVE_DIR="${QWEN3_4B_BASE_HF_RELATIVE_DIR:-Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}"
export QWEN3_4B_BASE_HF_MODEL="${QWEN3_4B_BASE_HF_MODEL:-${QWEN3_4B_BASE_HF_RELATIVE_DIR}/iter_0004000}"
export QWEN3_4B_BASE_HF_ROOT="${QWEN3_4B_BASE_HF_ROOT:-${HF_CKPT_DIR}/${QWEN3_4B_BASE_HF_RELATIVE_DIR}}"
export QWEN3_8B_BASE_HF_RELATIVE_DIR="${QWEN3_8B_BASE_HF_RELATIVE_DIR:-Qwen3-8B-Base/LR1.5e-5-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000}"
export QWEN3_8B_BASE_HF_MODEL="${QWEN3_8B_BASE_HF_MODEL:-${QWEN3_8B_BASE_HF_RELATIVE_DIR}/iter_0004000}"
export QWEN3_8B_BASE_HF_ROOT="${QWEN3_8B_BASE_HF_ROOT:-${HF_CKPT_DIR}/${QWEN3_8B_BASE_HF_RELATIVE_DIR}}"
export QWEN3_30B_A3B_BASE_HF_RELATIVE_DIR="${QWEN3_30B_A3B_BASE_HF_RELATIVE_DIR:-Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000}"
export QWEN3_30B_A3B_BASE_HF_MODEL="${QWEN3_30B_A3B_BASE_HF_MODEL:-${QWEN3_30B_A3B_BASE_HF_RELATIVE_DIR}/iter_0004000}"
export QWEN3_30B_A3B_BASE_HF_ROOT="${QWEN3_30B_A3B_BASE_HF_ROOT:-${HF_CKPT_DIR}/${QWEN3_30B_A3B_BASE_HF_RELATIVE_DIR}}"

# Checkpoints are keyed by configuration, so group-writable defaults could let
# two users silently resume the same optimizer state.
umask 0022

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
export DOCKER_IMAGE="${DOCKER_IMAGE:-radixark/miles:latest}"
# The upstream OCI tag can move independently of the Miles SGLang fork. The
# Singularity build therefore reapplies and verifies the scheduler-authoritative
# policy-provenance revision instead of trusting whichever checkout the tag has.
export SGLANG_REPO="${SGLANG_REPO:-okoge-kaz/sglang}"
export SGLANG_BRANCH="${SGLANG_BRANCH:-miles-staleness-weight-boundaries}"
export SGLANG_COMMIT="${SGLANG_COMMIT:-f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-${CONTAINER_DIR}/miles.sif}"
export ASYNC_CONTAINER_IMAGE_OVERRIDE="${ASYNC_CONTAINER_IMAGE_OVERRIDE:-${CONTAINER_IMAGE}}"

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

# In-container paths. Keep these stable: every train.sh references them.
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
    NUMBA_CACHE_DIR SGLANG_DG_CACHE_DIR SGLANG_DG_CACHE_DIR_PER_PROCESS; do
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

# --- Scheduler and container helpers ---------------------------------------
_miles_experiments_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=experiments/common/pbs.sh
source "${_miles_experiments_dir}/common/pbs.sh"
# shellcheck source=experiments/common/singularity.sh
source "${_miles_experiments_dir}/common/singularity.sh"
unset _miles_experiments_dir
