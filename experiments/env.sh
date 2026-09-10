#!/bin/bash
# Shared configuration for every job under experiments/.
# Source this from an sbatch script; it resolves overrides and creates this user's output/cache directories.

# The miles checkout that gets mounted over /root/miles inside the container.
# Derived from this file's own location, so a second checkout mounts itself
# rather than whichever path happened to be baked in. A hardcoded default here is
# the worst kind of wrong: the job runs, but against someone else's code.
export MILES_REPO="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"

# --- Secrets and local overrides --------------------------------------------
# `$MILES_REPO/.env`, if present. $HOME is not mounted into the container
# (--no-container-mount-home), so anything a job needs has to be resolved here on
# the host and carried in by --export=ALL; .env is the one file that does that.
# It is git-ignored (.gitignore:193) and must stay that way -- this is a checkout
# of an open-source tree.
#
# Values already in the environment win, so a sweep can override a single key per
# run without editing the file:
#   TAU_USER_MODEL=other-model experiments/submit_training.sh ...
#
# See .env.example for the keys the recipes look for.
if [[ -f "${MILES_REPO}/.env" ]]; then
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        _line="${_line%%$'\r'}"
        # Skip blanks, comments, and anything that is not KEY=VALUE.
        [[ -z "${_line}" || "${_line}" =~ ^[[:space:]]*# ]] && continue
        [[ "${_line}" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        _key="${BASH_REMATCH[2]}"
        _val="${BASH_REMATCH[3]}"
        # Strip one layer of matching quotes, so both KEY=v and KEY="v" work.
        [[ "${_val}" == \"*\" || "${_val}" == \'*\' ]] && _val="${_val:1:${#_val}-2}"
        # Already set in the environment? leave it alone.
        [[ -n "${!_key:-}" ]] || export "${_key}=${_val}"
    done < "${MILES_REPO}/.env"
    unset _line _key _val
fi

# --- Slurm ------------------------------------------------------------------
export SLURM_ACCOUNT_NAME="${SLURM_ACCOUNT_NAME:-nemotron_sw_post}"
export GPU_PARTITION="${GPU_PARTITION:-batch}"       # batch 4h / batch_long 7d
export CPU_PARTITION="${CPU_PARTITION:-cpu}"         # CPU limits are selected with QoS
export GPU_QOS="${GPU_QOS:-normal}"
export INTERACTIVE_QOS="${INTERACTIVE_QOS:-interactive}"
export CPU_QOS="${CPU_QOS:-cpu-normal}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"           # GB200 x4, aarch64
export ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
export CONTAINER_ARCH="${CONTAINER_ARCH:-aarch64}"
export ENROOT_LOCAL_SCRATCH_ROOT="${ENROOT_LOCAL_SCRATCH_ROOT:-/tmp}"
# Managed listener range; see experiments/notes/cluster.md.
export MILES_WORKER_PORT_START="${MILES_WORKER_PORT_START:-2000}"
export MILES_WORKER_PORT_END="${MILES_WORKER_PORT_END:-8999}"

# --- Workspace on lustre ----------------------------------------------------
# Split by whether a job READS or WRITES the directory.
#
# Read-only assets are shared from one workspace so that a second person does not
# re-download 8 GB of weights or re-run the difficulty filter to get a
# byte-identical prompt file. They are world-readable and stay put.
export WS="${WS:-/lustre/fsw/portfolios/coreai/users/${USER}}"
export SHARED_WS="${SHARED_WS:-${WS}}"
export DATASET_DIR="${DATASET_DIR:-${SHARED_WS}/datasets}"           # prompt files, eval benchmarks
export HF_CKPT_DIR="${HF_CKPT_DIR:-${QWEN3_4B_BASE_HF_ROOT:-${SHARED_WS}/checkpoints/hf}}"  # HuggingFace-format weights
export MEGATRON_CKPT_DIR="${MEGATRON_CKPT_DIR:-${SHARED_WS}/checkpoints/megatron}"  # torch_dist weights
export CONTAINER_DIR="${CONTAINER_DIR:-${SHARED_WS}/container}"        # architecture-qualified SQSH images

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

export CKPT_ROOT="${WS}/checkpoints"
# Overridable so offline evaluation can read another user's run: point it at
# their checkpoints/training and the container mount follows.
export TRAIN_CKPT_DIR="${TRAIN_CKPT_DIR:-${CKPT_ROOT}/training}"
export CACHE_DIR="${CACHE_DIR:-${WS}/cache}"                       # compile / JIT caches

# Inference Hub is OpenAI-compatible, so litellm reaches it as provider "openai"
# with the base URL overridden rather than as a gemini/anthropic provider.
export NVIDIA_INFERENCE_BASE_URL="${NVIDIA_INFERENCE_BASE_URL:-https://inference-api.nvidia.com/v1}"

# Where sbatch logs land (stdout and stderr combined, one file per job).
# The #SBATCH --output directives use the relative path experiments/outputs/,
# so submit from the repo root. Git-ignored.
export OUTPUT_DIR="${OUTPUT_DIR:-${MILES_REPO}/experiments/outputs}"

# --- Container --------------------------------------------------------------
export DOCKER_IMAGE="${DOCKER_IMAGE:-radixark/miles:latest}"
# See notes/containers.md for architecture and prefill-provenance qualification.
export SQSH_IMAGE="${SQSH_IMAGE:-${CONTAINER_DIR}/miles-oci-aarch64.sqsh}"

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
#   torch inductor  /tmp/torchinductor_$USER   <- node-local, not persistent across jobs
#   triton          ~/.triton/cache            <- /root/.triton, not mounted
#   CUDA PTX JIT    ~/.nv/ComputeCache         <- /root/.nv, not mounted
#
# Left alone, every job recompiles from scratch and the inductor cache eats
# node-local space while doing it. The values are container paths; --export=ALL carries
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
# — on node-local /tmp, discarded with the job/container.
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
