#!/bin/bash
# NeMo-Gym mini_swe_agent_2 server on a CPU-only Slurm allocation.
#
# Runs the agent harness and every Apptainer sandbox here, so SWE-bench test
# execution never competes for CPU with the SGLang rollout workers on the GPU
# nodes. The trainer reaches this over the cluster network via NEMO_GYM_URL.
#
# Sizing follows the per-sandbox figures NVIDIA aligned on internally for ECS
# (#swdl-nemo-gym-core-devs, 2026-07-10): SWE-bench Verified = 4 vCPU / 8 GB /
# 50 GiB scratch per concurrent sandbox. CONCURRENCY below must stay under
# (allocated cpus / 4) or the sandboxes thrash.
#
#SBATCH --job-name=swe-rl-gym-server
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/gym-server-%j.out
#SBATCH --error=logs/gym-server-%j.err
### TODO: set these for your cluster (CPU partition, not a GPU one).
#SBATCH --account=<account>
#SBATCH --partition=<cpu-partition>

set -euo pipefail

# ── Configure ────────────────────────────────────────────────────────────────
EXP_DIR="${EXP_DIR:-$PWD}"                       # this experiment directory
GYM_DIR="${GYM_DIR:-/lustre/fsw/portfolios/coreai/users/$USER/Gym}"
SIF_LINK_DIR="${SIF_LINK_DIR:-/lustre/fsw/portfolios/coreai/users/$USER/sif_by_id}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?set CONTAINER_IMAGE to your enroot/squashfs image}"
PORT="${PORT:-12000}"
CONCURRENCY="${CONCURRENCY:-16}"
URL_FILE="${URL_FILE:-$EXP_DIR/.nemo_gym_url}"   # trainer reads the address here

# GOLDEN=1 makes the harness apply each task's gold patch instead of calling a
# policy: validates sandbox + image + SWE-bench grading with no model at all.
GOLDEN_ARG=""
if [[ "${GOLDEN:-0}" == "1" ]]; then
    GOLDEN_ARG="++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.run_golden=true"
    echo "GOLDEN MODE: every gold patch must score reward 1.0"
fi

mkdir -p logs "$(dirname "$URL_FILE")"

# The node name changes every job, so publish the address instead of hardcoding
# it. The trainer does: export NEMO_GYM_URL=$(cat .nemo_gym_url)
SERVER_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
echo "http://${SERVER_HOST}:${PORT}" > "$URL_FILE"
echo "NeMo-Gym will listen at $(cat "$URL_FILE")"

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="/lustre:/lustre,${HOME}:${HOME}" \
     --no-container-mount-home \
     bash -c "
set -euo pipefail

# ── Apptainer inside the enroot container ────────────────────────────────────
# This is the standard pattern on NVIDIA clusters ('We normally install
# Apptainer inside enroot container on clusters', #swdl-nemo-gym 2026-06-26).
# fuse/squashfuse must be usable on this node: an EOS run failed with
# 'squashfuse_ll exited: fuse: device not found'. Checked explicitly below so
# the job dies here rather than 400 episodes later.
if ! command -v apptainer >/dev/null; then
    apt-get update && apt-get install -y wget git build-essential gcc
    cd /tmp
    wget -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb
    apt-get install -y ./apptainer_1.3.1_amd64.deb
    ln -sf /usr/bin/apptainer /usr/bin/singularity
fi
apptainer --version
test -e /dev/fuse || { echo 'FATAL: /dev/fuse missing - Apptainer cannot mount SIFs here'; exit 1; }

# Smoke-mount one real image before starting the server.
SAMPLE_SIF=\$(ls ${SIF_LINK_DIR}/*.sif | head -n1)
echo \"Probing \$SAMPLE_SIF ...\"
apptainer exec --userns --no-mount cwd \"\$SAMPLE_SIF\" /bin/true
echo 'SIF mount OK'

# ── NeMo-Gym ─────────────────────────────────────────────────────────────────
cd ${GYM_DIR}
export PATH=\"\$HOME/.local/bin:\$PATH\"
source .venv/bin/activate

exec gym env start \
    --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
    --config nemo_gym/sandbox/providers/apptainer/configs/apptainer.yaml \
    --config ${EXP_DIR}/configs/apptainer_overrides.yaml \
    --model-type vllm_model \
    '++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.port=${PORT}' \
    '++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.concurrency=${CONCURRENCY}' \
    ${GOLDEN_ARG}
"
