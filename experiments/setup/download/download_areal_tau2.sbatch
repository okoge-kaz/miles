#!/bin/bash
#SBATCH --job-name=dl-areal-tau2-rl
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log
#SBATCH --export=NIL

# Fetch only the AReaL RL task file and its nine DB snapshots. The 33,531-row
# SFT file is intentionally excluded from this RL-only training path.

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="${USER:-$(id -un)}"
WANDB_MODE=disabled
export PATH USER WANDB_MODE

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly HF_REPO=inclusionAI/AReaL-tau2-data
readonly HF_REVISION=1322eae337f836fe0e19bae14dab1eefc26bc983
readonly LOCAL_NAME=areal-tau2-data
export HF_REPO HF_REVISION LOCAL_NAME

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=USER,WANDB_MODE,HF_REPO,HF_REVISION,LOCAL_NAME \
    bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
target="/data/${LOCAL_NAME}"
hf download \
    --repo-type dataset \
    --revision "${HF_REVISION}" \
    --include tau2_rl_train.jsonl \
    --include "tau2_rl_database/*" \
    "${HF_REPO}" \
    --local-dir "${target}"

cd "${target}"
sha256sum --check /root/miles/experiments/setup/manifests/areal_tau2_rl.sha256
test "$(wc -l < tau2_rl_train.jsonl)" -eq 1982
test "$(find tau2_rl_database -maxdepth 1 -type f | wc -l)" -eq 9
printf "repo=%s\nrevision=%s\nscope=%s\n" \
    "${HF_REPO}" \
    "${HF_REVISION}" \
    rl-tasks-and-databases-only > MILES_SOURCE_PROVENANCE
du -sh "${target}"
'
