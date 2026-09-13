#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=download-assets
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/lightning-setup/%x-%j.log

set -euo pipefail
if (( $# == 0 )); then
    echo "Usage: sbatch experiments/scripts/download.sbatch scripts.download.<module> [arguments...]" >&2
    exit 2
fi
case "$1" in
    scripts.download.checkpoints.*|scripts.download.datasets.training.*|scripts.download.datasets.eval.*) ;;
    *) echo "Expected a checkpoint, training dataset, or evaluation dataset download module" >&2; exit 2 ;;
esac
REPO_ROOT="${MILES_REPO:-${SLURM_SUBMIT_DIR}}"
export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/${USER}/checkpoints/hf}"
source "${REPO_ROOT}/experiments/env.sh"
srun --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" --container-writable \
    --no-container-mount-home --export=ALL \
    bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
cd /root/miles
python3 -u -m "$@"
' bash "$@"
