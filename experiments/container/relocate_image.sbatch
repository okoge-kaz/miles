#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=miles-relocate-image
#SBATCH --partition=cpu
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=experiments/outputs/maintenance/container-relocation/%x-%j.log

# Submit with --dependency=afterany:<consumer-job-ids> and a specification path.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
python3 experiments/container/relocate_image.py "${1:?pass the relocation specification}"
