#!/bin/bash
#SBATCH --job-name=convert-blend
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert one Nemotron RL blend component to the prompt JSONL the recipes read.
#
#   sbatch -A coreai_horizon_dilations \
#     --export=ALL,DATASET=knowledge-mcqa,INPUT='/data/nemotron-rl-mcqa/data/*.parquet',\
# OUTPUT=/data/nemotron-rl-mcqa/knowledge-mcqa-miles.jsonl \
#     experiments/src/nemo_blends/convert.sbatch
#
# CPU only. Parquet inputs need pyarrow, which lives in the container rather
# than on the login node, so this runs inside the image even though it is pure
# data munging.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${DATASET:?set DATASET, e.g. knowledge-mcqa}"
: "${INPUT:?set INPUT, a file or glob under /data}"
: "${OUTPUT:?set OUTPUT, a path under /data}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc "
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
python3 -m experiments.src.nemo_blends.convert \
    --dataset ${DATASET} \
    --input ${INPUT} \
    --output ${OUTPUT} \
    ${LIMIT:+--limit ${LIMIT}}
wc -l ${OUTPUT}
head -1 ${OUTPUT} | cut -c1-400
"
