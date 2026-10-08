#!/bin/bash
#PBS -N miles-v011-preflight
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=01:00:00
#PBS -j oe
# setup.sh validate supplies the log file with qsub -o.
set -euo pipefail
REPO_ROOT="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
[[ -r "${CONTAINER_IMAGE}" ]] || { echo "missing container: ${CONTAINER_IMAGE}" >&2; exit 2; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
export PYTHONPATH=/root/miles:/root/Megatron-LM
miles_container_exec --image "${CONTAINER_IMAGE}" --nv --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles --export="GPUS_PER_NODE,SGLANG_COMMIT,MEGATRON_COMMIT,PYTHONPATH" \
    python3 /root/miles/experiments/src/cluster.py
