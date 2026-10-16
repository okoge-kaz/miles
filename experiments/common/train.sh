#!/bin/bash
# Container-side entry: configure the experiment, form Ray, run the driver.
set -euo pipefail
train_script="${1:?usage: train.sh <recipe train.sh>}"
export MILES_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${MILES_REPO}"
source "${train_script}"
source "${MILES_REPO}/experiments/common/run_identity.sh"
export PYTHONPATH="${MILES_REPO}:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
# Exclusive PBS nodes expose all GPUs; use the numeric IDs Miles expects.
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$(( GPUS_PER_NODE - 1 ))")"
nvlink_count="$(nvidia-smi topo -m 2>/dev/null | awk '{count += gsub(/NV[0-9]+/, "")} END {print count+0}')"
export NCCL_NVLS_ENABLE="$(( nvlink_count > 0 ? 1 : 0 ))"
if declare -F miles_recipe_preflight >/dev/null; then
    miles_recipe_preflight
fi
source "${MILES_REPO}/experiments/common/ray_cluster.sh"
python3 -m experiments.common.training
