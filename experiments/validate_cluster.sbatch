#!/bin/bash
#PBS -N miles-cluster-preflight
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=01:00:00
#PBS -j oe

# Host, Singularity, CUDA, GPU, and native-extension preflight. This is intentionally
# read-only; SGLang generation and a Miles optimizer update are separate gates.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
export WANDB_MODE=offline

mkdir -p "${OUTPUT_DIR}/validation"
[[ -r "${CONTAINER_IMAGE}" ]] || {
    echo "missing CONTAINER_IMAGE=${CONTAINER_IMAGE}" >&2
    exit 1
}

echo "host=$(hostname) arch=$(uname -m)"
echo "image=${CONTAINER_IMAGE}"
echo "wandb_mode=${WANDB_MODE}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
nvidia-smi topo -m

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
python3 - <<"PY"
import json
from pathlib import Path

import flash_attn
import ray
import sglang
import torch
import transformer_engine

assert torch.cuda.is_available(), "CUDA is unavailable inside the container"
assert torch.cuda.device_count() == 8, torch.cuda.device_count()

device = torch.device("cuda:0")
left = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
right = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
result = left @ right
torch.cuda.synchronize()
assert torch.isfinite(result).all()

checkpoint_configs = {
    "qwen3-4b": Path("/ckpt/hf/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000/config.json"),
    "qwen3-8b": Path("/ckpt/hf/Qwen3-8B-Base/LR1.5e-5-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000/iter_0004000/config.json"),
    "qwen3-30b-a3b": Path("/ckpt/hf/Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000/iter_0004000/config.json"),
}
missing_configs = [str(path) for path in checkpoint_configs.values() if not path.is_file()]
assert not missing_configs, missing_configs
dataset = Path("/data/dapo-math-17k/dapo-math-17k.jsonl")
assert dataset.is_file(), dataset

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "device_count": torch.cuda.device_count(),
            "transformer_engine": transformer_engine.__version__,
            "flash_attn": flash_attn.__version__,
            "sglang": sglang.__version__,
            "ray": ray.__version__,
            "wandb_mode": __import__("os").environ.get("WANDB_MODE"),
            "bf16_matmul_mean": result.float().mean().item(),
            "checkpoint_configs": {name: str(path) for name, path in checkpoint_configs.items()},
            "dataset": str(dataset),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
'

echo "cluster preflight passed"
