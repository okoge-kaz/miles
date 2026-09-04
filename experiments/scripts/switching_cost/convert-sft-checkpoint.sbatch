#!/bin/bash
#SBATCH --job-name=miles-convert-switch-sft
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/convert/%x-%j.log

# Validate a transferred SFT HuggingFace export, then delegate the actual
# conversion to the shared HF -> Megatron torch_dist conversion job.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  MODEL_VARIANT=qwen3-8b \
    sbatch experiments/scripts/switching_cost/convert-sft-checkpoint.sbatch

  MODEL_VARIANT=qwen3-30b-a3b \
    sbatch experiments/scripts/switching_cost/convert-sft-checkpoint.sbatch

The job uses the batch partition with interactive QoS. It refuses to start
conversion until all safetensors shards listed by the HuggingFace index are
present and nonempty.
EOF
}

if [[ "${1:-}" == --help ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT="${MILES_REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)}}"
[[ -f "${REPO_ROOT}/experiments/setup/convert_checkpoint.sbatch" ]] || {
    echo "cannot find the miles checkout at ${REPO_ROOT}; submit from the repository root or set MILES_REPO" >&2
    exit 1
}

: "${MODEL_VARIANT:?set MODEL_VARIANT=qwen3-8b or qwen3-30b-a3b}"

export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface}"
case "${MODEL_VARIANT}" in
    qwen3-8b)
        export HF_MODEL_NAME="Qwen3-8B-Base/LR1.5e-5-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000"
        export MODEL_NAME="Qwen3-8B-Base-LR1.5e-5-Step4000"
        export MEGATRON_MODEL_TYPE=qwen3-8B
        ;;
    qwen3-30b-a3b)
        export HF_MODEL_NAME="Qwen3-30B-A3B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP2-EP8-PACK1-standard-cp-STEPS4000"
        export MODEL_NAME="Qwen3-30B-A3B-Base-LR2.0e-5-Step4000"
        export MEGATRON_MODEL_TYPE=qwen3-30B-A3B
        ;;
    *)
        echo "unsupported MODEL_VARIANT=${MODEL_VARIANT}" >&2
        usage >&2
        exit 1
        ;;
esac

HF_CHECKPOINT="${HF_CKPT_DIR%/}/${HF_MODEL_NAME}"
python3 - "${HF_CHECKPOINT}" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
config_path = checkpoint / "config.json"
single_weights_path = checkpoint / "model.safetensors"
index_path = checkpoint / "model.safetensors.index.json"

if not config_path.is_file() or config_path.stat().st_size == 0:
    raise SystemExit(f"missing or empty HuggingFace config: {config_path}")

with config_path.open(encoding="utf-8") as config_file:
    json.load(config_file)

if single_weights_path.is_file() and single_weights_path.stat().st_size > 0:
    print(f"validated complete single-file checkpoint: {single_weights_path}")
    raise SystemExit(0)

if not index_path.is_file() or index_path.stat().st_size == 0:
    raise SystemExit(f"missing or empty safetensors index: {index_path}")

with index_path.open(encoding="utf-8") as index_file:
    index = json.load(index_file)

shard_names = sorted(set(index.get("weight_map", {}).values()))
if not shard_names:
    raise SystemExit(f"safetensors index has an empty weight_map: {index_path}")

missing_shards = []
for shard_name in shard_names:
    shard_path = checkpoint / shard_name
    if not shard_path.is_file() or shard_path.stat().st_size == 0:
        missing_shards.append(shard_path)

if missing_shards:
    preview = "\n".join(f"  {path}" for path in missing_shards[:20])
    remainder = len(missing_shards) - 20
    suffix = f"\n  ... and {remainder} more" if remainder > 0 else ""
    raise SystemExit(
        f"checkpoint transfer is incomplete; {len(missing_shards)} shard(s) are missing or empty:\n"
        f"{preview}{suffix}"
    )

print(f"validated complete sharded checkpoint: {checkpoint} ({len(shard_names)} shards)")
PY

echo "source=${HF_CHECKPOINT}"
echo "destination=${MEGATRON_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/megatron}/${MODEL_NAME}_torch_dist"

export MILES_REPO="${REPO_ROOT}"
exec bash "${REPO_ROOT}/experiments/setup/convert_checkpoint.sbatch"
