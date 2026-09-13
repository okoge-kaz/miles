#!/bin/bash
#SBATCH --account=nemotron_sw_post
#SBATCH --job-name=convert-lightning-sft6000
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/lightning-sft-convert/%x-%j.log

set -euo pipefail
REPO_ROOT="${MILES_REPO:-${SLURM_SUBMIT_DIR}}"
: "${SFT_SOURCE:=/lustre/fsw/portfolios/llmservice/users/venkats/nemo-evaluator-rundirs/nano_v35_sft/conversions/upsampled-iter6000/hf}"
: "${MODEL_NAME:=Nemotron-3.5-Lightning-30B-A3B-SFT-upsampled-iter6000}"
: "${HF_MODEL_NAME:=${MODEL_NAME}-policy}"
export SFT_SOURCE MODEL_NAME HF_MODEL_NAME
export HF_CKPT_DIR="${HF_CKPT_DIR:-/lustre/fsw/portfolios/coreai/users/${USER}/checkpoints/hf}"
export SQSH_IMAGE="${SQSH_IMAGE:-/lustre/fsw/portfolios/coreai/users/${USER}/container/miles-oci-aarch64-7035384.sqsh}"
source "${REPO_ROOT}/experiments/env.sh"
[[ -r "${SFT_SOURCE}/model.safetensors.index.json" ]]
srun --nodes=1 --ntasks=1 --mem=0 --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${SFT_SOURCE}:/source/hf:ro" \
    --container-name="lightning-sft-convert-${SLURM_JOB_ID}" \
    --container-writable --no-container-mount-home --export=ALL bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles:/root/Megatron-LM
export HF_HOME=/root/.cache/huggingface
export CUDA_DEVICE_MAX_CONNECTIONS=1
python3 -u -m experiments.src.checkpoints.lightning_sft \
    --source /source/hf --source-id "${SFT_SOURCE}" --destination "/ckpt/hf/${HF_MODEL_NAME}"
target="/ckpt/megatron/${MODEL_NAME}_torch_dist"
if [[ -e "${target}" ]]; then
    echo "Destination already exists; inspect it before another conversion: ${target}" >&2
    exit 2
fi
staging="${target}.converting-${SLURM_JOB_ID}"
[[ ! -e "${staging}" ]]
read -r -a MODEL_ARGS <<< "$(python -m miles.utils.external_utils.model_args_utils nemotron-3-nano-30b-a3b)"
torchrun --standalone --nproc-per-node=1 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --megatron-to-hf-mode bridge --moe-token-dispatcher-type alltoall \
    --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}" --save "${staging}"
python3 tests/manual/check_lightning_assets.py \
    --hf-checkpoint "/ckpt/hf/${HF_MODEL_NAME}" --megatron-checkpoint "${staging}" \
    --prompt-data /data/dapo-math-17k/dapo-math-17k.jsonl
cp "/ckpt/hf/${HF_MODEL_NAME}/staging_manifest.json" "${staging}/source_manifest.json"
mv "${staging}" "${target}"
echo "SFT conversion complete: ${target}"
'
