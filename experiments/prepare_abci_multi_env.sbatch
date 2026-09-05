#!/bin/bash
#PBS -N miles-v01-multi-data
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe

# Prepare the complete equal-domain blend from the training data staged on ABCI.
set -euo pipefail
cd "${PBS_O_WORKDIR:?submit from the Miles checkout}"
export WANDB_MODE=offline
source experiments/env.sh
miles_srun --image="${CONTAINER_IMAGE}" --bind="${CONTAINER_MOUNTS}" \
    --no-home --export=ALL bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
python3 -m experiments.src.datasets.nemotron.convert --dataset knowledge-mcqa \
    --input /data/nemotron-rl-mcqa/data/train-*.parquet \
    --output /data/nemotron-rl-mcqa/miles-train.jsonl
python3 -m experiments.src.datasets.nemotron.convert --dataset reasoning-gym \
    --input /data/nemotron-rl-reasoninggym/data/train.jsonl \
    --output /data/nemotron-rl-reasoninggym/miles-train.jsonl
python3 -m experiments.src.datasets.nemotron.convert --dataset competitive-coding \
    --input /data/nemotron-rl-comp-coding/opencodereasoning_filtered_25k_train.jsonl \
    --output /data/nemotron-rl-comp-coding/miles-train.jsonl
python3 -m experiments.src.datasets.nemotron.convert --dataset skywork-or1-code \
    --input /data/skywork-or1-rl/data/code-*.parquet \
    --output /data/skywork-or1-rl/miles-code-train.jsonl
'
bash experiments/setup/datasets/prepare_performance_transfer_blends.sbatch
export MATH_PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
bash experiments/setup/datasets/prepare_math_code_stem_blend.sbatch
bash experiments/setup/environments/prepare_reasoning_gym_dependencies.sbatch
