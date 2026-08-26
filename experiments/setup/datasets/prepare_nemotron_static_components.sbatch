#!/bin/bash
#SBATCH --job-name=prep-nemotron-static
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert every attached component with a local deterministic verifier. Judge,
# Lean, preference-model, and stateful-environment datasets remain staged in raw
# form and are documented separately rather than assigned a misleading reward.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

convert() {
    python3 -m experiments.src.datasets.nemotron.convert "$@"
}

convert --dataset dapo-math \
    --input /data/dapo-math-17k-byted/data/dapo-math-17k.parquet \
    --output /data/dapo-math-17k-byted/miles-train.jsonl
convert --dataset knowledge-mcqa \
    --input /data/nemotron-rl-mcqa/data/train-*.parquet \
    --output /data/nemotron-rl-mcqa/miles-train.jsonl
convert --dataset knowledge-mcqa \
    --input /data/nemotron-rl-mcqa/data/validation-*.parquet \
    --output /data/nemotron-rl-mcqa/miles-validation.jsonl
convert --dataset reasoning-gym \
    --input /data/nemotron-rl-reasoninggym/data/train.jsonl \
    --output /data/nemotron-rl-reasoninggym/miles-train.jsonl
convert --dataset structured-outputs \
    --input /data/nemotron-rl-ifollow-struct/structured_outputs_251027_nano_v3_sdg_json_train.jsonl \
    --output /data/nemotron-rl-ifollow-struct/miles-train.jsonl
convert --dataset structured-outputs \
    --input /data/nemotron-rl-ifollow-struct/structured_outputs_251027_nano_v3_sdg_json_val.jsonl \
    --output /data/nemotron-rl-ifollow-struct/miles-validation.jsonl
convert --dataset instruction-following \
    --input /data/nemotron-rl-ifollow/instruction_following.jsonl \
    --output /data/nemotron-rl-ifollow/miles-train.jsonl
convert --dataset conv-tooluse \
    --input /data/nemotron-rl-conv-tooluse/train.jsonl \
    --output /data/nemotron-rl-conv-tooluse/miles-train.jsonl
convert --dataset fncall-pivot \
    --input /data/nemotron-rl-fncall-pivot/train.jsonl \
    --output /data/nemotron-rl-fncall-pivot/miles-train.jsonl
convert --dataset swe-pivot \
    --input /data/nemotron-rl-swe-pivot/train.jsonl \
    --output /data/nemotron-rl-swe-pivot/miles-pivot-train.jsonl
convert --dataset competitive-coding \
    --input /data/nemotron-rl-comp-coding/opencodereasoning_filtered_25k_train.jsonl \
    --output /data/nemotron-rl-comp-coding/miles-train.jsonl
convert --dataset skywork-or1-math \
    --input /data/skywork-or1-rl/data/math-*.parquet \
    --output /data/skywork-or1-rl/miles-math-train.jsonl
convert --dataset skywork-or1-code \
    --input /data/skywork-or1-rl/data/code-*.parquet \
    --output /data/skywork-or1-rl/miles-code-train.jsonl

test "$(wc -l < /data/dapo-math-17k-byted/miles-train.jsonl)" -eq 1791700
test "$(wc -l < /data/nemotron-rl-reasoninggym/miles-train.jsonl)" -eq 15000
test "$(wc -l < /data/skywork-or1-rl/miles-math-train.jsonl)" -eq 105045
test "$(wc -l < /data/skywork-or1-rl/miles-code-train.jsonl)" -eq 14057
'
