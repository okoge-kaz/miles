#!/bin/bash
#SBATCH --job-name=prep-performance-blends
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Build benchmark-safe transfer-training files. LiveCodeBench and GPQA rows are
# deliberately absent: they stay evaluation-only, while related public train
# sets provide code and knowledge/reasoning practice.

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

OUT=/data/nemotron-performance-transfer

python3 -m experiments.src.datasets.common.merge \
    --input /data/nemotron-rl-comp-coding/miles-train.jsonl \
            /data/skywork-or1-rl/miles-code-train.jsonl \
    --output "${OUT}/nemotron3-nano-competitive-code-train.jsonl" \
    --expected-rows 38028 \
    --require-verifiers python_code \
    --summary "${OUT}/nemotron3-nano-competitive-code-summary.json"

python3 -m experiments.src.datasets.common.merge \
    --input /data/nemotron-rl-mcqa/miles-train.jsonl \
            /data/nemotron-rl-reasoninggym/miles-train.jsonl \
    --output "${OUT}/nemotron3-nano-knowledge-mcqa-reasoning-gym-train.jsonl" \
    --expected-rows 632020 \
    --require-verifiers mcqa_regex reasoning_gym \
    --summary "${OUT}/nemotron3-nano-knowledge-mcqa-reasoning-gym-summary.json"

python3 -m experiments.src.datasets.common.audit \
    --input "${OUT}/nemotron3-nano-competitive-code-train.jsonl" \
    --expected-rows 38028 --require-verifiers python_code \
    --output "${OUT}/nemotron3-nano-competitive-code-audit.json"
python3 -m experiments.src.datasets.common.audit \
    --input "${OUT}/nemotron3-nano-knowledge-mcqa-reasoning-gym-train.jsonl" \
    --expected-rows 632020 --require-verifiers mcqa_regex reasoning_gym \
    --output "${OUT}/nemotron3-nano-knowledge-mcqa-reasoning-gym-audit.json"
'
