#!/bin/bash
#SBATCH --job-name=prep-tool-call-pivot-env
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Build balanced, disjoint function-call-only train/eval splits. Message actions
# are excluded because exact next-message semantics require a judge model.

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
mkdir -p /data/nemotron-agentic-tool-call-pivot
python3 -m experiments.src.datasets.nemotron.convert \
    --dataset conv-tooluse-pivot \
    --input /data/nemotron-rl-conv-tooluse-pivot/train.jsonl \
    --output /data/nemotron-rl-conv-tooluse-pivot/miles-train.jsonl \
    --summary /data/nemotron-rl-conv-tooluse-pivot/miles-conversion-summary.json
python3 -m experiments.src.datasets.nemotron.convert \
    --dataset fncall-pivot \
    --input /data/nemotron-rl-fncall-pivot/train.jsonl \
    --output /data/nemotron-rl-fncall-pivot/miles-train.jsonl \
    --summary /data/nemotron-rl-fncall-pivot/miles-conversion-summary.json
python3 -m experiments.src.datasets.tool_call_pivot.prepare \
    --source conv-tooluse-pivot=/data/nemotron-rl-conv-tooluse-pivot/miles-train.jsonl \
    --source fncall-pivot=/data/nemotron-rl-fncall-pivot/miles-train.jsonl \
    --train-output /data/nemotron-agentic-tool-call-pivot/nemotron-agentic-tool-call-pivot-train.jsonl \
    --eval-output /data/nemotron-agentic-tool-call-pivot/nemotron-agentic-tool-call-pivot-heldout.jsonl \
    --summary /data/nemotron-agentic-tool-call-pivot/manifest.json \
    --train-per-source 4700 \
    --eval-per-source 200
test "$(wc -l < /data/nemotron-agentic-tool-call-pivot/nemotron-agentic-tool-call-pivot-train.jsonl)" -eq 9400
test "$(wc -l < /data/nemotron-agentic-tool-call-pivot/nemotron-agentic-tool-call-pivot-heldout.jsonl)" -eq 400
python3 - <<"PY"
import json
from pathlib import Path

manifest = json.loads(
    Path("/data/nemotron-agentic-tool-call-pivot/manifest.json").read_text(encoding="utf-8")
)
assert manifest["policy"] == "static-single-turn-pivot-exact-function-call-only"
assert manifest["interaction_mode"] == "static_single_turn_pivot"
assert manifest["stateful_environment"] is False
assert manifest["train_counts"] == {"conv-tooluse-pivot": 4700, "fncall-pivot": 4700}
assert manifest["eval_counts"] == {"conv-tooluse-pivot": 200, "fncall-pivot": 200}
assert manifest["overlap"] == 0
PY
'
