#!/bin/bash
#SBATCH --job-name=prep-tool-call-env
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
mkdir -p /data/nemotron-agentic-tool-call
python3 -m experiments.src.datasets.tool_call.prepare \
    --source conv-tooluse=/data/nemotron-rl-conv-tooluse/miles-train.jsonl \
    --source fncall-pivot=/data/nemotron-rl-fncall-pivot/miles-train.jsonl \
    --train-output /data/nemotron-agentic-tool-call/nemotron-agentic-tool-call-train.jsonl \
    --eval-output /data/nemotron-agentic-tool-call/nemotron-agentic-tool-call-heldout.jsonl \
    --summary /data/nemotron-agentic-tool-call/manifest.json \
    --train-per-source 4700 \
    --eval-per-source 200
test "$(wc -l < /data/nemotron-agentic-tool-call/nemotron-agentic-tool-call-train.jsonl)" -eq 9400
test "$(wc -l < /data/nemotron-agentic-tool-call/nemotron-agentic-tool-call-heldout.jsonl)" -eq 400
python3 - <<"PY"
import json
from pathlib import Path

manifest = json.loads(
    Path("/data/nemotron-agentic-tool-call/manifest.json").read_text(encoding="utf-8")
)
assert manifest["policy"] == "exact-function-call-only"
assert manifest["train_counts"] == {"conv-tooluse": 4700, "fncall-pivot": 4700}
assert manifest["eval_counts"] == {"conv-tooluse": 200, "fncall-pivot": 200}
assert manifest["overlap"] == 0
PY
'
