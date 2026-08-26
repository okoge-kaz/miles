#!/bin/bash
#SBATCH --job-name=test-tool-call-env
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
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
export WANDB_MODE=offline

python3 -m pytest -q tests/fast/experiments/test_tool_call_env.py
python3 - <<"PY"
import asyncio
import collections
import json
from pathlib import Path

from experiments.src.reward_sets.tool_call import reward
from miles.utils.types import Sample

root = Path("/data/nemotron-agentic-tool-call")
train_path = root / "nemotron-agentic-tool-call-train.jsonl"
eval_path = root / "nemotron-agentic-tool-call-heldout.jsonl"
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def audit(path, *, eval_only):
    counts = collections.Counter()
    fingerprints = set()
    first = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            metadata = row["metadata"]
            assert metadata["verifier"] == "expert_action", (path, line_number)
            assert metadata["expected_kind"] == "function_call", (path, line_number)
            assert bool(metadata.get("eval_only")) is eval_only, (path, line_number)
            assert row["tools"], (path, line_number)
            fingerprint = metadata["split_fingerprint"]
            assert fingerprint not in fingerprints, (path, line_number, fingerprint)
            fingerprints.add(fingerprint)
            counts[metadata["source"]] += 1
            if first is None:
                first = row
    return counts, fingerprints, first


train_counts, train_ids, row = audit(train_path, eval_only=False)
eval_counts, eval_ids, _ = audit(eval_path, eval_only=True)
assert train_counts == {"conv-tooluse": 4700, "fncall-pivot": 4700}
assert eval_counts == {"conv-tooluse": 200, "fncall-pivot": 200}
assert train_ids.isdisjoint(eval_ids)
assert manifest["policy"] == "exact-function-call-only"
assert manifest["train_counts"] == dict(train_counts)
assert manifest["eval_counts"] == dict(eval_counts)
assert manifest["overlap"] == 0

expected = row["metadata"]["expected_action"]
arguments = expected["arguments"]
if isinstance(arguments, str):
    arguments = json.loads(arguments)
correct_response = "<tool_call>" + json.dumps(
    {"name": expected["name"], "arguments": arguments}, separators=(",", ":")
) + "</tool_call>"
correct = Sample(response=correct_response, metadata=row["metadata"])
wrong = Sample(
    response="<tool_call>{\"name\":\"definitely_wrong\",\"arguments\":{}}</tool_call>",
    metadata=row["metadata"],
)
assert asyncio.run(reward(None, correct)) == 1.0
assert asyncio.run(reward(None, wrong)) == 0.0
print(
    "Tool-call actual artifact: train=9400 heldout=400 balanced=true "
    "overlap=0 exact-reward-correct=1 wrong=0"
)
PY
'
