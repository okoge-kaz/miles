#!/bin/bash
#SBATCH --job-name=prepare-agentic-tool-pivot
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log
#SBATCH --export=NIL

# Convert the pinned NVIDIA behavior-cloning pivots and reserve a deterministic
# held-out split. Only exact function-call actions enter RL; free-form message
# actions need a semantic judge and are reported but excluded.

set -euo pipefail

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
WANDB_MODE=offline
export PATH USER WANDB_MODE
source "${REPO_ROOT}/experiments/env.sh"

readonly SOURCE_ROOT="${DATASET_DIR}/nemotron-rl-conv-tooluse-pivot"
readonly SOURCE_PROVENANCE="${SOURCE_ROOT}/MILES_SOURCE_PROVENANCE"
readonly EXPECTED_REPO=nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1
readonly EXPECTED_REVISION=9643c8103d7bfbc2d7fc4d15991d6739c612ff58
readonly EXPECTED_SHA256=0909e9d7c24eee73ab0e2a9cf25bc462075cee6dd1a1f9502577a22c2a7a1f0a
[[ -s "${SOURCE_ROOT}/train.jsonl" && -s "${SOURCE_PROVENANCE}" ]] || {
    echo "pinned tool-use Pivot source is missing; run stage_swe_tool_rl_datasets.sh" >&2
    exit 1
}
grep -Fxq "repo=${EXPECTED_REPO}" "${SOURCE_PROVENANCE}"
grep -Fxq "revision=${EXPECTED_REVISION}" "${SOURCE_PROVENANCE}"
printf '%s  %s\n' "${EXPECTED_SHA256}" "${SOURCE_ROOT}/train.jsonl" \
    | sha256sum --check --status

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=NONE \
    env -i PATH="${PATH}" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
    bash --noprofile --norc -c '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
source_root=/data/nemotron-rl-conv-tooluse-pivot
output_root=/data/nemotron-agentic-conv-tooluse-pivot
mkdir -p "${output_root}"

python3 -m experiments.src.datasets.nemotron.convert \
    --dataset conv-tooluse-pivot \
    --input "${source_root}/train.jsonl" \
    --output "${source_root}/miles-train.jsonl" \
    --summary "${source_root}/miles-conversion-summary.json"

python3 -m experiments.src.datasets.tool_call_pivot.prepare \
    --source "conv-tooluse-pivot=${source_root}/miles-train.jsonl" \
    --train-output "${output_root}/nemotron-agentic-conv-tooluse-pivot-train.jsonl" \
    --eval-output "${output_root}/nemotron-agentic-conv-tooluse-pivot-heldout.jsonl" \
    --summary "${output_root}/manifest.json" \
    --train-per-source all \
    --eval-per-source 2000

python3 - <<"PY"
import json
from pathlib import Path

source_root = Path("/data/nemotron-rl-conv-tooluse-pivot")
output_root = Path("/data/nemotron-agentic-conv-tooluse-pivot")
conversion = json.loads(
    (source_root / "miles-conversion-summary.json").read_text(encoding="utf-8")
)
manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
assert conversion["dataset"] == "conv-tooluse-pivot"
assert conversion["kept"] == 96968
assert conversion["skipped"] == 0
assert manifest["policy"] == "static-single-turn-pivot-exact-function-call-only"
assert manifest["interaction_mode"] == "static_single_turn_pivot"
assert manifest["stateful_environment"] is False
assert manifest["eval_counts"] == {"conv-tooluse-pivot": 2000}
assert manifest["train_counts"] == {"conv-tooluse-pivot": 63559}
assert manifest["audit"]["conv-tooluse-pivot"]["function_call"] == 65559
assert manifest["audit"]["conv-tooluse-pivot"]["message"] == 31409
assert manifest["audit"]["conv-tooluse-pivot"]["eligible"] == 65559
assert len(manifest["train_sha256"]) == 64
assert len(manifest["eval_sha256"]) == 64
assert manifest["overlap"] == 0
print(
    json.dumps(
        {
            "converted": conversion["kept"],
            "eligible_train": manifest["train_total"],
            "heldout": manifest["eval_total"],
            "excluded_ineligible": manifest["audit"]["conv-tooluse-pivot"].get(
                "ineligible", 0
            ),
        },
        sort_keys=True,
    )
)
PY
'
