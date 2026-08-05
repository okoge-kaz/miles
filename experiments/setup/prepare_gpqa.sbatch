#!/bin/bash
#SBATCH --job-name=prep-gpqa
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Download and convert GPQA. Needs HF_TOKEN in .env and the terms accepted by
# that token's account -- Idavidrein/gpqa is gated (see .env.example).
#
#   sbatch -A coreai_horizon_dilations experiments/setup/prepare_gpqa.sbatch
#
# Produces three splits under $DATASET_DIR/gpqa:
#   gpqa-diamond-miles.jsonl   198 questions, the split every model card reports
#   gpqa-main-miles.jsonl      448
#   gpqa-extended-miles.jsonl  546
#
# Eval only. 198 questions is far too few to train on, and the diamond split is
# the reported benchmark -- training on it would make the eval meaningless.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

# Check on the host, before burning a job step, and name the exact fix.
if [[ -z "${HF_TOKEN:-}" ]]; then
    cat >&2 <<'MSG'
HF_TOKEN is not set.

Idavidrein/gpqa is gated. Two steps, both against the same account:
  1. https://huggingface.co/datasets/Idavidrein/gpqa   -- accept the terms
  2. https://huggingface.co/settings/tokens            -- create a read token
Then put it in .env at the repo root:
  HF_TOKEN=hf_...
.env is git-ignored (.gitignore:193) and must stay that way.
MSG
    exit 1
fi

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH=/root/miles

hf download --repo-type dataset Idavidrein/gpqa --local-dir /data/gpqa

# The gate fails softly: without an accepted licence the download returns only
# README.md and exits 0. Check for the payload rather than the status.
if [[ ! -s /data/gpqa/gpqa_diamond.csv ]]; then
    echo "ERROR: gpqa_diamond.csv is missing after download."
    echo "  The token is valid but the terms are probably not accepted by that"
    echo "  account: open https://huggingface.co/datasets/Idavidrein/gpqa"
    exit 1
fi

for split in diamond main extended; do
    python3 -m experiments.src.nemo_blends.convert \
        --dataset gpqa \
        --input "/data/gpqa/gpqa_${split}.csv" \
        --output "/data/gpqa/gpqa-${split}-miles.jsonl"
done

echo
echo "answer-letter balance (should be near-uniform; a spike means the options"
echo "were not shuffled and the eval is guessable):"
python3 - <<PY
import collections, json
for split in ("diamond", "main", "extended"):
    labels = [json.loads(l)["label"] for l in open(f"/data/gpqa/gpqa-{split}-miles.jsonl")]
    print(f"  {split:9s} n={len(labels):4d}  {dict(sorted(collections.Counter(labels).items()))}")
PY
'

echo
echo "Add to experiments/configs/eval_knowledge.yaml:"
echo "    - name: gpqa_diamond"
echo "      path: /data/gpqa/gpqa-diamond-miles.jsonl"
