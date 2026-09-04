#!/bin/bash
#SBATCH --job-name=test-transfer-input
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

# Exercise the exact Miles Dataset/chat-template path on both benchmark-safe
# transfer-training blends. Submit after prepare_performance_transfer_blends.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
export CODE_EXEC_SANDBOX=bubblewrap
export WANDB_MODE=offline

python3 - <<"PY"
import asyncio
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from experiments.src.reward_sets.all_domains import blend_reward, build_preflight_probes
from miles.utils.data import Dataset

tokenizer_dir = Path("/ckpt/hf/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
paths = (
    "/data/nemotron-performance-transfer/nemotron3-nano-competitive-code-train.jsonl@[0:2]",
    "/data/nemotron-performance-transfer/nemotron3-nano-knowledge-mcqa-reasoning-gym-train.jsonl@[0:2]",
)

args = SimpleNamespace()
for path in paths:
    dataset = Dataset(
        path,
        tokenizer=tokenizer,
        processor=None,
        max_length=None,
        prompt_key="prompt",
        label_key="label",
        tool_key="tools",
        apply_chat_template=True,
    )
    assert len(dataset.origin_samples) == 2
    assert all(sample.prompt and sample.label is not None for sample in dataset.origin_samples)
    tested = 0
    for sample in dataset.origin_samples:
        probes = build_preflight_probes(sample.label, sample.metadata)
        if probes is None:
            continue
        correct, wrong = probes
        sample.response = correct
        good = asyncio.run(blend_reward(args, sample))
        sample.response = wrong
        bad = asyncio.run(blend_reward(args, sample))
        assert good == 1.0 and bad == 0.0, (path, sample.metadata["verifier"], good, bad)
        tested += 1
    print(f"Miles transfer input: {path} rows=2 verifier-probes={tested}")
print("Miles performance-transfer Dataset/chat-template validation: ok")
PY
'
