#!/bin/bash
#PBS -N test-nemotron-input
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Exercise the actual Miles Dataset/chat-template path and verifier imports on a
# stratified sample. Submit with dependencies on Nano conversion and
# allenai/open-instruct staging.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap" \
    --writable-tmpfs \
    --no-home \
    --export=USER,WANDB_MODE=offline \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
export OPEN_INSTRUCT_PATH=/data/open-instruct
export CODE_EXEC_SANDBOX=bubblewrap
export WANDB_MODE=offline

NANO=/data/nemotron-3-nano-rl-training-blend
python3 -m experiments.src.datasets.common.audit \
    --input "${NANO}/miles-train-static.jsonl" \
    --expected-rows 83013 \
    --require-verifiers math mcqa_regex python_code json_schema ifeval_g \
    --sample-output "${NANO}/miles-static-validation.jsonl" \
    --samples-per-verifier 2 \
    --output "${NANO}/miles-audit.json"

python3 - <<"PY"
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from experiments.src.environments.instruction_following.verifier import _load_registry
from experiments.src.reward_sets.all_domains import blend_reward, build_preflight_probes
from miles.utils.data import Dataset

tokenizer_dir = Path("/ckpt/hf/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
dataset = Dataset(
    "/data/nemotron-3-nano-rl-training-blend/miles-static-validation.jsonl",
    tokenizer=tokenizer,
    processor=None,
    max_length=None,
    prompt_key="prompt",
    label_key="label",
    tool_key="tools",
    apply_chat_template=True,
)
assert len(dataset.origin_samples) == 10
assert all(isinstance(sample.prompt, str) and sample.prompt for sample in dataset.origin_samples)

registry = _load_registry()
instruction_ids = set()
with open("/data/nemotron-3-nano-rl-training-blend/miles-train-static.jsonl") as handle:
    for line in handle:
        row = json.loads(line)
        metadata = row["metadata"]
        if metadata["verifier"] == "ifeval_g":
            instruction_ids.update(metadata["instruction_id_list"])
missing = sorted(instruction_ids - registry.INSTRUCTION_DICT.keys())
assert not missing, f"IFEvalG registry misses {missing}"
print(f"IFEvalG coverage: {len(instruction_ids)}/{len(instruction_ids)} ids")

args = SimpleNamespace()
ifeval_probe = SimpleNamespace(
    response="A cat and a dog.",
    label="keywords:existence",
    metadata={
        "verifier": "ifeval_g",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["cat", "dog"]}],
        "prompt_text": "Include cat and dog.",
    },
)
good = asyncio.run(blend_reward(args, ifeval_probe))
ifeval_probe.response = "A cat."
bad = asyncio.run(blend_reward(args, ifeval_probe))
assert good == 1.0 and bad == 0.0, (good, bad)
print(f"preflight ifeval_g: correct={good} wrong={bad}")

for sample in dataset.origin_samples:
    verifier = sample.metadata["verifier"]
    probe = build_preflight_probes(sample.label, sample.metadata)
    if probe is None:
        continue
    correct, wrong = probe
    sample.response = correct
    good = asyncio.run(blend_reward(args, sample))
    sample.response = wrong
    bad = asyncio.run(blend_reward(args, sample))
    assert good == 1.0 and bad == 0.0, (verifier, good, bad)
    print(f"preflight {verifier}: correct={good} wrong={bad}")
print("Miles Dataset/chat template/verifier validation: ok")
PY
'
