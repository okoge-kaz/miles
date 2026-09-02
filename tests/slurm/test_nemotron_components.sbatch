#!/bin/bash
#PBS -N test-nemotron-data
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
export CODE_EXEC_SANDBOX=bubblewrap
export WANDB_MODE=offline

python3 - <<"PY"
import datasets
import jsonschema
import pyarrow
import pytest

print("versions", datasets.__version__, jsonschema.__version__, pyarrow.__version__, pytest.__version__)
PY
python3 -m compileall -q \
    experiments/src/datasets \
    experiments/src/environments \
    experiments/src/evaluators \
    experiments/src/protocols \
    experiments/src/reward_sets
pytest -q \
    tests/fast/experiments/test_experiments_env.py \
    tests/fast/experiments/test_experiment_src_layout.py \
    tests/fast/experiments/test_notes_consistency.py \
    tests/fast/experiments/test_nemotron_components.py \
    tests/fast/experiments/test_calendar_env.py \
    tests/fast/experiments/test_domain_training_recipes.py \
    tests/fast/experiments/test_reward_sets.py \
    tests/fast/experiments/test_setup_layout.py \
    tests/fast/experiments/test_tau_bench.py \
    tests/fast/experiments/test_tool_call_pivot.py \
    tests/fast/experiments/test_workplace_env.py \
    tests/fast/utils/test_arguments.py \
    tests/fast/utils/test_wandb_compat.py

python3 - <<"PY"
import json

from experiments.src.environments.reasoning_gym.verifier import score_reasoning_gym_sample
from miles.utils.types import Sample

with open("/data/nemotron-rl-reasoninggym/miles-train.jsonl", encoding="utf-8") as stream:
    row = json.loads(next(stream))
correct = Sample(
    response="<answer>" + row["label"] + "</answer>",
    label=row["label"],
    metadata=row["metadata"],
)
wrong = Sample(response="<answer>definitely-wrong</answer>", label=row["label"], metadata=row["metadata"])
assert score_reasoning_gym_sample(correct) == 1.0
assert score_reasoning_gym_sample(wrong) == 0.0
print("Official Reasoning Gym scorer: correct=1 wrong=0")
PY

python3 - <<"PY"
import asyncio
import json

from experiments.src.reward_sets.all_domains import blend_reward
from miles.utils.types import Sample

with open("/data/gpqa/gpqa-diamond-miles.jsonl", encoding="utf-8") as stream:
    row = json.loads(next(stream))
assert row["metadata"]["eval_only"] is True
assert row["metadata"]["source_format"] == "nemo-skills-preprocessed"
label = row["label"]
wrong = next(letter for letter in "ABCD" if letter != label)
correct_sample = Sample(response=f"Answer: {label}", label=label, metadata=row["metadata"])
wrong_sample = Sample(response=f"Answer: {wrong}", label=label, metadata=row["metadata"])
assert asyncio.run(blend_reward(None, correct_sample)) == 1.0
assert asyncio.run(blend_reward(None, wrong_sample)) == 0.0
print("Real GPQA row/verifier: correct=1 wrong=0 eval_only=true")
PY

export LCB_ALLOW_LOCAL_EXECUTION=1
export LCB_REPO_PATH=/data/livecodebench-code
python3 - <<"PY"
import asyncio
import json

from experiments.src.evaluators.livecodebench import livecodebench_reward
from miles.utils.types import Sample

metadata = {
    "eval_only": True,
    "verifier": "livecodebench",
    "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]),
    "private_test_cases": json.dumps([{"input": "4 5\n", "output": "9\n", "testtype": "stdin"}]),
    "lcb_metadata": json.dumps({"func_name": None}),
}
correct = Sample(response="```python\na, b = map(int, input().split())\nprint(a + b)\n```", metadata=metadata)
wrong = Sample(response="```python\nprint(0)\n```", metadata=metadata)
rewards = asyncio.run(livecodebench_reward(None, [correct, wrong]))
assert rewards == [1.0, 0.0], rewards
print("Official LiveCodeBench evaluator: correct=1 wrong=0")
PY

python3 - <<"PY"
import json

from experiments.src.environments.instruction_following.verifier import (
    score_ifeval_sample,
    validate_ifeval_metadata,
)
from miles.utils.types import Sample

path = "/data/nemotron-rl-ifollow/miles-train.jsonl"
rows = 0
constraints = 0
fractional_row = None
with open(path, encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        row = json.loads(line)
        try:
            constraints += validate_ifeval_metadata(row["metadata"])
        except Exception as exc:
            raise AssertionError(f"{path}:{line_number}: incompatible IFEvalG metadata") from exc
        if row["metadata"].get("record_id") == 33964:
            fractional_row = row
        rows += 1
assert rows == 46391, rows
assert fractional_row is not None
metadata = fractional_row["metadata"]
half = Sample(response="<think>reasoning, ignored</think>fact telephone", metadata=metadata)
full = Sample(response="<think>reasoning, ignored</think>fact telephone\nP.P.S note", metadata=metadata)
assert score_ifeval_sample(half) == 0.5
assert score_ifeval_sample(full) == 1.0
print(f"Official IFEvalG construction: rows={rows} constraints={constraints}")
print("Official IFEvalG semantics: partial=0.5 complete=1 reasoning-section=ignored")
PY

python3 - <<"PY"
from experiments.src.environments.competitive_programming.verifier import _run, run_tests

tests = {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]}
assert run_tests("a, b = map(int, input().split())\nprint(a + b)", tests, timeout=2) == 1.0
assert run_tests("print(0)", tests, timeout=2) == 0.0
harness = {
    "entry_point": "Solution().add",
    "import_prefix": "from typing import *\n",
    "test_code": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
}
correct = "class Solution:\n    def add(self, left, right):\n        return left + right"
wrong = "class Solution:\n    def add(self, left, right):\n        return 0"
assert run_tests(correct, harness, timeout=2) == 1.0
assert run_tests(wrong, harness, timeout=2) == 0.0
probe = (
    "import socket\n"
    "socket.setdefaulttimeout(0.5)\n"
    "socket.create_connection((\"1.1.1.1\", 53))\n"
    "print(123)"
)
output, succeeded = _run(probe, "", timeout=2, memory_gb=1)
assert not succeeded and "123" not in output
print("Bubblewrap verifier and network isolation: ok")
PY
'
