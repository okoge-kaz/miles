#!/bin/bash
#PBS -N test-nemotron-rewards
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
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

python3 -m compileall -q experiments/src/reward_sets
pytest -q \
    tests/fast/experiments/test_calendar_env.py \
    tests/fast/experiments/test_nemotron_components.py \
    tests/fast/experiments/test_reward_sets.py \
    tests/fast/experiments/test_workplace_env.py

python3 - <<"PY"
import asyncio
import json

from experiments.src.reward_sets.all_domains import blend_reward
from miles.utils.types import Sample

with open("/data/gpqa/gpqa-diamond-miles.jsonl", encoding="utf-8") as stream:
    row = json.loads(next(stream))
label = row["label"]
wrong = next(letter for letter in "ABCD" if letter != label)
assert row["metadata"]["eval_only"] is True
assert asyncio.run(
    blend_reward(None, Sample(response=f"Answer: {label}", label=label, metadata=row["metadata"]))
) == 1.0
assert asyncio.run(
    blend_reward(None, Sample(response=f"Answer: {wrong}", label=label, metadata=row["metadata"]))
) == 0.0
print("Real GPQA row/verifier: correct=1 wrong=0 eval_only=true")
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
PY

python3 - <<"PY"
from experiments.src.environments.competitive_programming.verifier import _run, run_tests

tests = {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]}
assert run_tests("a, b = map(int, input().split())\nprint(a + b)", tests, timeout=2) == 1.0
assert run_tests("print(0)", tests, timeout=2) == 0.0
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
