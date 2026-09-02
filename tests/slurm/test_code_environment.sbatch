#!/bin/bash
#PBS -N test-code-env
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
export WANDB_MODE=offline
export CODE_EXEC_SANDBOX=bubblewrap

python3 -m compileall -q \
    experiments/src/environments/competitive_programming \
    experiments/src/reward_sets/code.py
python3 -m pytest -q \
    tests/fast/experiments/test_code_environment.py \
    tests/fast/experiments/test_code_training_recipe.py \
    tests/fast/utils/test_arguments.py

python3 - <<"PY"
from experiments.src.environments.competitive_programming.verifier import _run, run_tests

tests = {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]}
correct = "a, b = map(int, input().split())\nprint(a + b)"
wrong = "print(0)"
assert run_tests(correct, tests, timeout=2) == 1.0
assert run_tests(wrong, tests, timeout=2) == 0.0

probe = (
    "import socket\n"
    "socket.setdefaulttimeout(0.5)\n"
    "socket.create_connection((\"1.1.1.1\", 53))\n"
    "print(123)"
)
output, succeeded = _run(probe, "", timeout=2, memory_gb=1)
assert not succeeded and "123" not in output
print("Code Bubblewrap verifier: correct=1 wrong=0 network-isolated=true")
PY
'
