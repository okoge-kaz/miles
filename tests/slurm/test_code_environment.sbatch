#!/bin/bash
#SBATCH --job-name=test-code-env
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

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
