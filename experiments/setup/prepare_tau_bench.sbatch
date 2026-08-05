#!/bin/bash
#SBATCH --job-name=prep-tau-bench
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Install tau-bench and emit the task files the recipe trains and evaluates on.
#
#   sbatch -A coreai_horizon_dilations experiments/setup/prepare_tau_bench.sbatch
#
# tau-bench is not on PyPI under a version miles pins, so it is cloned at the
# branch the miles example was written against. Its domain data (retail and
# airline products, orders, users, flights) ships inside the package as Python
# and JSON -- there is no database server to stand up, which is what makes this
# runnable here at all.
#
# What needs an external service is the *user simulator*: tau-bench drives it
# through litellm at rollout time, not here. This job is offline.
#
# Writes into $DATASET_DIR/tau-bench, and installs into the container's
# site-packages -- which is ephemeral, so the recipe reinstalls per job. The
# clone is cached on lustre so only the pip step repeats.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

TAU_DIR="${DATASET_DIR}/tau-bench"
TAU_SRC="${WS}/src/tau-bench"
mkdir -p "${TAU_DIR}" "$(dirname "${TAU_SRC}")"

if [[ -d "${TAU_SRC}/.git" ]]; then
    echo "skip clone: ${TAU_SRC} exists"
else
    # The branch the miles example targets; main has since changed the env API.
    git clone --branch feature/litellm-retry https://github.com/JD-ETH/tau-bench.git "${TAU_SRC}"
fi

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},${TAU_SRC}:/root/tau-bench" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
cd /root/tau-bench
pip install --no-input --quiet -e . --no-deps
pip install --no-input --quiet litellm

cd /root/miles/examples/experimental/tau-bench
# tau1_mock.py walks tau-bench own task lists and writes one row per task.
python3 tau1_mock.py --local_dir /root/tau-bench/

echo "--- files produced ---"
find /root/tau-bench -maxdepth 2 -name "*.jsonl" -newermt "-1 hour" -printf "  %8s %p\n"
'

echo
echo "copy the produced task files into ${TAU_DIR} and confirm:"
echo "  tau1_train.jsonl  tau1_retail_test.jsonl  tau1_airline_test.jsonl"
ls -la "${TAU_DIR}"
