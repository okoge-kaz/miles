#!/bin/bash
#SBATCH --job-name=miles-prefill-sqsh-check
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${VALIDATION_IMAGE:?set VALIDATION_IMAGE to the immutable sqsh to check}"
[[ "${VALIDATION_IMAGE}" == /* ]] || {
    echo "VALIDATION_IMAGE must be absolute: ${VALIDATION_IMAGE}" >&2
    exit 1
}
[[ -r "${VALIDATION_IMAGE}" ]] || {
    echo "VALIDATION_IMAGE is not readable: ${VALIDATION_IMAGE}" >&2
    exit 1
}

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --mem=0 \
    --container-image="${VALIDATION_IMAGE}" \
    --container-mounts="${MILES_REPO}:/root/miles" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        cd /sgl-workspace/sglang
        git rev-parse HEAD
        python3 -m pytest -q test/registered/unit/managers/test_policy_weight_version.py
        cd /root/miles
        python3 train_async.py --help > /tmp/train-async-help.txt
        grep -F -- "--sglang-enable-response-weight-version-segments" /tmp/train-async-help.txt
    '

sha256sum "${VALIDATION_IMAGE}"
