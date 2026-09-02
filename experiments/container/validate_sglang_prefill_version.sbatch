#!/bin/bash
#PBS -N miles-prefill-sif-check
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

: "${VALIDATION_IMAGE:=${CONTAINER_IMAGE}}"
[[ "${VALIDATION_IMAGE}" == /* ]] || {
    echo "VALIDATION_IMAGE must be absolute: ${VALIDATION_IMAGE}" >&2
    exit 1
}
[[ -r "${VALIDATION_IMAGE}" ]] || {
    echo "VALIDATION_IMAGE is not readable: ${VALIDATION_IMAGE}" >&2
    exit 1
}

miles_srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gpus-per-task=1 \
    --image="${VALIDATION_IMAGE}" \
    --bind="${MILES_REPO}:/root/miles" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        cd /sgl-workspace/sglang
        test "$(git -c safe.directory=/sgl-workspace/sglang rev-parse HEAD)" \
            = "${SGLANG_COMMIT:?SGLANG_COMMIT is required}"
        python3 -m pytest -q -p no:cacheprovider \
            test/registered/unit/managers/test_policy_weight_version.py
        cd /root/miles
        python3 train_async.py --help > /tmp/train-async-help.txt
        grep -F -- "--sglang-enable-response-weight-version-segments" /tmp/train-async-help.txt
    '

sha256sum "${VALIDATION_IMAGE}"
