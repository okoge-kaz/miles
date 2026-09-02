#!/bin/bash
#PBS -N miles-check-sglang-image
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Compatibility validation for the former Enroot derivation job. SGLang is
# pinned while the OCI image is built; an unprivileged Singularity job does not
# mutate a deployed SIF.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

expected_commit="${SGLANG_COMMIT:-f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce}"
[[ -r "${CONTAINER_IMAGE}" ]] || {
    echo "missing CONTAINER_IMAGE=${CONTAINER_IMAGE}; run experiments/setup.sh container --submit" >&2
    exit 1
}

singularity exec --no-eval --no-home "${CONTAINER_IMAGE}" bash -lc '
set -euo pipefail
test "$(git -c safe.directory=/sgl-workspace/sglang \
    -C /sgl-workspace/sglang rev-parse HEAD)" = "'"${expected_commit}"'"
python3 -m pytest -q -p no:cacheprovider \
    /sgl-workspace/sglang/test/registered/unit/managers/test_policy_weight_version.py
python3 -c "import sglang; print(sglang.__version__)"
'
echo "SGLang ${expected_commit} is present in ${CONTAINER_IMAGE}"
