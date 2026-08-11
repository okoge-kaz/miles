#!/bin/bash
#SBATCH --job-name=miles-prefill-sqsh
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Derive a reproducible SQSH from the currently deployed Miles image by replacing
# only the editable SGLang checkout. All enroot extraction/export work stays on a
# CPU node's local NVMe; only the final immutable image is written to Lustre.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "${REPO_ROOT}/experiments/env.sh"

BASE_IMAGE="${BASE_IMAGE:-${CONTAINER_DIR}/miles-latest-20260803.sqsh}"
SGLANG_REPO="${SGLANG_REPO:-okoge-kaz/sglang}"
SGLANG_BRANCH="${SGLANG_BRANCH:-miles-prefill-weight-version}"
SGLANG_COMMIT="${SGLANG_COMMIT:-23aaf6597dbbe47f9ceb6090e10372c844fed960}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${CONTAINER_DIR}/miles-prefill-weight-version-${SGLANG_COMMIT:0:9}.sqsh}"

test -f "${BASE_IMAGE}"
test ! -e "${OUTPUT_IMAGE}"

LOCAL_SCRATCH="/raid/enroot/tmp/${USER}-${SLURM_JOB_ID}"
export ENROOT_TEMP_PATH="${LOCAL_SCRATCH}/tmp"
export ENROOT_DATA_PATH="${LOCAL_SCRATCH}/data"
export ENROOT_CACHE_PATH="${CACHE_DIR}/enroot/cache"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_TEMP_PATH}" "${ENROOT_DATA_PATH}"

CONTAINER_NAME="miles-prefill-${USER}-${SLURM_JOB_ID}"
TEMP_OUTPUT="${OUTPUT_IMAGE}.partial-${SLURM_JOB_ID}"

cleanup() {
    enroot remove -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    rm -rf "${LOCAL_SCRATCH}"
    rm -f "${TEMP_OUTPUT}"
}
trap cleanup EXIT

echo "Creating ${CONTAINER_NAME} from ${BASE_IMAGE}"
time enroot create -n "${CONTAINER_NAME}" "${BASE_IMAGE}"

echo "Checking out SGLang ${SGLANG_REPO}@${SGLANG_COMMIT}"
time enroot start -r -w "${CONTAINER_NAME}" bash -lc "
    set -euo pipefail
    cd /sgl-workspace/sglang
    git fetch https://github.com/${SGLANG_REPO}.git ${SGLANG_BRANCH}
    git checkout -f ${SGLANG_COMMIT}
    test \"\$(git rev-parse HEAD)\" = ${SGLANG_COMMIT}
    python -c 'import sglang; print(sglang.__file__)'
"

echo "Exporting ${OUTPUT_IMAGE}"
time enroot export -o "${TEMP_OUTPUT}" "${CONTAINER_NAME}"
mv "${TEMP_OUTPUT}" "${OUTPUT_IMAGE}"

ls -lh "${OUTPUT_IMAGE}"
sha256sum "${OUTPUT_IMAGE}"
echo "done. Use it with: srun --container-image=${OUTPUT_IMAGE}"
