#!/bin/bash
#SBATCH --job-name=miles-prefill-sqsh
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Derive a reproducible SQSH from the currently deployed Miles image by replacing
# the editable SGLang checkout and installing the EFA userspace stack. All enroot
# extraction/export work stays on a CPU node's local NVMe; only the final
# immutable image is written to Lustre.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "${REPO_ROOT}/experiments/env.sh"

BASE_IMAGE="${BASE_IMAGE:-${CONTAINER_DIR}/miles-latest-20260803.sqsh}"
SGLANG_REPO="${SGLANG_REPO:-okoge-kaz/sglang}"
SGLANG_BRANCH="${SGLANG_BRANCH:-miles-staleness-weight-boundaries}"
SGLANG_COMMIT="${SGLANG_COMMIT:-f994b9aedfd0b1465dbb8f4e2a02eb789fc76dce}"
EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION:-1.49.0}"
EFA_INSTALLER_SHA256="${EFA_INSTALLER_SHA256:-cf2e9281a2328a243c76f911a490faed43ca0fecfe4733c25e34b2e92a32c309}"
EFA_INSTALL_SCRIPT="${REPO_ROOT}/docker/install_aws_efa.sh"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${CONTAINER_DIR}/miles-staleness-weight-boundaries-${SGLANG_COMMIT:0:9}-efa-${EFA_INSTALLER_VERSION}.sqsh}"

test -f "${BASE_IMAGE}"
test -r "${EFA_INSTALL_SCRIPT}"
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
    python -m pytest -q test/registered/unit/managers/test_policy_weight_version.py
"

echo "Installing AWS EFA userspace ${EFA_INSTALLER_VERSION}"
time enroot start \
    -r \
    -w \
    --mount "${EFA_INSTALL_SCRIPT}:/tmp/install_aws_efa.sh" \
    "${CONTAINER_NAME}" \
    env \
        EFA_INSTALLER_VERSION="${EFA_INSTALLER_VERSION}" \
        EFA_INSTALLER_SHA256="${EFA_INSTALLER_SHA256}" \
        bash /tmp/install_aws_efa.sh

echo "Validating named OFI plugin without enabling it globally"
time enroot start -r "${CONTAINER_NAME}" bash --noprofile --norc -c '
    set -euo pipefail
    test -s /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
    test -s /opt/amazon/ofi-nccl/lib/libnccl-tuner-ofi.so
    test ! -e /opt/amazon/ofi-nccl/lib/libnccl-net.so
    ! ldd /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | grep -q "not found"
    /opt/amazon/efa/bin/fi_info --version
    dpkg-query --show --showformat="\${Package}=\${Version}\\n" \
        libfabric1-aws libnccl-ofi-ngc-v3 rdma-core
'

echo "Exporting ${OUTPUT_IMAGE}"
time enroot export -o "${TEMP_OUTPUT}" "${CONTAINER_NAME}"
mv "${TEMP_OUTPUT}" "${OUTPUT_IMAGE}"

ls -lh "${OUTPUT_IMAGE}"
sha256sum "${OUTPUT_IMAGE}"
echo "done. Use it with: srun --container-image=${OUTPUT_IMAGE}"
