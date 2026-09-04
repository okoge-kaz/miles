#!/bin/bash
#SBATCH --job-name=miles-search-r1-sqsh
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Add the pinned CPU Faiss wheel to an already-qualified Miles SQSH. The output
# is immutable and provenance-stamped; extraction/export stays on node-local
# NVMe.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the Miles repository root}"
source "${REPO_ROOT}/experiments/env.sh"

BASE_IMAGE="${BASE_IMAGE:-${CONTAINER_DIR}/miles-search-r1-b300-20260815.sqsh}"
FAISS_CPU_VERSION="${FAISS_CPU_VERSION:-1.12.0}"
FAISS_CPU_WHEEL_FILENAME="${FAISS_CPU_WHEEL_FILENAME:-faiss_cpu-1.12.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl}"
FAISS_CPU_WHEEL_URL="${FAISS_CPU_WHEEL_URL:-https://files.pythonhosted.org/packages/4c/c9/be4e52fd96be601fefb313c26e1259ac2e6b556fb08cc392db641baba8c7/faiss_cpu-1.12.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl}"
FAISS_CPU_WHEEL_SHA256="${FAISS_CPU_WHEEL_SHA256:-c2e4963c7188f57cfba248f09ebd8a14c76b5ffb87382603ccd4576f2da39d74}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${CONTAINER_DIR}/miles-search-r1-b300-faiss-${FAISS_CPU_VERSION}-20260826.sqsh}"

[[ -f "${BASE_IMAGE}" ]] || { echo "base image not found: ${BASE_IMAGE}" >&2; exit 1; }
[[ "$(uname -m)" == x86_64 ]] || { echo "this pinned derivation is x86_64-only" >&2; exit 1; }
[[ "${FAISS_CPU_WHEEL_FILENAME}" =~ ^[A-Za-z0-9._+-]+\.whl$ ]] || {
    echo "invalid wheel filename: ${FAISS_CPU_WHEEL_FILENAME}" >&2
    exit 1
}
[[ "${OUTPUT_IMAGE}" == /* ]] || { echo "OUTPUT_IMAGE must be absolute" >&2; exit 1; }
[[ ! -e "${OUTPUT_IMAGE}" ]] || { echo "refusing to overwrite ${OUTPUT_IMAGE}" >&2; exit 1; }
[[ ! -e "${OUTPUT_IMAGE}.sha256" && ! -e "${OUTPUT_IMAGE}.provenance.env" ]] || {
    echo "refusing to overwrite image provenance for ${OUTPUT_IMAGE}" >&2
    exit 1
}

LOCAL_SCRATCH="/raid/enroot/tmp/${USER}-${SLURM_JOB_ID}"
export ENROOT_TEMP_PATH="${LOCAL_SCRATCH}/tmp"
export ENROOT_DATA_PATH="${LOCAL_SCRATCH}/data"
export ENROOT_CACHE_PATH="${CACHE_DIR}/enroot/cache"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_TEMP_PATH}" "${ENROOT_DATA_PATH}"

CONTAINER_NAME="miles-search-r1-${USER}-${SLURM_JOB_ID}"
WHEEL_DIR="${LOCAL_SCRATCH}/wheels"
WHEEL_PATH="${WHEEL_DIR}/${FAISS_CPU_WHEEL_FILENAME}"
TEMP_OUTPUT="${OUTPUT_IMAGE}.partial-${SLURM_JOB_ID}"
TEMP_SHA256="${OUTPUT_IMAGE}.sha256.partial-${SLURM_JOB_ID}"
TEMP_PROVENANCE="${OUTPUT_IMAGE}.provenance.env.partial-${SLURM_JOB_ID}"

cleanup() {
    enroot remove -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    rm -rf -- "${LOCAL_SCRATCH}"
    rm -f -- "${TEMP_OUTPUT}" "${TEMP_SHA256}" "${TEMP_PROVENANCE}"
}
trap cleanup EXIT

mkdir -p "${WHEEL_DIR}"
curl -fsSL --retry 5 --retry-all-errors -o "${WHEEL_PATH}" "${FAISS_CPU_WHEEL_URL}"
echo "${FAISS_CPU_WHEEL_SHA256}  ${WHEEL_PATH}" | sha256sum --check -

echo "Creating ${CONTAINER_NAME} from ${BASE_IMAGE}"
time enroot create -n "${CONTAINER_NAME}" "${BASE_IMAGE}"

echo "Installing faiss-cpu ${FAISS_CPU_VERSION}"
# CPU build nodes do not provide the mount namespace device needed by
# `enroot start --mount`. Stage the checksum-verified wheel into the extracted
# root, install it there, and remove the build input before export.
CONTAINER_WHEEL_DIR="${ENROOT_DATA_PATH}/${CONTAINER_NAME}/opt/miles-wheels"
mkdir -p "${CONTAINER_WHEEL_DIR}"
install -m 0444 "${WHEEL_PATH}" "${CONTAINER_WHEEL_DIR}/${FAISS_CPU_WHEEL_FILENAME}"
time enroot start \
    -r \
    -w \
    "${CONTAINER_NAME}" \
    bash --noprofile --norc -c '
set -euo pipefail
python3 -m pip install --no-deps --break-system-packages /opt/miles-wheels/'"${FAISS_CPU_WHEEL_FILENAME}"'
python3 -c '\''import faiss, fastapi, numpy, uvicorn; assert faiss.__version__ == "1.12.0"; index = faiss.IndexFlatIP(4); vectors = numpy.eye(4, dtype="float32"); index.add(vectors); assert index.search(vectors[:1], 1)[1].tolist() == [[0]]'\''
'
rm -f -- "${CONTAINER_WHEEL_DIR}/${FAISS_CPU_WHEEL_FILENAME}"
rmdir -- "${CONTAINER_WHEEL_DIR}"

installed_versions="$(
    enroot start -r "${CONTAINER_NAME}" python3 -c \
        'import faiss, fastapi, uvicorn; print(f"faiss={faiss.__version__}\nfastapi={fastapi.__version__}\nuvicorn={uvicorn.__version__}")' \
        | grep -E '^(faiss|fastapi|uvicorn)='
)"
base_sha256="$(sha256sum "${BASE_IMAGE}")"
base_sha256="${base_sha256%% *}"

echo "Exporting ${OUTPUT_IMAGE}"
time enroot export -o "${TEMP_OUTPUT}" "${CONTAINER_NAME}"
image_sha256="$(sha256sum "${TEMP_OUTPUT}")"
image_sha256="${image_sha256%% *}"
printf '%s  %s\n' "${image_sha256}" "$(basename -- "${OUTPUT_IMAGE}")" > "${TEMP_SHA256}"
cat > "${TEMP_PROVENANCE}" <<EOF
base_image=${BASE_IMAGE}
base_image_sha256=${base_sha256}
faiss_cpu_wheel_url=${FAISS_CPU_WHEEL_URL}
faiss_cpu_wheel_sha256=${FAISS_CPU_WHEEL_SHA256}
${installed_versions}
image_sha256=${image_sha256}
slurm_job_id=${SLURM_JOB_ID}
created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

mv -- "${TEMP_OUTPUT}" "${OUTPUT_IMAGE}"
mv -- "${TEMP_SHA256}" "${OUTPUT_IMAGE}.sha256"
mv -- "${TEMP_PROVENANCE}" "${OUTPUT_IMAGE}.provenance.env"

ls -lh "${OUTPUT_IMAGE}" "${OUTPUT_IMAGE}.sha256" "${OUTPUT_IMAGE}.provenance.env"
(cd "$(dirname -- "${OUTPUT_IMAGE}")" && sha256sum --check "$(basename -- "${OUTPUT_IMAGE}.sha256")")
echo "done. Validate with: SQSH_IMAGE=${OUTPUT_IMAGE} sbatch experiments/search_r1/evaluation/run.sbatch"
