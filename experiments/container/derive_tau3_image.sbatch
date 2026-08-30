#!/bin/bash
#SBATCH --job-name=miles-tau3-sqsh
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Embed the official Tau three v1.0.1 source and its frozen Gym runtime into an
# already EFA-qualified Miles image. Build inputs and output provenance are
# immutable; no training job installs packages dynamically.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the Miles repository root}"
source "${REPO_ROOT}/experiments/env.sh"

readonly TAU_REPOSITORY=https://github.com/sierra-research/tau2-bench.git
readonly TAU_RELEASE=v1.0.1
readonly TAU_COMMIT=fc0055dc4e0a316c3f83133267fbd6faaa770992
BASE_IMAGE="${BASE_IMAGE:-${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0.sqsh}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0-tau3-1.0.1.sqsh}"

[[ -f "${BASE_IMAGE}" ]] || { echo "base image not found: ${BASE_IMAGE}" >&2; exit 1; }
[[ "${OUTPUT_IMAGE}" == /* ]] || { echo "OUTPUT_IMAGE must be absolute" >&2; exit 1; }
[[ ! -e "${OUTPUT_IMAGE}" ]] || { echo "refusing to overwrite ${OUTPUT_IMAGE}" >&2; exit 1; }
[[ ! -e "${OUTPUT_IMAGE}.sha256" && ! -e "${OUTPUT_IMAGE}.provenance.env" ]] || {
    echo "refusing to overwrite image provenance for ${OUTPUT_IMAGE}" >&2
    exit 1
}

LOCAL_SCRATCH="/raid/enroot/tmp/${USER}-${SLURM_JOB_ID}-tau3"
export ENROOT_TEMP_PATH="${LOCAL_SCRATCH}/tmp"
export ENROOT_DATA_PATH="${LOCAL_SCRATCH}/data"
export ENROOT_CACHE_PATH="${CACHE_DIR}/enroot/cache"
SOURCE_DIR="${LOCAL_SCRATCH}/tau3-source"
CONTAINER_NAME="miles-tau3-${USER}-${SLURM_JOB_ID}"
TEMP_OUTPUT="${OUTPUT_IMAGE}.partial-${SLURM_JOB_ID}"
TEMP_SHA256="${OUTPUT_IMAGE}.sha256.partial-${SLURM_JOB_ID}"
TEMP_PROVENANCE="${OUTPUT_IMAGE}.provenance.env.partial-${SLURM_JOB_ID}"

cleanup() {
    enroot remove -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    rm -rf -- "${LOCAL_SCRATCH}"
    rm -f -- "${TEMP_OUTPUT}" "${TEMP_SHA256}" "${TEMP_PROVENANCE}"
}
trap cleanup EXIT

mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_TEMP_PATH}" "${ENROOT_DATA_PATH}"
git clone --filter=blob:none --no-checkout "${TAU_REPOSITORY}" "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" checkout --detach "${TAU_COMMIT}"
[[ "$(git -C "${SOURCE_DIR}" describe --tags --exact-match)" == "${TAU_RELEASE}" ]]
[[ -z "$(git -C "${SOURCE_DIR}" status --short)" ]]

time enroot create -n "${CONTAINER_NAME}" "${BASE_IMAGE}"
CONTAINER_ROOT="${ENROOT_DATA_PATH}/${CONTAINER_NAME}"
mkdir -p "${CONTAINER_ROOT}/opt/tau3"
cp -a "${SOURCE_DIR}/." "${CONTAINER_ROOT}/opt/tau3/"

time enroot start -r -w "${CONTAINER_NAME}" bash --noprofile --norc -c '
set -euo pipefail
command -v uv >/dev/null
cd /opt/tau3
uv export --frozen --extra gym --no-dev --no-emit-project --output-file /tmp/tau3-requirements.txt
uv venv --system-site-packages --python python3 /opt/tau3/.venv
uv pip sync --python /opt/tau3/.venv/bin/python /tmp/tau3-requirements.txt
uv pip install --python /opt/tau3/.venv/bin/python --no-deps /opt/tau3
/opt/tau3/.venv/bin/python - <<"PY"
import importlib.metadata
from tau2.runner.helpers import load_tasks

assert importlib.metadata.version("tau2") == "1.0.1"
assert len(load_tasks("retail", "train")) == 74
assert len(load_tasks("airline", "train")) == 30
assert len(load_tasks("telecom", "train")) == 74
print("Tau three v1.0.1 Gym runtime: ok")
PY
rm -f /tmp/tau3-requirements.txt
'

installed_versions="$(
    enroot start -r "${CONTAINER_NAME}" /opt/tau3/.venv/bin/python - <<'PY'
import importlib.metadata
for package in ("tau2", "gymnasium", "litellm"):
    print(f"{package}={importlib.metadata.version(package)}")
PY
)"
base_sha256="$(sha256sum "${BASE_IMAGE}")"
base_sha256="${base_sha256%% *}"

time enroot export -o "${TEMP_OUTPUT}" "${CONTAINER_NAME}"
image_sha256="$(sha256sum "${TEMP_OUTPUT}")"
image_sha256="${image_sha256%% *}"
printf '%s  %s\n' "${image_sha256}" "$(basename -- "${OUTPUT_IMAGE}")" > "${TEMP_SHA256}"
cat > "${TEMP_PROVENANCE}" <<EOF
base_image=${BASE_IMAGE}
base_image_sha256=${base_sha256}
tau_repository=${TAU_REPOSITORY}
tau_release=${TAU_RELEASE}
tau_commit=${TAU_COMMIT}
${installed_versions}
image_sha256=${image_sha256}
slurm_job_id=${SLURM_JOB_ID}
created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

mv -- "${TEMP_OUTPUT}" "${OUTPUT_IMAGE}"
mv -- "${TEMP_SHA256}" "${OUTPUT_IMAGE}.sha256"
mv -- "${TEMP_PROVENANCE}" "${OUTPUT_IMAGE}.provenance.env"
trap - EXIT
cleanup

(cd "$(dirname -- "${OUTPUT_IMAGE}")" && sha256sum --check "$(basename -- "${OUTPUT_IMAGE}.sha256")")
echo "Tau three image ready: ${OUTPUT_IMAGE}"
