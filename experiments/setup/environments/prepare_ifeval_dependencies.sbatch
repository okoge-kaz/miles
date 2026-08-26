#!/bin/bash
#SBATCH --job-name=prep-ifeval-deps
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Stage the small, pinned IFEvalG-only dependency closure in shared storage.
# This keeps GPU jobs offline and avoids mutating an immutable SQSH at launch.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail

TARGET=/data/open-instruct-deps
MARKER="${TARGET}/.miles-ifeval-deps-v1"
if [[ -f "${MARKER}" ]]; then
    echo "already staged: ${TARGET}"
    exit 0
fi
[[ ! -e "${TARGET}" ]] || {
    echo "refusing to replace unmarked dependency directory: ${TARGET}" >&2
    exit 1
}

TEMP_TARGET="/data/.open-instruct-deps-${SLURM_JOB_ID}"
cleanup() {
    case "${TEMP_TARGET}" in
        /data/.open-instruct-deps-*) rm -rf -- "${TEMP_TARGET}" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT
mkdir -p "${TEMP_TARGET}" "${TEMP_TARGET}/nltk_data"
python3 -m pip install --no-input --target "${TEMP_TARGET}" \
    absl-py==2.3.1 \
    immutabledict==1.2.0 \
    langdetect==1.0.9 \
    nltk==3.9.2
PYTHONPATH="${TEMP_TARGET}" python3 -m nltk.downloader \
    -d "${TEMP_TARGET}/nltk_data" punkt punkt_tab
PYTHONPATH="${TEMP_TARGET}" NLTK_DATA="${TEMP_TARGET}/nltk_data" python3 - <<"PY"
import absl
import immutabledict
import langdetect
import nltk

assert nltk.word_tokenize("Miles is ready.") == ["Miles", "is", "ready", "."]
print("IFEvalG dependencies import and tokenize: ok")
PY
touch "${TEMP_TARGET}/.miles-ifeval-deps-v1"
mv "${TEMP_TARGET}" "${TARGET}"
trap - EXIT
echo "staged: ${TARGET}"
'
