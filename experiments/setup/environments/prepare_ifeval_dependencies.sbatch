#!/bin/bash
#PBS -N prep-ifeval-deps
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Stage the small, pinned IFEvalG-only dependency closure in shared storage.
# This keeps GPU jobs offline and avoids mutating an immutable SIF at launch.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
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

TEMP_TARGET="/data/.open-instruct-deps-${MILES_JOB_ID}"
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
