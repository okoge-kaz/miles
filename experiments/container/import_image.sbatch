#!/bin/bash
#SBATCH --job-name=miles-import-image
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert the miles docker image into a squashfs file that pyxis can run.
#
#   sbatch -A coreai_horizon_dilations experiments/container/import_image.sbatch
#
# On another cluster, override the CPU partition on the sbatch command line and
# point ENROOT_LOCAL_SCRATCH_ROOT at node-local ext4/XFS scratch. For example:
#
#   sbatch -A <account> -p <cpu-partition> \
#     --export=ALL,DOCKER_IMAGE=radixark/miles:<immutable-tag>,ENROOT_LOCAL_SCRATCH_ROOT=/local/enroot \
#     experiments/container/import_image.sbatch
#
# Runs on a CPU node (egress to registries is available there; verified against
# ghcr.io / huggingface.co). Docker Hub applies anonymous pull rate limits — if
# the import fails with a 429, set ENROOT_CREDENTIALS (see README).

set -euo pipefail

# sbatch copies this script into Slurm's spool directory, so BASH_SOURCE does
# not point at the repo. Resolve through the submit directory (submit from the
# repo root), falling back to the path for a direct `bash` invocation.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

# The final squashfs step stacks every extracted layer with overlayfs, and
# **overlayfs cannot use Lustre as a lower dir** — pointing the scratch paths at
# lustre fails with:
#   enroot-mksquashovlfs: failed to mount overlay: 0:1:...:105: Invalid argument
# after the (successful) 106-layer download. So the working directory has to sit
# on a node-local POSIX filesystem. /raid is node-local NVMe (7 TB, ext4) and
# /raid/enroot is writable; scope it per user+job so concurrent jobs on the same
# node never share state, and clean up on exit.
LOCAL_SCRATCH_ROOT="${ENROOT_LOCAL_SCRATCH_ROOT:-/raid/enroot/tmp}"
LOCAL_SCRATCH="${LOCAL_SCRATCH_ROOT%/}/${USER}-${SLURM_JOB_ID:-manual}"
export ENROOT_TEMP_PATH="${LOCAL_SCRATCH}/tmp"
export ENROOT_DATA_PATH="${LOCAL_SCRATCH}/data"
# The cache only holds downloaded layer tarballs (no overlay), so it stays on
# lustre: a retry then skips the ~10 min download.
export ENROOT_CACHE_PATH="${CACHE_DIR}/enroot/cache"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_TEMP_PATH}" "${ENROOT_DATA_PATH}"
trap 'rm -rf "${LOCAL_SCRATCH}"' EXIT

# Import into a dated file, then repoint a dedicated stable symlink. A failed
# import therefore never destroys the image the running jobs are using, and
# every image on disk says when it was pulled. IMPORT_OUTPUT_IMAGE makes the
# immutable filename explicit for cross-cluster qualification; SQSH_LINK can be
# empty to skip updating a link. Do not reuse SQSH_IMAGE as the link: experiments
# often pin it to a real, immutable file.
DATED_IMAGE="${IMPORT_OUTPUT_IMAGE:-${CONTAINER_DIR}/miles-latest-$(date +%Y%m%d).sqsh}"
SQSH_LINK="${SQSH_LINK-${CONTAINER_DIR}/miles-latest.sqsh}"

[[ "${DATED_IMAGE}" == /* ]] || {
    echo "IMPORT_OUTPUT_IMAGE must be absolute: ${DATED_IMAGE}" >&2
    exit 1
}
[[ ! -e "${DATED_IMAGE}" ]] || {
    echo "refusing to overwrite existing image: ${DATED_IMAGE}" >&2
    exit 1
}
if [[ -n "${SQSH_LINK}" ]]; then
    [[ "${SQSH_LINK}" == /* ]] || {
        echo "SQSH_LINK must be absolute or empty: ${SQSH_LINK}" >&2
        exit 1
    }
    [[ "${SQSH_LINK}" != "${DATED_IMAGE}" ]] || {
        echo "SQSH_LINK must differ from IMPORT_OUTPUT_IMAGE" >&2
        exit 1
    }
    if [[ -e "${SQSH_LINK}" && ! -L "${SQSH_LINK}" ]]; then
        echo "refusing to replace non-symlink SQSH_LINK: ${SQSH_LINK}" >&2
        exit 1
    fi
fi

echo "importing ${DOCKER_IMAGE} -> ${DATED_IMAGE}"
time enroot import -o "${DATED_IMAGE}" "docker://${DOCKER_IMAGE}"

if [[ -n "${SQSH_LINK}" ]]; then
    ln -sfn "${DATED_IMAGE}" "${SQSH_LINK}"
fi

ls -lh "${DATED_IMAGE}"
[[ -z "${SQSH_LINK}" ]] || ls -lh "${SQSH_LINK}"
echo "done. Use it with: SQSH_IMAGE=${DATED_IMAGE} experiments/submit_training.sh ..."
