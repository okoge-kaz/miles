#!/bin/bash
#SBATCH --job-name=lock-swe-oci-images
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# Dataset-independent OCI provenance stage for private SWE task manifests.
# Optional Docker Hub credentials are read only from the submitting process
# environment.  No .env file is loaded and credentials are never arguments or
# output fields.

set -euo pipefail
umask 077

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_lock_swe_oci_images.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || exit 2
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached OCI locking" >&2
    exit 2
}
: "${SWE_PRIVATE_MANIFEST:?set an owner-only prepared SWE task manifest}"
: "${SWE_LOCKED_MANIFEST:?set the owner-only immutable task-manifest output}"
: "${SWE_IMAGE_LOCK_MANIFEST:?set the owner-only OCI evidence output}"

for toggle in SWE_RESOLVE_MISSING_IMAGE_LOCKS SWE_REFRESH_MISSING_IMAGE_LOCKS; do
    value="${!toggle:-}"
    [[ -z "${value}" || "${value}" == 0 || "${value}" == 1 ]] || {
        echo "${toggle} must be 0 or 1" >&2
        exit 2
    }
done
unset toggle value

args=(
    --private-manifest "${SWE_PRIVATE_MANIFEST}"
    --locked-manifest "${SWE_LOCKED_MANIFEST}"
    --image-lock-manifest "${SWE_IMAGE_LOCK_MANIFEST}"
    --concurrency "${SWE_IMAGE_LOCK_CONCURRENCY:-8}"
    --checkpoint-batch-size "${SWE_IMAGE_LOCK_CHECKPOINT_BATCH_SIZE:-32}"
)
if [[ "${SWE_RESOLVE_MISSING_IMAGE_LOCKS:-1}" == 1 ]]; then
    args+=(--resolve-missing)
fi
if [[ "${SWE_REFRESH_MISSING_IMAGE_LOCKS:-0}" == 1 ]]; then
    args+=(--refresh-missing)
fi
[[ -z "${SWE_IMAGE_LOCK_INSTANCE_ID:-}" ]] || args+=(--instance-id "${SWE_IMAGE_LOCK_INSTANCE_ID}")
[[ -z "${SWE_IMAGE_LOCK_LIMIT:-}" ]] || args+=(--limit "${SWE_IMAGE_LOCK_LIMIT}")

export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}"

python3 -m experiments.src.environments.swe.oci_image_lock "${args[@]}"
