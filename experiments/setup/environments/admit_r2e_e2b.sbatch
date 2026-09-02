#!/bin/bash
#PBS -N admit-r2e-e2b
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -j oe

# Live golden admission for immutable R2E-Gym images.  This job is resumable:
# every fully checked task gets an owner-only per-record checkpoint before the
# bounded worker pool advances.  It never reads .env and never places
# E2B_API_KEY in argv, logs, task data, or an output manifest.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
USER="$(id -un)"
export USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_admit_r2e_e2b.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || {
    echo "MILES_SWE_FIXED_EXPORTS must be 1" >&2
    exit 2
}
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached R2E admission" >&2
    exit 2
}
export PYTHON_DOTENV_DISABLED=1 WANDB_DISABLED=true WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"
umask 077
: "${HARBOR_ROOT:?set HARBOR_ROOT to the pinned Harbor checkout}"
: "${E2B_API_KEY:?inject E2B_API_KEY into the submitting process; .env is not read}"
SWE_TASKSET="${SWE_TASKSET:-r2e-gym-v1}"
case "${SWE_TASKSET}" in
    r2e-gym-v1|nemotron3-super-swe2-r2e) ;;
    *)
        echo "Unsupported R2E SWE_TASKSET: ${SWE_TASKSET}" >&2
        exit 2
        ;;
esac

readonly HARBOR_PYTHON="${HARBOR_PYTHON:-${HARBOR_ROOT}/.venv/bin/python}"
readonly OVERLAY_HELPER="${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/apply_harbor_e2b_overlays.sh"

[[ -x "${HARBOR_PYTHON}" ]] || {
    echo "Missing pinned Harbor Python with the e2b extra: ${HARBOR_PYTHON}" >&2
    exit 1
}
[[ -f "${OVERLAY_HELPER}" && ! -L "${OVERLAY_HELPER}" ]] || {
    echo "Missing shared Harbor E2B overlay helper: ${OVERLAY_HELPER}" >&2
    exit 1
}

# Admission and training share one checksum/stage-verified application path for
# the provider, collection, NoNewPrivs, template-ID, and late-tests overlays.
bash "${OVERLAY_HELPER}" "${HARBOR_ROOT}"

private_root="${SWE_R2E_PRIVATE_ROOT:-${DATASET_DIR}/miles-swe/private}"
private_manifest="${SWE_R2E_PRIVATE_MANIFEST:-${private_root}/${SWE_TASKSET}.tasks.private.jsonl}"
output_scope="${SWE_TASKSET}"
if [[ -n "${R2E_ADMISSION_INSTANCE_ID:-}" ]]; then
    [[ "${R2E_ADMISSION_INSTANCE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,239}$ ]] || {
        echo "Invalid R2E_ADMISSION_INSTANCE_ID" >&2
        exit 2
    }
    instance_sha="$(printf '%s' "${R2E_ADMISSION_INSTANCE_ID}" | sha256sum)"
    instance_sha="${instance_sha%% *}"
    output_scope+=".instance-${instance_sha}"
fi
if [[ -n "${R2E_ADMISSION_LIMIT:-}" ]]; then
    [[ "${R2E_ADMISSION_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Invalid R2E_ADMISSION_LIMIT" >&2
        exit 2
    }
    output_scope+=".limit-${R2E_ADMISSION_LIMIT}"
fi
admission_manifest="${R2E_ADMISSION_MANIFEST:-${private_root}/${output_scope}.admission.private.jsonl}"
admitted_manifest="${R2E_ADMITTED_MANIFEST:-${private_root}/${output_scope}.admitted.tasks.private.jsonl}"
quarantine_manifest="${R2E_QUARANTINE_MANIFEST:-${private_root}/${output_scope}.semantic-quarantine.private.jsonl}"
locked_manifest="${R2E_LOCKED_TASK_MANIFEST:-${private_root}/${output_scope}.locked.tasks.private.jsonl}"
work_root="${R2E_ADMISSION_WORK_ROOT:-${private_root}/${output_scope}.admission-work}"
parser="${R2E_EXECUTION_LOG_PARSER:-${DATASET_DIR}/miles-swe/verifiers/r2e-gym/execution_log_parser.py}"
image_lock_manifest="${R2E_IMAGE_LOCK_MANIFEST:-${private_root}/${output_scope}.images.private.jsonl}"

for toggle in R2E_RESOLVE_MISSING_IMAGE_LOCKS R2E_REFRESH_MISSING_IMAGE_LOCKS; do
    value="${!toggle:-}"
    [[ -z "${value}" || "${value}" == 0 || "${value}" == 1 ]] || {
        echo "${toggle} must be 0 or 1" >&2
        exit 2
    }
done
unset toggle value

[[ -r "${private_manifest}" ]] || {
    echo "Missing private R2E manifest: ${private_manifest}" >&2
    exit 1
}

args=(
    --private-manifest "${locked_manifest}"
    --admission-manifest "${admission_manifest}"
    --admitted-manifest "${admitted_manifest}"
    --quarantine-manifest "${quarantine_manifest}"
    --work-root "${work_root}"
    --r2e-execution-log-parser "${parser}"
    --concurrency "${R2E_ADMISSION_CONCURRENCY:-4}"
)
[[ -z "${R2E_ADMISSION_INSTANCE_ID:-}" ]] || args+=(--instance-id "${R2E_ADMISSION_INSTANCE_ID}")
[[ -z "${R2E_ADMISSION_LIMIT:-}" ]] || args+=(--limit "${R2E_ADMISSION_LIMIT}")

lock_args=(
    --private-manifest "${private_manifest}"
    --locked-manifest "${locked_manifest}"
    --image-lock-manifest "${image_lock_manifest}"
    --concurrency "${R2E_IMAGE_LOCK_CONCURRENCY:-8}"
    --checkpoint-batch-size "${R2E_IMAGE_LOCK_CHECKPOINT_BATCH_SIZE:-32}"
)
if [[ "${R2E_RESOLVE_MISSING_IMAGE_LOCKS:-1}" == 1 ]]; then
    lock_args+=(--resolve-missing)
fi
if [[ "${R2E_REFRESH_MISSING_IMAGE_LOCKS:-0}" == 1 ]]; then
    lock_args+=(--refresh-missing)
fi
[[ -z "${R2E_ADMISSION_INSTANCE_ID:-}" ]] || lock_args+=(--instance-id "${R2E_ADMISSION_INSTANCE_ID}")
[[ -z "${R2E_ADMISSION_LIMIT:-}" ]] || lock_args+=(--limit "${R2E_ADMISSION_LIMIT}")

export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export HARBOR_ENV_TYPE=e2b
export HARBOR_KEEP_SANDBOX=false
export HARBOR_E2B_NO_NEW_PRIVS_USERS=1000
export PYTHONPATH="${REPO_ROOT}:${HARBOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset HARBOR_E2B_REQUIRE_PREBUILT HARBOR_E2B_REQUIRE_TEMPLATE_PIN \
    HARBOR_E2B_TEMPLATE_PINS_FILE

"${HARBOR_PYTHON}" -m experiments.src.environments.swe.oci_image_lock "${lock_args[@]}"
"${HARBOR_PYTHON}" -m experiments.src.environments.swe.admit_r2e "${args[@]}"
