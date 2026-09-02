#!/bin/bash
#PBS -N admit-repository-swe-e2b
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -j oe

# Resolve OCI tags and run resumable live E2B semantic admission for one
# SWE-ReBench, SWE-Gym, or eval-only SWE-bench Verified task set. The host data
# root below is mounted as /data by Miles containers; Harbor's E2B Python must
# run on the host.
# Credentials are inherited only from the submitting process.  No .env file is
# read and no secret is placed in argv, logs, task trees, or evidence records.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
USER="$(id -un)"
export USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_admit_repository_swe_e2b.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || {
    echo "MILES_SWE_FIXED_EXPORTS must be 1" >&2
    exit 2
}
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached repository admission" >&2
    exit 2
}
export PYTHON_DOTENV_DISABLED=1 WANDB_DISABLED=true WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"
umask 077
: "${SWE_TASKSET:?set SWE_TASKSET to a supported prepared task set}"
: "${HARBOR_ROOT:?set HARBOR_ROOT to the pinned Harbor checkout}"
: "${E2B_API_KEY:?inject E2B_API_KEY into the submitting process; .env is not read}"

readonly EXPECTED_HARBOR_COMMIT=2ce5ba2af33a00c9fba0463f6403313996373f85
readonly HARBOR_PYTHON="${HARBOR_PYTHON:-${HARBOR_ROOT}/.venv/bin/python}"
readonly OVERLAY_HELPER="${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/apply_harbor_e2b_overlays.sh"
readonly DATA_ROOT="${SWE_DATA_ROOT:-${DATASET_DIR}/miles-swe}"

[[ "$(git -C "${HARBOR_ROOT}" rev-parse HEAD)" == "${EXPECTED_HARBOR_COMMIT}" ]] || {
    echo "Harbor must be pinned to ${EXPECTED_HARBOR_COMMIT}" >&2
    exit 1
}
[[ -x "${HARBOR_PYTHON}" ]] || {
    echo "Missing pinned Harbor Python with the e2b extra: ${HARBOR_PYTHON}" >&2
    exit 1
}
[[ -x "${OVERLAY_HELPER}" ]] || {
    echo "Missing Harbor E2B overlay helper: ${OVERLAY_HELPER}" >&2
    exit 1
}
"${OVERLAY_HELPER}" "${HARBOR_ROOT}"

private_root="${DATA_ROOT}/private"
verifier_root="${DATA_ROOT}/verifiers"
private_manifest="${SWE_PRIVATE_MANIFEST:-${private_root}/${SWE_TASKSET}.tasks.private.jsonl}"
output_scope="${SWE_TASKSET}"
if [[ -n "${SWE_ADMISSION_INSTANCE_ID:-}" ]]; then
    [[ "${SWE_ADMISSION_INSTANCE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,239}$ ]] || {
        echo "SWE_ADMISSION_INSTANCE_ID is invalid" >&2
        exit 2
    }
    instance_sha="$(printf '%s' "${SWE_ADMISSION_INSTANCE_ID}" | sha256sum)"
    instance_sha="${instance_sha%% *}"
    output_scope+=".instance-${instance_sha}"
fi
if [[ -n "${SWE_ADMISSION_LIMIT:-}" ]]; then
    [[ "${SWE_ADMISSION_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
        echo "SWE_ADMISSION_LIMIT must be a positive integer" >&2
        exit 2
    }
    output_scope+=".limit-${SWE_ADMISSION_LIMIT}"
fi
locked_manifest="${SWE_LOCKED_MANIFEST:-${private_root}/${output_scope}.locked.tasks.private.jsonl}"
image_locks="${SWE_IMAGE_LOCK_MANIFEST:-${private_root}/${output_scope}.images.private.jsonl}"
admission_manifest="${SWE_ADMISSION_MANIFEST:-${private_root}/${output_scope}.admission.private.jsonl}"
admitted_manifest="${SWE_ADMITTED_MANIFEST:-${private_root}/${output_scope}.admitted.tasks.private.jsonl}"
quarantine_manifest="${SWE_QUARANTINE_MANIFEST:-${private_root}/${output_scope}.semantic-quarantine.private.jsonl}"
work_root="${SWE_ADMISSION_WORK_ROOT:-${private_root}/${output_scope}.admission-work}"

case "${SWE_TASKSET}" in
    swe-rebench-v2|swe-rebench-v2-filtered-verified|nemotron3-ultra-swe-rebench-v2)
        admission_module=experiments.src.environments.swe.admit_swe_rebench
        environment_args=(
            --swe-rebench-log-parsers
            "${verifier_root}/swe-rebench-v2/lib/agent/log_parsers.py"
            --swe-rebench-constants
            "${verifier_root}/swe-rebench-v2/lib/agent/swe_constants.py"
            --swe-rebench-eval
            "${verifier_root}/swe-rebench-v2/eval.py"
        )
        ;;
    swe-gym|nemotron3-super-swe2-swe-gym|nemotron3-ultra-swe-gym)
        admission_module=experiments.src.environments.swe.admit_swe_gym
        environment_args=(
            --swe-gym-harness-root
            "${verifier_root}/swe-gym/SWE-Bench-Package"
        )
        ;;
    swe-bench-verified)
        admission_module=experiments.src.environments.swe.admit_swebench_verified
        environment_args=(
            --swebench-harness-root
            "${verifier_root}/swebench/SWE-bench"
        )
        ;;
    *)
        echo "Unsupported repository SWE_TASKSET=${SWE_TASKSET}" >&2
        exit 2
        ;;
esac

[[ -r "${private_manifest}" ]] || {
    echo "Missing owner-only prepared manifest: ${private_manifest}" >&2
    exit 1
}

lock_args=(
    --private-manifest "${private_manifest}"
    --locked-manifest "${locked_manifest}"
    --image-lock-manifest "${image_locks}"
    --concurrency "${SWE_IMAGE_LOCK_CONCURRENCY:-8}"
)
if [[ "${SWE_RESOLVE_MISSING_IMAGE_LOCKS:-1}" == 1 ]]; then
    lock_args+=(--resolve-missing)
fi
if [[ "${SWE_REFRESH_MISSING_IMAGE_LOCKS:-0}" == 1 ]]; then
    lock_args+=(--refresh-missing)
fi
[[ -z "${SWE_ADMISSION_INSTANCE_ID:-}" ]] || lock_args+=(--instance-id "${SWE_ADMISSION_INSTANCE_ID}")
[[ -z "${SWE_ADMISSION_LIMIT:-}" ]] || lock_args+=(--limit "${SWE_ADMISSION_LIMIT}")

admission_args=(
    --private-manifest "${locked_manifest}"
    --admitted-manifest "${admitted_manifest}"
    --admission-manifest "${admission_manifest}"
    --quarantine-manifest "${quarantine_manifest}"
    --work-root "${work_root}"
    --concurrency "${SWE_ADMISSION_CONCURRENCY:-4}"
    "${environment_args[@]}"
)
[[ -z "${SWE_ADMISSION_INSTANCE_ID:-}" ]] || admission_args+=(--instance-id "${SWE_ADMISSION_INSTANCE_ID}")
[[ -z "${SWE_ADMISSION_LIMIT:-}" ]] || admission_args+=(--limit "${SWE_ADMISSION_LIMIT}")

export WANDB_MODE=offline
export WANDB_DISABLED=true
unset WANDB_API_KEY
export PYTHON_DOTENV_DISABLED=1
export PYTHONDONTWRITEBYTECODE=1
export HARBOR_ENV_TYPE=e2b
export HARBOR_KEEP_SANDBOX=false
export HARBOR_E2B_NO_NEW_PRIVS_USERS=1000
export PYTHONPATH="${REPO_ROOT}:${HARBOR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset HARBOR_E2B_REQUIRE_PREBUILT HARBOR_E2B_REQUIRE_TEMPLATE_PIN \
    HARBOR_E2B_TEMPLATE_PINS_FILE

"${HARBOR_PYTHON}" -m experiments.src.environments.swe.oci_image_lock "${lock_args[@]}"
"${HARBOR_PYTHON}" -m "${admission_module}" "${admission_args[@]}"
