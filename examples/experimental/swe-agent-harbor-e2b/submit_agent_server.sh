#!/bin/bash

# Submit the durable Harbor server without copying the submission shell's
# unrelated credentials into the allocation. Secret values remain in the
# environment and only fixed variable names appear in sbatch argv.

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:-$(realpath "${SCRIPT_DIR}/../../..")}"
: "${E2B_API_KEY:?export E2B_API_KEY before submission}"
: "${HARBOR_ROOT:?set HARBOR_ROOT}"
: "${HARBOR_TASKS_DIR:?set HARBOR_TASKS_DIR}"
: "${HARBOR_E2B_PREBUILD_TASK_IDS_FILE:?set the admitted task-id file}"
: "${HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS:?set semantic admission manifests}"
: "${HARBOR_RUN_SECRET:?export HARBOR_RUN_SECRET}"
: "${HARBOR_ADMIN_SECRET:?export HARBOR_ADMIN_SECRET}"
: "${MAX_CONCURRENT:?set MAX_CONCURRENT}"

MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="AGENT_MAX_INPUT_TOKENS,AGENT_MAX_OUTPUT_TOKENS,AGENT_SETUP_TIMEOUT,AGENT_TIMEOUT,ASYNC_MAX_CONCURRENT_SAMPLES,E2B_ACCESS_TOKEN,E2B_API_KEY,E2B_API_URL,E2B_DOMAIN,E2B_SANDBOX_URL,HARBOR_ADMIN_SECRET,HARBOR_AGENT_ALLOWED_HOSTS,HARBOR_AGENT_MAX_ITERATIONS,HARBOR_E2B_PREBUILD_CONCURRENCY,HARBOR_E2B_PREBUILD_TASK_IDS_FILE,HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS,HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER,HARBOR_MAX_SEQ_LEN,HARBOR_PYTHON,HARBOR_RESPONSE_LENGTH_POLICY,HARBOR_ROOT,HARBOR_RUN_SECRET,HARBOR_TASKS_DIR,HARBOR_TERMINUS_2_ENABLE_SUMMARIZE,HARBOR_TERMINUS_2_LINEAR_HISTORY,HARBOR_TIMEOUT_MULTIPLIER,HARBOR_TRIAL_WALL_TIMEOUT_SEC,HARBOR_VERIFIER_TIMEOUT_SEC,HARBOR_WORKER_CANCEL_GRACE_SEC,MAX_CONCURRENT,MILES_SWE_FIXED_EXPORTS,PORT,TRIALS_DIR"

submission="$(
    sbatch \
        --parsable \
        --chdir="${REPO_ROOT}" \
        --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/run_agent_server.sbatch"
)"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
    echo "Slurm returned an invalid server job ID" >&2
    exit 2
}
printf 'server_job_id=%s\n' "${job_id}"
