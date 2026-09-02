#!/bin/bash

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(realpath "${SCRIPT_DIR}/../../..")}}" # submission root
source "${REPO_ROOT}/experiments/env.sh"
: "${SWE_TASKSET:?set SWE_TASKSET}"
MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="MILES_SWE_FIXED_EXPORTS,SWE_ADMITTED_OUTPUT,SWE_ADMITTED_SUMMARY,SWE_CANDIDATE_DATASET,SWE_FINALIZE_ALLOW_SUBSET,SWE_MATERIALIZATION_EVIDENCE,SWE_MATERIALIZE_DRY_RUN,SWE_MATERIALIZE_LIMIT,SWE_R2E_ADMITTED_MANIFEST,SWE_SEMANTIC_ADMISSION_MANIFEST,SWE_TASKSET,SWE_TASK_IDS_OUTPUT,SWE_TASK_MANIFEST,SWE_TASK_OUTPUT"

submission="$(
    pbs_submit --parsable --profile cpu --chdir="${REPO_ROOT}" --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/materialize_harbor_swe_tasks.sbatch"
)"
job_id="${submission}"
[[ "${job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || exit 2
printf 'swe_materialization_job_id=%s\n' "${job_id}"
