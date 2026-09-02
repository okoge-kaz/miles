#!/bin/bash

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(realpath "${SCRIPT_DIR}/../../..")}}" # submission root
source "${REPO_ROOT}/experiments/env.sh"
MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="MILES_SWE_FIXED_EXPORTS,SWE_EVAL_ALLOW_SUBSET,SWE_EVAL_CANDIDATE,SWE_EVAL_INPUT,SWE_EVAL_INPUT_SUMMARY,SWE_EVAL_MATERIALIZATION_EVIDENCE,SWE_EVAL_SEMANTIC_ADMISSION,SWE_EVAL_TASK_IDS,SWE_EVAL_TASK_MANIFEST,SWE_EVAL_TASK_OUTPUT,SWE_MATERIALIZE_DRY_RUN,SWE_MATERIALIZE_LIMIT"

submission="$(
    pbs_submit --parsable --profile cpu --chdir="${REPO_ROOT}" --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/materialize_swebench_verified_eval.sbatch"
)"
job_id="${submission}"
[[ "${job_id}" =~ ^[0-9]+(\[[^]]*\])?(\.[A-Za-z0-9._-]+)?$ ]] || exit 2
printf 'swebench_verified_materialization_job_id=%s\n' "${job_id}"
