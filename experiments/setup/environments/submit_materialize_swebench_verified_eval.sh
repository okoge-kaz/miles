#!/bin/bash

set -euo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:-$(realpath "${SCRIPT_DIR}/../../..")}" # submission root
MILES_SWE_FIXED_EXPORTS=1
export MILES_SWE_FIXED_EXPORTS
readonly EXPORT_NAMES="MILES_SWE_FIXED_EXPORTS,SWE_EVAL_ALLOW_SUBSET,SWE_EVAL_CANDIDATE,SWE_EVAL_INPUT,SWE_EVAL_INPUT_SUMMARY,SWE_EVAL_MATERIALIZATION_EVIDENCE,SWE_EVAL_SEMANTIC_ADMISSION,SWE_EVAL_TASK_IDS,SWE_EVAL_TASK_MANIFEST,SWE_EVAL_TASK_OUTPUT,SWE_MATERIALIZE_DRY_RUN,SWE_MATERIALIZE_LIMIT"

submission="$(
    sbatch --parsable --chdir="${REPO_ROOT}" --export="${EXPORT_NAMES}" \
        "${SCRIPT_DIR}/materialize_swebench_verified_eval.sbatch"
)"
job_id="${submission%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || exit 2
printf 'swebench_verified_materialization_job_id=%s\n' "${job_id}"
