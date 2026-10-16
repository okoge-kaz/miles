#!/bin/bash

# Internal single-job submission shared by the global 16-slot scheduler.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MILES_REPO="$(cd -- "${script_dir}/../../.." && pwd)"
source "${MILES_REPO}/experiments/common/pbs.sh"
source "${MILES_REPO}/experiments/common/job_env.sh"
run_name="${1:?}" action="${2:?}" shard="${3:?}" dependency="${4:-}" walltime="${5:?}" smoke="${6:?}"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
[[ "${action}" =~ ^(smoke|filter|merge)$ && "${shard}" =~ ^[0-9]+$ ]] || exit 2
run_dir="${MILES_REPO}/experiments/outputs/difficulty_filtering/${run_name}"
profile=gpu script="${run_dir}/run.sh"
if [[ "${action}" == merge ]]; then profile=cpu; script="${run_dir}/merge.sh"; walltime=00:30:00; fi
options=()
[[ -z "${dependency}" ]] || options+=(--dependency="${dependency}")
pbs_submit --parsable --profile="${profile}" --nodes=1 --job-kind=prep \
    --time="${walltime}" --chdir="${MILES_REPO}" --job-name="df-${action}-${shard}" \
    --output="${run_dir}/${action}-${shard}.ou" --export="$(miles_job_export_spec)" \
    "${options[@]}" "${script}" "/root/miles/experiments/outputs/difficulty_filtering/${run_name}/config.json" \
    "${action}" "${shard}" "${smoke}"
