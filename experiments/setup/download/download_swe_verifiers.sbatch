#!/bin/bash
#SBATCH --job-name=miles-download-swe-verifiers
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Fetch the exact upstream verifier modules accepted by materialize.py.
# Checksums are mandatory; repository-local .env files and credentials are not
# read or copied into provenance.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    env -u BASH_ENV -u ENV bash --noprofile --norc -c '
set -euo pipefail
target=/data/miles-swe/verifiers
mkdir -p "${target}/r2e-gym" "${target}/swe-rebench-v2/lib/agent"

fetch() {
    local url="$1"
    local expected="$2"
    local destination="$3"
    local partial
    partial=$(mktemp "${destination}.partial.XXXXXX")
    if ! curl --fail --location --silent --show-error "${url}" --output "${partial}"; then
        rm -f -- "${partial}"
        return 1
    fi
    if ! printf "%s  %s\n" "${expected}" "${partial}" | sha256sum --check --status; then
        rm -f -- "${partial}"
        return 1
    fi
    chmod 0444 "${partial}"
    mv -f -- "${partial}" "${destination}"
}

r2e_commit=0d94c4eb9431cd195c55a7ea3abd54006c9a1735
rebench_commit=c71902a8cf8d2b725f63d51f199f4d3e56f68d2d
fetch \
    "https://raw.githubusercontent.com/R2E-Gym/R2E-Gym/${r2e_commit}/src/r2egym/repo_analysis/execution_log_parser.py" \
    395f637f4b8d68160948f95097f861506f123978852a9da77258aa3ba3fe1904 \
    "${target}/r2e-gym/execution_log_parser.py"
fetch \
    "https://raw.githubusercontent.com/SWE-rebench/SWE-rebench-V2/${rebench_commit}/lib/agent/log_parsers.py" \
    a717b03efde1cb79dfb11e2a57d0262c0057d352a347a9fb09667ef6e5f6f20c \
    "${target}/swe-rebench-v2/lib/agent/log_parsers.py"
fetch \
    "https://raw.githubusercontent.com/SWE-rebench/SWE-rebench-V2/${rebench_commit}/lib/agent/swe_constants.py" \
    823dd1ef512d363ed5d4dce05d70f22d7f93b25722cda5b0971f17010f5168a5 \
    "${target}/swe-rebench-v2/lib/agent/swe_constants.py"
fetch \
    "https://raw.githubusercontent.com/SWE-rebench/SWE-rebench-V2/${rebench_commit}/scripts/eval.py" \
    4768c0c3e2adf3540c2228f819f4b073e4665ada06fa00f2234a1f7620d69eda \
    "${target}/swe-rebench-v2/eval.py"
printf "r2e_commit=%s\nrebench_commit=%s\n" "${r2e_commit}" "${rebench_commit}" \
    >"${target}/MILES_SOURCE_PROVENANCE"
'
