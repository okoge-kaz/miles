#!/bin/bash
#PBS -N miles-download-swe-verifiers
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Fetch the exact upstream verifier modules accepted by materialize.py.
# Checksums are mandatory; repository-local .env files and credentials are not
# read or copied into provenance.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
unset WANDB_API_KEY
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is missing or unsafe" >&2
    exit 2
}
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${SWE_CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=NONE \
    env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    WANDB_MODE=offline WANDB_DISABLED=true PYTHON_DOTENV_DISABLED=1 \
    bash --noprofile --norc -c '
set -euo pipefail
target=/data/miles-swe/verifiers
gym_root="${target}/swe-gym/SWE-Bench-Package"
verified_root="${target}/swebench/SWE-bench"
mkdir -p "${target}/r2e-gym" "${target}/swe-rebench-v2/lib/agent" \
    "${gym_root}/swegym/harness" "${verified_root}/swebench/harness"
active_partial=
cleanup_active_partial() {
    [[ -z "${active_partial}" ]] || rm -f -- "${active_partial}"
}
trap cleanup_active_partial EXIT
trap "exit 129" HUP
trap "exit 130" INT
trap "exit 143" TERM

fetch() {
    local url="$1"
    local expected="$2"
    local destination="$3"
    local partial
    partial=$(mktemp "${destination}.partial.XXXXXX")
    active_partial="${partial}"
    if ! curl --fail --location --silent --show-error \
        --connect-timeout 15 --max-time 120 --retry 3 --retry-delay 2 \
        --retry-max-time 180 "${url}" --output "${partial}"; then
        rm -f -- "${partial}"
        return 1
    fi
    if ! printf "%s  %s\n" "${expected}" "${partial}" | sha256sum --check --status; then
        rm -f -- "${partial}"
        return 1
    fi
    chmod 0444 "${partial}"
    mv -f -- "${partial}" "${destination}"
    active_partial=
}

r2e_commit=0d94c4eb9431cd195c55a7ea3abd54006c9a1735
rebench_commit=c71902a8cf8d2b725f63d51f199f4d3e56f68d2d
swe_gym_commit=16dd480cce9b27bf111a362d280881c6def5d2a7
swe_gym_version=2.0.13
swebench_commit=7336033d65d32ec62f9ce2419aa8f3a757b06ce2
swebench_version=2.0.13
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
fetch \
    "https://raw.githubusercontent.com/SWE-Gym/SWE-Bench-Package/${swe_gym_commit}/swegym/harness/constants.py" \
    5bd655172c9a9dfcb494d9c52bcbcc0d79a59cd20c39b8bbdc41ab8b0a9baf14 \
    "${gym_root}/swegym/harness/constants.py"
fetch \
    "https://raw.githubusercontent.com/SWE-Gym/SWE-Bench-Package/${swe_gym_commit}/swegym/harness/log_parsers.py" \
    6d1bce4088dc4d7cb614783dcd63eeb5c20478906158a82cc97ce0b886367b7c \
    "${gym_root}/swegym/harness/log_parsers.py"
fetch \
    "https://raw.githubusercontent.com/SWE-Gym/SWE-Bench-Package/${swe_gym_commit}/swegym/harness/grading.py" \
    51eb584b85ffbc245042332a080ff20ba2529c712e76db14da16f246b0e0675d \
    "${gym_root}/swegym/harness/grading.py"
fetch \
    "https://raw.githubusercontent.com/SWE-bench/SWE-bench/${swebench_commit}/swebench/harness/constants.py" \
    6d189fcea0459897741eb241407b25e728467d3120f9feb8deaf2c61c030bc3e \
    "${verified_root}/swebench/harness/constants.py"
fetch \
    "https://raw.githubusercontent.com/SWE-bench/SWE-bench/${swebench_commit}/swebench/harness/log_parsers.py" \
    43ce7f06a562177ef82126f547e40fe19c51874148aae6027760d8b49b74bc89 \
    "${verified_root}/swebench/harness/log_parsers.py"
fetch \
    "https://raw.githubusercontent.com/SWE-bench/SWE-bench/${swebench_commit}/swebench/harness/grading.py" \
    c953793204d52a7ac67b197e0479987efcdf13f381f87b685d711b2e156e3ce3 \
    "${verified_root}/swebench/harness/grading.py"

provenance_partial=$(mktemp "${target}/MILES_SOURCE_PROVENANCE.partial.XXXXXX")
active_partial="${provenance_partial}"
printf "r2e_commit=%s\nrebench_commit=%s\nswe_gym_repo=%s\nswe_gym_commit=%s\nswe_gym_version=%s\nswe_gym_constants_sha256=%s\nswe_gym_log_parsers_sha256=%s\nswe_gym_grading_sha256=%s\n" \
    "${r2e_commit}" "${rebench_commit}" \
    SWE-Gym/SWE-Bench-Package "${swe_gym_commit}" "${swe_gym_version}" \
    5bd655172c9a9dfcb494d9c52bcbcc0d79a59cd20c39b8bbdc41ab8b0a9baf14 \
    6d1bce4088dc4d7cb614783dcd63eeb5c20478906158a82cc97ce0b886367b7c \
    51eb584b85ffbc245042332a080ff20ba2529c712e76db14da16f246b0e0675d \
    >"${provenance_partial}"
printf "swebench_repo=%s\nswebench_commit=%s\nswebench_version=%s\nswebench_constants_sha256=%s\nswebench_log_parsers_sha256=%s\nswebench_grading_sha256=%s\n" \
    SWE-bench/SWE-bench "${swebench_commit}" "${swebench_version}" \
    6d189fcea0459897741eb241407b25e728467d3120f9feb8deaf2c61c030bc3e \
    43ce7f06a562177ef82126f547e40fe19c51874148aae6027760d8b49b74bc89 \
    c953793204d52a7ac67b197e0479987efcdf13f381f87b685d711b2e156e3ce3 \
    >>"${provenance_partial}"
chmod 0444 "${provenance_partial}"
mv -f -- "${provenance_partial}" "${target}/MILES_SOURCE_PROVENANCE"
active_partial=
'
