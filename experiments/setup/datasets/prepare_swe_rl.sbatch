#!/bin/bash
#PBS -N prepare-swe-rl
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -j oe

# Normalize one pinned SWE source into gold-free Miles rows and owner-only task
# manifests.  Schema-normalized candidates remain under a clearly marked
# not-admitted directory until OCI locking and live semantic E2B admission both
# succeed. Schema normalization alone never makes ReBench, R2E, or SWE-Gym
# eligible for RL. Unsupported schemas remain owner-only under
# /data/miles-swe/private/unadmitted.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_prepare_swe_rl.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || {
    echo "MILES_SWE_FIXED_EXPORTS must be 1" >&2
    exit 2
}
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached SWE preparation" >&2
    exit 2
}
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is missing or unsafe" >&2
    exit 2
}
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"
: "${SWE_SOURCE:?set SWE_SOURCE to a selector from download_swe_datasets.sbatch}"
export PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1
export SWE_SOURCE WANDB_DISABLED=true WANDB_MODE=offline

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${SWE_CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=PATH,PYTHON_DOTENV_DISABLED,PYTHONDONTWRITEBYTECODE,SWE_SOURCE,WANDB_DISABLED,WANDB_MODE \
    env -u BASH_ENV -u ENV bash --noprofile --norc -c '
set -euo pipefail
umask 077
cd /root/miles
export PYTHONPATH=/root/miles

public_root=/data/miles-swe
candidate_root=${public_root}/candidates/not-admitted
private_root=/data/miles-swe/private
unadmitted_root=${private_root}/unadmitted
mkdir -p "${candidate_root}" "${private_root}" "${unadmitted_root}"
chmod 0700 "${candidate_root}"
chmod 0700 "${private_root}" "${unadmitted_root}"

prepare_split() {
    local name="$1"
    local schema="$2"
    local usage="$3"
    local disposition="$4"
    local input_pattern="$5"
    local dataset_id="${6:-}"
    local invalid_policy="${7:-error}"
    local output manifest summary invalid_report
    if [[ "${disposition}" == materializable-candidate || "${disposition}" == official-eval ]]; then
        output="${candidate_root}/${name}-${usage}.not-admitted.jsonl"
    else
        output="${unadmitted_root}/${name}-${usage}.gold-free.jsonl"
    fi
    manifest="${private_root}/${name}.tasks.private.jsonl"
    summary="${private_root}/${name}.prepare-summary.json"
    invalid_report="${private_root}/${name}.invalid.private.jsonl"
    args=(
        --input "${input_pattern}"
        --source-name "${SWE_SOURCE}"
        --output "${output}"
        --task-manifest "${manifest}"
        --usage "${usage}"
        --agent-name terminus-2
        --include-schema "${schema}"
        --on-invalid "${invalid_policy}"
        --summary "${summary}"
    )
    if [[ "${invalid_policy}" == quarantine ]]; then
        args+=(--invalid-report "${invalid_report}")
    fi
    if [[ "${usage}" == train ]]; then
        args+=(
            --downstream-holdout
            /data/miles-swe/sources/swe-bench-verified/data/test-00000-of-00001.parquet
        )
    fi
    [[ -z "${dataset_id}" ]] || args+=(--dataset-id "${dataset_id}")
    if [[ "${schema}" == r2e-gym-v1 ]]; then
        args+=(--r2e-id-map "${private_root}/r2e-routing.private.json")
    fi
    python3 -m experiments.src.datasets.swe.prepare "${args[@]}"
    [[ -s "${output}" && -s "${manifest}" ]] || {
        echo "${name}: source produced no ${schema} tasks" >&2
        exit 1
    }
    chmod 0600 "${manifest}" "${summary}"
    if [[ "${disposition}" != materializable-candidate && "${disposition}" != official-eval ]]; then
        chmod 0600 "${output}"
        echo "quarantined ${name}: ${disposition}"
    else
        chmod 0600 "${output}"
        echo "prepared schema-normalized, environment-not-admitted candidate ${name}: ${output}"
        echo "live OCI + E2B semantic admission and an admitted-task join are required before RL"
    fi
}

case "${SWE_SOURCE}" in
    nemotron3-super-swe2)
        input=/data/miles-swe/sources/nemotron-3-super-swe2/swe2.jsonl
        prepare_split nemotron3-super-swe2-r2e r2e-gym-v1 train \
            unadmitted-r2e-runtime-and-live-admission "${input}"
        prepare_split nemotron3-super-swe2-swe-gym swe-gym train \
            unadmitted-swe-gym-live-admission "${input}"
        ;;
    nemotron3-ultra-swe)
        input=/data/miles-swe/sources/nemotron-3-ultra-swe/swe.jsonl
        prepare_split nemotron3-ultra-swe-rebench-v2 swe-rebench-v2 train \
            materializable-candidate "${input}"
        prepare_split nemotron3-ultra-swe-gym swe-gym train \
            unadmitted-swe-gym-live-admission "${input}"
        ;;
    r2e-gym-v1)
        prepare_split r2e-gym-v1 r2e-gym-v1 train \
            unadmitted-r2e-runtime-and-live-admission \
            "/data/miles-swe/sources/r2e-gym-v1/data/*.parquet" \
            R2E-Gym/R2E-Gym-V1 quarantine
        ;;
    swe-rebench-v2)
        prepare_split swe-rebench-v2 swe-rebench-v2 train materializable-candidate \
            "/data/miles-swe/sources/swe-rebench-v2/data/*.parquet" \
            nebius/SWE-rebench-V2 quarantine
        ;;
    swe-rebench-v2-filtered-verified)
        prepare_split swe-rebench-v2-filtered-verified swe-rebench-v2 train \
            materializable-candidate \
            "/data/miles-swe/sources/swe-rebench-v2-filtered-verified/data/*.parquet" \
            PrimeIntellect/SWE-rebench-V2-Filtered-Verified quarantine
        ;;
    swe-gym)
        prepare_split swe-gym swe-gym train unadmitted-swe-gym-live-admission \
            "/data/miles-swe/sources/swe-gym/data/*.parquet" SWE-Gym/SWE-Gym quarantine
        ;;
    swe-bench-verified)
        prepare_split swe-bench-verified swebench eval official-eval \
            "/data/miles-swe/sources/swe-bench-verified/data/*.parquet" \
            princeton-nlp/SWE-bench_Verified
        ;;
    *)
        echo "unknown SWE_SOURCE=${SWE_SOURCE}" >&2
        exit 2
        ;;
esac
'
