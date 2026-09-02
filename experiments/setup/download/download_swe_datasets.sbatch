#!/bin/bash
#PBS -N miles-download-swe
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=24:00:00
#PBS -j oe

# Selectively stage one pinned SWE source.  In particular, this avoids pulling
# the complete 26.8 GB Super or 17.1 GB Ultra blend when only SWE is needed.
# Repository-local .env files are never read.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/singularity.sh"

: "${SWE_SOURCE:?set SWE_SOURCE; see the case list in this file}"
case "${SWE_SOURCE}" in
    nemotron3-super-swe2)
        HF_REPO=nvidia/Nemotron-RL-Super-Training-Blends
        HF_REVISION=b90f74f1d0bafeec6d1f1321173f6775ba5bda2e
        LOCAL_NAME=nemotron-3-super-swe2
        HF_INCLUDE=$'swe2.jsonl\nREADME.md'
        ;;
    nemotron3-ultra-swe)
        HF_REPO=nvidia/Nemotron-RL-Ultra-Training-Blends
        HF_REVISION=79f8eda15ea12e1adf7bb14dcb338a29d391b80e
        LOCAL_NAME=nemotron-3-ultra-swe
        HF_INCLUDE=$'swe.jsonl\nREADME.md'
        ;;
    r2e-gym-v1)
        HF_REPO=R2E-Gym/R2E-Gym-V1
        HF_REVISION=903d405799ac435061c41e72260c81ca5100f964
        LOCAL_NAME=r2e-gym-v1
        HF_INCLUDE=$'data/*.parquet\nREADME.md\nLICENSE*'
        ;;
    swe-rebench-v2)
        HF_REPO=nebius/SWE-rebench-V2
        HF_REVISION=475dd5e8703bb5fb22dd3c60b5d038b019eba1e0
        LOCAL_NAME=swe-rebench-v2
        HF_INCLUDE=$'data/*.parquet\nREADME.md\nLICENSE*'
        ;;
    swe-rebench-v2-filtered-verified)
        HF_REPO=PrimeIntellect/SWE-rebench-V2-Filtered-Verified
        HF_REVISION=03cc767ee33126b7fc7890ad57047e9dd6914cca
        LOCAL_NAME=swe-rebench-v2-filtered-verified
        HF_INCLUDE=$'data/*.parquet\nREADME.md\nLICENSE*\n*.json\n*.jsonl'
        ;;
    swe-gym)
        HF_REPO=SWE-Gym/SWE-Gym
        HF_REVISION=bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb
        LOCAL_NAME=swe-gym
        HF_INCLUDE=$'data/*.parquet\nREADME.md\nLICENSE*'
        ;;
    swe-bench-verified)
        HF_REPO=princeton-nlp/SWE-bench_Verified
        HF_REVISION=c104f840cc67f8b6eec6f759ebc8b2693d585d4a
        LOCAL_NAME=swe-bench-verified
        HF_INCLUDE=$'data/*.parquet\nREADME.md\nLICENSE*'
        ;;
    *)
        echo "unknown SWE_SOURCE=${SWE_SOURCE}" >&2
        exit 2
        ;;
esac

export HF_REPO HF_REVISION LOCAL_NAME HF_INCLUDE
[[ -r "${CONTAINER_IMAGE}" ]] || {
    echo "missing CONTAINER_IMAGE=${CONTAINER_IMAGE}; run experiments/setup.sh container --submit" >&2
    exit 1
}
miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    env -u BASH_ENV -u ENV bash --noprofile --norc -c '
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
target="/data/miles-swe/sources/${LOCAL_NAME}"
mkdir -p "${target}"
download_args=()
while IFS= read -r pattern; do
    [[ -z "${pattern}" ]] || download_args+=(--include "${pattern}")
done <<<"${HF_INCLUDE}"
hf download \
    --repo-type dataset \
    --revision "${HF_REVISION}" \
    "${download_args[@]}" \
    --local-dir "${target}" \
    "${HF_REPO}"
payload_count=$(find "${target}" -type f \
    ! -path "*/.cache/*" ! -name README.md ! -name .gitattributes | wc -l)
[[ "${payload_count}" -gt 0 ]] || {
    echo "pinned download produced no dataset payload" >&2
    exit 1
}
printf "repo=%s\nrevision=%s\ninclude=%s\n" \
    "${HF_REPO}" "${HF_REVISION}" "${HF_INCLUDE}" \
    >"${target}/MILES_SOURCE_PROVENANCE"
du -sh "${target}"
'
