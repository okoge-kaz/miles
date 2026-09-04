#!/bin/bash
# Stage the pinned public datasets requested for SWE and tool-use RL.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(realpath "${SCRIPT_DIR}/../../..")"
source "${REPO_ROOT}/experiments/env.sh"

readonly SWE_REBENCH_REVISION=475dd5e8703bb5fb22dd3c60b5d038b019eba1e0
readonly SWE_REBENCH_SHA256=0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad
readonly SWE_GYM_REVISION=bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb
readonly SWE_GYM_SHA256=60569cea74bb281f7a5579467436a2bc1932c6e0c5f2f7fa0d084392abd9ad97
readonly TOOL_PIVOT_REVISION=9643c8103d7bfbc2d7fc4d15991d6739c612ff58
readonly TOOL_PIVOT_SHA256=0909e9d7c24eee73ab0e2a9cf25bc462075cee6dd1a1f9502577a22c2a7a1f0a

has_pinned_payload() {
    local target="$1"
    local repository="$2"
    local revision="$3"
    local provenance="${target}/MILES_SOURCE_PROVENANCE"
    [[ -s "${provenance}" ]] || return 1
    grep -Fxq "repo=${repository}" "${provenance}" || return 1
    grep -Fxq "revision=${revision}" "${provenance}" || return 1
    find "${target}" -maxdepth 2 -type f \
        ! -path '*/.cache/*' ! -name README.md ! -name .gitattributes \
        ! -name MILES_SOURCE_PROVENANCE -print -quit | grep -q .
}

stage_swe() {
    local selector="$1"
    local repository="$2"
    local local_name="$3"
    local revision="$4"
    local expected_sha256="$5"
    local target="${DATASET_DIR}/miles-swe/sources/${local_name}"
    local payload="${target}/data/train-00000-of-00001.parquet"
    if has_pinned_payload "${target}" "${repository}" "${revision}" \
        && printf '%s  %s\n' "${expected_sha256}" "${payload}" \
            | sha256sum --check --status; then
        printf 'verified\t%s\t%s\n' "${selector}" "${target}"
        return
    fi
    local submission job_id
    submission="$(sbatch --parsable \
        --chdir="${REPO_ROOT}" \
        --export="USER,SWE_SOURCE=${selector}" \
        "${SCRIPT_DIR}/download_swe_datasets.sbatch")"
    job_id="${submission%%;*}"
    printf 'submitted\t%s\t%s\n' "${selector}" "${job_id}"
}

stage_tool_pivot() {
    local repository=nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1
    local local_name=nemotron-rl-conv-tooluse-pivot
    local target="${DATASET_DIR}/${local_name}"
    if has_pinned_payload "${target}" "${repository}" "${TOOL_PIVOT_REVISION}" \
        && printf '%s  %s\n' "${TOOL_PIVOT_SHA256}" "${target}/train.jsonl" \
            | sha256sum --check --status; then
        printf 'verified\tconv-tooluse-pivot\t%s\n' "${target}"
        return
    fi
    local submission job_id
    submission="$(sbatch --parsable \
        --chdir="${REPO_ROOT}" \
        --export="USER,HF_REPO=${repository},LOCAL_NAME=${local_name},HF_REVISION=${TOOL_PIVOT_REVISION}" \
        "${SCRIPT_DIR}/download_dataset.sbatch")"
    job_id="${submission%%;*}"
    printf 'submitted\tconv-tooluse-pivot\t%s\n' "${job_id}"
}

stage_swe \
    swe-rebench-v2 \
    nebius/SWE-rebench-V2 \
    swe-rebench-v2 \
    "${SWE_REBENCH_REVISION}" \
    "${SWE_REBENCH_SHA256}"
stage_swe \
    swe-gym \
    SWE-Gym/SWE-Gym \
    swe-gym \
    "${SWE_GYM_REVISION}" \
    "${SWE_GYM_SHA256}"
stage_tool_pivot
