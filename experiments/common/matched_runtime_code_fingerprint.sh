#!/bin/bash
# Content identity for the live checkout used by the matched partial/concurrency
# cohort. This file is sourced by the launcher and both allocation recipes.

matched_runtime_code_paths() {
    local repo_root="${1:?repository root is required}"
    local miles_root="${repo_root}/miles"
    local miles_python_paths
    [[ -d "${miles_root}" ]] || {
        echo "runtime package directory is missing: ${miles_root}" >&2
        return 1
    }
    miles_python_paths="$(find "${miles_root}" -type f -name '*.py' -printf 'miles/%P\n')" || return 1

    {
        printf '%s\n' \
            train.py \
            train_async.py \
            experiments/env.sh \
            experiments/staleness_ratio_sweep.sh \
            experiments/common/clean_checkpoint.sh \
            experiments/common/matched_runtime_code_fingerprint.sh \
            experiments/common/placement.sh \
            experiments/common/ray_cluster.sh \
            experiments/common/run_identity.sh \
            experiments/common/wait_for_ray_nodes.py \
            experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/run.sbatch \
            experiments/scripts/math/async/dapo-math-p10-90/qwen3-4b/train.sh \
            experiments/scripts/math/sync/dapo-math-p10-90/qwen3-4b/run.sbatch \
            experiments/scripts/math/sync/dapo-math-p10-90/qwen3-4b/train.sh \
            scripts/models/qwen3-4B.sh
        printf '%s\n' "${miles_python_paths}"
    } | LC_ALL=C sort -u
}

matched_runtime_code_fingerprint() {
    local repo_root="${1:?repository root is required}"
    local relative_path file_digest digest_output
    local fingerprint_material=""
    local relative_path_output
    local -a relative_paths=()

    relative_path_output="$(matched_runtime_code_paths "${repo_root}")" || return 1
    mapfile -t relative_paths <<< "${relative_path_output}"
    (( ${#relative_paths[@]} > 0 )) || {
        echo "runtime fingerprint file list is empty" >&2
        return 1
    }
    for relative_path in "${relative_paths[@]}"; do
        [[ -f "${repo_root}/${relative_path}" ]] || {
            echo "runtime fingerprint input is missing: ${relative_path}" >&2
            return 1
        }
        digest_output="$(sha256sum -- "${repo_root}/${relative_path}")"
        file_digest="${digest_output%% *}"
        fingerprint_material+="${relative_path}"$'\t'"${file_digest}"$'\n'
    done
    printf '%s' "${fingerprint_material}" | sha256sum | awk '{print $1}'
}

verify_matched_runtime_code_fingerprint() {
    local repo_root="${1:?repository root is required}"
    local expected="${MATCHED_RUNTIME_CODE_FINGERPRINT:-}"
    local actual

    # Direct recipe submissions and every pre-existing launcher remain a no-op.
    [[ "${MATCHED_PARTIAL_CONCURRENCY_COHORT:-0}" == 1 ]] || return 0
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "matched cohort requires a 64-character runtime code fingerprint" >&2
        return 1
    }
    actual="$(matched_runtime_code_fingerprint "${repo_root}")"
    [[ "${actual}" == "${expected}" ]] || {
        echo "matched cohort live checkout changed after submission:" \
             "expected ${expected}, got ${actual}" >&2
        return 1
    }
    echo "matched runtime code fingerprint ${actual}"
}
