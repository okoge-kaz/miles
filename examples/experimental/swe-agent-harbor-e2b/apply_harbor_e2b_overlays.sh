#!/bin/bash

# Apply the exact Miles/E2B overlays to the pinned Harbor checkout.  The
# canonical stage digests make this safe for both a clean checkout and a
# checkout left at any earlier overlay stage; every other state fails closed.

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
HARBOR_CHECKOUT="${1:-${HARBOR_ROOT:-}}"
EXPECTED_COMMIT=2ce5ba2af33a00c9fba0463f6403313996373f85

[[ -n "${HARBOR_CHECKOUT}" ]] || {
    echo "usage: $0 /absolute/path/to/pinned-harbor" >&2
    exit 2
}
[[ "${HARBOR_CHECKOUT}" == /* && -d "${HARBOR_CHECKOUT}/.git" && \
    ! -L "${HARBOR_CHECKOUT}" ]] || {
    echo "Harbor checkout must be an absolute, symlink-free Git directory" >&2
    exit 2
}
[[ "$(stat -c '%u' "${HARBOR_CHECKOUT}")" == "$(id -u)" ]] || {
    echo "Harbor checkout must be owned by the launching user" >&2
    exit 2
}
[[ "$(git -C "${HARBOR_CHECKOUT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
    echo "Harbor must be pinned to ${EXPECTED_COMMIT}" >&2
    exit 2
}

PATCH_FILES=(
    "${SCRIPT_DIR}/harbor-miles-e2b.patch"
    "${SCRIPT_DIR}/harbor-swe-collect-hardening.patch"
    "${SCRIPT_DIR}/harbor-e2b-no-new-privs.patch"
    "${SCRIPT_DIR}/harbor-e2b-late-verifier-tests.patch"
    "${SCRIPT_DIR}/harbor-agent-server-auth-attestation.patch"
)
PATCH_HASHES=(
    8f8305e8e91a0a802908f4c9691607178d2bca21f10e7ecbe1998230b3a62438
    4c3c7c384b339edc44698385f21778f138dfff654649e77c249084eaebe6d818
    16ab79612486f77efae847eed9f20c4722a8751836dd179d3a5777e31a425f34
    5cac5e94ee6be5474a17700f11d89fbb3cec2c00e2c28b230f0cd9f73603b714
    949d1e3930eb0b5d64b1f60b865e59b92e04a82c56a47f59bccb253b01f851fc
)
STAGE_HASHES=(
    763d7ca81481a947a486cb9d420f6999edb35e425ccae5d5174f97471ee1db66
    ef8439ae798b8716aa8f073f2bf41b124775223d12382a9931757fbfabce09fe
    a2ab8b35dffd7fdd63ad2d2e511039691e1b75cfa3c0f14ba620ebeb8aeda35e
    68189951d9df0050ecd4cbe5a8b297e339a30890a3bdc55d12692e36195da048
    22c9979920d05180444218b57e401b0af2eaca856dd42e324b32ac52049ab57c
    0249f6f072db2fcd43230867298ae50da4a4b02fcfccaf1742ee60f8b1e8a8cb
)
OVERLAY_PATHS=(
    agent_server/models.py
    agent_server/trial_runner.py
    miles_agent_server.py
    src/harbor/environments/e2b.py
    src/harbor/models/task/config.py
    src/harbor/trial/errors.py
    src/harbor/trial/private_verifier_package.py
    src/harbor/trial/single_step.py
    src/harbor/trial/trial.py
    tests/unit/environments/test_e2b.py
    tests/unit/models/test_artifact_validation.py
    tests/unit/test_miles_agent_server_results.py
    tests/unit/test_single_step_trial.py
    tests/unit/test_trial_verifier_artifact_transfer.py
)

for index in "${!PATCH_FILES[@]}"; do
    patch_file="${PATCH_FILES[$index]}"
    [[ -f "${patch_file}" && ! -L "${patch_file}" ]] || {
        echo "Harbor overlay is missing or unsafe: $(basename -- "${patch_file}")" >&2
        exit 2
    }
    actual_hash="$(sha256sum "${patch_file}")"
    actual_hash="${actual_hash%% *}"
    [[ "${actual_hash}" == "${PATCH_HASHES[$index]}" ]] || {
        echo "Harbor overlay checksum mismatch: $(basename -- "${patch_file}")" >&2
        exit 2
    }
done

overlay_tree_digest() {
    {
        for relative in "${OVERLAY_PATHS[@]}"; do
            path="${HARBOR_CHECKOUT}/${relative}"
            if [[ -f "${path}" && ! -L "${path}" ]]; then
                mode="$(stat -c '%a' "${path}")"
                [[ "$(stat -c '%u' "${path}")" == "$(id -u)" ]] && \
                    (( (8#${mode} & 8#022) == 0 )) || {
                    echo "Harbor overlay path is not safely owned: ${relative}" >&2
                    return 9
                }
                file_hash="$(sha256sum "${path}")"
                file_hash="${file_hash%% *}"
                printf 'file %s %s\n' "${relative}" "${file_hash}"
            elif [[ ! -e "${path}" && ! -L "${path}" ]]; then
                printf 'missing %s\n' "${relative}"
            else
                echo "Harbor overlay path is a symlink or special file: ${relative}" >&2
                return 9
            fi
        done
    } | sha256sum | cut -d' ' -f1
}

path_is_overlay_owned() {
    local candidate="$1"
    local expected
    for expected in "${OVERLAY_PATHS[@]}"; do
        [[ "${candidate}" == "${expected}" ]] && return 0
    done
    return 1
}

while IFS= read -r -d '' untracked_path; do
    [[ "${untracked_path}" == "src/harbor/trial/private_verifier_package.py" ]] || {
        echo "Pinned Harbor checkout has an unrelated untracked file" >&2
        exit 2
    }
done < <(git -C "${HARBOR_CHECKOUT}" ls-files --others --exclude-standard -z)

while IFS= read -r changed_path; do
    [[ -z "${changed_path}" ]] && continue
    path_is_overlay_owned "${changed_path}" || {
        echo "Pinned Harbor checkout has an unrelated tracked change: ${changed_path}" >&2
        exit 2
    }
done < <(git -C "${HARBOR_CHECKOUT}" diff --name-only HEAD --)

current_hash="$(overlay_tree_digest)"
stage=-1
for index in "${!STAGE_HASHES[@]}"; do
    if [[ "${current_hash}" == "${STAGE_HASHES[$index]}" ]]; then
        stage="${index}"
        break
    fi
done
(( stage >= 0 )) || {
    echo "Harbor overlay tree is neither clean nor an exact admitted stage" >&2
    exit 2
}

for index in "${!PATCH_FILES[@]}"; do
    (( index < stage )) && continue
    patch_file="${PATCH_FILES[$index]}"
    git -C "${HARBOR_CHECKOUT}" apply --unidiff-zero --check "${patch_file}"
    git -C "${HARBOR_CHECKOUT}" apply --unidiff-zero "${patch_file}"
    current_hash="$(overlay_tree_digest)"
    [[ "${current_hash}" == "${STAGE_HASHES[$((index + 1))]}" ]] || {
        echo "Harbor overlay post-apply tree mismatch" >&2
        exit 2
    }
done

echo "Harbor E2B overlays verified at pinned commit ${EXPECTED_COMMIT}."
