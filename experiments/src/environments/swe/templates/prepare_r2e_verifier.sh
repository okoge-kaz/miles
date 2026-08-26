#!/bin/bash

# Pin a fresh R2E verifier image to the unique parent of the private gold
# commit. The source image's default HEAD is never accepted as a grading base.

set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/opt/miles-swe/root-home
export XDG_CONFIG_HOME=/opt/miles-swe/root-home/xdg
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0='*'
export GIT_CONFIG_KEY_1=core.fsmonitor
export GIT_CONFIG_VALUE_1=false
export GIT_CONFIG_KEY_2=core.hooksPath
export GIT_CONFIG_VALUE_2=/dev/null
export GIT_ATTR_NOSYSTEM=1
unset BASH_ENV CDPATH ENV GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR \
    GIT_DIR GIT_EXTERNAL_DIFF GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
    GIT_SSH GIT_SSH_COMMAND GIT_WORK_TREE SSH_ASKPASS

for command in awk git getent setpriv setsid timeout; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "R2E verifier image lacks required command: ${command}" >&2
        exit 1
    }
done
for command in /bin/bash /usr/bin/env /usr/bin/setpriv; do
    [[ -x "${command}" ]] || {
        echo "R2E verifier image lacks required hardening executable: ${command}" >&2
        exit 1
    }
done

base_commit="${MILES_SWE_BASE_COMMIT:-}"
gold_commit="${MILES_SWE_GOLD_COMMIT:-}"
repo=/testbed
[[ -d "${repo}/.git" && ! -L "${repo}/.git" ]] || {
    echo "R2E verifier source must use a real .git directory" >&2
    exit 1
}
source_gitdir="$(cd "${repo}/.git" && pwd -P)"
[[ "${source_gitdir}" == "${repo}/.git" ]] || {
    echo "R2E verifier source Git directory escapes the repository" >&2
    exit 1
}
install -o root -g root -m 0600 /dev/null "${source_gitdir}/config"
rm -f -- "${source_gitdir}/config.worktree" "${source_gitdir}/info/attributes"
[[ -f "${repo}/run_tests.sh" && ! -L "${repo}/run_tests.sh" ]] || {
    echo "R2E source image lacks a regular trusted test runner" >&2
    exit 1
}
install -d -o root -g root -m 0700 /opt/miles-swe/private
install -o root -g root -m 0500 "${repo}/run_tests.sh" \
    /opt/miles-swe/private/run_tests.sh
[[ "${gold_commit}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "R2E verifier requires the private gold-commit binding" >&2
    exit 1
}
git -C "${repo}" cat-file -e "${gold_commit}^{commit}" || {
    echo "R2E verifier source does not contain the bound gold commit" >&2
    exit 1
}
export GIT_CONFIG_VALUE_0="${repo}"
gold_parent_line="$(git -C "${repo}" show -s --format='%P' "${gold_commit}")"
gold_parents=()
if [[ -n "${gold_parent_line}" ]]; then
    read -r -a gold_parents <<<"${gold_parent_line}"
fi
[[ "${#gold_parents[@]}" == 1 ]] || {
    echo "R2E gold commit must have exactly one parent" >&2
    exit 1
}
derived_base="${gold_parents[0]}"
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${derived_base}" ]] || {
    echo "R2E verifier source HEAD is not the bound gold parent" >&2
    exit 1
}
if [[ -n "${base_commit}" && "${base_commit}" != "${derived_base}" ]]; then
    echo "published R2E base_commit does not match the gold parent" >&2
    exit 1
fi
base_tree="$(git -C "${repo}" rev-parse "${derived_base}^{tree}")"

git -C "${repo}" checkout --detach "${derived_base}"
git -C "${repo}" reset --hard "${derived_base}"
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${derived_base}" ]] || {
    echo "fresh R2E verifier did not reach the derived base" >&2
    exit 1
}

secret_names=(
    syn_issue.json
    expected_test_output.json
    execution_result.json
    parsed_commit.json
    modified_files.json
    modified_entities.json
)
for secret_name in "${secret_names[@]}"; do
    rm -f -- "/${secret_name}" "/root/${secret_name}" "/testbed/${secret_name}"
done
find "${repo}" -type f \( \
    -name syn_issue.json -o -name expected_test_output.json -o \
    -name execution_result.json -o -name parsed_commit.json -o \
    -name modified_files.json -o -name modified_entities.json -o \
    -name '*.pyc' \
\) -delete
find "${repo}" -type d -name __pycache__ -prune -exec rm -rf -- {} +
rm -f -- "${repo}/run_tests.sh"
git -C "${repo}" clean -ffdx -e .venv -e '.venv/**'

# Delete the source object database wholesale so alternates, replace refs, and
# kept packs cannot preserve the answer. The image-owned virtualenv remains
# untracked and excluded from the fresh synthetic base.
rm -rf -- "${repo}/.git"
git -C "${repo}" init -q --template=
git -C "${repo}" config user.email verifier@miles.invalid
git -C "${repo}" config user.name Miles
printf '/.venv/\n' >"${repo}/.git/info/exclude"
git -C "${repo}" add -A >/dev/null
tree="$(git -C "${repo}" write-tree)"
[[ "${tree}" == "${base_tree}" ]] || {
    echo "fresh R2E verifier tree differs from the derived base tree" >&2
    exit 1
}
commit="$(printf 'Verifier base state\n' | git -C "${repo}" \
    -c user.email=verifier@miles.invalid -c user.name=Miles commit-tree "${tree}")"
git -C "${repo}" symbolic-ref HEAD refs/heads/__miles_swe_verifier_base
git -C "${repo}" update-ref refs/heads/__miles_swe_verifier_base "${commit}"
if git -C "${repo}" cat-file -e "${gold_commit}^{commit}"; then
    echo "gold commit remains readable in the fresh verifier" >&2
    exit 1
fi
if git -C "${repo}" cat-file -e "${derived_base}^{commit}"; then
    echo "source-image base commit remains readable in the fresh verifier" >&2
    exit 1
fi
if [[ -n "$(git -C "${repo}" fsck --no-reflogs --unreachable --no-progress 2>&1)" ]]; then
    echo "fresh R2E verifier repository contains unreachable Git objects" >&2
    exit 1
fi

if ! getent passwd 1000 >/dev/null; then
    useradd --create-home --uid 1000 --shell /bin/bash miles-verifier
fi
verifier_user="$(getent passwd 1000 | cut -d: -f1)"
verifier_group="$(getent passwd 1000 | cut -d: -f4)"
[[ "$(id -u "${verifier_user}")" == 1000 && "${verifier_group}" != 0 ]] || {
    echo "verifier UID 1000 must have a non-root primary group" >&2
    exit 1
}
usermod -G "${verifier_group}" "${verifier_user}"
if command -v sudo >/dev/null 2>&1 && \
    su -s /bin/sh "${verifier_user}" -c 'sudo -n true' >/dev/null 2>&1; then
    echo "verifier UID 1000 retains passwordless sudo privileges" >&2
    exit 1
fi

install -d -m 0755 /opt/miles-swe
printf '%s\n' "${commit}" >/opt/miles-swe/verifier-base-commit
printf '%s\n' "${verifier_group}" >/opt/miles-swe/verifier-gid
chmod 0444 /opt/miles-swe/verifier-base-commit
chmod 0444 /opt/miles-swe/verifier-gid
chown -R root:root "${repo}/.git" /opt/miles-swe
chmod -R a+rX "${repo}/.git"
chmod -R go-w "${repo}/.git"
chmod -R go-w /opt/miles-swe
find "${repo}" -path "${repo}/.git" -prune -o \
    -exec chown -h "1000:${verifier_group}" {} +
