#!/bin/bash

# Create the only identity permitted to execute model-controlled repository
# code in a verifier. Trusted patch application and grading remain root-only.

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

runtime_excludes="$(mktemp)"
trap 'rm -f -- "${runtime_excludes}"' EXIT

for command in awk git getent setpriv setsid timeout; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "SWE verifier image lacks required command: ${command}" >&2
        exit 1
    }
done
for command in /bin/bash /usr/bin/env /usr/bin/setpriv; do
    [[ -x "${command}" ]] || {
        echo "SWE verifier image lacks required hardening executable: ${command}" >&2
        exit 1
    }
done

if [[ -d /testbed/.git && ! -L /testbed/.git ]]; then
    repo=/testbed
else
    repo="$(pwd -P)"
    while [[ "${repo}" != / && ! -d "${repo}/.git" ]]; do
        repo="${repo%/*}"
        [[ -n "${repo}" ]] || repo=/
    done
fi
[[ -d "${repo}/.git" && ! -L "${repo}/.git" ]] || {
    echo "SWE verifier source must use a real .git directory" >&2
    exit 1
}
repo="$(cd "${repo}" && pwd -P)"
source_gitdir="$(cd "${repo}/.git" && pwd -P)"
[[ "${source_gitdir}" == "${repo}/.git" ]] || {
    echo "SWE verifier source Git directory escapes the repository" >&2
    exit 1
}
install -o root -g root -m 0600 /dev/null "${source_gitdir}/config"
rm -f -- "${source_gitdir}/config.worktree" "${source_gitdir}/info/attributes"
base_commit="${MILES_SWE_BASE_COMMIT:?}"
export GIT_CONFIG_VALUE_0="${repo}"
base_tree="$(git -C "${repo}" rev-parse "${base_commit}^{tree}")"
git -C "${repo}" checkout --detach "${base_commit}"
git -C "${repo}" reset --hard "${base_commit}"
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${base_commit}" ]] || {
    echo "SWE verifier image did not reach the exact task base" >&2
    exit 1
}

while IFS= read -r -d '' path; do
    [[ "${path}" != *$'\n'* ]] || {
        echo "verifier image has an untracked path containing a newline" >&2
        exit 1
    }
    printf '/%s\n' "${path}" >>"${runtime_excludes}"
done < <(git -C "${repo}" ls-files --others --exclude-standard -z)

# Remove the source repository object database wholesale. Deleting refs plus
# gc is insufficient when source images contain alternates, replace refs, or
# pack .keep files. Only the exact published base tree is committed into the
# fresh verifier repository; image-owned runtimes remain untracked/excluded.
rm -rf -- "${repo}/.git"
git -C "${repo}" init -q --template=
git -C "${repo}" config user.email verifier@miles.invalid
git -C "${repo}" config user.name Miles
if [[ -s "${runtime_excludes}" ]]; then
    install -m 0600 "${runtime_excludes}" "${repo}/.git/info/exclude"
fi
git -C "${repo}" add -A >/dev/null
tree="$(git -C "${repo}" write-tree)"
[[ "${tree}" == "${base_tree}" ]] || {
    echo "fresh verifier tree differs from the published base tree" >&2
    exit 1
}
commit="$(printf 'Verifier base state\n' | git -C "${repo}" \
    -c user.email=verifier@miles.invalid -c user.name=Miles commit-tree "${tree}")"
git -C "${repo}" symbolic-ref HEAD refs/heads/__miles_swe_verifier_base
git -C "${repo}" update-ref refs/heads/__miles_swe_verifier_base "${commit}"
if git -C "${repo}" cat-file -e "${base_commit}^{commit}"; then
    echo "source-image Git history remains readable in the fresh verifier" >&2
    exit 1
fi
if [[ -n "$(git -C "${repo}" fsck --no-reflogs --unreachable --no-progress 2>&1)" ]]; then
    echo "fresh verifier repository contains unreachable Git objects" >&2
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
printf '%s\n' "${repo}" >/opt/miles-swe/verifier-workdir
printf '%s\n' "${verifier_group}" >/opt/miles-swe/verifier-gid
printf '%s\n' "${commit}" >/opt/miles-swe/verifier-base-commit
chmod 0444 /opt/miles-swe/verifier-workdir /opt/miles-swe/verifier-gid \
    /opt/miles-swe/verifier-base-commit
find "${repo}" -path "${repo}/.git" -prune -o \
    -exec chown -h "1000:${verifier_group}" {} +
chown -R root:root "${repo}/.git" /opt/miles-swe
chmod -R a+rX "${repo}/.git"
chmod -R go-w "${repo}/.git"
chmod -R go-w /opt/miles-swe
