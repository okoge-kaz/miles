#!/bin/bash

# Required Harbor collect hook. A root-owned failure sentinel is baked into the
# image; only a complete, bounded, atomic snapshot replaces it.

set -u

# This hook runs as root after an untrusted UID 1000 agent.  Do not inherit an
# agent-writable shell/Git configuration or permit repository attributes to
# select external diff/textconv programs.  The local repository config is
# root-owned by prepare_agent.sh; the explicit options below disable the two
# local config features that can launch background hooks during collection.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/opt/miles-swe/root-home
export XDG_CONFIG_HOME=/opt/miles-swe/root-home/xdg
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_COUNT=0
export GIT_ATTR_NOSYSTEM=1
unset BASH_ENV CDPATH ENV GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR \
    GIT_DIR GIT_EXTERNAL_DIFF GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
    GIT_SSH GIT_SSH_COMMAND GIT_WORK_TREE SSH_ASKPASS

target=/opt/miles-swe/collected/model.patch
invalid() {
    exit 0
}
fatal() {
    exit 2
}
[[ -d "${HOME}" && ! -L "${HOME}" && "$(stat -c '%u:%a' "${HOME}")" == 0:700 ]] \
    || fatal
repo="$(cat /opt/miles-swe/workdir)" || fatal
gitdir="$(cat /opt/miles-swe/gitdir)" || fatal
[[ -d "${gitdir}" && ! -L "${gitdir}" && "$(stat -c '%u' "${gitdir}")" == 0 ]] \
    || fatal
gitdir_mode="$(stat -c '%a' "${gitdir}")" || fatal
(( (8#${gitdir_mode} & 8#022) == 0 )) || fatal
safe_git() {
    command git -c core.fsmonitor=false -c core.hooksPath=/dev/null \
        --git-dir="${gitdir}" --work-tree="${repo}" "$@"
}
tmp="$(mktemp /opt/miles-swe/collected/.model.patch.XXXXXX)" || fatal
cleanup() {
    rm -f -- "${tmp}"
}
trap cleanup EXIT

for proc in /proc/[0-9]*; do
    uid="$(awk '/^Uid:/ { print $2; exit }' "${proc}/status" 2>/dev/null)" || continue
    [[ "${uid}" == 1000 ]] || continue
    kill -STOP "${proc##*/}" 2>/dev/null || fatal
done
for proc in /proc/[0-9]*; do
    uid="$(awk '/^Uid:/ { print $2; exit }' "${proc}/status" 2>/dev/null)" || continue
    [[ "${uid}" == 1000 ]] || continue
    state="$(awk '/^State:/ { print $2; exit }' "${proc}/status" 2>/dev/null)" || fatal
    [[ "${state}" == T || "${state}" == t ]] || fatal
done

safe_git rev-parse --show-toplevel >/dev/null 2>&1 || invalid
untracked_count="$(safe_git ls-files --others --exclude-standard -z \
    | tr -cd '\000' | wc -c)" || invalid
[[ "${untracked_count}" -le 200 ]] || invalid
safe_git add -A >/dev/null 2>&1 || invalid
file_count="$(safe_git diff --cached --name-only -z \
    --no-ext-diff --no-textconv \
    | tr -cd '\000' | wc -c)" || invalid
[[ "${file_count}" -le 200 ]] || invalid
safe_git diff --cached --binary --full-index \
    --no-ext-diff --no-textconv HEAD -- >"${tmp}" || invalid
[[ "$(stat -c '%s' "${tmp}")" -le 16777216 ]] || invalid
chmod 0600 "${tmp}" || fatal
mv -f "${tmp}" "${target}" || fatal
trap - EXIT
