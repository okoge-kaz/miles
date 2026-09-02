#!/bin/bash
#PBS -N r2e-gold-exposure
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Inspect an R2E source image, whose checkout must be the privately bound gold
# commit's unique parent, for exact copies of gold-touched files outside the
# repository. A clean result narrows one leakage class; it is not proof that an
# arbitrary renamed or encoded oracle artifact is absent.

set -euo pipefail
umask 077

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
umask 077

: "${SWE_SOURCE_IMAGE:?set an immutable R2E docker.io/...@sha256:... reference}"
: "${SWE_EXPECTED_GOLD:?set the private task gold commit}"
[[ "${SWE_SOURCE_IMAGE}" =~ ^docker\.io/namanjain12/[a-z0-9._-]+_final@sha256:[0-9a-f]{64}$ ]] || {
    echo "SWE_SOURCE_IMAGE is not an approved immutable R2E image" >&2
    exit 2
}
if [[ ! "${SWE_EXPECTED_GOLD}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "SWE_EXPECTED_GOLD is not a full Git commit" >&2
    exit 2
fi

miles_srun \
    --image="docker://${SWE_SOURCE_IMAGE}" \
    --writable-tmpfs \
    --no-home \
    --export=SWE_EXPECTED_GOLD \
    env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        SWE_EXPECTED_GOLD="${SWE_EXPECTED_GOLD:-}" \
        bash --noprofile --norc -c '
set -euo pipefail
export HOME=/tmp/miles-r2e-exposure-home
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
mkdir -m 0700 "${HOME}"

if [[ -d /testbed/.git && ! -L /testbed/.git ]]; then
    repo=/testbed
else
    repo="$(pwd -P)"
    while [[ "${repo}" != / && ! -d "${repo}/.git" ]]; do
        repo="${repo%/*}"
        [[ -n "${repo}" ]] || repo=/
    done
fi
[[ -d "${repo}/.git" && ! -L "${repo}/.git" ]] || exit 3
safe_git=(git --no-pager -c "safe.directory=${repo}" -C "${repo}")
image_head="$("${safe_git[@]}" rev-parse HEAD)"
gold="${SWE_EXPECTED_GOLD}"
if ! "${safe_git[@]}" cat-file -e "${gold}^{commit}" 2>/dev/null; then
    printf "repo=%s\nimage_head=%s\nexpected_gold=%s\n" \
        "${repo}" "${image_head}" "${gold}"
    echo "expected gold commit is absent from the image Git object database" >&2
    exit 4
fi
parent_line="$("${safe_git[@]}" show -s --format="%P" "${gold}")"
parents=()
if [[ -n "${parent_line}" ]]; then
    read -r -a parents <<<"${parent_line}"
fi
printf "repo=%s\n" "${repo}"
printf "image_head=%s\n" "${image_head}"
printf "gold=%s\n" "${gold}"
printf "gold_parent_count=%s\n" "${#parents[@]}"
printf "gold_parents=%s\n" "${parents[*]:-none}"
printf "gold_subject=%s\n" "$("${safe_git[@]}" show -s --format=%s "${gold}")"
printf "gold_author_time=%s\n" "$("${safe_git[@]}" show -s --format=%aI "${gold}")"
printf "gold_tree=%s\n" "$("${safe_git[@]}" show -s --format=%T "${gold}")"
printf "ref_count=%s\n" "$("${safe_git[@]}" for-each-ref --format="%(refname)" | wc -l)"
printf "commit_count_all_refs=%s\n" "$("${safe_git[@]}" rev-list --all --count)"
printf "tracked_file_count=%s\n" "$("${safe_git[@]}" ls-files -z | tr -cd "\0" | wc -c)"
printf "untracked_file_count=%s\n" "$("${safe_git[@]}" ls-files --others --exclude-standard -z | tr -cd "\0" | wc -c)"
printf "ignored_file_count=%s\n" "$("${safe_git[@]}" ls-files --others --ignored --exclude-standard -z | tr -cd "\0" | wc -c)"
printf "worktree_tree=%s\n" "$("${safe_git[@]}" write-tree)"
echo "refs_begin"
"${safe_git[@]}" for-each-ref --format="%(refname) %(objectname)"
echo "refs_end"
echo "recent_commits_begin"
"${safe_git[@]}" log --all --max-count=10 --format="%H %P %T %s"
echo "recent_commits_end"
echo "known_artifact_paths_begin"
find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \
    \( -name r2e_tests -o -name syn_issue.json -o \
       -name expected_test_output.json -o -name execution_result.json -o \
       -name parsed_commit.json -o -name modified_files.json -o \
       -name modified_entities.json -o -name run_tests.sh \) -print 2>/dev/null
echo "known_artifact_paths_end"
printf "venv_size_bytes=%s\n" "$(du -sb -- "${repo}/.venv" | cut -f1)"
printf "venv_entry_count=%s\n" "$(find "${repo}/.venv" -xdev -printf . | wc -c)"
printf "venv_symlink_count=%s\n" "$(find "${repo}/.venv" -xdev -type l -printf . | wc -c)"
echo "venv_symlinks_begin"
find "${repo}/.venv" -xdev -type l -printf "%p -> %l\n" | head -n 100
echo "venv_symlinks_end"
[[ "${#parents[@]}" == 1 ]] || {
    echo "R2E gold commit does not have exactly one parent" >&2
    exit 4
}
base="${parents[0]}"
[[ "${image_head}" == "${base}" ]] || {
    echo "R2E source checkout is not the privately bound gold parent" >&2
    exit 4
}
changed_nul=/tmp/miles-r2e-changed.nul
"${safe_git[@]}" diff --name-only -z "${base}" "${gold}" >"${changed_nul}"

changed_count=0
duplicate_count=0
echo "exact_gold_duplicates_begin"
while IFS= read -r -d "" relative; do
    [[ "${relative}" != *$'"'"'\r'"'"'* && "${relative}" != *$'"'"'\n'"'"'* ]] || exit 5
    changed_count=$((changed_count + 1))
    blob="$("${safe_git[@]}" rev-parse "${gold}:${relative}" 2>/dev/null)" || continue
    [[ "$("${safe_git[@]}" cat-file -t "${blob}")" == blob ]] || continue
    gold_file="/tmp/miles-r2e-gold-file-${changed_count}"
    "${safe_git[@]}" cat-file blob "${blob}" >"${gold_file}"
    size="$(stat -c %s "${gold_file}")"
    digest="$(sha256sum "${gold_file}" | cut -d" " -f1)"
    while IFS= read -r -d "" candidate; do
        [[ "${candidate}" == "${repo}/"* ]] && continue
        [[ "$(sha256sum "${candidate}" | cut -d" " -f1)" == "${digest}" ]] || continue
        printf "%s\n" "${candidate}"
        duplicate_count=$((duplicate_count + 1))
    done < <(find /usr /opt /root -xdev -type f \
        -size "${size}c" -readable -print0 2>/dev/null)
    rm -f -- "${gold_file}"
done <"${changed_nul}"
echo "exact_gold_duplicates_end"
gold_text_count=0
echo "gold_commit_text_paths_begin"
while IFS= read -r -d "" candidate; do
    [[ "${candidate}" == "${repo}/"* ]] && continue
    if grep -a -F -l -- "${gold}" "${candidate}" 2>/dev/null; then
        gold_text_count=$((gold_text_count + 1))
    fi
done < <(find /usr /opt /root -xdev -type f -readable -print0 2>/dev/null)
echo "gold_commit_text_paths_end"
printf "gold=%s\nbase=%s\n" "${gold}" "${base}"
printf "gold_changed_file_count=%s\n" "${changed_count}"
printf "exact_gold_duplicate_count=%s\n" "${duplicate_count}"
printf "gold_commit_text_path_count=%s\n" "${gold_text_count}"
[[ "${duplicate_count}" == 0 ]] || exit 6
[[ "${gold_text_count}" == 0 ]] || exit 7
echo "r2e-gold-image-exposure-narrow-check-ok"
'
