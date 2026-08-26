#!/bin/bash
#SBATCH --job-name=swe-source-inventory
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

# Inspect an immutable repository-SWE source image without exposing the
# submission environment inside it. This is an admission-policy diagnostic,
# not a substitute for the live E2B empty=0/oracle=1 checks.

set -euo pipefail
umask 077

: "${SWE_SOURCE_IMAGE:?set an immutable docker.io/...@sha256:... reference}"
: "${SWE_SOURCE_SCHEMA:?set r2e-gym-v1, swe-rebench-v2, or swe-gym}"

[[ "${SWE_SOURCE_IMAGE}" =~ ^docker\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$ ]] || {
    echo "SWE_SOURCE_IMAGE must be an immutable docker.io digest" >&2
    exit 2
}
case "${SWE_SOURCE_SCHEMA}" in
    r2e-gym-v1)
        [[ "${SWE_SOURCE_IMAGE}" == docker.io/namanjain12/*_final@sha256:* ]] || exit 2
        ;;
    swe-rebench-v2)
        [[ "${SWE_SOURCE_IMAGE}" == docker.io/swerebenchv2/* ]] || exit 2
        ;;
    swe-gym)
        [[ "${SWE_SOURCE_IMAGE}" == docker.io/xingyaoww/sweb.eval.x86_64.* ]] || exit 2
        ;;
    *)
        echo "unsupported SWE_SOURCE_SCHEMA=${SWE_SOURCE_SCHEMA}" >&2
        exit 2
        ;;
esac

srun \
    --container-image="${SWE_SOURCE_IMAGE}" \
    --container-writable \
    --no-container-mount-home \
    --export=SWE_SOURCE_SCHEMA \
    env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        SWE_SOURCE_SCHEMA="${SWE_SOURCE_SCHEMA}" \
        bash --noprofile --norc -c '
set -euo pipefail
export HOME=/tmp/miles-swe-inventory-home
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
[[ -d "${repo}/.git" && ! -L "${repo}/.git" ]] || {
    echo "source image working directory has no regular Git ancestor" >&2
    exit 1
}
inventory_nul=/tmp/miles-swe-untracked.nul
inventory=/tmp/miles-swe-untracked.txt
git -C "${repo}" -c safe.directory="${repo}" \
    ls-files --others -z >"${inventory_nul}"
: >"${inventory}"
untracked_count=0
while IFS= read -r -d "" path; do
    [[ "${path}" != *$'"'"'\r'"'"'* && "${path}" != *$'"'"'\n'"'"'* ]] || exit 3
    printf "%s\n" "${path}" >>"${inventory}"
    untracked_count=$((untracked_count + 1))
done <"${inventory_nul}"

printf "schema=%s\n" "${SWE_SOURCE_SCHEMA}"
printf "repo=%s\n" "${repo}"
printf "head=%s\n" "$(git -C "${repo}" -c safe.directory="${repo}" rev-parse HEAD)"
printf "untracked_count=%s\n" "${untracked_count}"
printf "untracked_inventory_sha256=%s\n" "$(sha256sum "${inventory_nul}" | cut -d" " -f1)"
echo "suspicious_untracked_paths_begin"
awk '"'"'BEGIN {IGNORECASE=1}
    /(^|\/)(test(s|bed)?|gold(en)?|oracle|answer|solution|expected|patch|issue)([._\/-]|$)/ ||
    /(^|\/)(fail_to_pass|pass_to_pass|test_patch|gold_patch|expected_output)([._\/-]|$)/ {
        if (printed < 200) print
        printed++
    }'"'"' "${inventory}"
echo "suspicious_untracked_paths_end"
'
