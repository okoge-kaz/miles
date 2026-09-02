#!/bin/bash
#PBS -N r2e-agent-hardening
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Exercise the exact R2E image/runtime transformation against one immutable
# publisher image. This is not a security-hardening admission pass: Singularity may
# expose only a single UID-0 mapping. Live admission must repeat the UID-1000,
# no-network, no-new-privileges checks in a fresh E2B agent sandbox.

set -euo pipefail
umask 077

: "${SWE_SOURCE_IMAGE:?set an immutable R2E source image}"
: "${SWE_BASE_COMMIT:?set the private task base commit}"
: "${SWE_GOLD_COMMIT:?set the private task gold commit}"
[[ "${SWE_SOURCE_IMAGE}" =~ ^docker\.io/namanjain12/[a-z0-9._-]+_final@sha256:[0-9a-f]{64}$ ]] || exit 2
[[ "${SWE_BASE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "${SWE_GOLD_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2

repo_root="${PBS_O_WORKDIR:-$(pwd -P)}"
source "${repo_root}/experiments/env.sh"
umask 077
assets="${repo_root}/experiments/src/environments/swe/templates"
[[ -d "${assets}" && ! -L "${assets}" ]] || exit 2

miles_srun \
    --fakeroot \
    --image="docker://${SWE_SOURCE_IMAGE}" \
    --bind="${assets}:/mnt/miles-swe-templates:ro" \
    --writable-tmpfs \
    --no-home \
    --export=SWE_BASE_COMMIT,SWE_GOLD_COMMIT \
    env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        SWE_BASE_COMMIT="${SWE_BASE_COMMIT}" \
        SWE_GOLD_COMMIT="${SWE_GOLD_COMMIT}" \
        bash --noprofile --norc -c '
set -euo pipefail
umask 077
oracle_seeded=0
while IFS= read -r -d "" relative; do
    if git -C /testbed cat-file blob "${SWE_GOLD_COMMIT}:${relative}" \
        >/tmp/arbitrary-renamed-oracle-copy; then
        oracle_seeded=1
        break
    fi
done < <(git -C /testbed diff --name-only -z \
    "${SWE_BASE_COMMIT}" "${SWE_GOLD_COMMIT}" --)
[[ "${oracle_seeded}" == 1 && -s /tmp/arbitrary-renamed-oracle-copy ]]
printf "%s\n" "${SWE_GOLD_COMMIT}" >/tmp/arbitrary-gold-commit-copy
install -d -o root -g root -m 0700 \
    /opt/miles-swe/collected \
    /opt/miles-swe/root-home \
    /opt/miles-swe/root-home/xdg
printf "%s\n" MILES_SWE_AGENT_STATE_INVALID \
    >/opt/miles-swe/collected/model.patch
install -o root -g root -m 0555 \
    /mnt/miles-swe-templates/prepare_agent.sh \
    /opt/miles-swe/prepare_agent.sh
install -o root -g root -m 0500 \
    /mnt/miles-swe-templates/collect_agent_patch.sh \
    /opt/miles-swe/collect_agent_patch.sh
install -o root -g root -m 0500 \
    /mnt/miles-swe-templates/strip_agent_privileges.py \
    /opt/miles-swe/strip_agent_privileges.py
MILES_SWE_SCHEMA=r2e-gym-v1 \
MILES_SWE_BASE_COMMIT="${SWE_BASE_COMMIT}" \
MILES_SWE_GOLD_COMMIT="${SWE_GOLD_COMMIT}" \
MILES_SWE_RUNTIME_POLICY=none \
    /opt/miles-swe/prepare_agent.sh
[[ ! -e /tmp/arbitrary-renamed-oracle-copy && \
    ! -e /tmp/arbitrary-gold-commit-copy ]]
echo r2e-agent-prepare-ok
stat -c "repo_after_prepare_owner=%u:%g repo_after_prepare_mode=%a" /testbed
python3 /opt/miles-swe/strip_agent_privileges.py /
echo r2e-agent-privilege-strip-ok
stat -c "repo_after_strip_owner=%u:%g repo_after_strip_mode=%a" /testbed

repo="$(cat /opt/miles-swe/workdir)"
gitdir="$(cat /opt/miles-swe/gitdir)"
[[ "${repo}" == /testbed && -d "${gitdir}" ]]
[[ "$(cat /opt/miles-swe/r2e-rootfs-attestation)" == \
    miles-r2e-visible-rootfs-attestation-v1 ]]
[[ -L "${repo}/.venv" && -d /opt/miles-swe/runtime/venv ]]
[[ ! -e /r2e_tests && ! -e "${repo}/run_tests.sh" ]]
for commit in "${SWE_BASE_COMMIT}" "${SWE_GOLD_COMMIT}"; do
    ! git --git-dir="${gitdir}" --work-tree="${repo}" \
        cat-file -e "${commit}^{commit}" 2>/dev/null
done
[[ -z "$(git --git-dir="${gitdir}" fsck \
    --no-reflogs --unreachable --no-progress 2>&1)" ]]
echo r2e-agent-root-checks-ok

agent_group="$(getent passwd 1000 | cut -d: -f4)"
agent_env=(
    /usr/bin/setpriv --reuid=1000 --regid="${agent_group}" --clear-groups
    --no-new-privs -- /usr/bin/env -i
    HOME="$(getent passwd 1000 | cut -d: -f6)"
    PATH=/testbed/.venv/bin:/usr/local/bin:/usr/bin:/bin
)
agent_uid="$("${agent_env[@]}" /usr/bin/id -u)"
agent_proc_uid="$("${agent_env[@]}" /usr/bin/awk \
    '\''/^Uid:/ {print $2 ":" $3 ":" $4}'\'' /proc/self/status)"
printf "agent_uid=%s agent_proc_uid=%s\n" "${agent_uid}" "${agent_proc_uid}"
uid_map="$("${agent_env[@]}" /bin/cat /proc/self/uid_map)"
printf "%s\n" "${uid_map}"
if [[ "${agent_uid}" == 1000 && "${agent_proc_uid}" == 1000:1000:1000 ]]; then
    uid_contract=e2b-validated
elif [[ "${agent_uid}" == 0 && "${agent_proc_uid}" == 0:0:0 && \
    "${uid_map}" =~ ^[[:space:]]*0[[:space:]]+[0-9]+[[:space:]]+1[[:space:]]*$ ]]; then
    # Root-mapped fakeroot may map only the submitting host UID to container
    # UID 0, so it cannot
    # represent UID 1000. This job still validates the immutable-image layout;
    # production E2B admission separately requires a real UID 1000 sandbox.
    uid_contract=e2b-required-fakeroot-single-map
else
    echo "unexpected container UID mapping" >&2
    exit 8
fi
"${agent_env[@]}" /bin/grep -q "^NoNewPrivs:[[:space:]]*1$" /proc/self/status
"${agent_env[@]}" /testbed/.venv/bin/python -c \
    "import Orange; print(Orange.__name__)"
echo r2e-agent-import-ok
stat -c "repo_owner=%u:%g repo_mode=%a" /testbed
"${agent_env[@]}" /usr/bin/touch /testbed/.miles-r2e-smoke
stat -c "smoke_owner=%u:%g smoke_mode=%a" /testbed/.miles-r2e-smoke
smoke_owner="$(stat -c %u /testbed/.miles-r2e-smoke)"
[[ "${smoke_owner}" == 0 || "${smoke_owner}" == 1000 ]] || exit 8
if [[ "${smoke_owner}" == 0 ]]; then
    echo "fakeroot maps UID-1000 writes to root ownership; E2B admission still requires UID 1000"
fi
rm -f -- /testbed/.miles-r2e-smoke
if [[ "${uid_contract}" == e2b-required-fakeroot-single-map ]]; then
    echo "r2e-agent-image-runtime-ok uid-contract=e2b-required"
else
    echo "r2e-agent-image-runtime-ok uid-contract=validated"
fi
'
