#!/bin/bash
#PBS -N swe-agent-image-hardening
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Execute the exact materialized agent-image hardening scripts against an
# immutable source image through Singularity. This exercises the Dockerfile RUN path
# without using an external E2B credential.

set -euo pipefail
umask 077

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"
umask 077

: "${SWE_SOURCE_IMAGE:?set an immutable docker.io/...@sha256:... reference}"
: "${SWE_SOURCE_SCHEMA:?set swe-rebench-v2 or swe-gym}"
: "${SWE_BASE_COMMIT:?set the exact 40-hex task base commit}"
: "${SWE_TASK_ENV:?set the host path to a materialized task environment directory}"

[[ "${SWE_SOURCE_IMAGE}" =~ ^docker\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$ ]] || exit 2
[[ "${SWE_BASE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ -d "${SWE_TASK_ENV}" && ! -L "${SWE_TASK_ENV}" ]] || exit 2
task_env="$(realpath "${SWE_TASK_ENV}")"
sealed_task_root="${DATASET_DIR}/miles-swe/private/harbor-tasks-dry-run"
case "${task_env}" in
    "${sealed_task_root}"/*/environment) ;;
    *)
        echo "SWE_TASK_ENV is outside the sealed dry-run task root" >&2
        exit 2
        ;;
esac
task_root="$(dirname "${task_env}")"
[[ -f "${task_root}/tests/verifier_config.json" && \
    ! -L "${task_root}/tests/verifier_config.json" ]] || exit 2
runtime_policy="$(sed -n \
    's/.*MILES_SWE_RUNTIME_POLICY=\([^[:space:]]*\).*/\1/p' \
    "${task_env}/Dockerfile")"
case "${runtime_policy}" in
    none|npm-node-modules-v2|python-editable-metadata-v1) ;;
    *)
        echo "materialized agent runtime policy is missing or invalid" >&2
        exit 2
        ;;
esac
export SWE_RUNTIME_POLICY="${runtime_policy}"
case "${SWE_SOURCE_SCHEMA}" in
    swe-rebench-v2)
        [[ "${SWE_SOURCE_IMAGE}" == docker.io/swerebenchv2/* ]] || exit 2
        ;;
    swe-gym)
        [[ "${SWE_SOURCE_IMAGE}" == docker.io/xingyaoww/sweb.eval.x86_64.* ]] || exit 2
        ;;
    *) exit 2 ;;
esac

miles_srun \
    --fakeroot \
    --image="docker://${SWE_SOURCE_IMAGE}" \
    --bind="${task_root}:/mnt/miles-swe-task:ro" \
    --writable-tmpfs \
    --no-home \
    --export=SWE_BASE_COMMIT,SWE_RUNTIME_POLICY,SWE_SOURCE_SCHEMA \
    env -i \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        SWE_BASE_COMMIT="${SWE_BASE_COMMIT}" \
        SWE_RUNTIME_POLICY="${SWE_RUNTIME_POLICY}" \
        SWE_SOURCE_SCHEMA="${SWE_SOURCE_SCHEMA}" \
        bash --noprofile --norc -c '
set -euo pipefail
umask 077
install -d -o root -g root -m 0700 \
    /opt/miles-swe/collected \
    /opt/miles-swe/root-home \
    /opt/miles-swe/root-home/xdg
printf "%s\n" MILES_SWE_AGENT_STATE_INVALID \
    >/opt/miles-swe/collected/model.patch
install -o root -g root -m 0555 \
    /mnt/miles-swe-task/environment/prepare_agent.sh \
    /opt/miles-swe/prepare_agent.sh
install -o root -g root -m 0500 \
    /mnt/miles-swe-task/environment/seal_playwright_runtime.sh \
    /opt/miles-swe/seal_playwright_runtime.sh
install -o root -g root -m 0500 \
    /mnt/miles-swe-task/environment/collect_agent_patch.sh \
    /opt/miles-swe/collect_agent_patch.sh
install -o root -g root -m 0500 \
    /mnt/miles-swe-task/environment/strip_agent_privileges.py \
    /opt/miles-swe/strip_agent_privileges.py
MILES_SWE_SCHEMA="${SWE_SOURCE_SCHEMA}" \
MILES_SWE_BASE_COMMIT="${SWE_BASE_COMMIT}" \
MILES_SWE_RUNTIME_POLICY="${SWE_RUNTIME_POLICY}" \
    /opt/miles-swe/prepare_agent.sh
python3 /opt/miles-swe/strip_agent_privileges.py /
repo="$(cat /opt/miles-swe/workdir)"
gitdir="$(cat /opt/miles-swe/gitdir)"
[[ -d "${repo}" && -d "${gitdir}" ]]
[[ "$(stat -c "%u:%a" /opt/miles-swe/workdir)" == 0:444 ]]
[[ "$(stat -c "%u" "${gitdir}")" == 0 ]]
[[ -z "$(git --git-dir="${gitdir}" --work-tree="${repo}" \
    status --porcelain --untracked-files=all)" ]]
runtime_policy="$(cat /opt/miles-swe/runtime-policy)"
[[ "${runtime_policy}" == "${SWE_RUNTIME_POLICY}" ]]
playwright_path=/opt/miles-swe/runtime/no-playwright-runtime
if [[ "${runtime_policy}" == npm-node-modules-v2 ]]; then
    [[ -L "${repo}/node_modules" ]]
    [[ "$(readlink -f -- "${repo}/node_modules")" == \
        /opt/miles-swe/runtime/node_modules ]]
    /usr/bin/setpriv --reuid=1000 --regid="$(id -g 1000)" \
        --clear-groups --no-new-privs --inh-caps=-all \
        --ambient-caps=-all --bounding-set=-all \
        timeout --signal=TERM --kill-after=5s 30s \
        node -e "const fs=require(\"fs\"); for (const p of process.argv.slice(1)) JSON.parse(fs.readFileSync(p,\"utf8\"));" \
        "${repo}/package.json" "${repo}/package-lock.json"
    [[ "$(stat -c "%u:%a" /opt/miles-swe/npm-repo-runtime-paths)" == \
        0:444 ]]
    [[ "$(sha256sum /opt/miles-swe/npm-repo-runtime.inventory | cut -d" " -f1)" == \
        "$(cat /opt/miles-swe/npm-repo-runtime.inventory.sha256)" ]]
    : >/tmp/miles-swe-npm-repo-runtime.inventory
    while IFS= read -r relative; do
        [[ "${relative}" == dist ]]
        [[ -d "${repo}/${relative}" && ! -L "${repo}/${relative}" ]]
        [[ "$(stat -c "%u" "${repo}/${relative}")" == 0 ]]
        [[ -z "$(find "${repo}/${relative}" -xdev -perm /222 -print -quit)" ]]
        while IFS= read -r -d "" runtime_file; do
            runtime_relative="${runtime_file#"${repo}/"}"
            printf "%s\\0%s\\0%s\\0" "${runtime_relative}" \
                "$(stat -c %s "${runtime_file}")" \
                "$(sha256sum "${runtime_file}" | cut -d" " -f1)" \
                >>/tmp/miles-swe-npm-repo-runtime.inventory
        done < <(find "${repo}/${relative}" -xdev -type f -print0 | sort -z)
    done </opt/miles-swe/npm-repo-runtime-paths
    [[ "$(sha256sum /tmp/miles-swe-npm-repo-runtime.inventory | cut -d" " -f1)" == \
        "$(cat /opt/miles-swe/npm-repo-runtime.inventory.sha256)" ]]
    playwright_path="$(cat /opt/miles-swe/playwright-browsers-path)"
    if [[ "${playwright_path}" != none ]]; then
        [[ "${playwright_path}" == /opt/miles-swe/runtime/ms-playwright ]]
        read -r actual_inventory_digest _ < <(sha256sum \
            /opt/miles-swe/playwright-runtime.inventory)
        [[ "${actual_inventory_digest}" == "$(cat \
            /opt/miles-swe/playwright-runtime.inventory.sha256)" ]]
        : >/tmp/miles-swe-playwright-runtime.inventory
        while IFS= read -r -d "" runtime_file; do
            runtime_relative="${runtime_file#"${playwright_path}/"}"
            printf "%s\\0%s\\0%s\\0" "${runtime_relative}" \
                "$(stat -c %s "${runtime_file}")" \
                "$(sha256sum "${runtime_file}" | cut -d" " -f1)" \
                >>/tmp/miles-swe-playwright-runtime.inventory
        done < <(find "${playwright_path}" -xdev -type f -print0 | sort -z)
        [[ "$(sha256sum /tmp/miles-swe-playwright-runtime.inventory \
            | cut -d" " -f1)" == "${actual_inventory_digest}" ]]
        browser="$(find "${playwright_path}" -xdev -type f -name chrome \
            -perm /111 -print -quit)"
        [[ -n "${browser}" ]]
        HOME=/tmp/miles-swe-agent-baseline-home \
        PLAYWRIGHT_BROWSERS_PATH="${playwright_path}" \
            /usr/bin/setpriv --reuid=1000 --regid="$(id -g 1000)" \
            --clear-groups --no-new-privs --inh-caps=-all \
            --ambient-caps=-all --bounding-set=-all \
            timeout --signal=TERM --kill-after=5s 30s \
            "${browser}" --version >/tmp/miles-swe-browser-version
    else
        playwright_path=/opt/miles-swe/runtime/no-playwright-runtime
    fi
elif [[ "${runtime_policy}" == python-editable-metadata-v1 ]]; then
    [[ -f /opt/miles-swe/runtime-links ]]
    while IFS= read -r relative; do
        [[ -z "${relative}" || -L "${repo}/${relative}" ]]
    done </opt/miles-swe/runtime-links
    HOME=/tmp/miles-swe-agent-baseline-home \
        /usr/bin/setpriv --reuid=1000 --regid="$(id -g 1000)" \
        --clear-groups --no-new-privs --inh-caps=-all \
        --ambient-caps=-all --bounding-set=-all \
        bash --noprofile --norc -c \
        "source /opt/miniconda3/bin/activate && conda activate testbed && \
        cd \"${repo}\" && python -m pip check"
fi

if [[ "${SWE_SOURCE_SCHEMA}" == swe-rebench-v2 ]]; then
    while IFS= read -r -d "" baseline_command; do
        HOME=/tmp/miles-swe-agent-baseline-home \
        PLAYWRIGHT_BROWSERS_PATH="${playwright_path}" \
        PYTHON_DOTENV_DISABLED=1 WANDB_MODE=offline WANDB_DISABLED=true \
            /usr/bin/setpriv --reuid=1000 --regid="$(id -g 1000)" \
            --clear-groups --no-new-privs --inh-caps=-all \
            --ambient-caps=-all --bounding-set=-all \
            timeout --signal=TERM --kill-after=10s 1200s \
            bash --noprofile --norc -c \
            "cd \"${repo}\" && ${baseline_command}"
    done < <(python3 -c "import json, os; \
commands=json.load(open(\"/mnt/miles-swe-task/tests/verifier_config.json\", \
encoding=\"utf-8\")).get(\"test_commands\"); \
assert isinstance(commands, list) and commands and \
all(isinstance(command, str) and command and \"\\0\" not in command \
for command in commands), \"invalid materialized public baseline commands\"; \
[os.write(1, command.encode() + b\"\\0\") for command in commands]")
else
    HOME=/tmp/miles-swe-agent-baseline-home \
    PYTHON_DOTENV_DISABLED=1 WANDB_MODE=offline WANDB_DISABLED=true \
        /usr/bin/setpriv --reuid=1000 --regid="$(id -g 1000)" \
        --clear-groups --no-new-privs --inh-caps=-all \
        --ambient-caps=-all --bounding-set=-all \
        timeout --signal=TERM --kill-after=10s 1200s \
        bash --noprofile --norc -c \
        "PYTHONPATH=/mnt/miles-swe-task/tests/lib \
        python3 /mnt/miles-swe-task/tests/swe_gym_run.py \
        --public-baseline /mnt/miles-swe-task/tests/verifier_config.json \
        \"${repo}\" >/tmp/miles-swe-public-baseline.sh && \
        bash --noprofile --norc /tmp/miles-swe-public-baseline.sh"
fi
echo "swe-agent-image-hardening-ok"
'
