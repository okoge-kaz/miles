#!/bin/bash

set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/opt/miles-swe/root-home
export XDG_CONFIG_HOME=/opt/miles-swe/root-home/xdg
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_ATTR_NOSYSTEM=1
unset BASH_ENV CDPATH ENV GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR \
    GIT_DIR GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_WORK_TREE LD_LIBRARY_PATH \
    LD_PRELOAD

[[ "$(id -u)" == 0 ]] || {
    echo "R2E verifier bootstrap must run as root" >&2
    exit 2
}
install -d -o root -g root -m 0700 /opt/miles-swe/root-home \
    /opt/miles-swe/root-home/xdg /opt/miles-swe/collected
chown -R root:root /tests
find /tests -type d -exec chmod 0700 {} +
find /tests -type f -exec chmod 0600 {} +
chmod 0700 /tests/test.sh /tests/prepare_r2e_verifier.sh \
    /tests/strip_agent_privileges.py
base_commit="$(cat /tests/base_commit.txt)"
gold_commit="$(cat /tests/gold_commit.txt)"
[[ "${base_commit}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "R2E verifier base binding is invalid" >&2
    exit 2
}
[[ "${gold_commit}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "R2E verifier gold binding is invalid" >&2
    exit 2
}
python3 /tests/strip_agent_privileges.py /
MILES_SWE_BASE_COMMIT="${base_commit}" \
MILES_SWE_GOLD_COMMIT="${gold_commit}" \
    /tests/prepare_r2e_verifier.sh

install -d -o root -g root -m 0700 /logs/verifier
rm -f -- /logs/verifier/reward.txt /logs/verifier/report.json \
    /logs/verifier/test-output.log
artifact=/opt/miles-swe/collected/model.patch
[[ -f "${artifact}" && ! -L "${artifact}" ]] || {
    echo "model patch artifact is absent; trial is ungraded" >&2
    exit 2
}
if grep -qx 'MILES_SWE_AGENT_STATE_INVALID' "${artifact}"; then
    printf '{"resolved":false,"reason":"agent_repository_state_invalid","reward":0}\n' \
        >/logs/verifier/report.json
    printf '0\n' >/logs/verifier/reward.txt
    exit 0
fi

repo=/testbed
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${repo}"
export GIT_CONFIG_KEY_1=core.fsmonitor
export GIT_CONFIG_VALUE_1=false
export GIT_CONFIG_KEY_2=core.hooksPath
export GIT_CONFIG_VALUE_2=/dev/null
safe_git() {
    command git -c core.fsmonitor=false -c core.hooksPath=/dev/null \
        -C "${repo}" "$@"
}
base_commit="$(cat /opt/miles-swe/verifier-base-commit)"
[[ "$(safe_git rev-parse HEAD)" == "${base_commit}" ]] || {
    echo "R2E verifier image is not at its derived base; trial is ungraded" >&2
    exit 2
}
safe_git reset --hard "${base_commit}" >/dev/null
install -o root -g root -m 0600 /dev/null /logs/verifier/model-paths.json
set +e
python3 /tests/model_patch_policy.py \
    "${repo}" "${artifact}" /tests/model_path_policy.json \
    >/logs/verifier/model-paths.json
policy_status=$?
set -e
case "${policy_status}" in
    0) ;;
    10)
        printf '{"resolved":false,"reason":"model_patch_path_policy_rejected","reward":0}\n' \
            >/logs/verifier/report.json
        printf '0\n' >/logs/verifier/reward.txt
        exit 0
        ;;
    *)
        echo "model path-policy validation failed; trial is ungraded" >&2
        exit 2
        ;;
esac
if [[ -s "${artifact}" ]] && ! safe_git apply --check --binary "${artifact}"; then
    printf '{"resolved":false,"reason":"model_patch_did_not_apply","reward":0}\n' \
        >/logs/verifier/report.json
    printf '0\n' >/logs/verifier/reward.txt
    exit 0
fi
[[ ! -s "${artifact}" ]] || safe_git apply --binary "${artifact}"

# Re-create verifier-owned inputs after patch application.  The test runner is
# copied into /tests while the fresh verifier image is built and is never
# loaded from the model-modifiable repository.
rm -rf -- "${repo}/r2e_tests"
ln -s /r2e_tests "${repo}/r2e_tests"
for config in pytest.ini setup.cfg tox.ini pyproject.toml conftest.py; do
    if safe_git diff --name-only -- "${config}" | grep -qx "${config}"; then
        if safe_git ls-files --error-unmatch "${config}" >/dev/null 2>&1; then
            safe_git checkout HEAD -- "${config}"
        else
            rm -f -- "${repo}/${config}"
        fi
    fi
done

set +e
install -o root -g root -m 0600 /dev/null /logs/verifier/test-output.log
verifier_gid="$(cat /opt/miles-swe/verifier-gid)"
verifier_home="$(getent passwd 1000 | cut -d: -f6)"
/usr/bin/setsid --wait /usr/bin/setpriv --reuid=1000 \
    --regid="${verifier_gid}" --clear-groups --no-new-privs -- \
    /usr/bin/env -i HOME="${verifier_home}" LANG=C.UTF-8 \
        PATH=/testbed/.venv/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin \
    /usr/bin/timeout --signal=TERM --kill-after=30s 1800s \
        /bin/bash --noprofile --norc -c \
        'ulimit -c 0; ulimit -f 65536; ulimit -n 4096; ulimit -u 1024; grep -q "^NoNewPrivs:[[:space:]]*1$" /proc/self/status || exit 125; cd "$1" && exec /bin/bash --noprofile --norc /proc/self/fd/3' \
        _ "${repo}" 3</opt/miles-swe/private/run_tests.sh \
        > /logs/verifier/test-output.log 2>&1 &
test_group=$!
wait "${test_group}"
test_status=$?
set -e
# Kill/reap every remaining untrusted process before the root grader starts.
kill -TERM -- "-${test_group}" 2>/dev/null || true
for proc in /proc/[0-9]*; do
    uid="$(awk '/^Uid:/ { print $2; exit }' "${proc}/status" 2>/dev/null)" || continue
    [[ "${uid}" == 1000 ]] || continue
    kill -KILL "${proc##*/}" 2>/dev/null || true
done
wait "${test_group}" 2>/dev/null || true
test_output_size="$(stat -c '%s' /logs/verifier/test-output.log)" || {
    echo "cannot inspect R2E test output; trial is ungraded" >&2
    exit 2
}
if [[ "${test_status}" == 153 || "${test_output_size}" -ge 67108864 ]]; then
    printf '{"resolved":false,"reason":"model_test_output_limit_exceeded","reward":0}\n' \
        >/logs/verifier/report.json
    printf '0\n' >/logs/verifier/reward.txt
    exit 0
fi
if [[ "${test_status}" == 125 ]]; then
    echo "NoNewPrivs enforcement failed for untrusted R2E tests; trial is ungraded" >&2
    exit 2
fi
if [[ "${test_status}" == 124 || "${test_status}" == 137 ]]; then
    printf '{"resolved":false,"reason":"model_test_timeout","reward":0}\n' \
        >/logs/verifier/report.json
    printf '0\n' >/logs/verifier/reward.txt
    exit 0
fi
[[ -s /logs/verifier/test-output.log ]] || {
    printf '{"resolved":false,"reason":"model_test_produced_no_output","reward":0}\n' \
        >/logs/verifier/report.json
    printf '0\n' >/logs/verifier/reward.txt
    exit 0
}
python3 /tests/r2e_grader.py
exit 0
