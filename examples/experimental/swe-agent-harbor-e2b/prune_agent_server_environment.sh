#!/bin/bash

# Remove scheduler-shell credentials and hooks before starting the long-lived
# Harbor process. Arguments are executable paths/options only; secret values
# remain in the small fixed-name process environment.

set -euo pipefail
umask 077

(( $# > 0 )) || {
    echo "usage: $0 command [args ...]" >&2
    exit 2
}

: "${TRIALS_DIR:?TRIALS_DIR must identify the private server state root}"
[[ "$TRIALS_DIR" == /* && -d "$TRIALS_DIR" && ! -L "$TRIALS_DIR" && \
    "$(stat -c '%u' "$TRIALS_DIR")" == "$(id -u)" && \
    $((8#$(stat -c '%a' "$TRIALS_DIR") & 8#077)) -eq 0 ]] || {
    echo "TRIALS_DIR must be an absolute, owner-only directory" >&2
    exit 2
}

# Never expose the submission user's home-directory credentials or shell hooks
# to the long-lived service. These directories contain no inherited files.
HOME="${TRIALS_DIR}/server-home"
TMPDIR="${TRIALS_DIR}/server-tmp"
XDG_CACHE_HOME="${TRIALS_DIR}/server-xdg-cache"
XDG_CONFIG_HOME="${TRIALS_DIR}/server-xdg-config"
XDG_DATA_HOME="${TRIALS_DIR}/server-xdg-data"
XDG_STATE_HOME="${TRIALS_DIR}/server-xdg-state"
XDG_RUNTIME_DIR="${TRIALS_DIR}/server-xdg-runtime"
for private_directory in \
    "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
    "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR"; do
    if [[ -e "$private_directory" ]]; then
        [[ -d "$private_directory" && ! -L "$private_directory" && \
            "$(stat -c '%u' "$private_directory")" == "$(id -u)" ]] || {
            echo "Server-private environment directory is unsafe" >&2
            exit 2
        }
    else
        mkdir -m 0700 "$private_directory"
    fi
    chmod 0700 "$private_directory"
done
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME
export XDG_RUNTIME_DIR PATH
HARBOR_SERVER_ENV_PRUNED=1
export HARBOR_SERVER_ENV_PRUNED

is_allowed_export() {
    case "$1" in
        AGENT_MAX_INPUT_TOKENS|AGENT_MAX_OUTPUT_TOKENS|AGENT_SETUP_TIMEOUT|AGENT_TIMEOUT|ASYNC_MAX_CONCURRENT_SAMPLES|DASHBOARD_LOG_PATH|DASHBOARD_PORT|E2B_ACCESS_TOKEN|E2B_API_KEY|E2B_API_URL|E2B_DOMAIN|E2B_SANDBOX_URL|HARBOR_ADMIN_SECRET|HARBOR_AGENT_ALLOWED_HOSTS|HARBOR_AGENT_MAX_ITERATIONS|HARBOR_E2B_ADMISSION_REPORT|HARBOR_E2B_PREBUILD_CONCURRENCY|HARBOR_E2B_PREBUILD_TASK_IDS_FILE|HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS|HARBOR_E2B_TEMPLATE_PINS_FILE|HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER|HARBOR_MAX_SEQ_LEN|HARBOR_PYTHON|HARBOR_RESPONSE_LENGTH_POLICY|HARBOR_ROOT|HARBOR_RUN_SECRET|HARBOR_SERVER_ENV_PRUNED|HARBOR_TASKS_DIR|HARBOR_TERMINUS_2_ENABLE_SUMMARIZE|HARBOR_TERMINUS_2_LINEAR_HISTORY|HARBOR_TIMEOUT_MULTIPLIER|HARBOR_TRIAL_WALL_TIMEOUT_SEC|HARBOR_VERIFIER_TIMEOUT_SEC|HARBOR_WORKER_CANCEL_GRACE_SEC|HOME|LANG|LC_ALL|LOGNAME|MAX_CONCURRENT|MILES_ROOT|PATH|PORT|PYTHON_DOTENV_DISABLED|PYTHONDONTWRITEBYTECODE|TERM|TMPDIR|TRIALS_DIR|TZ|USER|WANDB_DISABLED|WANDB_MODE|XDG_CACHE_HOME|XDG_CONFIG_HOME|XDG_DATA_HOME|XDG_RUNTIME_DIR|XDG_STATE_HOME)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

while IFS= read -r variable_name; do
    is_allowed_export "$variable_name" || unset "$variable_name"
done < <(compgen -e)

unset -f is_allowed_export
unset variable_name
while IFS= read -r function_name; do
    unset -f "$function_name"
done < <(compgen -A function)
unset function_name private_directory
exec "$@"
