#!/bin/bash
# Start Miles' Harbor agent server with one ephemeral E2B sandbox per trial.
# This script never reads a .env file. Export credentials in its process
# environment or inject them through the scheduler/secret manager.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${HARBOR_SERVER_ENV_PRUNED:-}" != 1 ]]; then
    TRIALS_DIR="${TRIALS_DIR:-/tmp/harbor-e2b-trials}"
    if [[ -e "$TRIALS_DIR" ]]; then
        [[ -d "$TRIALS_DIR" && ! -L "$TRIALS_DIR" && \
            "$(stat -c '%u' "$TRIALS_DIR")" == "$(id -u)" ]] || {
            echo "TRIALS_DIR is unsafe" >&2
            exit 2
        }
    else
        mkdir -p -m 0700 "$TRIALS_DIR"
    fi
    chmod 0700 "$TRIALS_DIR"
    TRIALS_DIR="$(realpath "$TRIALS_DIR")"
    export TRIALS_DIR
    exec "$SCRIPT_DIR/prune_agent_server_environment.sh" /bin/bash "$0" "$@"
fi
readonly MILES_ROOT="${MILES_ROOT:-$(realpath "$SCRIPT_DIR/../../..")}"
readonly EXPECTED_HARBOR_COMMIT=2ce5ba2af33a00c9fba0463f6403313996373f85

: "${HARBOR_ROOT:?set HARBOR_ROOT to a harbor-framework/harbor checkout}"
: "${HARBOR_TASKS_DIR:?set HARBOR_TASKS_DIR to the materialized Harbor task directories}"
: "${HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS:?set colon-separated owner-only semantic admission JSONL files}"
: "${E2B_API_KEY:?export E2B_API_KEY in this process; this launcher never reads .env}"
: "${HARBOR_RUN_SECRET:?export a dedicated /run bearer secret in this process}"
: "${HARBOR_ADMIN_SECRET:?export a distinct admin bearer secret in this process}"
: "${MAX_CONCURRENT:?set MAX_CONCURRENT equal to the training ASYNC_MAX_CONCURRENT_SAMPLES}"

for secret_name in HARBOR_RUN_SECRET HARBOR_ADMIN_SECRET; do
    secret_value="${!secret_name}"
    (( ${#secret_value} >= 32 && ${#secret_value} <= 4096 )) && \
        [[ "$secret_value" != *$'\r'* && "$secret_value" != *$'\n'* ]] || {
        echo "${secret_name} must be 32-4096 characters without CR/LF" >&2
        exit 2
    }
done
[[ "$HARBOR_RUN_SECRET" != "$HARBOR_ADMIN_SECRET" ]] || {
    echo "HARBOR_RUN_SECRET and HARBOR_ADMIN_SECRET must be distinct" >&2
    exit 2
}
unset secret_value
export HARBOR_RUN_SECRET HARBOR_ADMIN_SECRET

[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] && (( MAX_CONCURRENT <= 256 )) || {
    echo "MAX_CONCURRENT must be an integer in [1, 256]" >&2
    exit 2
}
if [[ -n "${ASYNC_MAX_CONCURRENT_SAMPLES:-}" && \
    "$MAX_CONCURRENT" != "$ASYNC_MAX_CONCURRENT_SAMPLES" ]]; then
    echo "MAX_CONCURRENT must equal ASYNC_MAX_CONCURRENT_SAMPLES" >&2
    exit 2
fi

[[ "${HARBOR_TIMEOUT_MULTIPLIER:-1}" == 1 ]] || {
    echo "HARBOR_TIMEOUT_MULTIPLIER is fixed to 1 for production SWE trials" >&2
    exit 2
}
[[ "${HARBOR_TRIAL_WALL_TIMEOUT_SEC:-12600}" == 12600 ]] || {
    echo "HARBOR_TRIAL_WALL_TIMEOUT_SEC is fixed to 12600 seconds" >&2
    exit 2
}
[[ "${AGENT_TIMEOUT:-3600}" == 3600 ]] || {
    echo "AGENT_TIMEOUT is fixed to 3600 seconds" >&2
    exit 2
}
[[ "${AGENT_SETUP_TIMEOUT:-1800}" == 1800 ]] || {
    echo "AGENT_SETUP_TIMEOUT is fixed to 1800 seconds" >&2
    exit 2
}
[[ "${HARBOR_VERIFIER_TIMEOUT_SEC:-2100}" == 2100 ]] || {
    echo "HARBOR_VERIFIER_TIMEOUT_SEC is fixed to 2100 seconds" >&2
    exit 2
}
[[ "${HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER:-1}" == 1 ]] || {
    echo "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER is fixed to 1" >&2
    exit 2
}
HARBOR_TIMEOUT_MULTIPLIER=1
HARBOR_TRIAL_WALL_TIMEOUT_SEC=12600
AGENT_TIMEOUT=3600
AGENT_SETUP_TIMEOUT=1800
HARBOR_VERIFIER_TIMEOUT_SEC=2100
HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER=1
readonly HARBOR_TIMEOUT_MULTIPLIER HARBOR_TRIAL_WALL_TIMEOUT_SEC \
    AGENT_TIMEOUT AGENT_SETUP_TIMEOUT HARBOR_VERIFIER_TIMEOUT_SEC \
    HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER
export HARBOR_TIMEOUT_MULTIPLIER HARBOR_TRIAL_WALL_TIMEOUT_SEC \
    AGENT_TIMEOUT AGENT_SETUP_TIMEOUT HARBOR_VERIFIER_TIMEOUT_SEC \
    HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER

HARBOR_PYTHON="${HARBOR_PYTHON:-$HARBOR_ROOT/.venv/bin/python}"
TRIALS_DIR="${TRIALS_DIR:-/tmp/harbor-e2b-trials}"
PORT="${PORT:-11000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-0}"
DASHBOARD_LOG_PATH="${DASHBOARD_LOG_PATH:-$TRIALS_DIR/requests.jsonl}"
E2B_PREBUILD_CONCURRENCY="${HARBOR_E2B_PREBUILD_CONCURRENCY:-4}"
E2B_ADMISSION_REPORT="${HARBOR_E2B_ADMISSION_REPORT:-$TRIALS_DIR/e2b-template-admission.json}"
E2B_TEMPLATE_PINS="${HARBOR_E2B_TEMPLATE_PINS_FILE:-$TRIALS_DIR/e2b-template-pins.json}"

[[ "$DASHBOARD_PORT" == 0 ]] || {
    echo "The production Harbor server requires DASHBOARD_PORT=0" >&2
    exit 2
}

if [[ ! -x "$HARBOR_PYTHON" ]]; then
    echo "Harbor Python is missing: $HARBOR_PYTHON" >&2
    echo "Run 'uv sync --extra e2b' in HARBOR_ROOT or set HARBOR_PYTHON." >&2
    exit 2
fi
if [[ ! -d "$HARBOR_TASKS_DIR" ]]; then
    echo "Harbor task directory is missing: $HARBOR_TASKS_DIR" >&2
    exit 2
fi
if [[ "$(stat -c '%u' "$HARBOR_TASKS_DIR")" != "$(id -u)" ]] || \
    find "$HARBOR_TASKS_DIR" -xdev \( -type d -o -type f \) \
        ! -uid "$(id -u)" -print -quit | grep -q .; then
    echo "Harbor task tree must be owned by the launching user" >&2
    exit 2
fi
if [[ -L "$HARBOR_TASKS_DIR" ]] || \
    find "$HARBOR_TASKS_DIR" -xdev -type l -print -quit | grep -q .; then
    echo "Harbor task tree must not contain symlinks" >&2
    exit 2
fi
if find "$HARBOR_TASKS_DIR" -xdev ! -type d ! -type f -print -quit | grep -q .; then
    echo "Harbor task tree must not contain special files" >&2
    exit 2
fi
if find "$HARBOR_TASKS_DIR" -xdev -type f -links +1 -print -quit | grep -q .; then
    echo "Harbor task tree must not contain hardlinked files" >&2
    exit 2
fi
if find "$HARBOR_TASKS_DIR" -xdev \
    \( -type d -o -type f \) -perm /077 -print -quit | grep -q .; then
    echo "Harbor task tree must be owner-only" >&2
    exit 2
fi
if find "$HARBOR_TASKS_DIR" -xdev -mindepth 1 \
    \( -type d -o -type f \) -perm /222 -print -quit | grep -q .; then
    echo "Selected Harbor task subtrees must be sealed read-only" >&2
    exit 2
fi
if [[ "$(git -C "$HARBOR_ROOT" rev-parse HEAD)" != "$EXPECTED_HARBOR_COMMIT" ]]; then
    echo "Harbor must be pinned to $EXPECTED_HARBOR_COMMIT" >&2
    exit 2
fi

# Apply and attest all compatibility/security overlays. The shared helper
# accepts only the clean tree or an exact prior stage and verifies every patch
# checksum before mutating the checkout.
"$SCRIPT_DIR/apply_harbor_e2b_overlays.sh" "$HARBOR_ROOT"

export HARBOR_ENV_TYPE=e2b
export HARBOR_KEEP_SANDBOX=false
export HARBOR_E2B_NO_NEW_PRIVS_USERS=1000
export PYTHON_DOTENV_DISABLED=1
export AGENT_MAX_INPUT_TOKENS="${AGENT_MAX_INPUT_TOKENS:-65536}"
export AGENT_MAX_OUTPUT_TOKENS="${AGENT_MAX_OUTPUT_TOKENS:-16384}"
export HARBOR_RESPONSE_LENGTH_POLICY="${HARBOR_RESPONSE_LENGTH_POLICY:-abort}"
export PYTHONPATH="$MILES_ROOT:$HARBOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$TRIALS_DIR" ]]; then
    [[ -d "$TRIALS_DIR" && ! -L "$TRIALS_DIR" ]] || {
        echo "TRIALS_DIR must be a regular directory, not a symlink" >&2
        exit 2
    }
    [[ "$(stat -c '%u' "$TRIALS_DIR")" == "$(id -u)" ]] || {
        echo "TRIALS_DIR must be owned by the launching user" >&2
        exit 2
    }
    chmod 0700 "$TRIALS_DIR"
else
    mkdir -m 0700 -p "$TRIALS_DIR"
fi
trials_real="$(realpath "$TRIALS_DIR")"
for private_output in "$DASHBOARD_LOG_PATH" "$E2B_ADMISSION_REPORT" "$E2B_TEMPLATE_PINS"; do
    [[ "$(realpath -m "$(dirname -- "$private_output")")" == "$trials_real" ]] || {
        echo "Harbor reports and request logs must remain directly under TRIALS_DIR" >&2
        exit 2
    }
    if [[ -e "$private_output" ]]; then
        [[ -f "$private_output" && ! -L "$private_output" && \
            "$(stat -c '%h' "$private_output")" == 1 ]] || {
            echo "Harbor private output path must be a regular file" >&2
            exit 2
        }
        chmod 0600 "$private_output"
    fi
done
"$HARBOR_PYTHON" "$SCRIPT_DIR/preflight.py"

prebuild_args=(
    --tasks-dir "$HARBOR_TASKS_DIR"
    --report "$E2B_ADMISSION_REPORT"
    --pins "$E2B_TEMPLATE_PINS"
    --concurrency "$E2B_PREBUILD_CONCURRENCY"
)
IFS=: read -r -a semantic_admission_manifests \
    <<<"$HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS"
for semantic_manifest in "${semantic_admission_manifests[@]}"; do
    [[ -n "$semantic_manifest" ]] || {
        echo "Semantic admission manifest list contains an empty path" >&2
        exit 2
    }
    prebuild_args+=(--semantic-admission-manifest "$semantic_manifest")
done
if [[ -n "${HARBOR_E2B_PREBUILD_TASK_IDS_FILE:-}" ]]; then
    task_ids_mode="$(stat -c '%a' "$HARBOR_E2B_PREBUILD_TASK_IDS_FILE" 2>/dev/null || true)"
    [[ -f "$HARBOR_E2B_PREBUILD_TASK_IDS_FILE" && \
        ! -L "$HARBOR_E2B_PREBUILD_TASK_IDS_FILE" && \
        "$(stat -c '%h' "$HARBOR_E2B_PREBUILD_TASK_IDS_FILE")" == 1 && \
        "$(stat -c '%u' "$HARBOR_E2B_PREBUILD_TASK_IDS_FILE")" == "$(id -u)" && \
        -n "$task_ids_mode" ]] && \
        (( (8#${task_ids_mode} & 8#077) == 0 )) || {
        echo "E2B task-id file must be regular, owner-only, and owned by this user" >&2
        exit 2
    }
    prebuild_args+=(--task-ids-file "$HARBOR_E2B_PREBUILD_TASK_IDS_FILE")
fi
"$HARBOR_PYTHON" "$SCRIPT_DIR/prebuild_templates.py" "${prebuild_args[@]}"
export HARBOR_E2B_REQUIRE_PREBUILT=true
export HARBOR_E2B_REQUIRE_TEMPLATE_PIN=true
export HARBOR_E2B_TEMPLATE_PINS_FILE="$E2B_TEMPLATE_PINS"

exec "$HARBOR_PYTHON" "$HARBOR_ROOT/miles_agent_server.py" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-concurrent "$MAX_CONCURRENT" \
    --agent-timeout "$AGENT_TIMEOUT" \
    --agent-setup-timeout "$AGENT_SETUP_TIMEOUT" \
    --trials-dir "$TRIALS_DIR" \
    --dashboard-port "$DASHBOARD_PORT" \
    --dashboard-log-path "$DASHBOARD_LOG_PATH"
