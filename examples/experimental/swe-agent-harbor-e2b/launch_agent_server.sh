#!/bin/bash
# Start Miles' Harbor agent server with one ephemeral E2B sandbox per trial.
# This script never reads a .env file. Export credentials in its process
# environment or inject them through the scheduler/secret manager.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MILES_ROOT="${MILES_ROOT:-$(realpath "$SCRIPT_DIR/../../..")}"
PATCH_FILE="$SCRIPT_DIR/harbor-miles-e2b.patch"
EXPECTED_HARBOR_COMMIT=2ce5ba2af33a00c9fba0463f6403313996373f85

: "${HARBOR_ROOT:?set HARBOR_ROOT to a harbor-framework/harbor checkout}"
: "${HARBOR_TASKS_DIR:?set HARBOR_TASKS_DIR to the materialized Harbor task directories}"
: "${E2B_API_KEY:?export E2B_API_KEY in this process; this launcher never reads .env}"

HARBOR_PYTHON="${HARBOR_PYTHON:-$HARBOR_ROOT/.venv/bin/python}"
TRIALS_DIR="${TRIALS_DIR:-/tmp/harbor-e2b-trials}"
PORT="${PORT:-11000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-5400}"
AGENT_SETUP_TIMEOUT="${AGENT_SETUP_TIMEOUT:-1800}"
DASHBOARD_PORT="${DASHBOARD_PORT:-0}"
DASHBOARD_LOG_PATH="${DASHBOARD_LOG_PATH:-$TRIALS_DIR/requests.jsonl}"

if [[ ! -x "$HARBOR_PYTHON" ]]; then
    echo "Harbor Python is missing: $HARBOR_PYTHON" >&2
    echo "Run 'uv sync --extra e2b' in HARBOR_ROOT or set HARBOR_PYTHON." >&2
    exit 2
fi
if [[ ! -d "$HARBOR_TASKS_DIR" ]]; then
    echo "Harbor task directory is missing: $HARBOR_TASKS_DIR" >&2
    exit 2
fi
if [[ "$(git -C "$HARBOR_ROOT" rev-parse HEAD)" != "$EXPECTED_HARBOR_COMMIT" ]]; then
    echo "Harbor must be pinned to $EXPECTED_HARBOR_COMMIT" >&2
    exit 2
fi

# The pinned Miles server branch has Harbor's E2B provider but its server-side
# selector only wires Docker and Daytona. Apply the narrow, idempotent overlay;
# fail closed if upstream source drift makes the patch ambiguous.
if git -C "$HARBOR_ROOT" apply --unidiff-zero --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    : # already applied
elif git -C "$HARBOR_ROOT" apply --unidiff-zero --check "$PATCH_FILE" >/dev/null 2>&1; then
    git -C "$HARBOR_ROOT" apply --unidiff-zero "$PATCH_FILE"
else
    echo "Harbor E2B overlay does not match this checkout." >&2
    echo "Use the harbor-miles-v0.20.0 branch documented in README.md." >&2
    exit 2
fi

export HARBOR_ENV_TYPE=e2b
export HARBOR_KEEP_SANDBOX=false
export PYTHON_DOTENV_DISABLED=1
export AGENT_MAX_INPUT_TOKENS="${AGENT_MAX_INPUT_TOKENS:-65536}"
export AGENT_MAX_OUTPUT_TOKENS="${AGENT_MAX_OUTPUT_TOKENS:-16384}"
export HARBOR_RESPONSE_LENGTH_POLICY="${HARBOR_RESPONSE_LENGTH_POLICY:-abort}"
export PYTHONPATH="$MILES_ROOT:$HARBOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$TRIALS_DIR"
"$HARBOR_PYTHON" "$SCRIPT_DIR/preflight.py"

exec "$HARBOR_PYTHON" "$HARBOR_ROOT/miles_agent_server.py" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-concurrent "$MAX_CONCURRENT" \
    --agent-timeout "$AGENT_TIMEOUT" \
    --agent-setup-timeout "$AGENT_SETUP_TIMEOUT" \
    --trials-dir "$TRIALS_DIR" \
    --dashboard-port "$DASHBOARD_PORT" \
    --dashboard-log-path "$DASHBOARD_LOG_PATH"
