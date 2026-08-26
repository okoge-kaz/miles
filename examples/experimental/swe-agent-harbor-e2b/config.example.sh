#!/bin/bash
# Non-secret configuration example. The launcher does not source this file.
# Export E2B_API_KEY separately through your shell, scheduler, or secret manager.

export HARBOR_ROOT=/path/to/harbor
export HARBOR_TASKS_DIR=/path/to/materialized/harbor_tasks
export TRIALS_DIR=/path/to/harbor_trials
export MAX_CONCURRENT=32
export AGENT_TIMEOUT=5400
export AGENT_SETUP_TIMEOUT=1800
export DASHBOARD_PORT=0
export HARBOR_WORKER_CANCEL_GRACE_SEC=30
export PYTHON_DOTENV_DISABLED=1

# E2B Cloud is the SDK default. Self-hosted E2B-compatible endpoints may set:
# export E2B_API_URL=https://api.example.invalid
# export E2B_SANDBOX_URL=https://sandbox.example.invalid

# A sandbox-resident agent needs the Miles session endpoint on this allowlist.
# A host-process agent such as terminus-2 does not make model calls from E2B.
export HARBOR_AGENT_ALLOWED_HOSTS=trainer.example.invalid
