#!/bin/bash
# Non-secret configuration example. The launcher does not source this file.
# Export E2B_API_KEY, HARBOR_RUN_SECRET, and a distinct HARBOR_ADMIN_SECRET
# separately through your shell, scheduler, or secret manager. Never write
# those values in this file or pass them through argv.

export HARBOR_ROOT=/path/to/harbor
export HARBOR_TASKS_DIR=/path/to/materialized/harbor_tasks
export HARBOR_E2B_PREBUILD_TASK_IDS_FILE=/path/to/admitted-instance-ids.txt
export HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS=/path/to/r2e-admissions.jsonl:/path/to/rebench-admissions.jsonl:/path/to/swe-gym-admissions.jsonl
export TRIALS_DIR=/path/to/harbor_trials
export MAX_CONCURRENT=64
export ASYNC_MAX_CONCURRENT_SAMPLES=64
export AGENT_TIMEOUT=3600
export AGENT_SETUP_TIMEOUT=1800
export HARBOR_TIMEOUT_MULTIPLIER=1
export HARBOR_TRIAL_WALL_TIMEOUT_SEC=12600
export HARBOR_VERIFIER_TIMEOUT_SEC=2100
export HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER=1
export HARBOR_E2B_PREBUILD_CONCURRENCY=4
export DASHBOARD_PORT=0
export HARBOR_WORKER_CANCEL_GRACE_SEC=30
export PYTHON_DOTENV_DISABLED=1

# E2B Cloud is the SDK default. Self-hosted E2B-compatible endpoints may set:
# export E2B_API_URL=https://api.example.invalid
# export E2B_SANDBOX_URL=https://sandbox.example.invalid

# A sandbox-resident agent needs the Miles session endpoint on this allowlist.
# A host-process agent such as terminus-2 does not make model calls from E2B.
export HARBOR_AGENT_ALLOWED_HOSTS=trainer.example.invalid
