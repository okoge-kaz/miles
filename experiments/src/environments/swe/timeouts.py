"""Single-source timeout contract for repository-level SWE trials."""

from __future__ import annotations

HARBOR_TIMEOUT_MULTIPLIER = 1

AGENT_ENVIRONMENT_START_TIMEOUT_SEC = 1800
AGENT_SETUP_TIMEOUT_SEC = 1800
AGENT_EXECUTION_TIMEOUT_SEC = 3600
COLLECT_TIMEOUT_SEC = 120
VERIFIER_ENVIRONMENT_START_TIMEOUT_SEC = 1800
VERIFIER_EXECUTION_TIMEOUT_SEC = 2100

TRIAL_PHASE_BUDGET_SEC = (
    AGENT_ENVIRONMENT_START_TIMEOUT_SEC
    + AGENT_SETUP_TIMEOUT_SEC
    + AGENT_EXECUTION_TIMEOUT_SEC
    + COLLECT_TIMEOUT_SEC
    + VERIFIER_ENVIRONMENT_START_TIMEOUT_SEC
    + VERIFIER_EXECUTION_TIMEOUT_SEC
)

# Harbor owns this wall clock and cancels the worker before a client gives up.
# The gaps cover artifact transfer and both E2B sandbox teardown paths.
TRIAL_WALL_TIMEOUT_SEC = 12_600
TRIAL_REQUEST_TIMEOUT_SEC = 13_200
TRIAL_TIMEOUT_CEILING_SEC = 14_400


def validate_timeout_contract() -> None:
    """Fail when a future phase change no longer fits the trial ceiling."""

    if HARBOR_TIMEOUT_MULTIPLIER != 1:
        raise RuntimeError("SWE Harbor timeout multiplier must remain one")
    if not (
        TRIAL_PHASE_BUDGET_SEC
        < TRIAL_WALL_TIMEOUT_SEC
        < TRIAL_REQUEST_TIMEOUT_SEC
        < TRIAL_TIMEOUT_CEILING_SEC
    ):
        raise RuntimeError("SWE timeout budgets are inconsistent")


validate_timeout_contract()
