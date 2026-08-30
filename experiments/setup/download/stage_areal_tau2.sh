#!/bin/bash
# Stage the pinned AReaL RL-only payload, then materialize Miles training rows.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

ACCOUNT="${SLURM_ACCOUNT_NAME:-coreai_horizon_dilations}"
RAW="${DATASET_DIR}/areal-tau2-data/tau2_rl_train.jsonl"
PREPARED="${DATASET_DIR}/areal-tau2-data/miles-tau2-rl-train.jsonl"
SUMMARY="${DATASET_DIR}/areal-tau2-data/miles-tau2-rl-summary.json"
DOWNLOAD_JOB=""

prepared_ready() {
    [[ -r "${PREPARED}" && -r "${SUMMARY}" ]] || return 1
    python3 - "${PREPARED}" "${SUMMARY}" <<'PY'
import hashlib
import json
import sys

data_path, summary_path = sys.argv[1:]
with open(summary_path, encoding="utf-8") as stream:
    summary = json.load(stream)
digest = hashlib.sha256()
with open(data_path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
raise SystemExit(
    not (
        summary.get("rows") == 1982
        and summary.get("dataset_repo") == "inclusionAI/AReaL-tau2-data"
        and summary.get("dataset_revision") == "1322eae337f836fe0e19bae14dab1eefc26bc983"
        and summary.get("interaction_mode") == "stateful_multi_turn_user_simulator_environment"
        and summary.get("verifier") == "areal_tau2_environment"
        and summary.get("tau_package_version") == "1.0.1"
        and len(summary.get("db_sha256", {})) == 9
        and summary.get("output_sha256") == digest.hexdigest()
    )
)
PY
}

if [[ -r "${RAW}" ]] && (cd "${DATASET_DIR}/areal-tau2-data" && \
    sha256sum --status --check "${REPO_ROOT}/experiments/setup/manifests/areal_tau2_rl.sha256"); then
    echo "present AReaL Tau2 RL source"
else
    DOWNLOAD_JOB="$(sbatch --parsable -A "${ACCOUNT}" --export=NIL \
        "${REPO_ROOT}/experiments/setup/download/download_areal_tau2.sbatch")"
    echo "submitted download ${DOWNLOAD_JOB}"
fi

if prepared_ready; then
    echo "present AReaL Tau2 Miles training data"
    exit 0
fi

DEPENDENCY=()
[[ -z "${DOWNLOAD_JOB}" ]] || DEPENDENCY=(--dependency="afterok:${DOWNLOAD_JOB}")
PREPARE_JOB="$(sbatch --parsable -A "${ACCOUNT}" --export=NIL "${DEPENDENCY[@]}" \
    "${REPO_ROOT}/experiments/setup/environments/prepare_areal_tau2.sbatch")"
echo "submitted prepare ${PREPARE_JOB}"
