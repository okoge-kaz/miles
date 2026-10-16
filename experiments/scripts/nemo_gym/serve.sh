#!/bin/bash

set -euo pipefail

MILES_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${NEMO_GYM_ROOT:=/opt/nemo-gym}"
export PYTHONPATH="${MILES_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${NEMO_GYM_ROOT}/.venv/bin/python" -m experiments.src.nemo_gym.runtime.serve --root "${NEMO_GYM_ROOT}" "$@"
