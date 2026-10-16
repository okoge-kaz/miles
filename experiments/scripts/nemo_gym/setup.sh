#!/bin/bash

set -euo pipefail

MILES_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${NEMO_GYM_ROOT:?set a writable directory for the CPU runtime}"
export PYTHONPATH="${MILES_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m experiments.src.nemo_gym.runtime.setup --root "${NEMO_GYM_ROOT}" "$@"
