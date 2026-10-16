#!/bin/bash

set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../../.."
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m experiments.src.difficulty_filtering.utils.submit_suite "$@"
