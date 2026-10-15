#!/bin/bash

set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../../.."
export PYTHONDONTWRITEBYTECODE=1
if (( $# == 0 )); then
    data_root="${RL_DATASET_DIR:-${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}/datasets/rl}"
    shopt -s nullglob
    directories=("${data_root}"/difficulty/*-sft-iter4000-n16-len16384-*parts)
    (( ${#directories[@]} )) || { echo 'No filtering measurements found' >&2; exit 1; }
    set -- "${directories[@]}"
fi
exec python3 -m experiments.src.difficulty_filtering.utils.progress "$@"
