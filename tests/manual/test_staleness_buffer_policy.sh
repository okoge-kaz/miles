#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/common/staleness_buffer_policy.sh"

[[ "$(staleness_training_buffer_queue_size 1 1000)" == 1000 ]]
[[ "$(staleness_training_buffer_queue_size 7 2048)" == 2048 ]]
[[ "$(staleness_training_buffer_queue_size 8 1000)" == 6000 ]]
[[ "$(staleness_training_buffer_queue_size 20 1000)" == 6000 ]]

[[ "$(staleness_required_training_buffer_queue_size 8 3072 16)" == 1536 ]]
[[ "$(staleness_required_training_buffer_queue_size 16 3072 16)" == 3072 ]]
[[ "$(staleness_required_training_buffer_queue_size 20 3072 16)" == 3840 ]]
[[ "$(staleness_required_training_buffer_queue_size 24 3072 16)" == 4608 ]]
[[ "$(staleness_required_training_buffer_queue_size 28 3072 16)" == 5376 ]]
[[ "$(staleness_required_training_buffer_queue_size 32 3072 16)" == 6144 ]]
[[ "$(staleness_required_training_buffer_queue_size 3 10 4)" == 8 ]]

require_staleness_training_buffer_queue_size 8 6000 3072 16
require_staleness_training_buffer_queue_size 20 6000 3072 16
require_staleness_training_buffer_queue_size 28 6000 3072 16
if require_staleness_training_buffer_queue_size 8 1000 3072 16 2>/dev/null; then
    echo "s8 unexpectedly accepted TRAINING_BUFFER_QUEUE_SIZE=1000" >&2
    exit 1
fi
if require_staleness_training_buffer_queue_size 32 6000 3072 16 2>/dev/null; then
    echo "s32 unexpectedly accepted undersized TRAINING_BUFFER_QUEUE_SIZE=6000" >&2
    exit 1
fi
