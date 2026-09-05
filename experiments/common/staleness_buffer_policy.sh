#!/bin/bash
# Shared queue-capacity policy for the DAPO math staleness experiments.

MILES_HIGH_STALENESS_THRESHOLD=8
MILES_HIGH_STALENESS_TRAINING_BUFFER_QUEUE_SIZE=6000

staleness_training_buffer_queue_size() {
    local max_weight_staleness="$1"
    local default_queue_size="$2"

    [[ "${max_weight_staleness}" =~ ^[1-9][0-9]*$ ]] || {
        echo "MAX_WEIGHT_STALENESS must be a positive integer, got '${max_weight_staleness}'" >&2
        return 1
    }
    [[ "${default_queue_size}" =~ ^[1-9][0-9]*$ ]] || {
        echo "TRAINING_BUFFER_QUEUE_SIZE must be a positive integer, got '${default_queue_size}'" >&2
        return 1
    }

    if (( max_weight_staleness >= MILES_HIGH_STALENESS_THRESHOLD )); then
        printf '%s\n' "${MILES_HIGH_STALENESS_TRAINING_BUFFER_QUEUE_SIZE}"
    else
        printf '%s\n' "${default_queue_size}"
    fi
}

staleness_required_training_buffer_queue_size() {
    local max_weight_staleness="$1"
    local global_batch_size="$2"
    local samples_per_prompt="$3"

    [[ "${max_weight_staleness}" =~ ^[1-9][0-9]*$ ]] || {
        echo "MAX_WEIGHT_STALENESS must be a positive integer, got '${max_weight_staleness}'" >&2
        return 1
    }
    [[ "${global_batch_size}" =~ ^[1-9][0-9]*$ ]] || {
        echo "GLOBAL_BATCH_SIZE must be a positive integer, got '${global_batch_size}'" >&2
        return 1
    }
    [[ "${samples_per_prompt}" =~ ^[1-9][0-9]*$ ]] || {
        echo "N_SAMPLES_PER_PROMPT must be a positive integer, got '${samples_per_prompt}'" >&2
        return 1
    }

    # The completed queue is measured in prompt groups, while one optimizer
    # update consumes GLOBAL_BATCH_SIZE trajectories. Reserve enough groups for
    # the requested number of weight updates, rounding up for general batch
    # shapes.
    printf '%s\n' "$((
        (global_batch_size * max_weight_staleness + samples_per_prompt - 1)
        / samples_per_prompt
    ))"
}

require_staleness_training_buffer_capacity() {
    local max_weight_staleness="$1"
    local actual_queue_size="$2"
    local global_batch_size="$3"
    local samples_per_prompt="$4"
    local required_queue_size
    required_queue_size="$(
        staleness_required_training_buffer_queue_size \
            "${max_weight_staleness}" \
            "${global_batch_size}" \
            "${samples_per_prompt}"
    )" || return 1

    (( actual_queue_size >= required_queue_size )) || {
        echo "TRAINING_BUFFER_QUEUE_SIZE=${actual_queue_size} is too small for" \
             "MAX_WEIGHT_STALENESS=${max_weight_staleness}: need at least" \
             "ceil(GLOBAL_BATCH_SIZE * MAX_WEIGHT_STALENESS /" \
             "N_SAMPLES_PER_PROMPT) = ceil(${global_batch_size} *" \
             "${max_weight_staleness} / ${samples_per_prompt}) =" \
             "${required_queue_size} completed groups" >&2
        return 1
    }
}

require_staleness_training_buffer_queue_size() {
    local max_weight_staleness="$1"
    local actual_queue_size="$2"
    local global_batch_size="$3"
    local samples_per_prompt="$4"
    local expected_queue_size
    expected_queue_size="$(
        staleness_training_buffer_queue_size \
            "${max_weight_staleness}" \
            "${actual_queue_size}"
    )" || return 1

    [[ "${actual_queue_size}" == "${expected_queue_size}" ]] || {
        echo "MAX_WEIGHT_STALENESS=${max_weight_staleness} requires" \
             "TRAINING_BUFFER_QUEUE_SIZE=${expected_queue_size}, got ${actual_queue_size}" >&2
        return 1
    }

    require_staleness_training_buffer_capacity \
        "${max_weight_staleness}" \
        "${actual_queue_size}" \
        "${global_batch_size}" \
        "${samples_per_prompt}"
}
