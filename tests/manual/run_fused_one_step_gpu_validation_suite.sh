#!/bin/bash

set -euo pipefail

: "${SUITE_ROOT:?in-container output root}"
: "${SUITE_FIXED_DUMP:?in-container fixed rollout dump}"
: "${SUITE_BASE_LOAD:=/ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist}"
: "${SUITE_WARMUP_STEPS:=2}"
: "${SUITE_MEASURED_STEPS:=20}"
: "${SUITE_LIVE_ROLLOUTS:=3}"

cd /root/miles

prepare_load_dir() {
    local load_dir=$1
    if [[ -e "${load_dir}" ]]; then
        echo "refusing to reuse validation directory ${load_dir}" >&2
        return 1
    fi
    mkdir -p "${load_dir}"
    cp "${SUITE_BASE_LOAD}/latest_checkpointed_iteration.txt" "${load_dir}/"
    ln -s "${SUITE_BASE_LOAD}/release" "${load_dir}/release"
}

run_with_gpu_monitor() {
    local output=$1
    shift
    mkdir -p "${output}"
    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,power.draw \
        --format=csv,noheader,nounits \
        --loop=1 >"${output}/gpu.csv" &
    local monitor_pid=$!
    set +e
    "$@" 2>&1 | tee "${output}/run.log"
    local run_status=${PIPESTATUS[0]}
    set -e
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    return "${run_status}"
}

run_fixed() {
    local mode=$1
    local tag=$2
    local num_rollout=$3
    local deterministic=$4
    local save=$5
    local verify=$6
    local save_train_data=$7
    local load_dir="${SUITE_ROOT}/${tag}/checkpoint"
    local output="${SUITE_ROOT}/${tag}/output"
    prepare_load_dir "${load_dir}"
    run_with_gpu_monitor "${output}" \
        env \
        VALIDATION_MODE="${mode}" \
        VALIDATION_LOAD="${load_dir}" \
        VALIDATION_DUMP="${SUITE_FIXED_DUMP}" \
        VALIDATION_OUTPUT="${output}" \
        VALIDATION_NUM_ROLLOUT="${num_rollout}" \
        VALIDATION_SAVE="${save}" \
        VALIDATION_VERIFY="${verify}" \
        VALIDATION_DETERMINISTIC="${deterministic}" \
        VALIDATION_SAVE_TRAIN_DATA="${save_train_data}" \
        VALIDATION_ROLLOUT_BATCH_SIZE=8 \
        VALIDATION_N_SAMPLES_PER_PROMPT=16 \
        VALIDATION_GLOBAL_BATCH_SIZE=128 \
        VALIDATION_MAX_TOKENS_PER_GPU=8192 \
        bash /root/miles/tests/manual/run_fused_one_step_validation.sh
}

run_live() {
    local mode=$1
    local tag=$2
    local load_dir="${SUITE_ROOT}/${tag}/checkpoint"
    local output="${SUITE_ROOT}/${tag}/output"
    prepare_load_dir "${load_dir}"
    run_with_gpu_monitor "${output}" \
        env \
        LIVE_MODE="${mode}" \
        LIVE_LOAD="${load_dir}" \
        LIVE_OUTPUT="${output}" \
        LIVE_NUM_ROLLOUT="${SUITE_LIVE_ROLLOUTS}" \
        bash /root/miles/tests/manual/run_fused_one_step_live_smoke.sh
}

total_perf_steps=$((SUITE_WARMUP_STEPS + SUITE_MEASURED_STEPS))

echo "START deterministic legacy $(date --iso-8601=seconds)"
run_fixed legacy deterministic-legacy 1 1 1 0 1
echo "DONE deterministic legacy $(date --iso-8601=seconds)"

echo "START deterministic fused shadow $(date --iso-8601=seconds)"
run_fixed fused deterministic-fused 1 1 1 1 1
echo "DONE deterministic fused shadow $(date --iso-8601=seconds)"

echo "START reverse-order perf fused $(date --iso-8601=seconds)"
run_fixed fused reverse-perf-fused "${total_perf_steps}" 0 0 0 0
echo "DONE reverse-order perf fused $(date --iso-8601=seconds)"

echo "START reverse-order perf legacy $(date --iso-8601=seconds)"
run_fixed legacy reverse-perf-legacy "${total_perf_steps}" 0 0 0 0
echo "DONE reverse-order perf legacy $(date --iso-8601=seconds)"

echo "START live fused $(date --iso-8601=seconds)"
run_live fused live-fused
echo "DONE live fused $(date --iso-8601=seconds)"

echo "START live legacy $(date --iso-8601=seconds)"
run_live legacy live-legacy
echo "DONE live legacy $(date --iso-8601=seconds)"

echo "SUITE_ROOT=${SUITE_ROOT}"
