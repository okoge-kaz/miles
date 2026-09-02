#!/bin/bash
#PBS -N miles-staleness-telemetry-gpu
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:?submit this test from the repository root}"
source "${REPO_ROOT}/experiments/env.sh"
BASELINE_REPO="${BASELINE_REPO:-${REPO_ROOT}-baseline}"
if [[ ! -d "${BASELINE_REPO}" ]]; then
    echo "missing clean baseline worktree: ${BASELINE_REPO}" >&2
    exit 1
fi
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${BASELINE_REPO}:/root/miles-baseline"

export STALENESS_VALIDATION_ROOT="/root/miles/experiments/outputs/staleness-telemetry-gpu-${MILES_JOB_ID_SHORT}"
export STALENESS_PARITY_STEPS="${STALENESS_PARITY_STEPS:-3}"
export STALENESS_WARMUP_STEPS="${STALENESS_WARMUP_STEPS:-2}"
export STALENESS_MEASURED_STEPS="${STALENESS_MEASURED_STEPS:-8}"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail

cd /root/miles
validation_root="${STALENESS_VALIDATION_ROOT}"
source_dump=/ckpt/training/manual/fused-one-step-20260811/perf/subset-128.pt
fixed_dump="${validation_root}/fixed-staleness.pt"
base_load=/ckpt/megatron/Qwen3-4B-Instruct-2507_torch_dist
mkdir -p "${validation_root}"

# Keep the offline stdout parser in the same final validation because it joins
# rollout, trainer, consumption, and throughput streams from the live smoke.
python -m pytest -q \
    tests/fast/test_offpolicy_log_source.py \
    tests/fast/test_analyze_staleness_telemetry.py
python -m black --check \
    experiments/src/offpolicy_acceleration/check_logging.py \
    experiments/src/offpolicy_acceleration/log_source.py \
    tests/fast/ray/rollout/real_ray/test_rollout_manager.py \
    tests/fast/test_offpolicy_log_source.py
python -m isort --check-only \
    experiments/src/offpolicy_acceleration/check_logging.py \
    experiments/src/offpolicy_acceleration/log_source.py \
    tests/fast/ray/rollout/real_ray/test_rollout_manager.py \
    tests/fast/test_offpolicy_log_source.py

# This production-manager test imports Transformer Engine and therefore needs
# libcuda even though its SGLang engines are Ray mocks. Keep it out of the CPU
# suite and run it once on the validation node before the training checks.
python -m pytest -q tests/fast/ray/rollout/real_ray/test_rollout_manager.py \
    | tee "${validation_root}/real-ray-rollout-manager.log"

python tests/manual/prepare_staleness_telemetry_dump.py \
    --input "${source_dump}" \
    --output "${fixed_dump}" \
    --group-size 16 \
    --lags 0 1 2 4 8 17 \
    | tee "${validation_root}/prepare-dump.json"

prepare_load_dir() {
    local load_dir=$1
    if [[ -e "${load_dir}" ]]; then
        echo "refusing to reuse validation directory ${load_dir}" >&2
        return 1
    fi
    mkdir -p "${load_dir}"
    cp "${base_load}/latest_checkpointed_iteration.txt" "${load_dir}/"
    ln -s "${base_load}/release" "${load_dir}/release"
}

run_with_gpu_monitor() {
    local output=$1
    shift
    mkdir -p "${output}"
    local start_ns
    start_ns=$(date +%s%N)
    nvidia-smi \
        --query-gpu=timestamp,index,memory.used,power.draw,utilization.gpu \
        --format=csv,noheader,nounits \
        --loop=1 >"${output}/gpu.csv" &
    local monitor_pid=$!
    set +e
    "$@" 2>&1 | tee "${output}/run.log"
    local run_status=${PIPESTATUS[0]}
    local end_ns
    end_ns=$(date +%s%N)
    python -c "import sys; print((int(sys.argv[2]) - int(sys.argv[1])) / 1e9)" \
        "${start_ns}" "${end_ns}" >"${output}/wall_seconds.txt"
    set -e
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    return "${run_status}"
}

run_validation() {
    local tag=$1
    local steps=$2
    local deterministic=$3
    local metrics=$4
    local histogram=$5
    local code_root=${6:-/root/miles}
    local run_root="${validation_root}/${tag}"
    prepare_load_dir "${run_root}/checkpoint"
    run_with_gpu_monitor "${run_root}/output" \
        env \
        VALIDATION_MODE=fused \
        VALIDATION_LOAD="${run_root}/checkpoint" \
        VALIDATION_DUMP="${fixed_dump}" \
        VALIDATION_OUTPUT="${run_root}/output" \
        VALIDATION_NUM_ROLLOUT="${steps}" \
        VALIDATION_SAVE=0 \
        VALIDATION_VERIFY=0 \
        VALIDATION_DETERMINISTIC="${deterministic}" \
        VALIDATION_SAVE_TRAIN_DATA=0 \
        VALIDATION_FULLY_ASYNC=1 \
        VALIDATION_STALENESS_METRICS="${metrics}" \
        VALIDATION_STALENESS_RATIO_HISTOGRAM="${histogram}" \
        VALIDATION_CODE_ROOT="${code_root}" \
        VALIDATION_DISABLE_KL_CHECKER=1 \
        VALIDATION_ROLLOUT_BATCH_SIZE=8 \
        VALIDATION_N_SAMPLES_PER_PROMPT=16 \
        VALIDATION_GLOBAL_BATCH_SIZE=128 \
        VALIDATION_MAX_TOKENS_PER_GPU=8192 \
        bash tests/manual/run_fused_one_step_validation.sh
}

run_validation parity-baseline "${STALENESS_PARITY_STEPS}" 1 0 0 /root/miles-baseline
run_validation parity-off "${STALENESS_PARITY_STEPS}" 1 0 0
run_validation parity-full "${STALENESS_PARITY_STEPS}" 1 1 1

perf_steps=$((STALENESS_WARMUP_STEPS + STALENESS_MEASURED_STEPS))
run_validation perf-baseline "${perf_steps}" 0 0 0 /root/miles-baseline
run_validation perf-off "${perf_steps}" 0 0 0
run_validation perf-base "${perf_steps}" 0 1 0
run_validation perf-histogram "${perf_steps}" 0 1 1

python tests/manual/analyze_staleness_telemetry_gpu_validation.py \
    --deterministic-baseline "${validation_root}/parity-baseline/output/run.log" \
    --deterministic-off "${validation_root}/parity-off/output/run.log" \
    --deterministic-on "${validation_root}/parity-full/output/run.log" \
    --deterministic-baseline-dir "${validation_root}/parity-baseline/output" \
    --deterministic-off-dir "${validation_root}/parity-off/output" \
    --deterministic-on-dir "${validation_root}/parity-full/output" \
    --performance-baseline "${validation_root}/perf-baseline/output/run.log" \
    --performance-off "${validation_root}/perf-off/output/run.log" \
    --performance-on "${validation_root}/perf-base/output/run.log" \
    --performance-histogram "${validation_root}/perf-histogram/output/run.log" \
    --performance-baseline-gpu "${validation_root}/perf-baseline/output/gpu.csv" \
    --performance-off-gpu "${validation_root}/perf-off/output/gpu.csv" \
    --performance-on-gpu "${validation_root}/perf-base/output/gpu.csv" \
    --performance-histogram-gpu "${validation_root}/perf-histogram/output/gpu.csv" \
    --warmup-steps "${STALENESS_WARMUP_STEPS}" \
    --output "${validation_root}/result.json"

python tests/manual/benchmark_staleness_telemetry_overhead.py \
    | tee "${validation_root}/rollout-hotpath-benchmark.json"
python tests/manual/benchmark_sglang_forward_version_stamp.py \
    | tee "${validation_root}/sglang-hotpath-benchmark.json"

echo "STALENESS_VALIDATION_ROOT=${validation_root}"
'
