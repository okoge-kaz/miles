#!/bin/bash
#SBATCH --job-name=miles-staleness-telemetry-cpu
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/staleness-telemetry-cpu-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
SQSH_IMAGE="${SQSH_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/container/miles-staleness-weight-boundaries-f994b9aed.sqsh}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${REPO_ROOT}:/workspace/miles" \
    --container-writable \
    --no-container-mount-home \
    bash -lc '
set -euo pipefail
cd /workspace/miles
export PYTHONPATH=/workspace/miles
python -m pytest -q \
    tests/fast/backends/training_utils/loss/test_staleness_metrics.py \
    tests/fast/backends/training_utils/test_debug_dump.py \
    tests/fast/test_analyze_staleness_telemetry.py \
    tests/fast/test_offpolicy_log_source.py \
    tests/fast/rollout/test_recycle_compute_metrics.py \
    tests/fast/ray/rollout/test_metrics.py \
    tests/fast/ray/rollout/test_train_data_conversion.py \
    tests/fast/rollout/test_fully_async_checkpoint.py \
    tests/fast/rollout/session/test_samples_codec.py \
    tests/fast/rollout/test_queue_policy.py \
    tests/fast/rollout/test_queue_telemetry.py \
    tests/fast/utils/test_types.py \
    tests/fast/rollout/test_fully_async_rollout.py \
    tests/fast/test_train_async_checkpoint.py

python_files=(
    experiments/src/offpolicy_acceleration/analyze_staleness_telemetry.py
    experiments/src/offpolicy_acceleration/check_logging.py
    experiments/src/offpolicy_acceleration/log_source.py
    miles/backends/experimental/fsdp_utils/actor.py
    miles/backends/megatron_utils/actor.py
    miles/backends/megatron_utils/model.py
    miles/backends/training_utils/debug_dump.py
    miles/backends/training_utils/log_utils.py
    miles/backends/training_utils/loss.py
    miles/backends/training_utils/loss_hub/losses.py
    miles/backends/training_utils/loss_hub/staleness_metrics.py
    miles/ray/rollout/metrics.py
    miles/ray/rollout/rollout_manager.py
    miles/ray/rollout/train_data_conversion.py
    miles/rollout/base_types.py
    miles/rollout/fully_async_rollout.py
    miles/rollout/fully_async_telemetry.py
    miles/rollout/generate_utils/sample_utils.py
    miles/rollout/inference_rollout/inference_rollout_common.py
    miles/rollout/recycle_compute_metrics.py
    miles/rollout/session/samples/codec.py
    miles/utils/arguments.py
    miles/utils/train_metric_utils.py
    miles/utils/types.py
    tests/fast/backends/training_utils/loss/test_staleness_metrics.py
    tests/fast/backends/training_utils/test_debug_dump.py
    tests/fast/ray/rollout/real_ray/test_rollout_manager.py
    tests/fast/ray/rollout/test_train_data_conversion.py
    tests/fast/rollout/test_fully_async_checkpoint.py
    tests/fast/rollout/session/test_samples_codec.py
    tests/fast/rollout/test_recycle_compute_metrics.py
    tests/fast/test_analyze_staleness_telemetry.py
    tests/fast/test_offpolicy_log_source.py
    tests/fast/test_train_async_checkpoint.py
    tests/fast/utils/test_types.py
    tests/manual/analyze_staleness_telemetry_gpu_validation.py
    tests/manual/benchmark_sglang_forward_version_stamp.py
    tests/manual/benchmark_staleness_telemetry_overhead.py
    tests/manual/prepare_staleness_telemetry_dump.py
    tests/manual/sglang_prefill_weight_version_smoke.py
    train_async.py
)
if [[ "${STALENESS_FORMAT_CHANGED:-0}" == 1 ]]; then
    python -m black "${python_files[@]}"
    python -m isort "${python_files[@]}"
fi
python -m black --check "${python_files[@]}"
python -m isort --check-only "${python_files[@]}"
if python -c "import ruff" 2>/dev/null; then
    python -m ruff check "${python_files[@]}"
elif command -v ruff >/dev/null 2>&1; then
    ruff check "${python_files[@]}"
else
    echo "ruff unavailable in validation image; pytest, Black, and isort completed" >&2
fi
'
