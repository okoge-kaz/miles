#!/bin/bash
#SBATCH --job-name=queue-unit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/queue-unit-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
MODE="${1:-check}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL,QUEUE_VALIDATION_MODE="${MODE}" \
    bash -lc '
        set -euo pipefail
        cd /root/miles
        python_files=(
            experiments/analyze_queue_lifecycle.py
            experiments/bench_queue_lifecycle_logging.py
            miles/ray/rollout/metrics.py
            miles/ray/rollout/rollout_manager.py
            miles/rollout/base_types.py
            miles/rollout/fully_async_rollout.py
            miles/rollout/queue_policy.py
            miles/rollout/queue_telemetry.py
            miles/utils/arguments.py
            tests/fast/ray/rollout/real_ray/test_rollout_manager.py
            tests/fast/ray/rollout/test_metrics.py
            tests/fast/rollout/test_fully_async_rollout.py
            tests/fast/rollout/test_queue_lifecycle_analysis.py
            tests/fast/rollout/test_queue_policy.py
            train_async.py
        )
        if [[ "${QUEUE_VALIDATION_MODE}" == format ]]; then
            python -m isort "${python_files[@]}"
            python -m black "${python_files[@]}"
            exit 0
        fi
        python -m isort --check-only "${python_files[@]}"
        python -m black --check "${python_files[@]}"
        pytest -q \
            tests/fast/rollout/test_fully_async_rollout.py \
            tests/fast/rollout/test_queue_lifecycle_analysis.py \
            tests/fast/rollout/test_queue_policy.py \
            tests/fast/ray/rollout/test_debug_data.py \
            tests/fast/ray/rollout/test_metrics.py
        python experiments/bench_queue_lifecycle_logging.py
    '
