#!/bin/bash
#SBATCH --job-name=sglang-prefill-smoke
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
IMAGE="/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/container/miles-latest-20260803.sqsh"
CHECKPOINT_ROOT="/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/checkpoints/hf"

srun \
    --container-image="${IMAGE}" \
    --container-mounts="${REPO_ROOT}:/root/miles,${CHECKPOINT_ROOT}:/ckpt/hf" \
    --container-writable \
    --no-container-mount-home \
    bash -lc '
        set -euo pipefail
        cd /sgl-workspace/sglang
        git fetch https://github.com/okoge-kaz/sglang.git miles-prefill-weight-version
        git checkout -f 23aaf6597dbbe47f9ceb6090e10372c844fed960

        python -m sglang.launch_server \
            --model-path /ckpt/hf/Qwen3-1.7B \
            --host 127.0.0.1 \
            --port 30000 \
            --weight-version 10 \
            --disable-cuda-graph \
            --mem-fraction-static 0.7 \
            >/tmp/sglang-prefill-smoke-server.log 2>&1 &
        server_pid=$!

        cleanup() {
            kill "${server_pid}" 2>/dev/null || true
            wait "${server_pid}" 2>/dev/null || true
            tail -200 /tmp/sglang-prefill-smoke-server.log
        }
        trap cleanup EXIT

        for _ in $(seq 1 180); do
            if curl -fsS http://127.0.0.1:30000/health >/dev/null; then
                break
            fi
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                echo "SGLang server exited during startup" >&2
                exit 1
            fi
            sleep 1
        done
        curl -fsS http://127.0.0.1:30000/health >/dev/null

        python /root/miles/tests/manual/sglang_prefill_weight_version_smoke.py
    '
