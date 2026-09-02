#!/bin/bash
#PBS -N sglang-prefill-smoke
#PBS -q R9920261300
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=00:15:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:?submit this test from the repository root}"
source "${REPO_ROOT}/experiments/env.sh"
export SGLANG_SMOKE_MODEL_PATH="${SGLANG_SMOKE_MODEL_PATH:-/ckpt/hf/Qwen3-4B}"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    bash -lc '
        set -euo pipefail
        test "$(git -c safe.directory=/sgl-workspace/sglang \
            -C /sgl-workspace/sglang rev-parse HEAD)" = "${SGLANG_COMMIT}"
        test -r "${SGLANG_SMOKE_MODEL_PATH}/config.json"

        python -m sglang.launch_server \
            --model-path "${SGLANG_SMOKE_MODEL_PATH}" \
            --host 127.0.0.1 \
            --port 30000 \
            --weight-version 10 \
            --enable-response-weight-version-segments \
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
