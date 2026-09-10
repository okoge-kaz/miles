#!/bin/bash
# Inside the writable OCI image, with the staged Step4000 HF parent at /ckpt/hf.
# Qualifies scheduler provenance only, not multi-node training.
set -euo pipefail
cd /root/miles
bash experiments/container/apply_oci_sglang_provenance.sh
python -m sglang.launch_server \
    --model-path /ckpt/hf/iter_0004000 \
    --host 127.0.0.1 --port 4080 --weight-version 10 \
    --enable-response-weight-version-segments \
    --disable-cuda-graph --mem-fraction-static 0.7 \
    >/tmp/oci-prefill-server.log 2>&1 &
server_pid=$!
cleanup() {
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    tail -100 /tmp/oci-prefill-server.log
}
trap cleanup EXIT
for _ in $(seq 1 360); do
    if curl -fsS http://127.0.0.1:4080/health >/dev/null 2>&1; then
        break
    fi
    kill -0 "${server_pid}" || exit 1
    sleep 1
done
curl -fsS http://127.0.0.1:4080/health >/dev/null
python tests/manual/sglang_prefill_weight_version_smoke.py --base-url http://127.0.0.1:4080
