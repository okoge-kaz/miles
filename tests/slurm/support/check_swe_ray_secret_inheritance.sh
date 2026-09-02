#!/bin/bash

set -euo pipefail
umask 077

: "${HARBOR_RUN_SECRET:?generated Harbor run secret is missing}"
: "${RAY_TEST_ROOT:?private Ray test root is missing}"
[[ "${RAY_DASHBOARD_HOST:?}" == 127.0.0.1 ]] || {
    echo "Ray dashboard must be bound to loopback" >&2
    exit 2
}

mkdir -p -m 0700 "${RAY_TEST_ROOT}"
chmod 0700 "${RAY_TEST_ROOT}"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
ray stop --force >/dev/null 2>&1 || true
ray start --head \
    --node-ip-address=127.0.0.1 \
    --port=6379 \
    --num-cpus=2 \
    --num-gpus=0 \
    --disable-usage-stats \
    --dashboard-host="${RAY_DASHBOARD_HOST}" \
    --dashboard-port=8265 \
    --temp-dir="${RAY_TEST_ROOT}/ray" >/dev/null

# No runtime-env secret and no secret-bearing command argument is used here.
# Ray head/driver/workers inherit the fixed-name PBS export instead.
ray job submit --address=http://127.0.0.1:8265 -- \
    python3 /root/miles/tests/slurm/support/check_swe_ray_secret_inheritance.py
