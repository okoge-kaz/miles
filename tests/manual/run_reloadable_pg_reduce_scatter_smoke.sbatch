#!/bin/bash
#PBS -N miles-rpg-rs-smoke
#PBS -q R9920261300
#PBS -l select=2:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=00:05:00
#PBS -j oe

set -euo pipefail

REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}}"
source "${REPO_ROOT}/experiments/env.sh"

: "${RPG_SMOKE_IMAGE:?set RPG_SMOKE_IMAGE to an immutable dated SIF path}"
[[ "${RPG_SMOKE_IMAGE}" == /* && -f "${RPG_SMOKE_IMAGE}" && ! -L "${RPG_SMOKE_IMAGE}" ]] || {
    echo "RPG_SMOKE_IMAGE must be an absolute, dated regular SIF: ${RPG_SMOKE_IMAGE}" >&2
    exit 2
}
sha256sum "${RPG_SMOKE_IMAGE}"

unset NCCL_NET NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
export RPG_EXPECTED_CPU_COUNT="${PBS_GPU_CPUS_PER_NODE}"

miles_srun \
    --nodes="${MILES_JOB_NUM_NODES}" \
    --ntasks="${MILES_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${PBS_GPU_CPUS_PER_NODE}" \
    --image="${RPG_SMOKE_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --nv \
    --export=ALL \
    bash -lc '
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 7)"
        export PYTHONPATH=/root/Megatron-LM:/root/miles
        cd /root/miles
        timeout 180s /usr/local/bin/torchrun \
            --standalone \
            --nproc-per-node=2 \
            tests/manual/reloadable_pg_reduce_scatter_smoke.py
    '
