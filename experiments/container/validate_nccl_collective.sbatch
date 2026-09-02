#!/bin/bash
#PBS -N miles-nccl-check
#PBS -q R9920261300
#PBS -l select=2:ncpus=192:ngpus=8:mpiprocs=1
#PBS -l place=scatter:excl
#PBS -l walltime=00:20:00
#PBS -j oe

# Run a 16-GPU collective over the cluster-provided NCCL/InfiniBand stack and,
# by default, repeat it with Socket transport as a diagnostic control.

set -euo pipefail

REPO_ROOT="${MILES_SUBMIT_DIR:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}}"
source "${REPO_ROOT}/experiments/env.sh"

: "${COLLECTIVE_IMAGE:=${CONTAINER_IMAGE}}"
: "${COMPARE_TCP:=1}"
: "${MILES_NCCL_TRANSPORT:=system}"

[[ -r "${COLLECTIVE_IMAGE}" ]] || {
    echo "missing COLLECTIVE_IMAGE=${COLLECTIVE_IMAGE}" >&2
    exit 1
}

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
mapfile -t ALLOCATED_HOSTS < <(awk '!seen[$1]++ { sub(/\..*/, "", $1); print $1 }' "${PBS_NODEFILE:?}")
(( ${#ALLOCATED_HOSTS[@]} == MILES_JOB_NUM_NODES )) || {
    echo "could not resolve every allocated hostname" >&2
    exit 1
}
export MASTER_ADDR="${ALLOCATED_HOSTS[0]}"

common_run=(
    --nodes="${MILES_JOB_NUM_NODES}"
    --ntasks="${MILES_JOB_NUM_NODES}"
    --ntasks-per-node=1
    --image="${COLLECTIVE_IMAGE}"
    --bind="${CONTAINER_MOUNTS}"
    --no-home
    --export=ALL
)

run_collective() {
    local transport="$1"
    export MILES_NCCL_TRANSPORT="${transport}"
    case "${transport}" in
        system)
            export MASTER_PORT="${SYSTEM_MASTER_PORT:-29640}"
            unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN NCCL_NET FI_PROVIDER
            export NCCL_IB_DISABLE=0
            ;;
        tcp)
            export MASTER_PORT="${TCP_MASTER_PORT:-29641}"
            unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
            export NCCL_NET=Socket NCCL_IB_DISABLE=1
            ;;
        *)
            echo "MILES_NCCL_TRANSPORT must be system or tcp" >&2
            return 2
            ;;
    esac

    echo "running ${transport} collective"
    miles_srun \
        "${common_run[@]}" \
        --gpus-per-task=8 \
        --cpus-per-task=64 \
        bash --noprofile --norc -c '
            set -euo pipefail
            node_rank="${MILES_NODE_RANK:?}"
            echo "torchrun host=$(hostname -s) node_rank=${node_rank} master=${MASTER_ADDR}:${MASTER_PORT}"
            timeout --signal=TERM "${NCCL_COLLECTIVE_TIMEOUT:-300}" torchrun \
                --nnodes="${MILES_JOB_NUM_NODES}" \
                --nproc-per-node=8 \
                --node-rank="${node_rank}" \
                --master-addr="${MASTER_ADDR}" \
                --master-port="${MASTER_PORT}" \
                /root/miles/experiments/container/validate_nccl_collective.py
        '
}

run_collective "${MILES_NCCL_TRANSPORT}"
if [[ "${COMPARE_TCP}" == 1 && "${MILES_NCCL_TRANSPORT}" != tcp ]]; then
    run_collective tcp
fi
