#!/bin/bash
#SBATCH --job-name=miles-efa-nccl
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Run one 16-GPU all-reduce with forced AWS Libfabric, then an identical Socket
# control. Success requires the EFA provider and named OFI plugin on every node,
# plus direct proof that every rank loaded the named OFI plugin and AWS
# libfabric while NCCL_NET forced the registered AWS Libfabric network.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
source "${REPO_ROOT}/experiments/env.sh"

: "${EFA_SQSH_IMAGE:=${CONTAINER_DIR}/miles-staleness-weight-boundaries-f994b9aed-efa-1.49.0.sqsh}"
: "${COMPARE_TCP:=1}"

test -r "${EFA_SQSH_IMAGE}"

export NCCL_NET_PLUGIN=ofi
export NCCL_TUNER_PLUGIN=ofi
export NCCL_NET="AWS Libfabric"
export NCCL_IB_DISABLE=0
export FI_PROVIDER=efa
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mapfile -t ALLOCATED_HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#ALLOCATED_HOSTS[@]} == SLURM_JOB_NUM_NODES )) || {
    echo "could not resolve every allocated hostname" >&2
    exit 1
}
printf -v ALLOCATED_HOSTS_TEXT '%s ' "${ALLOCATED_HOSTS[@]}"
export ALLOCATED_HOSTS_TEXT
# Slurm does not guarantee that task 0 is placed on NodeList[0], even with a
# block distribution.  Every task derives its torchrun node rank from its own
# hostname below, so rank 0 and MASTER_ADDR always identify the same node.
export MASTER_ADDR="${ALLOCATED_HOSTS[0]}"
export MASTER_PORT="${MASTER_PORT:-29640}"

common_srun=(
    --nodes="${SLURM_JOB_NUM_NODES}"
    --container-image="${EFA_SQSH_IMAGE}"
    --container-mounts="${CONTAINER_MOUNTS}"
    --no-container-mount-home
    --export=ALL
)

echo "Validating EFA devices and plugin on every node"
srun \
    "${common_srun[@]}" \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    bash /root/miles/experiments/common/check_efa.sh

run_collective() {
    local transport="$1"
    export MILES_NCCL_TRANSPORT="${transport}"
    if [[ "${transport}" == efa ]]; then
        export MASTER_PORT="${EFA_MASTER_PORT:-29640}"
    else
        export MASTER_PORT="${TCP_MASTER_PORT:-29641}"
    fi
    echo "Running ${transport} collective"
    srun \
        "${common_srun[@]}" \
        --ntasks="${SLURM_JOB_NUM_NODES}" \
        --ntasks-per-node=1 \
        --gpus-per-task=8 \
        --cpus-per-task=64 \
        bash --noprofile --norc -c '
            set -euo pipefail
            export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            if [[ "${MILES_NCCL_TRANSPORT}" == tcp ]]; then
                unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
            fi
            read -r -a allocated_hosts <<< "${ALLOCATED_HOSTS_TEXT}"
            current_host="$(hostname -s)"
            node_rank=""
            for index in "${!allocated_hosts[@]}"; do
                if [[ "${allocated_hosts[index]}" == "${current_host}" ]]; then
                    node_rank="${index}"
                    break
                fi
            done
            [[ -n "${node_rank}" ]] || {
                echo "host ${current_host} is absent from ${SLURM_JOB_NODELIST}" >&2
                exit 1
            }
            echo "torchrun host=${current_host} node_rank=${node_rank} master=${MASTER_ADDR}:${MASTER_PORT}"
            timeout --signal=TERM "${EFA_COLLECTIVE_TIMEOUT:-300}" torchrun \
                --nnodes="${SLURM_JOB_NUM_NODES}" \
                --nproc-per-node=8 \
                --node-rank="${node_rank}" \
                --master-addr="${MASTER_ADDR}" \
                --master-port="${MASTER_PORT}" \
                /root/miles/experiments/container/validate_efa_collective.py
        '
}

run_collective efa

if [[ "${COMPARE_TCP}" == 1 ]]; then
    unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN FI_PROVIDER
    export NCCL_NET=Socket
    # This is an explicit Socket-only control measurement, not a training
    # workaround. EFA training always runs through run_with_efa_env.sh above.
    export NCCL_IB_DISABLE=1
    run_collective tcp
fi
