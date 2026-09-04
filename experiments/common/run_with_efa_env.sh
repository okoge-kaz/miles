#!/bin/bash
# Pyxis constructs LD_LIBRARY_PATH while entering the container. Set the EFA
# paths and select the AWS OFI network after that boundary. NCCL_NET is
# deliberate: unlike NCCL_NET_PLUGIN, it makes a missing OFI network fatal
# instead of allowing NCCL to fall back silently to another transport.

set -euo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NCCL_NET="AWS Libfabric"
export NCCL_NET_PLUGIN=ofi
export NCCL_TUNER_PLUGIN=ofi
export NCCL_IB_DISABLE=0
export FI_PROVIDER=efa
exec "$@"
