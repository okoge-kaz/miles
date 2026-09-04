#!/bin/bash
# Fail closed before a multi-node job can silently fall back to TCP or NET/IB.

set -euo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

: "${NCCL_NET_PLUGIN:?EFA jobs must set NCCL_NET_PLUGIN=ofi}"
: "${NCCL_IB_DISABLE:?EFA jobs must set NCCL_IB_DISABLE=0}"
: "${NCCL_NET:?EFA jobs must set NCCL_NET='AWS Libfabric'}"

[[ "${NCCL_NET_PLUGIN}" == ofi ]] || {
    echo "NCCL_NET_PLUGIN must be ofi, got ${NCCL_NET_PLUGIN}" >&2
    exit 1
}
[[ "${NCCL_IB_DISABLE}" == 0 ]] || {
    echo "NCCL_IB_DISABLE must be 0 for EFA, got ${NCCL_IB_DISABLE}" >&2
    exit 1
}
[[ "${NCCL_NET}" == "AWS Libfabric" ]] || {
    echo "NCCL_NET must be 'AWS Libfabric', got ${NCCL_NET}" >&2
    exit 1
}

plugin=/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
fi_info=/opt/amazon/efa/bin/fi_info

test -s "${plugin}"
test -x "${fi_info}"
plugin_dependencies="$(ldd "${plugin}")"
if grep -q 'not found' <<<"${plugin_dependencies}"; then
    printf '%s\n' "${plugin_dependencies}" >&2
    exit 1
fi
loader_cache="$(ldconfig -p)"
grep -q 'libnccl-net-ofi.so' <<<"${loader_cache}"

provider_info="$("${fi_info}" -p efa -t FI_EP_RDM)"
grep -q 'provider: efa' <<<"${provider_info}"
provider_count="$(grep -c 'provider: efa' <<<"${provider_info}")"
plugin_version="$(dpkg-query --show --showformat='${Version}' libnccl-ofi-ngc-v3)"
libfabric_version="$(dpkg-query --show --showformat='${Version}' libfabric1-aws)"

printf 'EFA preflight host=%s network=%q provider_count=%s ofi_nccl=%s libfabric=%s\n' \
    "$(hostname)" "${NCCL_NET}" "${provider_count}" "${plugin_version}" "${libfabric_version}"
