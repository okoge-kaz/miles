#!/bin/bash
# Install the EFA userspace stack in a container without making OFI NCCL the
# default network plugin. The host supplies the EFA kernel module and devices.

set -euo pipefail

: "${EFA_INSTALLER_VERSION:=1.49.0}"
: "${EFA_INSTALLER_SHA256:=cf2e9281a2328a243c76f911a490faed43ca0fecfe4733c25e34b2e92a32c309}"

archive="aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
work_dir="$(mktemp -d /tmp/miles-efa-install.XXXXXX)"

cleanup() {
    case "${work_dir}" in
        /tmp/miles-efa-install.*) rm -rf -- "${work_dir}" ;;
        *) return 1 ;;
    esac
}
trap cleanup EXIT

curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --show-error \
    --silent \
    --output "${work_dir}/${archive}" \
    "https://efa-installer.amazonaws.com/${archive}"
printf '%s  %s\n' "${EFA_INSTALLER_SHA256}" "${work_dir}/${archive}" | sha256sum --check --strict

tar -xzf "${work_dir}/${archive}" -C "${work_dir}"
(
    cd "${work_dir}/aws-efa-installer"
    # --build-ngc is the upstream container-build mode. It implies --skip-kmod,
    # --skip-limit-conf, --no-verify, and --yes. Keep MPI out of the image: Miles
    # launches collectives with Slurm/Pyxis and only needs EFA libfabric + OFI NCCL.
    ./efa_installer.sh --build-ngc --skip-mpi
)

plugin_dir=/opt/amazon/ofi-nccl/lib
plugin="${plugin_dir}/libnccl-net-ofi.so"
tuner="${plugin_dir}/libnccl-tuner-ofi.so"

test -s "${plugin}"
test -s "${tuner}"
test -x /opt/amazon/efa/bin/fi_info

# A generic plugin name is auto-discovered by NCCL and would silently change
# transport selection on non-EFA hosts. EFA jobs opt in with the named plugin
# and NCCL_NET="AWS Libfabric"; all other jobs retain their previous NCCL
# transport selection.
if [[ -e "${plugin_dir}/libnccl-net.so" || -L "${plugin_dir}/libnccl-net.so" ]]; then
    echo "unexpected generic NCCL net plugin: ${plugin_dir}/libnccl-net.so" >&2
    exit 1
fi

if ldd "${plugin}" | grep -q 'not found'; then
    ldd "${plugin}" >&2
    exit 1
fi

dpkg-query --show \
    --showformat='${Package}=${Version}\n' \
    libfabric1-aws libnccl-ofi-ngc-v3 rdma-core
echo "AWS EFA userspace installed; enable explicitly with NCCL_NET_PLUGIN=ofi and NCCL_NET='AWS Libfabric'"
