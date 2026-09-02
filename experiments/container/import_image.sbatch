#!/bin/bash
#PBS -N miles-import-image
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Build the configured OCI image through the Miles definition file into an
# immutable SIF, then update the stable miles.sif symlink. The definition fixes
# non-root permissions and creates every static bind destination.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

command -v singularity >/dev/null || {
    echo "singularity is not available on $(hostname)" >&2
    exit 1
}

pbs_job_id="${PBS_JOBID:-manual-$$}"
job_id="${pbs_job_id%%.*}"
dated_image="${IMPORT_OUTPUT_IMAGE:-${CONTAINER_DIR}/miles-$(date +%Y%m%d-%H%M%S).sif}"
stable_image="${SINGULARITY_LINK-${CONTAINER_DIR}/miles.sif}"
publishing_image="${dated_image}.publishing-${job_id}"
checksum_file="${dated_image}.sha256"
provenance_file="${dated_image}.provenance.env"
partial_checksum="${checksum_file}.partial-${job_id}"
partial_provenance="${provenance_file}.partial-${job_id}"
definition_file="${MILES_REPO}/experiments/container/miles.def"
oci_image="${DOCKER_IMAGE#docker://}"
sglang_repo="${SGLANG_REPO}"
sglang_branch="${SGLANG_BRANCH}"
sglang_commit="${SGLANG_COMMIT}"
dotenv_mask="${MILES_REPO}/experiments/common/dotenv.disabled"
smoke_mounts="${CONTAINER_MOUNTS},${dotenv_mask}:/root/miles/.env:ro,/usr/bin/bwrap:/usr/local/bin/bwrap:ro"
pbs_local_job_dir="/local/${pbs_job_id}"
pbs_local_short_job_dir="/local/${job_id}"
if [[ -n "${PBS_JOBID:-}" ]]; then
    if [[ -d "${pbs_local_job_dir}" && -w "${pbs_local_job_dir}" ]]; then
        local_scratch_root="${pbs_local_job_dir}"
    elif [[ "${pbs_local_short_job_dir}" != "${pbs_local_job_dir}" \
        && -d "${pbs_local_short_job_dir}" && -w "${pbs_local_short_job_dir}" ]]; then
        local_scratch_root="${pbs_local_short_job_dir}"
    elif [[ -n "${PBS_LOCALDIR:-}" && "${PBS_LOCALDIR}" == /local/* \
        && -d "${PBS_LOCALDIR}" && -w "${PBS_LOCALDIR}" ]]; then
        local_scratch_root="${PBS_LOCALDIR}"
    elif [[ -d /local && -w /local ]] && mkdir -p -- "${pbs_local_job_dir}"; then
        local_scratch_root="${pbs_local_job_dir}"
    else
        echo "PBS jobs require writable node-local storage at ${pbs_local_job_dir}" >&2
        exit 1
    fi
elif [[ -n "${PBS_LOCALDIR:-}" && -d "${PBS_LOCALDIR}" && -w "${PBS_LOCALDIR}" ]]; then
    local_scratch_root="${PBS_LOCALDIR}"
else
    local_scratch_root="${TMPDIR:-/tmp}"
fi
[[ -d "${local_scratch_root}" && -w "${local_scratch_root}" ]] || {
    echo "node-local scratch is not writable: ${local_scratch_root}" >&2
    exit 1
}
local_build_dir=""
local_image=""
local_definition=""
owns_local_build_dir=0

[[ "${dated_image}" == /* ]] || {
    echo "IMPORT_OUTPUT_IMAGE must be absolute: ${dated_image}" >&2
    exit 1
}
[[ -n "${oci_image}" && "${oci_image}" != *[[:space:]]* ]] || {
    echo "DOCKER_IMAGE must be a non-empty OCI reference without whitespace" >&2
    exit 1
}
[[ "${sglang_repo}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || {
    echo "SGLANG_REPO must be an owner/repository GitHub path: ${sglang_repo}" >&2
    exit 1
}
[[ "${sglang_branch}" =~ ^[A-Za-z0-9._/-]+$ ]] || {
    echo "SGLANG_BRANCH contains unsupported characters: ${sglang_branch}" >&2
    exit 1
}
[[ "${sglang_commit}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "SGLANG_COMMIT must be a full lowercase SHA-1: ${sglang_commit}" >&2
    exit 1
}
if [[ "${oci_image}" != *@sha256:* ]]; then
    echo "warning: DOCKER_IMAGE is mutable; use repository@sha256:digest for a reproducible SIF" >&2
fi
if [[ -n "${stable_image}" ]]; then
    [[ "${stable_image}" == /* && "${stable_image}" != "${dated_image}" ]] || {
        echo "SINGULARITY_LINK must be an absolute path different from the output" >&2
        exit 1
    }
    if [[ -e "${stable_image}" && ! -L "${stable_image}" ]]; then
        echo "refusing to replace non-symlink: ${stable_image}" >&2
        exit 1
    fi
fi

mkdir -p "${CONTAINER_DIR}" "$(dirname -- "${dated_image}")"
exec 9> "${CONTAINER_DIR}/.miles-image-build.lock"
flock --nonblock 9 || {
    echo "another Miles image build is active" >&2
    exit 5
}
[[ -r "${definition_file}" ]] || {
    echo "missing Singularity definition: ${definition_file}" >&2
    exit 1
}
definition_sha256="$(sha256sum "${definition_file}")"
definition_sha256="${definition_sha256%% *}"
[[ -r "${dotenv_mask}" && -x /usr/bin/bwrap ]] || {
    echo "nested-bind smoke-test sources are unavailable" >&2
    exit 1
}

local_build_dir="$(mktemp -d "${local_scratch_root%/}/miles-singularity-${job_id}.XXXXXX")"
owns_local_build_dir=1
local_image="${local_build_dir}/miles-build-${job_id}.sif"
local_definition="${local_build_dir}/miles.def"
install -m 0644 -- "${definition_file}" "${local_definition}"
local_definition_sha256="$(sha256sum "${local_definition}")"
local_definition_sha256="${local_definition_sha256%% *}"
[[ "${local_definition_sha256}" == "${definition_sha256}" ]] || {
    echo "job-local Singularity definition copy failed checksum validation" >&2
    exit 1
}

cleanup() {
    rm -f -- \
        "${local_image}" "${publishing_image}" \
        "${partial_checksum}" "${partial_provenance}"
    if [[ ! -e "${dated_image}" ]]; then
        rm -f -- "${checksum_file}" "${provenance_file}"
    fi
    if (( owns_local_build_dir != 0 )); then
        rm -rf -- "${local_build_dir}"
        owns_local_build_dir=0
    fi
}
trap cleanup EXIT

singularity_env() {
    local current_dir
    current_dir="$(pwd)"
    export SINGULARITY_TMPDIR="${current_dir}"
    export APPTAINER_TMPDIR="${current_dir}"
    export SINGULARITY_CACHEDIR="${current_dir}"
    export APPTAINER_CACHEDIR="${current_dir}"
    printf 'Singularity build environment: %s\n' "${current_dir}"
    printf '  TMPDIR=%s\n  CACHEDIR=%s\n' "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"
}

cd "${local_build_dir}"
singularity_env

smoke_test() {
    local image="$1"
    singularity inspect "${image}" >/dev/null
    # Verify image-resident code before the checkout bind shadows /root/miles.
    singularity exec \
        --no-eval \
        --no-home \
        "${image}" \
        /bin/bash --noprofile --norc -c '
	set -euo pipefail
	test -x /root
	test ! -e /opt/amazon
	for package in libfabric1-aws libnccl-ofi-ngc-v3; do
	    ! dpkg-query -W "${package}" >/dev/null 2>&1
	done
	test ! -e /etc/libibverbs.d/efa.driver
	test ! -e /usr/include/infiniband/efadv.h
	test ! -e /usr/include/rdma/efa-abi.h
	! find /usr/lib /usr/local/lib \
	    \( -type f -o -type l \) \
	    \( -name "libefa.a" -o -name "libefa.so*" \
	       -o -name "libefa-rdmav*.so" -o -name "libefa.pc" \) \
	    -print -quit 2>/dev/null | grep -q .
	test -r /etc/libibverbs.d/mlx5.driver
	test -r /usr/lib/x86_64-linux-gnu/libibverbs.so.1
	find /usr/lib/x86_64-linux-gnu/libibverbs \( -type f -o -type l \) \
	    -name "libmlx5-rdmav*.so" -print -quit | grep -q .
	! find /usr/lib /usr/local/lib \( -type f -o -type l \) \
	    \( -name "libnccl-net-ofi.so*" -o -name "libnccl-tuner-ofi.so*" \) \
	    -print -quit 2>/dev/null | grep -q .
	test "$(git -c safe.directory=/sgl-workspace/sglang -C /sgl-workspace/sglang rev-parse HEAD)" = "'"${sglang_commit}"'"
for path in /root/Megatron-LM /root/miles /opt/tau3 /sgl-workspace/sglang; do
    test -r "${path}" && test -x "${path}"
    if find "${path}" \( -type d ! -executable -o -type f ! -readable \) -print -quit | grep -q .; then
        echo "non-root permission failure below ${path}" >&2
        exit 1
    fi
done
'

    # Then verify the production mounts, including the nested .env mask. The
    # writable tmpfs intentionally supplies a target when the checkout has no
    # .env; every maintained runtime launch uses the same overlay mode.
    singularity exec \
        --no-eval \
        --writable-tmpfs \
        --no-home \
        --bind "${smoke_mounts}" \
        "${image}" \
        /bin/bash --noprofile --norc -c '
	set -euo pipefail
	test -x /root
	test ! -e /opt/amazon
	for package in libfabric1-aws libnccl-ofi-ngc-v3; do
	    ! dpkg-query -W "${package}" >/dev/null 2>&1
	done
	test ! -e /etc/libibverbs.d/efa.driver
	test ! -e /usr/include/infiniband/efadv.h
	test ! -e /usr/include/rdma/efa-abi.h
	! find /usr/lib /usr/local/lib \
	    \( -type f -o -type l \) \
	    \( -name "libefa.a" -o -name "libefa.so*" \
	       -o -name "libefa-rdmav*.so" -o -name "libefa.pc" \) \
	    -print -quit 2>/dev/null | grep -q .
	test -r /etc/libibverbs.d/mlx5.driver
	test -r /usr/lib/x86_64-linux-gnu/libibverbs.so.1
	find /usr/lib/x86_64-linux-gnu/libibverbs \( -type f -o -type l \) \
	    -name "libmlx5-rdmav*.so" -print -quit | grep -q .
	! find /usr/lib /usr/local/lib \( -type f -o -type l \) \
	    \( -name "libnccl-net-ofi.so*" -o -name "libnccl-tuner-ofi.so*" \) \
	    -print -quit 2>/dev/null | grep -q .
	test -r /root/Megatron-LM && test -x /root/Megatron-LM
test -r /root/miles && test -x /root/miles
test -r /root/miles/.env && test ! -w /root/miles/.env
test -r /opt/tau3 && test -x /opt/tau3
test "${TAU2_DATA_DIR}" = /opt/tau3/data
test -r "${TAU2_DATA_DIR}/tau2/domains/retail/tasks.json"
test -x /usr/local/bin/bwrap
test "${HOME}" = /cache/home
for path in \
    /root/miles /root/.cache \
    /data /data/pre-train /data/sft \
    /ckpt/hf /ckpt/megatron /ckpt/training \
    /cache "${HOME}"; do
    test -w "${path}"
done
PYTHONPATH=/root/Megatron-LM:/root/miles python3 -c "import megatron, miles, sglang"
python3 - <<PY
import inspect
from dataclasses import fields
from pathlib import Path

import sglang
from sglang.srt.server_args import ServerArgs

source = Path(inspect.getfile(sglang)).resolve()
assert source.is_relative_to(Path("/sgl-workspace/sglang/python")), source
assert "enable_response_weight_version_segments" in {field.name for field in fields(ServerArgs)}
PY
python3 -m pytest -q -p no:cacheprovider \
    /sgl-workspace/sglang/test/registered/unit/managers/test_policy_weight_version.py
/opt/tau3/.venv/bin/python - <<PY
from tau2.runner.helpers import load_tasks

assert len(load_tasks("retail", "train")) == 74
PY
'
}

if [[ -e "${dated_image}" ]]; then
    [[ -f "${checksum_file}" && -f "${provenance_file}" ]] || {
        echo "existing image has incomplete provenance: ${dated_image}" >&2
        exit 1
    }
    grep -Fqx "oci_image=${oci_image}" "${provenance_file}" || {
        echo "existing image was built from a different OCI reference: ${dated_image}" >&2
        exit 1
    }
    grep -Fqx "sglang_repo=${sglang_repo}" "${provenance_file}" || {
        echo "existing image was built from a different SGLang repository: ${dated_image}" >&2
        exit 1
    }
    grep -Fqx "sglang_branch=${sglang_branch}" "${provenance_file}" || {
        echo "existing image was built from a different SGLang branch: ${dated_image}" >&2
        exit 1
    }
    grep -Fqx "sglang_commit=${sglang_commit}" "${provenance_file}" || {
        echo "existing image was built from a different SGLang commit: ${dated_image}" >&2
        exit 1
    }
    grep -Fqx "definition_sha256=${definition_sha256}" "${provenance_file}" || {
        echo "existing image was built from a different Singularity definition: ${dated_image}" >&2
        exit 1
    }
    (cd "$(dirname -- "${dated_image}")" && sha256sum --check "$(basename -- "${checksum_file}")")
    smoke_test "${dated_image}"
    [[ -z "${stable_image}" ]] || ln -sfn "${dated_image}" "${stable_image}"
    echo "reused verified image: ${dated_image}"
    exit 0
fi

echo "building docker://${oci_image} -> ${dated_image}"
time singularity build \
    --fakeroot \
    --fix-perms \
    --build-arg "DOCKER_IMAGE=${oci_image}" \
    --build-arg "SGLANG_REPO=${sglang_repo}" \
    --build-arg "SGLANG_BRANCH=${sglang_branch}" \
    --build-arg "SGLANG_COMMIT=${sglang_commit}" \
    "${local_image}" \
    "${local_definition}"
smoke_test "${local_image}"

image_sha256="$(sha256sum "${local_image}")"
image_sha256="${image_sha256%% *}"
printf '%s  %s\n' "${image_sha256}" "$(basename -- "${dated_image}")" > "${partial_checksum}"
cat > "${partial_provenance}" <<EOF
oci_image=${oci_image}
sglang_repo=${sglang_repo}
sglang_branch=${sglang_branch}
sglang_commit=${sglang_commit}
definition_file=${definition_file}
definition_sha256=${definition_sha256}
image_sha256=${image_sha256}
singularity_version=$(singularity --version)
builder_uid=$(id -u)
pbs_job_id=${PBS_JOBID:-manual}
created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

# The first move may cross filesystems, so target a hidden shared-filesystem
# staging name. Only its final rename makes the complete SIF visible.
mv -- "${local_image}" "${publishing_image}"
printf '%s  %s\n' "${image_sha256}" "${publishing_image}" | sha256sum --check --strict
mv -- "${partial_checksum}" "${checksum_file}"
mv -- "${partial_provenance}" "${provenance_file}"
mv -- "${publishing_image}" "${dated_image}"

if [[ -n "${stable_image}" ]]; then
    ln -sfn "${dated_image}" "${stable_image}"
fi

cd "${REPO_ROOT}"
cleanup
trap - EXIT
ls -lh "${dated_image}" "${checksum_file}" "${provenance_file}"
[[ -z "${stable_image}" ]] || ls -lh "${stable_image}"
echo "ready: CONTAINER_IMAGE=${dated_image}"
