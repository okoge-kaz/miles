#!/bin/bash
#PBS -N miles-v011-import-image
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

set -euo pipefail
REPO_ROOT="${MILES_REPO:-${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "${REPO_ROOT}/experiments/env.sh"
command -v "${SINGULARITY_BIN}" >/dev/null

job_id="${MILES_JOB_ID_SHORT}"
dated_image="${IMPORT_OUTPUT_IMAGE:-${CONTAINER_DIR}/miles-v0.1.1-$(date +%Y%m%d-%H%M%S)-${job_id}.sif}"
stable_image="${SINGULARITY_LINK-${CONTAINER_IMAGE}}"
definition_file="${MILES_REPO}/experiments/container/miles.def"
sglang_patch_file="${MILES_REPO}/experiments/container/sglang-v0.1.1-provenance.patch"
: "${APPLY_SGLANG_PROVENANCE_PATCH:=0}"
: "${INSTALL_CODE_SANDBOX:=0}"
: "${INSTALL_NEMO_GYM:=0}"
[[ "${INSTALL_NEMO_GYM}" == 0 || "${INSTALL_NEMO_GYM}" == 1 ]] || {
    echo "INSTALL_NEMO_GYM must be 0 or 1" >&2; exit 2
}
[[ "${INSTALL_CODE_SANDBOX}" == 0 || "${INSTALL_CODE_SANDBOX}" == 1 ]] || {
    echo "INSTALL_CODE_SANDBOX must be 0 or 1" >&2; exit 2
}
[[ "${APPLY_SGLANG_PROVENANCE_PATCH}" == 0 || "${APPLY_SGLANG_PROVENANCE_PATCH}" == 1 ]] || {
    echo "APPLY_SGLANG_PROVENANCE_PATCH must be 0 or 1" >&2; exit 2
}
[[ -r "${sglang_patch_file}" ]] || { echo "missing SGLang provenance patch: ${sglang_patch_file}" >&2; exit 2; }
sglang_patch_sha256="$(sha256sum "${sglang_patch_file}")"
sglang_patch_sha256="${sglang_patch_sha256%% *}"
[[ -r "${MILES_REPO}/experiments/common/dotenv.disabled" ]] || {
    echo "missing container dotenv mask: experiments/common/dotenv.disabled" >&2; exit 2
}
oci_image="${DOCKER_IMAGE#docker://}"
[[ "${dated_image}" == /* && "${dated_image}" != *[[:space:]]* ]] || {
    echo "IMPORT_OUTPUT_IMAGE must be absolute and contain no whitespace" >&2; exit 2
}
[[ -n "${oci_image}" && "${oci_image}" != *[[:space:]]* ]] || {
    echo "DOCKER_IMAGE must be a non-empty OCI reference without whitespace" >&2; exit 2
}
if [[ -n "${stable_image}" ]]; then
    [[ "${stable_image}" == /* && "${stable_image}" != "${dated_image}" ]] || {
        echo "SINGULARITY_LINK must be absolute and different from IMPORT_OUTPUT_IMAGE" >&2; exit 2
    }
    if [[ -e "${stable_image}" && ! -L "${stable_image}" ]]; then
        echo "refusing to replace non-symlink: ${stable_image}" >&2; exit 2
    fi
fi
[[ ! -e "${dated_image}" ]] || { echo "output already exists: ${dated_image}" >&2; exit 2; }
mkdir -p "$(dirname -- "${dated_image}")"
[[ -z "${stable_image}" ]] || mkdir -p "$(dirname -- "${stable_image}")"
exec 9> "${CONTAINER_DIR}/.miles-v0.1.1-image-build.lock"
flock --nonblock 9 || { echo "another v0.1.1 image build is active" >&2; exit 5; }

scratch="${PBS_LOCALDIR:-${TMPDIR:-/tmp}}"
for candidate in "/local/${PBS_JOBID:-missing}" "/local/${job_id}"; do
    if [[ -d "${candidate}" && -w "${candidate}" ]]; then
        scratch="${candidate}"
        break
    fi
done
[[ -d "${scratch}" && -w "${scratch}" ]] || { echo "scratch is not writable: ${scratch}" >&2; exit 2; }
build_dir="$(mktemp -d "${scratch%/}/miles-v011-${job_id}.XXXXXX")"
local_image="${build_dir}/miles.sif"
publishing_image="${dated_image}.publishing-${job_id}"
partial_checksum="${dated_image}.sha256.partial-${job_id}"
partial_provenance="${dated_image}.provenance.env.partial-${job_id}"
partial_link="${stable_image}.partial-${job_id}"
cleanup() {
    rm -rf -- "${build_dir}"
    rm -f -- "${publishing_image}" "${partial_checksum}" "${partial_provenance}"
    [[ -z "${stable_image}" ]] || rm -f -- "${partial_link}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
install -m 0644 -- "${definition_file}" "${build_dir}/miles.def"
install -m 0644 -- "${sglang_patch_file}" "${build_dir}/sglang-provenance.patch"
for asset in prepare_code_sandbox.py verify_code_sandbox.py code_sandbox.requirements.json; do
    install -m 0644 -- "${MILES_REPO}/experiments/container/${asset}" "${build_dir}/${asset}"
done
mkdir -p "${build_dir}/nemo_gym/runtime" "${build_dir}/nemo_gym/patches"
if [[ "${INSTALL_NEMO_GYM}" == 1 ]]; then
    cp -- "${MILES_REPO}/experiments/src/nemo_gym/runtime/"{setup.py,sources.lock.json,requirements.lock.txt} \
        "${build_dir}/nemo_gym/runtime/"
    cp -- "${MILES_REPO}/experiments/src/nemo_gym/patches/"*.patch "${build_dir}/nemo_gym/patches/"
fi
code_sandbox_requirements_sha256="$(sha256sum "${build_dir}/code_sandbox.requirements.json")"
code_sandbox_requirements_sha256="${code_sandbox_requirements_sha256%% *}"
definition_sha256="$(sha256sum "${build_dir}/miles.def")"
definition_sha256="${definition_sha256%% *}"
export SINGULARITY_TMPDIR="${build_dir}" APPTAINER_TMPDIR="${build_dir}"
export SINGULARITY_CACHEDIR="${build_dir}" APPTAINER_CACHEDIR="${build_dir}"
cd "${build_dir}"
echo "building docker://${oci_image} in ${build_dir}"
"${SINGULARITY_BIN}" build --fakeroot --fix-perms \
    --build-arg "DOCKER_IMAGE=${oci_image}" \
    --build-arg "SGLANG_COMMIT=${SGLANG_COMMIT}" \
    --build-arg "MEGATRON_COMMIT=${MEGATRON_COMMIT}" \
    --build-arg "SGLANG_PATCH_FILE=${build_dir}/sglang-provenance.patch" \
    --build-arg "SGLANG_PATCH_SHA256=${sglang_patch_sha256}" \
    --build-arg "APPLY_SGLANG_PROVENANCE_PATCH=${APPLY_SGLANG_PROVENANCE_PATCH}" \
    --build-arg "INSTALL_CODE_SANDBOX=${INSTALL_CODE_SANDBOX}" \
    --build-arg "CODE_SANDBOX_BUILD_DIR=${build_dir}" \
    --build-arg "INSTALL_NEMO_GYM=${INSTALL_NEMO_GYM}" \
    --build-arg "NEMO_GYM_BUILD_DIR=${build_dir}/nemo_gym" \
    "${local_image}" "${build_dir}/miles.def"

# Test the actual mounts as the ordinary submitting UID before publication.
miles_container_exec --image "${local_image}" --bind "${CONTAINER_MOUNTS}" \
    --cwd /root/miles --no-nv --cleanenv \
    --export "SGLANG_COMMIT,MEGATRON_COMMIT" \
    bash --noprofile --norc -c '
set -euo pipefail
test -x /root
test ! -e /opt/amazon
test ! -e /etc/libibverbs.d/efa.driver
test -r /etc/libibverbs.d/mlx5.driver
test -r /root/miles/train_async.py
test -r /root/miles/.env && test ! -w /root/miles/.env
test "${HOME}" = /cache/home
for path in /data /ckpt/hf /ckpt/megatron /ckpt/training /cache "${HOME}"; do test -w "${path}"; done
test "$(git -c safe.directory=/sgl-workspace/sglang -C /sgl-workspace/sglang rev-parse HEAD)" = "${SGLANG_COMMIT}"
test "$(git -c safe.directory=/root/Megatron-LM -C /root/Megatron-LM rev-parse HEAD)" = "${MEGATRON_COMMIT}"
PYTHONPATH=/root/miles:/root/Megatron-LM python3 -c "import megatron, miles, sglang, ray"
'
if [[ "${INSTALL_CODE_SANDBOX}" == 1 ]]; then
    miles_container_exec --image "${local_image}" --bind "${CONTAINER_MOUNTS}" \
        --cwd /root/miles --no-nv --cleanenv \
        env PYTHONPATH=/root/miles:/root/Megatron-LM python3 /opt/miles/verify_code_sandbox.py --sandbox
fi
if [[ "${INSTALL_NEMO_GYM}" == 1 ]]; then
    miles_container_exec --image "${local_image}" --bind "${CONTAINER_MOUNTS}" \
        --cwd /root/miles --no-nv --cleanenv \
        env PYTHONPATH=/root/miles:/opt/nemo-gym OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /opt/nemo-gym/.venv/bin/python -m experiments.src.nemo_gym.runtime.verify \
        --state-dir "/root/miles/experiments/outputs/validation/nemo-gym-image-${job_id}"
fi
image_sha256="$(sha256sum "${local_image}")"
image_sha256="${image_sha256%% *}"
printf '%s  %s\n' "${image_sha256}" "$(basename -- "${dated_image}")" > "${partial_checksum}"
cat > "${partial_provenance}" <<EOF
oci_image=${oci_image}
sglang_commit=${SGLANG_COMMIT}
megatron_commit=${MEGATRON_COMMIT}
sglang_provenance_patch_enabled=${APPLY_SGLANG_PROVENANCE_PATCH}
sglang_provenance_patch_sha256=${sglang_patch_sha256}
code_sandbox_enabled=${INSTALL_CODE_SANDBOX}
nemo_gym_enabled=${INSTALL_NEMO_GYM}
code_sandbox_requirements_sha256=${code_sandbox_requirements_sha256}
definition_sha256=${definition_sha256}
image_sha256=${image_sha256}
singularity_version=$("${SINGULARITY_BIN}" --version)
pbs_job_id=${MILES_JOB_ID}
created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
mv -- "${local_image}" "${publishing_image}"
printf '%s  %s\n' "${image_sha256}" "${publishing_image}" | sha256sum --check --strict
mv -- "${partial_checksum}" "${dated_image}.sha256"
mv -- "${partial_provenance}" "${dated_image}.provenance.env"
mv -- "${publishing_image}" "${dated_image}"
if [[ -n "${stable_image}" ]]; then
    ln -s -- "${dated_image}" "${partial_link}"
    mv -Tf -- "${partial_link}" "${stable_image}"
fi
echo "ready: CONTAINER_IMAGE=${dated_image}"
