#!/bin/bash
#PBS -N reasoning-eval-images
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -j oe

# Pull evaluator images by OCI digest into immutable SIF files. Digest-qualified
# sources remove the separate mutable-tag registry probe used by the old runtime.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

container_root="${REASONING_EVAL_CONTAINER_ROOT:-${CONTAINER_DIR}/reasoning-eval}"
image_names=(
    "vllm-openai-v0.20.2-70a098d9.sif"
    "nemo-evaluator-nemo-skills-26.03-ac1b048e.sif"
)
image_digests=(
    "sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923"
    "sha256:ac1b048e13fe7f2a59751b528fc23f5f471452197ad9ae40b715a77cda0a9612"
)
image_sources=(
    "docker://vllm/vllm-openai@sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923"
    "docker://nvcr.io/nvidia/eval-factory/nemo-skills@sha256:ac1b048e13fe7f2a59751b528fc23f5f471452197ad9ae40b715a77cda0a9612"
)

mkdir -p "${container_root}" "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
exec 9> "${container_root}/.image-import.lock"
flock --nonblock 9 || {
    echo "another reasoning-evaluation image import is active" >&2
    exit 5
}

job_id="${PBS_JOBID:-manual-$$}"
job_id="${job_id%%.*}"
for index in "${!image_names[@]}"; do
    final_image="${container_root}/${image_names[${index}]}"
    partial_image="${final_image%.sif}.partial-${job_id}.sif"
    checksum_file="${final_image}.sha256"
    digest_file="${final_image}.oci-digest"

    if [[ -f "${final_image}" ]]; then
        [[ -f "${checksum_file}" && -f "${digest_file}" ]] || {
            echo "existing image has incomplete provenance: ${final_image}" >&2
            exit 2
        }
        [[ "$(< "${digest_file}")" == "${image_digests[${index}]}" ]] || {
            echo "existing image has the wrong OCI digest: ${final_image}" >&2
            exit 3
        }
        (cd "${container_root}" && sha256sum --check "$(basename -- "${checksum_file}")")
        singularity inspect "${final_image}" >/dev/null
        continue
    fi

    cleanup_partial() {
        rm -f -- "${partial_image}"
    }
    trap cleanup_partial EXIT
    echo "pulling ${image_sources[${index}]}"
    singularity pull "${partial_image}" "${image_sources[${index}]}"
    singularity inspect "${partial_image}" >/dev/null
    mv -- "${partial_image}" "${final_image}"
    (cd "${container_root}" && sha256sum "$(basename -- "${final_image}")" > "$(basename -- "${checksum_file}")")
    printf '%s\n' "${image_digests[${index}]}" > "${digest_file}"
    trap - EXIT
done

echo "Pinned evaluation images are ready under ${container_root}"
