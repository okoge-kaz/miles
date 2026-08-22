#!/bin/bash

#SBATCH --job-name=reason-eval-images
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/setup/%x-%j.log

set -euo pipefail

SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${SCRIPT_PATH}")/../../.." && pwd)"
fi
source "${REPO_ROOT}/experiments/env.sh"

CONTAINER_ROOT="${REASONING_EVAL_CONTAINER_ROOT:-${SHARED_WS}/containers}"
ENROOT_CACHE_ROOT="${CONTAINER_ROOT}/.enroot-cache"
IMAGE_NAMES=(
    "vllm-openai-v0.20.2-70a098d9.sqsh"
    "nemo-evaluator-nemo-skills-26.03-ac1b048e.sqsh"
)
IMAGE_URIS=(
    "docker://registry-1.docker.io#vllm/vllm-openai:v0.20.2"
    "docker://nvcr.io#nvidia/eval-factory/nemo-skills:26.03"
)
OCI_DIGESTS=(
    "sha256:70a098d90dbab428a001d9e852fc0fc8d67da5beb03e7851a22247653bf35923"
    "sha256:ac1b048e13fe7f2a59751b528fc23f5f471452197ad9ae40b715a77cda0a9612"
)
MANIFEST_URLS=(
    "https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/v0.20.2"
    "https://nvcr.io/v2/nvidia/eval-factory/nemo-skills/manifests/26.03"
)
TOKEN_URLS=(
    "https://auth.docker.io/token"
    "https://nvcr.io/proxy_auth"
)
TOKEN_SERVICES=(
    "registry.docker.io"
    ""
)
TOKEN_SCOPES=(
    "repository:vllm/vllm-openai:pull"
    "repository:nvidia/eval-factory/nemo-skills:pull"
)

resolve_tag_digest() {
    local index="$1"
    local token response_headers digest
    local -a token_arguments=(
        --get
        --data-urlencode "scope=${TOKEN_SCOPES[${index}]}"
    )
    if [[ -n "${TOKEN_SERVICES[${index}]}" ]]; then
        token_arguments+=(--data-urlencode "service=${TOKEN_SERVICES[${index}]}")
    fi
    token="$(
        curl --fail --silent --show-error --location \
            "${token_arguments[@]}" "${TOKEN_URLS[${index}]}" \
        | jq -r '.token // .access_token // empty'
    )"
    [[ -n "${token}" ]] || { echo "registry did not issue a pull token" >&2; return 1; }
    response_headers="$(
        curl --fail --silent --show-error --head --max-time 60 \
            --header "Authorization: Bearer ${token}" \
            --header 'Accept: application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json' \
            "${MANIFEST_URLS[${index}]}"
    )"
    digest="$(
        awk 'tolower($1) == "docker-content-digest:" { sub(/\r$/, "", $2); print $2; exit }' \
            <<< "${response_headers}"
    )"
    [[ -n "${digest}" ]] || { echo "registry response did not include a manifest digest" >&2; return 1; }
    printf '%s\n' "${digest}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cd "${REPO_ROOT}"
    mkdir -p experiments/outputs/reasoning_eval/setup
    sbatch --test-only --export=ALL "${SCRIPT_PATH}"
    exec sbatch --export=ALL "${SCRIPT_PATH}" "$@"
fi

ENROOT_TEMP_ROOT="${SLURM_TMPDIR:-/tmp}/reasoning-eval-enroot-${SLURM_JOB_ID}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_ROOT}"
export ENROOT_DATA_PATH="${ENROOT_TEMP_ROOT}/data"
export ENROOT_TEMP_PATH="${ENROOT_TEMP_ROOT}/tmp"
mkdir -p "${CONTAINER_ROOT}" "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_TEMP_PATH}"
exec 9> "${CONTAINER_ROOT}/.reasoning-eval-image-import.lock"
flock --nonblock 9 || { echo "another reasoning-evaluation image import is active" >&2; exit 5; }

for index in "${!IMAGE_NAMES[@]}"; do
    final_image="${CONTAINER_ROOT}/${IMAGE_NAMES[${index}]}"
    partial_image="${final_image}.partial-${SLURM_JOB_ID}"
    oci_digest="${OCI_DIGESTS[${index}]}"

    if [[ -f "${final_image}" ]]; then
        [[ -f "${final_image}.oci-digest" ]] || {
            echo "existing image has no OCI digest provenance: ${final_image}" >&2
            exit 2
        }
        [[ "$(< "${final_image}.oci-digest")" == "${oci_digest}" ]] || {
            echo "existing image OCI digest does not match: ${final_image}" >&2
            exit 3
        }
        [[ -f "${final_image}.sha256" ]] || {
            echo "existing image has no SquashFS checksum: ${final_image}" >&2
            exit 4
        }
        unsquashfs -s "${final_image}" >/dev/null
        sha256sum --check "${final_image}.sha256"
        echo "Validated existing image: ${final_image}"
        continue
    fi

    [[ ! -e "${partial_image}" ]] || {
        echo "partial image already exists: ${partial_image}" >&2
        exit 6
    }
    resolved_digest="$(resolve_tag_digest "${index}")"
    [[ "${resolved_digest}" == "${oci_digest}" ]] || {
        echo "refusing mutable tag with unexpected digest: ${IMAGE_URIS[${index}]}" >&2
        echo "  expected=${oci_digest}" >&2
        echo "  resolved=${resolved_digest}" >&2
        exit 7
    }
    echo "Importing ${IMAGE_URIS[${index}]}"
    enroot import --arch x86_64 --output "${partial_image}" "${IMAGE_URIS[${index}]}"
    unsquashfs -s "${partial_image}" >/dev/null
    mv -- "${partial_image}" "${final_image}"
    sha256sum "${final_image}" > "${final_image}.sha256"
    printf '%s\n' "${oci_digest}" > "${final_image}.oci-digest"
    echo "Completed image import: ${final_image}"
done

echo "Pinned evaluation images are ready under ${CONTAINER_ROOT}"
