#!/bin/bash
#PBS -N test-tau-nvidia-api
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -j oe

# One real, credential-bearing provider probe. The job reads only the named
# values from .env and never writes the key or response text to stdout.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
export WANDB_MODE=offline
source "${REPO_ROOT}/experiments/env.sh"
source "${REPO_ROOT}/experiments/common/load_job_env.sh"
load_job_env "${REPO_ROOT}/.env" NVIDIA_INFERENCE_API_KEY TAU_USER_MODEL
: "${NVIDIA_INFERENCE_API_KEY:?set NVIDIA_INFERENCE_API_KEY in the repository .env or job environment}"
: "${TAU_USER_MODEL:?set TAU_USER_MODEL in the repository .env or job environment}"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -r "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is not readable: ${DOTENV_MASK}" >&2
    exit 1
}

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro" \
    --writable-tmpfs \
    --no-home \
    --export=USER,WANDB_MODE,NVIDIA_INFERENCE_API_KEY,TAU_USER_MODEL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

python3 - <<"PY"
import asyncio
import os

import aiohttp

from experiments.src.environments.tau_bench.user_simulator import (
    DEFAULT_NVIDIA_MODEL,
    NvidiaRequestError,
    generate_nvidia_user,
)


async def main() -> None:
    requested_model = os.environ["TAU_USER_MODEL"]
    assert requested_model == DEFAULT_NVIDIA_MODEL, (
        f"Tau NVIDIA probe requires {DEFAULT_NVIDIA_MODEL}, got {requested_model}"
    )
    response_models = []
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def post_json(url, payload, headers):
            async with session.post(url, json=payload, headers=headers) as response:
                if not 200 <= response.status < 300:
                    raise NvidiaRequestError(
                        "NVIDIA API returned an HTTP error",
                        status_code=response.status,
                    )
                body = await response.json(content_type=None)
                if not isinstance(body, dict):
                    raise NvidiaRequestError("NVIDIA API returned a non-object response")
                response_models.append(str(body.get("model") or ""))
                return body

        result = await generate_nvidia_user(
            [
                {"role": "system", "content": "Reply as a concise simulated customer."},
                {"role": "user", "content": "Hello, how can I help?"},
            ],
            post_json=post_json,
            model=requested_model,
            max_output_tokens=32,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            request_timeout=120,
            max_retries=1,
        )
    assert result.text.strip()
    assert response_models == [requested_model], (
        f"NVIDIA response model mismatch: requested={requested_model} "
        f"returned={response_models}"
    )
    print(
        f"Tau NVIDIA user API probe: model={requested_model} "
        f"ok output_tokens={result.output_tokens} finish_reason={result.finish_reason}"
    )


asyncio.run(main())
PY
'
