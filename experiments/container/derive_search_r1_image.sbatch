#!/bin/bash
#PBS -N miles-check-search-r1-image
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -j oe

# Compatibility validation for the former Enroot Faiss derivation. The pinned
# Faiss wheel is installed by docker/Dockerfile before the SIF is pulled.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

[[ -r "${CONTAINER_IMAGE}" ]] || {
    echo "missing CONTAINER_IMAGE=${CONTAINER_IMAGE}; run experiments/setup.sh container --submit" >&2
    exit 1
}

singularity exec --no-eval --no-home "${CONTAINER_IMAGE}" python3 -c '
import faiss
import fastapi
import numpy
import uvicorn

assert faiss.__version__ == "1.12.0", faiss.__version__
index = faiss.IndexFlatIP(4)
vectors = numpy.eye(4, dtype="float32")
index.add(vectors)
assert index.search(vectors[:1], 1)[1].tolist() == [[0]]
print(f"faiss={faiss.__version__} fastapi={fastapi.__version__} uvicorn={uvicorn.__version__}")
'
echo "Search-R1 dependencies are present in ${CONTAINER_IMAGE}"
