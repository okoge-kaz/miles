#!/bin/bash
#PBS -N prep-ifbench
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Pin the official held-out IFBench checkout, stage its Python dependency
# closure, and convert all 300 prompts to an eval-only Miles JSONL file.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

readonly IFBENCH_COMMIT=db69a6f05689830b0068b8f1529ebcfd2f3b164c
readonly IFBENCH_DIR="${DATASET_DIR}/ifbench"
readonly IFBENCH_DEPS_DIR="${DATASET_DIR}/ifbench-deps"

[[ -d "${IFBENCH_DIR}/.git" ]] || {
    echo "missing ${IFBENCH_DIR}; run download_git_dependency.sbatch first" >&2
    exit 1
}
[[ "$(git -C "${IFBENCH_DIR}" rev-parse HEAD)" == "${IFBENCH_COMMIT}" ]] || {
    echo "IFBench checkout is not pinned to ${IFBENCH_COMMIT}" >&2
    exit 1
}

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail

readonly target=/data/ifbench-deps
readonly marker="${target}/.miles-ifbench-0.2.0-db69a6f"
if [[ ! -f "${marker}" ]]; then
    [[ ! -e "${target}" ]] || {
        echo "refusing to replace unmarked IFBench dependency directory: ${target}" >&2
        exit 1
    }
    readonly temp_target="/data/.ifbench-deps-${MILES_JOB_ID}"
    cleanup() {
        case "${temp_target}" in
            /data/.ifbench-deps-*) rm -rf -- "${temp_target}" ;;
            *) return 1 ;;
        esac
    }
    trap cleanup EXIT
    mkdir -p "${temp_target}" "${temp_target}/nltk_data"
    python3 -m pip install --no-input --target "${temp_target}" \
        absl-py==2.3.1 \
        emoji==2.14.1 \
        immutabledict==1.2.0 \
        langdetect==1.0.9 \
        nltk==3.9.2 \
        syllapy==0.7.2
    PYTHONPATH="${temp_target}" python3 -m nltk.downloader \
        -d "${temp_target}/nltk_data" punkt punkt_tab
    touch "${temp_target}/.miles-ifbench-0.2.0-db69a6f"
    mv "${temp_target}" "${target}"
    trap - EXIT
fi

cd /root/miles
export PYTHONPATH="/data/ifbench:/data/ifbench-deps:/root/miles"
export NLTK_DATA=/data/ifbench-deps/nltk_data
python3 -m experiments.src.datasets.nemotron.convert \
    --dataset ifbench \
    --input /data/ifbench/data/IFBench_test.jsonl \
    --output /data/ifbench/IFBench_test_miles.jsonl \
    --summary /data/ifbench/IFBench_test_miles.summary.json

python3 - <<"PY"
import json

from miles.rollout.rm_hub.ifbench import compute_ifbench_reward

with open("/data/ifbench/IFBench_test_miles.jsonl", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
assert len(rows) == 300, len(rows)
assert all(
    row["metadata"].get("eval_only") is True
    and row["metadata"].get("verifier") == "ifbench"
    for row in rows
)
metadata = {
    "instruction_id_list": ["keywords:existence"],
    "prompt_text": "Include cat and dog.",
    "kwargs": [{"keywords": ["cat", "dog"]}],
    "record_id": 0,
}
assert compute_ifbench_reward("A cat and a dog.", None, metadata) == 1.0
assert compute_ifbench_reward("A cat.", None, metadata) == 0.0
print("official IFBench verifier: correct=1 wrong=0; held-out rows=300")
PY
'

printf 'IFBench commit=%s deps=%s output=%s\n' \
    "${IFBENCH_COMMIT}" "${IFBENCH_DEPS_DIR}" "${IFBENCH_DIR}/IFBench_test_miles.jsonl"
