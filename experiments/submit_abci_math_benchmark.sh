#!/bin/bash
# Preview by default. Use the same BENCHMARK_ID for smoke and resume; give each
# fresh comparison a new ID. MILES_REPO can select a frozen source worktree.
set -euo pipefail
repo="${MILES_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${repo}"
source experiments/env.sh
stage="${1:-throughput}"
split="${2:-2:6}"
mode="${3:---test-only}"
[[ $# -le 3 && ( "${mode}" == --submit || "${mode}" == --test-only ) ]] || exit 2
case "${split}" in 2:6|3:5) ;; *) echo 'split must be 2:6 or 3:5' >&2; exit 2 ;; esac
case "${stage}" in
    convert) nodes=1 ;;
    smoke|resume) nodes=2 ;;
    fit|throughput) nodes=8 ;;
    *) echo 'stage must be convert, smoke, resume, fit, or throughput' >&2; exit 2 ;;
esac
if [[ "${stage}" == resume ]]; then
    : "${BENCHMARK_ID:?set the BENCHMARK_ID used for smoke}"
fi
export BENCHMARK_ID="${BENCHMARK_ID:-main-$(date -u +%Y%m%dT%H%M%SZ)}"
export BENCHMARK_STAGE="${stage}" BENCHMARK_ACTOR_NODES="${split%:*}"
export MILES_REPO="${repo}"
mkdir -p experiments/outputs/abci-throughput
options=(--test-only)
if [[ "${mode}" == --submit ]]; then
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
        echo 'commit the integration and use a frozen clean checkout before submitting' >&2; exit 1;
    }
    image="$(readlink -f -- "${CONTAINER_IMAGE}")"
    grep -Fqx "sglang_commit=${SGLANG_COMMIT}" "${image}.provenance.env" || {
        echo 'build/select the new managed SIF before submitting' >&2; exit 1;
    }
    options=(--parsable)
fi
# The compute-side env.sh reads the owner's ~/.netrc. Export only the selected
# assets and run identity, so unrelated shell overrides cannot alter a split.
exports="BENCHMARK_ID,BENCHMARK_STAGE,BENCHMARK_ACTOR_NODES,MILES_REPO,MILES_WORKSPACE_ROOT"
exports+=",CONTAINER_IMAGE,HF_CKPT_DIR,MEGATRON_CKPT_DIR,RL_DATASET_DIR,CACHE_DIR"
result="$(pbs_submit "${options[@]}" --profile gpu --nodes="${nodes}" --time=04:00:00 \
    --job-name="math-${stage}-t${BENCHMARK_ACTOR_NODES}" \
    --output="${repo}/experiments/outputs/abci-throughput/" --export="${exports}" \
    experiments/benchmark_abci_math.sbatch)"
printf 'BENCHMARK_ID=%s stage=%s split=%s\n%s\n' "${BENCHMARK_ID}" "${stage}" "${split}" "${result}"
if [[ "${mode}" == --submit ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${BENCHMARK_ID}" "${stage}" "${split}" "${result}" \
        "$(git rev-parse HEAD)" "${CONTAINER_IMAGE}" >> experiments/outputs/abci-throughput/jobs.tsv
fi
