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
case "${split}" in 1:7|2:6|3:5) ;; *) echo 'split must be 1:7, 2:6, or 3:5' >&2; exit 2 ;; esac
walltime=04:00:00
case "${stage}" in
    convert) nodes=1 ;;
    smoke|resume) nodes=2 ;;
    fit|throughput) nodes=8 ;;
    production)
        nodes=8; walltime=24:00:00
        [[ "${split}" != 1:7 ]] || walltime=36:00:00
        ;;
    *) echo 'stage must be convert, smoke, resume, fit, throughput, or production' >&2; exit 2 ;;
esac
walltime="${BENCHMARK_WALLTIME:-${walltime}}"
if [[ "${stage}" == resume ]]; then
    : "${BENCHMARK_ID:?set the BENCHMARK_ID used for smoke}"
fi
export BENCHMARK_ID="${BENCHMARK_ID:-main-$(date -u +%Y%m%dT%H%M%SZ)}"
export BENCHMARK_STAGE="${stage}" BENCHMARK_ACTOR_NODES="${split%:*}"
export MILES_REPO="${repo}"
export DUMP_ROLLOUT_DATA="${DUMP_ROLLOUT_DATA:-1}" LOG_ROLLOUT_REPETITION="${LOG_ROLLOUT_REPETITION:-1}"
export LOG_TOKEN_LAG_METRICS="${LOG_TOKEN_LAG_METRICS:-1}" LOG_ROLLOUT_TIMING="${LOG_ROLLOUT_TIMING:-0}"
export SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-cache_aware}"
export ASYNC_MAX_CONCURRENT_SAMPLES="${ASYNC_MAX_CONCURRENT_SAMPLES:-3072}"
export MAX_WEIGHT_STALENESS="${MAX_WEIGHT_STALENESS:-16}"
export ZERO_REWARD_ON_TRUNCATED="${ZERO_REWARD_ON_TRUNCATED:-1}"
export ZERO_LOSS_ON_TRUNCATED="${ZERO_LOSS_ON_TRUNCATED:-0}"
for truncation_flag in ZERO_REWARD_ON_TRUNCATED ZERO_LOSS_ON_TRUNCATED; do
    [[ "${!truncation_flag}" =~ ^[01]$ ]] || { echo "${truncation_flag} must be 0 or 1" >&2; exit 2; }
done
[[ "${ZERO_REWARD_ON_TRUNCATED}" != 1 || "${ZERO_LOSS_ON_TRUNCATED}" != 1 ]] || {
    echo 'ZERO_REWARD_ON_TRUNCATED and ZERO_LOSS_ON_TRUNCATED are mutually exclusive' >&2; exit 2;
}
[[ "${MAX_WEIGHT_STALENESS}" =~ ^[1-9][0-9]*$ ]] || {
    echo 'MAX_WEIGHT_STALENESS must be a positive integer' >&2; exit 2;
}
case "${SGLANG_ROUTER_POLICY}" in
    cache_aware|round_robin) ;;
    *) echo 'benchmark SGLANG_ROUTER_POLICY must be cache_aware or round_robin' >&2; exit 2 ;;
esac
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
exports+=",DUMP_ROLLOUT_DATA,LOG_ROLLOUT_REPETITION,LOG_TOKEN_LAG_METRICS,LOG_ROLLOUT_TIMING"
exports+=",SGLANG_ROUTER_POLICY,ASYNC_MAX_CONCURRENT_SAMPLES"
exports+=",MAX_WEIGHT_STALENESS"
exports+=",ZERO_REWARD_ON_TRUNCATED,ZERO_LOSS_ON_TRUNCATED"
result="$(pbs_submit "${options[@]}" --profile gpu --nodes="${nodes}" --time="${walltime}" \
    --job-name="math-${stage}-t${BENCHMARK_ACTOR_NODES}" \
    --output="${repo}/experiments/outputs/abci-throughput/" --export="${exports}" \
    experiments/benchmark_abci_math.sbatch)"
printf 'BENCHMARK_ID=%s stage=%s split=%s\n%s\n' "${BENCHMARK_ID}" "${stage}" "${split}" "${result}"
if [[ "${mode}" == --submit ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${BENCHMARK_ID}" "${stage}" "${split}" "${result}" \
        "$(git rev-parse HEAD)" "${CONTAINER_IMAGE}" >> experiments/outputs/abci-throughput/jobs.tsv
fi
