#!/bin/bash
# Find the rollout node count to run the study at, by measuring where the
# trainer stops waiting on generation.
#
#     experiments/node_ratio_sweep.sh [--submit]
#     experiments/node_ratio_sweep.sh --check
#
# Train is fixed at one node; rollout varies. The adopted point is the SMALLEST
# rollout allocation whose `train_wait` is acceptable -- rollout nodes are the
# cost, and a little waiting is cheaper than a node that sits idle.
#
# Only 1, 3 and 7 rollout nodes are adoptable. The colocated arm has to run at
# the same total GPU count, and there every GPU trains, so dp = 8*(1+R)/TP.
# With TP=2 and a 2048 global batch that divides only at R = 1, 3, 7:
#
#     R=1  total 2  dp  8  ->  256/rank
#     R=3  total 4  dp 16  ->  128/rank
#     R=7  total 8  dp 32  ->   64/rank
#     R=2,4,5,6     dp 12/20/24/28 -> not integral
#
# R=2 and R=4 are submitted as appendix points only; they are reportable but not
# adoptable. Set APPENDIX=1 to include them.
#
# The batch shape is the production one on purpose. Shrinking it would cut the
# KV pressure and the per-step cost together, and the whole question is where
# those two cross.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507"

ROLLOUT_NODES=(1 3 7)
[[ "${APPENDIX:-0}" == "1" ]] && ROLLOUT_NODES=(1 2 3 4 7)

# Enough steps to get past the drain and average a few steady ones. A step is
# ~15-20 min at this batch shape, and analyze_throughput.py drops step 1 plus
# every leading step whose train_wait is under 2 s.
: "${WALL:=03:00:00}"
: "${NUM_ROLLOUT:=12}"

if [[ "${1:-}" == "--check" ]]; then
    shopt -s nullglob
    logs=("${LOG_DIR}"/noderatio-r*.log)
    (( ${#logs[@]} )) || { echo "no noderatio-*.log under ${LOG_DIR}"; exit 1; }
    "${REPO_ROOT}/experiments/analyze_throughput.py" "${logs[@]}"
    exit 0
fi

if [[ "${1:-}" != "--submit" ]]; then
    echo "would submit ${#ROLLOUT_NODES[@]} points, ${WALL} each, ${NUM_ROLLOUT} rollouts:"
    for r in "${ROLLOUT_NODES[@]}"; do
        printf '  rollout %d node -> %d node total, dp %d colocated, %s\n' \
            "$r" "$((1+r))" "$(( 8*(1+r)/2 ))" \
            "$(( 2048 % (8*(1+r)/2) == 0 ? 0 : 1 ))X adoptable" 2>/dev/null || true
    done
    echo "re-run with --submit"
    exit 0
fi

for r in "${ROLLOUT_NODES[@]}"; do
    n=$(( 1 + r ))
    name="noderatio-r${r}"
    jid=$(sbatch --parsable \
        -A "${SLURM_ACCOUNT_NAME}" \
        --partition=batch \
        --job-name="${name}" \
        --nodes="${n}" --time="${WALL}" \
        --output="${LOG_DIR}/${name}-%j.log" \
        --export="ALL,RUN_NAME=${name},CONFIG_TAG=${name},NUM_ROLLOUT=${NUM_ROLLOUT},ACTOR_NUM_NODES=1,ROLLOUT_NUM_GPUS=$(( r * 8 )),SAVE_INTERVAL=1000,SAVE_RETAIN_INTERVAL=1000,SAVE_HF=0,EVAL_INTERVAL=1000,SKIP_EVAL_BEFORE_TRAIN=1" \
        "${REPO_ROOT}/${RECIPE}")
    echo "${jid}  rollout ${r} node (${n} total)"
done

echo
echo "check with:  experiments/node_ratio_sweep.sh --check"
