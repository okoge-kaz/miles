#!/bin/bash
# Step 1 of notes/node-ratio-procedure.md: measure the natural staleness and the
# train/rollout balance across both node axes, in one pass.
#
#     experiments/node_ratio_sweep.sh            # print the grid and the cost
#     experiments/node_ratio_sweep.sh --submit
#     experiments/node_ratio_sweep.sh --check
#
# The cap is 64 rather than unset. Leaving it unset skips the whole block
# (`fully_async_rollout.py:306` guards on `is not None`), dropping the
# measurement along with the enforcement; 64 is far above anything observed, so
# it measures without ever binding. Enforcing a real cap here would be worse than
# useless -- a recycled group is regenerated from scratch, so the run never
# reaches the lag it would have reached.
#
# Both node axes are swept because the adoptable rollout counts are a function of
# the train count. The colocated arm runs at the same total GPU count with every
# GPU training, so
#
#     dp = 8 * (T + R) / (TP * CP * PP)   must divide GLOBAL_BATCH_SIZE
#
# At gbs 3072, TP 2 and a cap of MAX_TOTAL_NODES=8 that gives
#
#     T=1  ->  R in 1, 2, 3, 5, 7
#     T=2  ->  R in 1, 2, 4, 6
#     T=4  ->  R in 2, 4
#
# The grid is derived, not written down, so changing the batch shape or the node
# cap re-derives it instead of silently leaving a stale list.
#
# Every point runs once per rollout seed. Generation order is the largest single
# source of run-to-run spread here, and one sample of a noisy quantity cannot say
# whether two node counts differ; the seeds land in separate checkpoint paths
# because CONFIG_TAG carries rseed.
#
# The batch shape is the production one. Shrinking it would move KV pressure and
# per-step cost together, and their crossing is the measurement.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507"

: "${TRAIN_NODES:=1}"           # one train node; override to sweep the other axis too
: "${ROLLOUT_SEEDS:=42 43}"
: "${MAX_TOTAL_NODES:=8}"      # slurm schedules small jobs far sooner
: "${NUM_ROLLOUT:=12}"         # the buffer fills by ~step 7, so 12 leaves a steady tail
: "${STALENESS:=64}"

read_default() {  # read a recipe default so the grid cannot drift from the recipe
    sed -n "s/^: \"\${$1:=\([^}]*\)}\".*/\1/p" "${REPO_ROOT}/${RECIPE}" | head -1
}
GBS=$(read_default GLOBAL_BATCH_SIZE)
TP=$(read_default TENSOR_PARALLEL_SIZE)
CP=$(read_default CONTEXT_PARALLEL_SIZE)

# interactive caps at 2 nodes, batch_short at 4 nodes and 2 h. Anything larger
# waits behind the whole cluster on batch, so place each point at the smallest
# partition that fits it.
placement() {
    local n=$1
    if   (( n <= 2 )); then echo "interactive 03:00:00"
    elif (( n <= 4 )); then echo "batch_short 02:00:00"
    else                    echo "batch       03:00:00"
    fi
}

points() {  # T R nodes partition wall
    for t in ${TRAIN_NODES}; do
        for (( r = 1; r + t <= MAX_TOTAL_NODES; r++ )); do
            local n=$(( t + r )) dp
            dp=$(( 8 * n / (TP * CP) ))
            (( GBS % dp == 0 )) || continue
            echo "${t} ${r} ${n} $(placement "${n}")"
        done
    done
}

if [[ "${1:-}" == "--check" ]]; then
    shopt -s nullglob
    logs=("${LOG_DIR}"/noderatio-s${STALENESS}-t*.log)
    (( ${#logs[@]} )) || { echo "no noderatio-s${STALENESS}-t*.log under ${LOG_DIR}"; exit 1; }
    "${REPO_ROOT}/experiments/analyze_throughput.py" "${logs[@]}"
    echo
    for f in "${logs[@]}"; do
        echo "=== $(basename "${f}") ==="
        grep -o "'staleness/rollout/count_[a-z0-9_]*': [0-9.]*" "${f}" | tail -11
    done
    exit 0
fi

total_jobs=0 total_nodeh=0
while read -r t r n part wall; do
    for _ in ${ROLLOUT_SEEDS}; do
        total_jobs=$(( total_jobs + 1 ))
        total_nodeh=$(( total_nodeh + n * ${wall%%:*} ))
    done
done < <(points)

if [[ "${1:-}" != "--submit" ]]; then
    printf 'gbs %s, tp %s, cp %s, staleness %s (never binds), %s rollouts, seeds: %s\n\n' \
        "${GBS}" "${TP}" "${CP}" "${STALENESS}" "${NUM_ROLLOUT}" "${ROLLOUT_SEEDS}"
    printf '  %-3s %-3s %-6s %-12s %-9s %s\n' T R nodes partition wall "colocated dp"
    while read -r t r n part wall; do
        printf '  %-3s %-3s %-6s %-12s %-9s %d\n' "$t" "$r" "$n" "$part" "$wall" "$(( 8 * n / (TP * CP) ))"
    done < <(points)
    printf '\n%d jobs, at most %d node-hours, largest job %d nodes\n' \
        "${total_jobs}" "${total_nodeh}" "${MAX_TOTAL_NODES}"
    echo "re-run with --submit"
    exit 0
fi

while read -r t r n part wall; do
    for seed in ${ROLLOUT_SEEDS}; do
        name="noderatio-s${STALENESS}-t${t}r${r}-rs${seed}"
        jid=$(sbatch --parsable \
            -A "${SLURM_ACCOUNT_NAME}" \
            --partition="${part}" \
            --job-name="${name}" \
            --nodes="${n}" --time="${wall}" \
            --output="${LOG_DIR}/${name}-%j.log" \
            --export="ALL,RUN_NAME=${name},CONFIG_TAG=${name},NUM_ROLLOUT=${NUM_ROLLOUT},MAX_WEIGHT_STALENESS=${STALENESS},ROLLOUT_SEED=${seed},ACTOR_NUM_NODES=${t},ROLLOUT_NUM_GPUS=$(( r * 8 )),SAVE_INTERVAL=1000,SAVE_RETAIN_INTERVAL=1000,SAVE_HF=0,EVAL_INTERVAL=1000,SKIP_EVAL_BEFORE_TRAIN=1" \
            "${REPO_ROOT}/${RECIPE}")
        echo "${jid}  T=${t} R=${r}  ${n} node  ${part}  rseed ${seed}"
    done
done < <(points)

echo
echo "check with:  experiments/node_ratio_sweep.sh --check"
