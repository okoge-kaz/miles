#!/bin/bash
# What the train:rollout node ratio does to the *realized* staleness, decomposed
# into the interval spent generating and the interval spent queued.
#
#     experiments/realized_staleness_sweep.sh                      # print the plan and the cost
#     experiments/realized_staleness_sweep.sh --submit
#     experiments/realized_staleness_sweep.sh --mode convergence
#     experiments/realized_staleness_sweep.sh --mode convergence --submit
#     experiments/realized_staleness_sweep.sh --check
#
# Two modes, because the question has a short and a long form:
#
#   verify       one 4 h job per point. The queue reaches its depth by ~step 7,
#                so a dozen rollouts is enough to read the steady state at a
#                fixed response length.
#   convergence  chained 4 h jobs to NUM_ROLLOUT, plus one on-policy colocated
#                arm as the reference. Training lengthens responses, which
#                lengthens generation, which moves `staleness/pre_queue` -- so the
#                realized staleness of a converged run is not the one `verify`
#                measures. Only this mode can say what it settles at.
#
# The cap is parked at STALENESS=64 in both: enforcing a real one would recycle,
# and a recycled group is regenerated from scratch, so the run never reaches the
# lag it would otherwise have reached. 64 is far above anything observed. It is
# not left unset, because `fully_async_rollout.py:407` gates
# `staleness/bound/{rollout,train}` on the flag being present.
#
# `staleness/{total,pre_queue,in_queue}` are not gated and are the readout here.
# See notes/telemetry.md for the decomposition and notes/node-ratio-procedure.md
# for what the balance means.
#
# Both node axes are swept because the adoptable rollout counts are a function of
# the train count. The colocated arm runs at the same total GPU count with every
# GPU training, so
#
#     dp = 8 * (T + R) / (TP * CP * PP)   must divide GLOBAL_BATCH_SIZE
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

ASYNC_RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b/run.sbatch
COLO_RECIPE=experiments/math_sync/dapo-math-p10-90/qwen3-4b/run.sbatch
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b"

: "${TRAIN_NODES:=1}"           # one train node; override to sweep the other axis too
: "${MAX_TOTAL_NODES:=8}"
: "${STALENESS:=64}"            # parked, never binds
: "${STALENESS_REFERENCE:=completion}"   # the bound never binds here, so this
                                        # only picks which quantity staleness/bound/* mirrors
: "${WALL:=04:00:00}"

# verify: the queue fills by ~step 7, so 12 leaves a steady tail.
# convergence: the full run, chained. N_JOBS is sized off the measured 335 s/step;
# surplus is nearly free because a job whose run already reached NUM_ROLLOUT
# resumes, finds nothing to do and exits in minutes.
: "${VERIFY_ROLLOUT:=12}"
: "${CONVERGENCE_ROLLOUT:=300}"
: "${N_JOBS:=10}"

# https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/2441648885
# Must go on the sbatch line, not in a #SBATCH directive: the directive form
# strips the double quotes and the reaper then sees invalid JSON.
IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async RL: the training node is idle while the rollout engines generate its next batch; one 192x16 batch at 32k response length exceeds the default threshold"}}'

read_default() {  # read a recipe default so the grid cannot drift from the recipe
    sed -n "s/^: \"\${$1:=\([^}]*\)}\".*/\1/p" "${REPO_ROOT}/${ASYNC_RECIPE}" | head -1
}
GBS=$(read_default GLOBAL_BATCH_SIZE)
TP=$(read_default TENSOR_PARALLEL_SIZE)
CP=$(read_default CONTEXT_PARALLEL_SIZE)
GPN=$(read_default ACTOR_GPUS_PER_NODE)

MODE=verify; SUBMIT=0; CHECK=0
while (( $# )); do
    case "$1" in
        --mode)   MODE="$2"; shift 2 ;;
        --submit) SUBMIT=1; shift ;;
        --check)  CHECK=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ "${MODE}" == verify || "${MODE}" == convergence ]] ||
    { echo "--mode must be verify or convergence, got '${MODE}'" >&2; exit 1; }

if [[ "${MODE}" == verify ]]; then
    NUM_ROLLOUT="${VERIFY_ROLLOUT}"
    # Two seeds: over twelve rollouts the generation order is the largest source
    # of spread, and one sample cannot say whether two ratios differ.
    : "${ROLLOUT_SEEDS:=42 43}"
    # Nothing is read back from these runs, so writing checkpoints is pure cost.
    SAVE_ENV="SAVE_INTERVAL=1000,SAVE_RETAIN_INTERVAL=1000,SAVE_HF=0"
else
    # One seed per ratio. notes/off-policy-variables.md puts the seed replicates
    # on the colocated arm, so paying for them per ratio here would double a
    # four-figure node-hour bill to measure the variance twice.
    : "${ROLLOUT_SEEDS:=42}"
    NUM_ROLLOUT="${CONVERGENCE_ROLLOUT}"
    # A chain resumes from --load, so it needs real checkpoints. The retain
    # interval must stay a multiple of the save interval or megatron asserts at
    # startup (validate_args: save_retain_interval % save_interval == 0).
    SAVE_ENV="SAVE_INTERVAL=10,SAVE_RETAIN_INTERVAL=100,SAVE_HF=0"
fi

# interactive allows 4 h but caps at 2 nodes, and it schedules ahead of
# everything else -- fine for a single verification job, wrong for a chain that
# occupies it for days. batch_short caps at 2 h so it cannot host a 4 h job at
# all. Convergence therefore goes to batch regardless of size.
partition_for() {  # nodes -> partition
    [[ "${MODE}" == verify ]] && (( $1 <= 2 )) && { echo interactive; return; }
    echo batch
}

points() {  # T R nodes
    for t in ${TRAIN_NODES}; do
        for (( r = 1; r + t <= MAX_TOTAL_NODES; r++ )); do
            local n=$(( t + r )) dp
            dp=$(( GPN * n / (TP * CP) ))
            (( GBS % dp == 0 )) || continue
            echo "${t} ${r} ${n}"
        done
    done
}

name_of() { echo "realstale-${MODE}-t$1r$2-rs$3"; }
colo_name_of() { echo "realstale-${MODE}-colocated-n$1-rs$2"; }

if (( CHECK == 1 )); then
    shopt -s nullglob
    logs=("${LOG_DIR}"/realstale-*.log)
    (( ${#logs[@]} )) || { echo "no realstale-*.log under ${LOG_DIR}"; exit 1; }
    "${REPO_ROOT}/experiments/analyze_throughput.py" "${logs[@]}"
    echo
    # The decomposition, not the bound: the bound is parked and reports nothing.
    for f in "${logs[@]}"; do
        echo "=== $(basename "${f}") ==="
        for fam in total pre_queue in_queue; do
            v=$(grep -o "'staleness/${fam}/mean': [0-9.]*" "${f}" | tail -1)
            printf '  %-10s %s\n' "${fam}" "${v:-<absent>}"
        done
    done
    exit 0
fi

n_points=$(points | wc -l)
n_seeds=$(wc -w <<<"${ROLLOUT_SEEDS}")
if [[ "${MODE}" == verify ]]; then
    jobs_per_arm=1
else
    jobs_per_arm="${N_JOBS}"
fi

printf 'mode %s: %d ratios x %d seeds x %d job(s), %s wall each\n' \
    "${MODE}" "${n_points}" "${n_seeds}" "${jobs_per_arm}" "${WALL}"
printf 'gbs %s, tp %s, cp %s; staleness %s (parked, never binds), reference %s; %s rollouts\n\n' \
    "${GBS}" "${TP}" "${CP}" "${STALENESS}" "${STALENESS_REFERENCE}" "${NUM_ROLLOUT}"

printf '  %-3s %-3s %-6s %-12s %-9s %s\n' T R nodes partition wall "colocated dp"
total_nodeh=0
while read -r t r n; do
    part=$(partition_for "${n}")
    printf '  %-3s %-3s %-6s %-12s %-9s %d\n' "$t" "$r" "$n" "$part" "${WALL}" "$(( GPN * n / (TP * CP) ))"
    total_nodeh=$(( total_nodeh + n * ${WALL%%:*} * n_seeds * jobs_per_arm ))
done < <(points)

if [[ "${MODE}" == convergence ]]; then
    printf '  %-3s %-3s %-6s %-12s %-9s %d   <- on-policy reference\n' \
        "${MAX_TOTAL_NODES}" 0 "${MAX_TOTAL_NODES}" batch "${WALL}" \
        "$(( GPN * MAX_TOTAL_NODES / (TP * CP) ))"
    total_nodeh=$(( total_nodeh + MAX_TOTAL_NODES * ${WALL%%:*} * jobs_per_arm ))
    printf '\none colocated arm only: it is the reference every ratio is read against,\n'
    printf 'and it does not vary with the ratio.\n'
fi

printf '\n%d node-hours at most, largest job %d nodes\n' "${total_nodeh}" "${MAX_TOTAL_NODES}"

if (( SUBMIT == 0 )); then
    echo "re-run with --submit"
    exit 0
fi

mkdir -p "${LOG_DIR}"
echo

submit_chain() {  # name nodes partition recipe extra_env
    local name=$1 nodes=$2 part=$3 recipe=$4 extra=$5 dep="" jid k
    for (( k = 1; k <= jobs_per_arm; k++ )); do
        local jobname="${name}$([[ "${jobs_per_arm}" == 1 ]] || echo "-p${k}")"
        jid=$(sbatch --parsable ${dep} \
            -A "${SLURM_ACCOUNT_NAME}" \
            --comment="${IDLE_EXEMPTION}" \
            --partition="${part}" \
            --job-name="${jobname}" \
            --nodes="${nodes}" --time="${WALL}" \
            --output="${LOG_DIR}/${jobname}-%j.log" \
            --export="ALL,RUN_NAME=${name},CONFIG_TAG=${name},NUM_ROLLOUT=${NUM_ROLLOUT},${SAVE_ENV},EVAL_INTERVAL=1000,SKIP_EVAL_BEFORE_TRAIN=1,${extra}" \
            "${REPO_ROOT}/${recipe}")
        printf '%s  %-40s%s\n' "${jid}" "${jobname}" \
            "$([[ -n "${dep}" ]] && echo "  after ${dep##*:}")"
        dep="--dependency=afterany:${jid}"
    done
}

while read -r t r n; do
    for seed in ${ROLLOUT_SEEDS}; do
        submit_chain "$(name_of "${t}" "${r}" "${seed}")" "${n}" "$(partition_for "${n}")" \
            "${ASYNC_RECIPE}" \
            "MAX_WEIGHT_STALENESS=${STALENESS},STALENESS_REFERENCE=${STALENESS_REFERENCE},PAUSE_GENERATION_MODE=in_place,ROLLOUT_SEED=${seed},ACTOR_NUM_NODES=${t},ROLLOUT_NUM_GPUS=$(( r * GPN ))"
    done
done < <(points)

if [[ "${MODE}" == convergence ]]; then
    seed=$(awk '{print $1}' <<<"${ROLLOUT_SEEDS}")
    # Set explicitly rather than inherited: --export=ALL would otherwise leak the
    # submitting shell's async values into run_identity.sh's colocated guard,
    # which rejects a non-default staleness, pause mode or reference on this path.
    submit_chain "$(colo_name_of "${MAX_TOTAL_NODES}" "${seed}")" "${MAX_TOTAL_NODES}" batch \
        "${COLO_RECIPE}" \
        "ACTOR_NUM_NODES=${MAX_TOTAL_NODES},ROLLOUT_NUM_GPUS=0,MAX_WEIGHT_STALENESS=0,PAUSE_GENERATION_MODE=none,STALENESS_REFERENCE=completion,ROLLOUT_SEED=${seed}"
fi

echo
echo "check with:  experiments/realized_staleness_sweep.sh --check"
