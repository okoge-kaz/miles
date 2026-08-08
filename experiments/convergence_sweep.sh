#!/bin/bash
# The convergence study: run each arm to NUM_ROLLOUT and read the quality curve.
#
#     experiments/convergence_sweep.sh                  # print the plan and the cost
#     experiments/convergence_sweep.sh --tier 1
#     experiments/convergence_sweep.sh --tier 1 --submit
#     experiments/convergence_sweep.sh --tier 1 --submit --force   # ignore a dirty CKPT_PATH
#     experiments/convergence_sweep.sh --tier 1 --staleness 0      # one arm, to resubmit it alone
#     experiments/convergence_sweep.sh --tier 3 --lr 1e-6          # half of tier 3, to stage it
#
# The split is fixed by notes/node-ratio-procedure.md: 4 nodes everywhere, async
# arms as 1 train + 3 rollout, the staleness-0 arm colocated across all 4.
#
# Tiers are submission order, not a grid. Tier 1 answers the paper's question;
# everything after it is a refinement that is only worth its GPU-hours once tier
# 1 has a curve.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

ASYNC_RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
COLO_RECIPE=experiments/math_sync/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507"

: "${TOTAL_ROLLOUT:=300}"
: "${N_JOBS:=10}"            # 300 x 335 s = 28 h against a 4 h wall, plus margin
: "${WALL:=04:00:00}"
: "${PARTITION:=batch}"
: "${NODES:=4}"
: "${TRAIN_SEED:=1234}"
: "${ROLLOUT_SEED:=42}"

# The third column selects the correction and, with it, the bounds the source
# paper uses; the fourth is the PPO ratio denominator; the fifth is the OPSM
# threshold, "-" for off. None of the three are free parameters -- see
# notes/algorithm-ablation.md.
#
# Tier 2 carries no s=0 arm. Every correction there exists to mitigate
# off-policyness, so on-policy it has nothing to act on and tier 1's s=0 run is
# the reference for all of them.
#
# lr 5e-6 is the primary. At 1e-6 the corrections do not fire -- measured
# `tis_clipfrac` 5e-06, IcePop 7.8e-06 -- so an algorithm sweep there compares
# methods that never act. See notes/algorithm-ablation.md.
#
# The two DeepSeek-V3.2 rows per staleness are a pair: OPSM needs the behaviour
# policy as the denominator, and miles rejects that alongside TIS, so the OPSM
# arm differs from tier 1 in two ways at once. The "-" row is the control that
# separates them.
#
# Tier 5 is not part of the plan. It is a rescue, run only for an arm that has
# already collapsed at a higher learning rate, and only for that arm -- never as
# a block. Submit it with --staleness to name the arm.
#
# tier | staleness | IS_CORRECTION | RATIO_DENOMINATOR | OPSM_DELTA | LR | LEN
ARMS=$(cat <<'EOF'
1 0 tis    actor            -    5e-6 32768
1 1 tis    actor            -    5e-6 32768
1 2 tis    actor            -    5e-6 32768
1 4 tis    actor            -    5e-6 32768
2 1 icepop actor            -    5e-6 32768
2 2 icepop actor            -    5e-6 32768
2 4 icepop actor            -    5e-6 32768
2 1 m2po   rollout-logprobs -    5e-6 32768
2 2 m2po   rollout-logprobs -    5e-6 32768
2 4 m2po   rollout-logprobs -    5e-6 32768
2 1 none   rollout-logprobs 1e-4 5e-6 32768
2 2 none   rollout-logprobs 1e-4 5e-6 32768
2 4 none   rollout-logprobs 1e-4 5e-6 32768
2 1 none   rollout-logprobs -    5e-6 32768
2 2 none   rollout-logprobs -    5e-6 32768
2 4 none   rollout-logprobs -    5e-6 32768
3 0 tis    actor            -    1e-6 32768
3 1 tis    actor            -    1e-6 32768
3 2 tis    actor            -    1e-6 32768
3 4 tis    actor            -    1e-6 32768
4 0 tis    actor            -    5e-6 4096
4 1 tis    actor            -    5e-6 4096
4 2 tis    actor            -    5e-6 4096
4 4 tis    actor            -    5e-6 4096
5 0 tis    actor            -    1e-7 32768
5 1 tis    actor            -    1e-7 32768
5 2 tis    actor            -    1e-7 32768
5 4 tis    actor            -    1e-7 32768
EOF
)

# The bounds each correction's source paper actually uses. TIS keeps miles'
# one-sided truncation; IcePop is two-sided.
# Chain length per arm. The slower the arm, the more 4 h jobs it takes to reach
# NUM_ROLLOUT, and a chain that ends early needs a manual top-up. Surplus is
# nearly free: a job whose run has already reached NUM_ROLLOUT resumes, finds
# nothing to do and exits in ~4 minutes (measured), so it costs one node-job, not
# one node-4-hours. Sized off the measured steady-state step times at lr 1e-6 --
# s0 604 s, s1 431 s, s2 346 s, s4 345 s -- with margin, because 5e-6 lengthens
# responses faster and no measurement of it exists yet.
n_jobs_of() {  # staleness -> chained job count
    case "$1" in
        0) echo $(( N_JOBS + 8 )) ;;   # colocated: no rollout/train overlap at all
        1) echo $(( N_JOBS + 5 )) ;;   # tightest bound, so the most recycling waste
        *) echo "${N_JOBS}" ;;
    esac
}

bounds_of() {  # is_correction -> "low high"
    case "$1" in
        icepop) echo "0.5 5.0" ;;
        m2po)   echo "0 0" ;;      # unused: M2PO derives its bounds per step
        *)      echo "0 2.0" ;;
    esac
}

TIER=""; STALENESS=""; LR_FILTER=""; SUBMIT=0; FORCE=0
while (( $# )); do
    case "$1" in
        --tier)      TIER="$2"; shift 2 ;;
        --staleness) STALENESS="$2"; shift 2 ;;   # comma-separated, e.g. 0 or 1,2,4
        --lr)        LR_FILTER="$2"; shift 2 ;;   # comma-separated, e.g. 1e-6
        --submit)    SUBMIT=1; shift ;;
        --force)     FORCE=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

arms_of_tier() {
    awk -v t="${TIER}" -v s="${STALENESS}" -v l="${LR_FILTER}" '
        BEGIN {
            if (s != "") { n = split(s, a, ","); for (i = 1; i <= n; i++) keep[a[i]] = 1 }
            if (l != "") { n = split(l, b, ","); for (i = 1; i <= n; i++) keeplr[b[i]] = 1 }
        }
        NF && ($1 == t || t == "") && (s == "" || $2 in keep) && (l == "" || $6 in keeplr)
    ' <<<"${ARMS}"
}

# CKPT_PATH is derived by common/run_identity.sh, so asking it is the only way to
# be sure the guard checks the directory the job will actually write to.
ckpt_path_of() {  # staleness is_correction ratio_denominator opsm_delta lr max_response_len
    (
        set -euo pipefail
        MODEL_NAME=Qwen3-4B-Instruct-2507 DATASET_TAG=dapo-math-p10-90
        ADVANTAGE_ESTIMATOR=grpo ENTROPY_COEF=0.00 KL_LOSS_COEF=0.00
        EPS_CLIP=0.2 EPS_CLIP_HIGH=0.28 EPS_CLIP_C= RATIO_DENOMINATOR=$3
        read -r TIS_CLIP_LOW TIS_CLIP < <(bounds_of "$2")
        if [[ "$4" == "-" ]]; then USE_OPSM=0; OPSM_DELTA=1e-4
        else USE_OPSM=1; OPSM_DELTA=$4; fi
        ROLLOUT_BATCH_SIZE=192 N_SAMPLES_PER_PROMPT=16 GLOBAL_BATCH_SIZE=3072
        NUM_STEPS_PER_ROLLOUT=1
        MAX_WEIGHT_STALENESS=$1 IS_CORRECTION=$2 LR=$5 MAX_RESPONSE_LEN=$6
        MIS_PROFILE=
        TRAIN_SEED="${TRAIN_SEED}" ROLLOUT_SEED="${ROLLOUT_SEED}"
        if (( $1 == 0 )); then PLACEMENT=colocated; unset MAX_WEIGHT_STALENESS
        else PLACEMENT=async; PAUSE_GENERATION_MODE=in_place; fi
        source "${REPO_ROOT}/experiments/common/run_identity.sh" >/dev/null
        printf '%s\t%s\n' "${CKPT_PATH}" "${RUN_NAME}"
    )
}

host_ckpt() { sed "s#^/ckpt/training#${TRAIN_CKPT_DIR}#" <<<"$1"; }

if [[ -z "${TIER}" ]]; then
    echo "pass --tier N. Tiers, in submission order:"
    echo "  1  primary: lr 5e-6, 32k, GRPO + DAPO clip-higher + TIS 2.0, token-level loss"
    echo "  2  algorithm: IcePop, M2PO, DeepSeek-V3.2 OPSM (+ its denominator control)"
    echo "  3  learning rate: 1e-6 -- the slower reference, not the primary"
    echo "  4  response length: 4k"
    echo "  5  RESCUE ONLY: lr 1e-7, for an arm that collapsed above. Not a block;"
    echo "     name the arm with --staleness."
    echo
    awk 'NF{c[$1]++} END{for(t in c) printf "  tier %s: %d arms\n", t, c[t]}' <<<"${ARMS}" | sort
    exit 0
fi

n_jobs=${N_JOBS}
n_arms=$(arms_of_tier | wc -l)

printf 'tier %s: %d arms x %d-%d chained jobs x %d nodes, %s wall each\n' \
    "${TIER}" "${n_arms}" "${n_jobs}" "$(( n_jobs + 8 ))" "${NODES}" "${WALL}"
printf '%d rollouts per arm, seeds tseed %s / rseed %s\n\n' \
    "${TOTAL_ROLLOUT}" "${TRAIN_SEED}" "${ROLLOUT_SEED}"

dirty=0
printf '  %-3s %-7s %-17s %-5s %-6s %-6s %-9s %s\n' s IS denom opsm lr len place "checkpoint state"
while read -r tier s isc denom prof lr len; do
    IFS=$'\t' read -r cpath rname < <(ckpt_path_of "${s}" "${isc}" "${denom}" "${prof}" "${lr}" "${len}")
    host=$(host_ckpt "${cpath}")
    place=$([[ "${s}" == 0 ]] && echo colocated || echo "async 1+3")
    if compgen -G "${host}/iter_*" >/dev/null 2>&1; then
        state="RESUMES from $(cat "${host}/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')"
        dirty=1
    else
        state="clean"
    fi
    printf '  %-3s %-7s %-17s %-5s %-6s %-6s %-9s %s\n' "${s}" "${isc}" "${denom}" "${prof}" "${lr}" "${len}" "${place}" "${state}"
done < <(arms_of_tier)

echo
total_jobs=0
while read -r _ s_ _ _ _ _ _; do total_jobs=$(( total_jobs + $(n_jobs_of "${s_}") )); done < <(arms_of_tier)
printf '%d jobs total; at most %d node-hours; measured 335 s/step gives ~%.0f h per arm\n' \
    "${total_jobs}" "$(( total_jobs * NODES * ${WALL%%:*} ))" \
    "$(awk -v n="${TOTAL_ROLLOUT}" 'BEGIN{print n*335/3600}')"

if (( SUBMIT == 0 )); then
    echo "re-run with --submit"
    exit 0
fi

if grep -qw TBD < <(arms_of_tier); then
    echo
    echo "refusing: an arm has OPSM_DELTA=TBD. The DeepSeek-V3.2 threshold is not" >&2
    echo "in the paper; measure opsm_clipfrac first. See notes/algorithm-ablation.md" >&2
    exit 1
fi

if (( dirty == 1 && FORCE == 0 )); then
    echo
    echo "refusing: an arm above already has a checkpoint and would resume from it" >&2
    echo "pass --force if resuming is what you want" >&2
    exit 1
fi

echo
while read -r tier s isc denom prof lr len; do
    IFS=$'\t' read -r cpath rname < <(ckpt_path_of "${s}" "${isc}" "${denom}" "${prof}" "${lr}" "${len}")
    if (( s == 0 )); then
        recipe="${COLO_RECIPE}"
        # Both are set explicitly: --export=ALL would otherwise leak whatever the
        # submitting shell holds into run_identity.sh's colocated guard.
        placement_env="ACTOR_NUM_NODES=${NODES},ROLLOUT_NUM_GPUS=0,MAX_WEIGHT_STALENESS=0,PAUSE_GENERATION_MODE=none"
    else
        recipe="${ASYNC_RECIPE}"
        placement_env="ACTOR_NUM_NODES=1,ROLLOUT_NUM_GPUS=$(( (NODES - 1) * 8 )),MAX_WEIGHT_STALENESS=${s},PAUSE_GENERATION_MODE=in_place"
    fi
    read -r tclow tchigh < <(bounds_of "${isc}")
    algo_env="TIS_CLIP_LOW=${tclow},TIS_CLIP=${tchigh}"
    algo_env="${algo_env},RATIO_DENOMINATOR=${denom}"
    [[ "${prof}" == "-" ]] || algo_env="${algo_env},USE_OPSM=1,OPSM_DELTA=${prof}"

    dep=""
    arm_jobs=$(n_jobs_of "${s}")
    for (( k = 1; k <= arm_jobs; k++ )); do
            name="conv-s${s}-${isc}$([[ "${denom}" == actor ]] || echo "-rolloutlp")$([[ "${prof}" == "-" ]] || echo "-opsm")-lr${lr}-${len}-p${k}"
        # Every job in the chain gets identical arguments. NUM_ROLLOUT feeds
        # train_iters and so lr_decay_steps (megatron_utils/model.py:78-80), and
        # OptimizerParamScheduler.load_state_dict asserts the checkpoint's value
        # matches. A per-job rollout budget would break resume on the first
        # boundary. The wall stops each job; the next one resumes from the last
        # checkpoint, so SAVE_INTERVAL bounds what a boundary costs.
        jid=$(sbatch --parsable ${dep} \
            -A "${SLURM_ACCOUNT_NAME}" \
            --partition="${PARTITION}" \
            --job-name="${name}" \
            --nodes="${NODES}" --time="${WALL}" \
            --output="${LOG_DIR}/${name}-%j.log" \
            --export="ALL,RUN_NAME=${rname},NUM_ROLLOUT=${TOTAL_ROLLOUT},LR=${lr},MAX_RESPONSE_LEN=${len},IS_CORRECTION=${isc},TRAIN_SEED=${TRAIN_SEED},ROLLOUT_SEED=${ROLLOUT_SEED},${placement_env},${algo_env}" \
            "${REPO_ROOT}/${recipe}")
        printf '%s  %-46s%s\n' "${jid}" "${name}" \
            "$([[ -n "${dep}" ]] && echo "  after ${dep##*:}")"
        dep="--dependency=afterany:${jid}"
    done
    echo
done < <(arms_of_tier)
