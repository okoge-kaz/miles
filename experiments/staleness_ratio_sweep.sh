#!/bin/bash
# The train:rollout balance as a function of the staleness bound, at a fixed
# 8-node allocation. See notes/node-ratio-procedure.md for what the readout means
# and why the bound and the balance cannot be read independently.
#
#     experiments/staleness_ratio_sweep.sh                 # print the grid and the cost
#     experiments/staleness_ratio_sweep.sh --submit
#     experiments/staleness_ratio_sweep.sh --staleness 2    # one row of the grid
#     experiments/staleness_ratio_sweep.sh --ratio 1:7,2:6  # one pair of columns
#     experiments/staleness_ratio_sweep.sh --check
#
# Unlike node_ratio_sweep.sh, the bound here is *enforced*, not parked at 64: the
# question is what balance each bound wants, so the recycling it causes is part
# of the measurement rather than something to keep out of it.
#
# --submit DELETES the checkpoint directory of every point it submits. A point is
# a fresh measurement, not a continuation: resuming would start the run with a
# warm queue and a policy that has already moved. The dry run lists what would go.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507"

: "${TOTAL_NODES:=8}"
: "${RATIOS:=1:7 2:6 3:5 4:4}"      # train:rollout, in nodes
: "${STALENESS_LEVELS:=1 2 4 8}"
: "${ROLLOUT_SEED:=42}"
: "${LR:=1e-6}"
: "${IS_CORRECTION:=tis}"
: "${WANDB_PROJECT:=async-rl-dapo-math-node-ratio}"
: "${PARTITION:=batch}"             # 8 nodes: batch_short caps at 4
: "${WALL:=04:00:00}"

# Both are pinned rather than inherited: this sweep names them as its fixed
# conditions, and a recipe default that moves later must not move the sweep.
: "${TIS_CLIP:=2.0}"
: "${TIS_CLIP_LOW:=0}"
: "${RATIO_DENOMINATOR:=actor}"

# https://nvidia.atlassian.net/wiki/spaces/HWINFCSSUP/pages/2441648885
# Must go on the sbatch line, not in a #SBATCH directive: the directive form
# strips the double quotes and the reaper then sees invalid JSON.
IDLE_EXEMPTION='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"data_loading","description":"Async RL: the training node is idle while the rollout engines generate its next batch; one 192x16 batch at 32k response length exceeds the default threshold"}}'

read_default() {  # read a recipe default so the grid cannot drift from the recipe
    sed -n "s/^: \"\${$1:=\([^}]*\)}\".*/\1/p" "${REPO_ROOT}/${RECIPE}" | head -1
}
GBS=$(read_default GLOBAL_BATCH_SIZE)
TP=$(read_default TENSOR_PARALLEL_SIZE)
CP=$(read_default CONTEXT_PARALLEL_SIZE)
GPN=$(read_default ACTOR_GPUS_PER_NODE)

STALENESS_FILTER=""; RATIO_FILTER=""; SUBMIT=0; CHECK=0
while (( $# )); do
    case "$1" in
        --staleness) STALENESS_FILTER="$2"; shift 2 ;;   # comma-separated, e.g. 1,2
        --ratio)     RATIO_FILTER="$2"; shift 2 ;;       # comma-separated, e.g. 1:7,2:6
        --submit)    SUBMIT=1; shift ;;
        --check)     CHECK=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

in_list() {  # value comma-separated-list -> 0 when the list is empty or contains it
    [[ -z "$2" ]] && return 0
    local x; for x in ${2//,/ }; do [[ "${x}" == "$1" ]] && return 0; done
    return 1
}

# dp is the trainer's alone under --fully-async: the rollout GPUs do not train.
# A shape megatron would reject is dropped here, at submission, with the reason.
points() {  # staleness T R dp
    local s t r dp
    for s in ${STALENESS_LEVELS}; do
        in_list "${s}" "${STALENESS_FILTER}" || continue
        for pair in ${RATIOS}; do
            in_list "${pair}" "${RATIO_FILTER}" || continue
            t="${pair%%:*}"; r="${pair##*:}"
            (( t + r == TOTAL_NODES )) || { echo "skip ${pair}: not ${TOTAL_NODES} nodes" >&2; continue; }
            (( (t * GPN) % (TP * CP) == 0 )) || { echo "skip ${pair}: tp${TP}*cp${CP} does not divide $(( t * GPN )) train GPUs" >&2; continue; }
            dp=$(( t * GPN / (TP * CP) ))
            (( GBS % dp == 0 )) || { echo "skip ${pair}: gbs ${GBS} not divisible by dp ${dp}" >&2; continue; }
            echo "${s} ${t} ${r} ${dp}"
        done
    done
}

name_of() { echo "s$1-t$2r$3"; }

# CKPT_PATH is derived by common/run_identity.sh, so asking it is the only way to
# be sure the guard checks the directory the job will actually write to.
ckpt_path_of() {  # staleness config_tag
    (
        set -euo pipefail
        MODEL_NAME=Qwen3-4B-Instruct-2507 DATASET_TAG=dapo-math-p10-90 PLACEMENT=async
        ADVANTAGE_ESTIMATOR=grpo ENTROPY_COEF=0.00 KL_LOSS_COEF=0.00
        EPS_CLIP=0.2 EPS_CLIP_HIGH=0.28 EPS_CLIP_C= RATIO_DENOMINATOR="${RATIO_DENOMINATOR}"
        IS_CORRECTION="${IS_CORRECTION}" TIS_CLIP="${TIS_CLIP}" TIS_CLIP_LOW="${TIS_CLIP_LOW}"
        MIS_PROFILE= USE_OPSM=0 OPSM_DELTA=1e-4
        ROLLOUT_BATCH_SIZE=192 N_SAMPLES_PER_PROMPT=16 GLOBAL_BATCH_SIZE="${GBS}"
        NUM_STEPS_PER_ROLLOUT=1 MAX_RESPONSE_LEN=32768 LR="${LR}"
        MAX_WEIGHT_STALENESS=$1 PAUSE_GENERATION_MODE=in_place
        TRAIN_SEED=1234 ROLLOUT_SEED="${ROLLOUT_SEED}" RUN_NAME=$2 CONFIG_TAG=$2
        source "${REPO_ROOT}/experiments/common/run_identity.sh" >/dev/null
        printf '%s\n' "${CKPT_PATH}"
    )
}

host_ckpt() { sed "s#^/ckpt/training#${TRAIN_CKPT_DIR}#" <<<"$1"; }

# RUN_NAME is the two swept axes and nothing else. run_identity.sh's derivation is
# longer, not shorter, so it is overridden rather than inherited. run_identity.sh
# rejects a bad identity by exiting, and a command substitution swallows that
# status -- checked explicitly, because the alternative fails 100 s into an
# 8-node allocation.
host_ckpt_of() {  # staleness name -> host path, or abort
    local cpath
    if ! cpath=$(ckpt_path_of "$1" "$2"); then
        echo "run identity rejected ${2}; not submitting" >&2
        exit 1
    fi
    host_ckpt "${cpath}"
}

if (( CHECK == 1 )); then
    shopt -s nullglob
    # One log per point: a resubmitted point leaves its previous log behind, and
    # reporting both would put two runs of the same configuration in the table as
    # if they were two configurations. Highest job id wins.
    declare -A newest_jid newest_log
    skipped=0
    for f in "${LOG_DIR}"/s*-t*r*-*.log; do
        b=$(basename "${f}" .log); point="${b%-*}"; jid="${b##*-}"
        [[ "${jid}" =~ ^[0-9]+$ ]] || continue
        if [[ -z "${newest_jid[${point}]:-}" ]] || (( jid > newest_jid[${point}] )); then
            [[ -z "${newest_jid[${point}]:-}" ]] || skipped=$(( skipped + 1 ))
            newest_jid[${point}]="${jid}"; newest_log[${point}]="${f}"
        else
            skipped=$(( skipped + 1 ))
        fi
    done
    (( ${#newest_log[@]} )) || { echo "no s*-t*r*.log under ${LOG_DIR}"; exit 1; }
    logs=(); for point in "${!newest_log[@]}"; do logs+=("${newest_log[${point}]}"); done
    IFS=$'\n' logs=($(sort <<<"${logs[*]}")); unset IFS
    (( skipped == 0 )) || echo "(${skipped} superseded log(s) skipped)"
    "${REPO_ROOT}/experiments/analyze_throughput.py" "${logs[@]}"
    echo
    python3 - "${logs[@]}" <<'PY'
import ast, re, sys, pathlib

# The last rollout's values: the queue needs several rollouts to reach its depth,
# so an average over the run mixes the fill-up with the steady state.
KEYS = [
    ("staleness/mean", "lag_mean"),
    ("staleness/max", "lag_max"),
    ("staleness/frac_at_bound", "at_bound"),
    ("staleness/offered/mean", "offered_mean"),
    ("rollout/fully_async/queue_size", "queue"),
    ("rollout/fully_async/stale_groups_recycled", "recycled"),
    ("rollout/fully_async/wasted_token_frac", "wasted"),
    ("staleness/retry_count_mean", "retries"),
]
print(f"{'run':28s}" + "".join(f"{label:>13s}" for _, label in KEYS))
for path in sorted(sys.argv[1:]):
    last = None
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        if "fully_async/queue_size" not in line:
            continue
        m = re.search(r"\{.*\}", line)
        try:
            last = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            continue
    if last is None:
        continue
    name = pathlib.Path(path).stem.rsplit("-", 1)[0]
    row = "".join(f"{last[k]:13.3f}" if k in last else f"{'-':>13s}" for k, _ in KEYS)
    print(f"{name:28s}{row}")
PY
    exit 0
fi

n_jobs=$(points | wc -l)

printf 'lr %s, %s, %d nodes per job, %s wall, rseed %s\n' \
    "${LR}" "${IS_CORRECTION}" "${TOTAL_NODES}" "${WALL}" "${ROLLOUT_SEED}"
printf 'wandb project %s\n' "${WANDB_PROJECT}"
printf 'NUM_ROLLOUT unset: the recipe default stands, so the wall stops each job and\n'
printf 'the run stays resumable. gbs %s, tp %s, cp %s; the bound is enforced, so\n' \
    "${GBS}" "${TP}" "${CP}"
printf 'recycling is part of the result.\n\n'

printf '  %-3s %-3s %-3s %-5s %-5s %-9s %s\n' s T R dp gbs/dp place "checkpoint state"
while read -r s t r dp; do
    name=$(name_of "${s}" "${t}" "${r}")
    host=$(host_ckpt_of "${s}" "${name}")
    if compgen -G "${host}/iter_*" >/dev/null 2>&1; then
        state="DELETED at --submit (has iter_$(cat "${host}/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?'))"
    else
        state="clean"
    fi
    printf '  %-3s %-3s %-3s %-5s %-5s %-9s %s\n' \
        "${s}" "${t}" "${r}" "${dp}" "$(( GBS / dp ))" "async ${t}+${r}" "${state}"
done < <(points)

printf '\n%d jobs x %d nodes x %s = %d node-hours\n' \
    "${n_jobs}" "${TOTAL_NODES}" "${WALL}" "$(( n_jobs * TOTAL_NODES * ${WALL%%:*} ))"
echo "one rollout seed: this orders the bounds; it does not separate two adjacent ratios."

if (( SUBMIT == 0 )); then
    echo "re-run with --submit"
    exit 0
fi

mkdir -p "${LOG_DIR}"

# A point already in the queue owns its checkpoint directory. Deleting it under a
# running job corrupts that run and produces a measurement of neither.
while read -r s t r dp; do
    name=$(name_of "${s}" "${t}" "${r}")
    if [[ -n "$(squeue -h -u "${USER}" -n "${name}" -o '%i' 2>/dev/null)" ]]; then
        echo "refusing: ${name} is already queued or running as" \
             "$(squeue -h -u "${USER}" -n "${name}" -o '%i' | tr '\n' ' ')" >&2
        echo "cancel it first, or narrow this submission with --staleness/--ratio" >&2
        exit 1
    fi
done < <(points)

echo
while read -r s t r dp; do
    name=$(name_of "${s}" "${t}" "${r}")
    host=$(host_ckpt_of "${s}" "${name}")
    # A point is a fresh measurement. Leaving the directory in place would resume
    # the run instead, and the first rollout would then be reported with a queue
    # and a policy that a cold start does not have. Guarded on the prefix because
    # this deletes recursively and the path comes from a shell expansion.
    if [[ -d "${host}" ]]; then
        case "${host}" in
            "${TRAIN_CKPT_DIR}"/*) rm -rf -- "${host}"; echo "removed ${host}" ;;
            *) echo "refusing to delete outside ${TRAIN_CKPT_DIR}: ${host}" >&2; exit 1 ;;
        esac
    fi
    jid=$(sbatch --parsable \
        -A "${SLURM_ACCOUNT_NAME}" \
        --comment="${IDLE_EXEMPTION}" \
        --partition="${PARTITION}" \
        --job-name="${name}" \
        --nodes="${TOTAL_NODES}" --time="${WALL}" \
        --output="${LOG_DIR}/${name}-%j.log" \
        --export="ALL,WANDB_PROJECT=${WANDB_PROJECT},RUN_NAME=${name},CONFIG_TAG=${name},LR=${LR},MAX_WEIGHT_STALENESS=${s},PAUSE_GENERATION_MODE=in_place,ACTOR_NUM_NODES=${t},ROLLOUT_NUM_GPUS=$(( r * GPN )),ROLLOUT_SEED=${ROLLOUT_SEED},IS_CORRECTION=${IS_CORRECTION},TIS_CLIP=${TIS_CLIP},TIS_CLIP_LOW=${TIS_CLIP_LOW},RATIO_DENOMINATOR=${RATIO_DENOMINATOR}" \
        "${REPO_ROOT}/${RECIPE}")
    printf '%s  s=%-2s T=%s R=%s  dp%-3s\n' "${jid}" "${s}" "${t}" "${r}" "${dp}"
done < <(points)

echo
echo "check with:  experiments/staleness_ratio_sweep.sh --check"
