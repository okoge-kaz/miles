#!/bin/bash
# The train:rollout balance as a function of the staleness bound, at a fixed
# 8-node allocation. See notes/node-ratio-procedure.md for what the readout means
# and why the bound and the balance cannot be read independently.
#
#     experiments/staleness_ratio_sweep.sh                 # print the grid and the cost
#     experiments/staleness_ratio_sweep.sh --submit        # fresh dependency chains; existing paths refuse
#     experiments/staleness_ratio_sweep.sh --resume        # dependency chains from valid checkpoints
#     experiments/staleness_ratio_sweep.sh --submit --clean-checkpoints  # explicitly start over
#     CHAIN_JOBS=3 experiments/staleness_ratio_sweep.sh --resume \
#         --staleness 1 --ratio 2:6 --append-after 12345678
#     experiments/staleness_ratio_sweep.sh --staleness 2    # one row of the grid
#     experiments/staleness_ratio_sweep.sh --ratio 1:7,2:6  # one pair of columns
#     experiments/staleness_ratio_sweep.sh --include-colocated  # add one s=0 on-policy arm
#     experiments/staleness_ratio_sweep.sh --colocated-only     # select only that s=0 arm
#     experiments/staleness_ratio_sweep.sh --check
#
# Unlike realized_staleness_sweep.sh, the bound here is *enforced*, not parked at 64: the
# question is what balance each bound wants, so the recycling it causes is part
# of the measurement rather than something to keep out of it.
#
# No mode deletes checkpoints by default. --submit refuses an existing path,
# --resume requires and preserves a valid checkpoint, and --clean-checkpoints is
# the explicit destructive opt-in for a fresh run. Every arm is over-provisioned
# as an afterany dependency chain; once NUM_ROLLOUT is reached, surplus jobs load
# the completed checkpoint and exit without training more steps.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${REPO_ROOT}/experiments/env.sh"

ASYNC_RECIPE=experiments/math_async/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
COLO_RECIPE=experiments/math_sync/dapo-math-p10-90/qwen3-4b-instruct-2507/run.sbatch
LOG_DIR="${OUTPUT_DIR}/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507"

: "${TOTAL_NODES:=8}"
: "${RATIOS:=1:7 2:6 3:5 4:4}"      # train:rollout, in nodes
: "${STALENESS_LEVELS:=1 2 4 8}"
# The bound starts at the first scheduler-authoritative prefill forward. This
# excludes client/router waiting but includes generation and trainer-queue lag.
STALENESS_REFERENCE=prefill
: "${ROLLOUT_SEED:=42}"
: "${LR:=1e-6}"
: "${IS_CORRECTION:=tis}"
: "${FUSE_ONE_STEP_ACTOR_LOGPROBS:=1}"
: "${LOG_PROBS_CHUNK_SIZE:=-1}"
: "${OBSERVE_TRAINING_ENTROPY:=0}"
: "${WANDB_PROJECT:=async-rl-dapo-math-node-ratio}"
: "${PARTITION:=batch}"             # 8 nodes: batch_short caps at 4
: "${WALL:=04:00:00}"
: "${CHAIN_JOBS:=10}"               # 300 rollouts need ~7 measured 4 h segments
: "${SAVE_INTERVAL:=10}"
: "${SAVE_RETAIN_INTERVAL:=10}"     # retain every distributed checkpoint
: "${SAVE_HF:=1}"                   # retain policy snapshots for offline eval
: "${HF_SAVE_INTERVAL:=10}"         # rollout cadence of the retained HF series

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
    sed -n "s/^: \"\${$1:=\([^}]*\)}\".*/\1/p" "${REPO_ROOT}/${ASYNC_RECIPE}" | head -1
}
GBS=$(read_default GLOBAL_BATCH_SIZE)
TP=$(read_default TENSOR_PARALLEL_SIZE)
CP=$(read_default CONTEXT_PARALLEL_SIZE)
GPN=$(read_default ACTOR_GPUS_PER_NODE)
NUM_ROLLOUT="${NUM_ROLLOUT:-$(read_default NUM_ROLLOUT)}"

STALENESS_FILTER=""; RATIO_FILTER=""; INCLUDE_COLOCATED=0; COLOCATED_ONLY=0
APPEND_AFTER=""
SUBMIT=0; RESUME=0; CLEAN_CHECKPOINTS=0; CHECK=0
while (( $# )); do
    case "$1" in
        --staleness) STALENESS_FILTER="$2"; shift 2 ;;   # comma-separated, e.g. 1,2
        --ratio)     RATIO_FILTER="$2"; shift 2 ;;       # comma-separated, e.g. 1:7,2:6
        --submit)    SUBMIT=1; shift ;;
        --resume)    RESUME=1; shift ;;
        --append-after) APPEND_AFTER="$2"; shift 2 ;;
        --clean-checkpoints) CLEAN_CHECKPOINTS=1; shift ;;
        --include-colocated) INCLUDE_COLOCATED=1; shift ;;
        --colocated-only) INCLUDE_COLOCATED=1; COLOCATED_ONLY=1; shift ;;
        --check)     CHECK=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if (( SUBMIT == 1 && RESUME == 1 )); then
    echo "--submit and --resume are mutually exclusive" >&2
    exit 1
fi
if (( CLEAN_CHECKPOINTS == 1 && SUBMIT == 0 )); then
    echo "--clean-checkpoints requires --submit and cannot be used with --resume" >&2
    exit 1
fi
if [[ -n "${APPEND_AFTER}" ]]; then
    (( RESUME == 1 )) || { echo "--append-after requires --resume" >&2; exit 1; }
    [[ "${APPEND_AFTER}" =~ ^[0-9]+$ ]] || {
        echo "--append-after must be a numeric Slurm job id" >&2
        exit 1
    }
fi
[[ "${CHAIN_JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "CHAIN_JOBS must be a positive integer" >&2; exit 1; }
[[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ROLLOUT must be a positive integer" >&2; exit 1; }
[[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_INTERVAL must be a positive integer" >&2; exit 1; }
[[ "${SAVE_RETAIN_INTERVAL}" =~ ^[1-9][0-9]*$ ]] ||
    { echo "SAVE_RETAIN_INTERVAL must be a positive integer" >&2; exit 1; }
[[ "${SAVE_HF}" =~ ^[01]$ ]] || { echo "SAVE_HF must be 0 or 1" >&2; exit 1; }
[[ "${HF_SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] ||
    { echo "HF_SAVE_INTERVAL must be a positive integer" >&2; exit 1; }
(( SAVE_RETAIN_INTERVAL % SAVE_INTERVAL == 0 )) ||
    { echo "SAVE_RETAIN_INTERVAL must be divisible by SAVE_INTERVAL" >&2; exit 1; }

in_list() {  # value comma-separated-list -> 0 when the list is empty or contains it
    [[ -z "$2" ]] && return 0
    local x; for x in ${2//,/ }; do [[ "${x}" == "$1" ]] && return 0; done
    return 1
}

# dp is the trainer's alone under --fully-async: the rollout GPUs do not train.
# A shape megatron would reject is dropped here, at submission, with the reason.
points() {  # staleness T R dp
    local s t r dp
    (( COLOCATED_ONLY == 0 )) || return 0
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
colo_name_of() { echo "s0-colocated"; }

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
        STALENESS_REFERENCE="${STALENESS_REFERENCE}"
        TRAIN_SEED=1234 ROLLOUT_SEED="${ROLLOUT_SEED}" RUN_NAME=$2 CONFIG_TAG=$2
        source "${REPO_ROOT}/experiments/common/run_identity.sh" >/dev/null
        printf '%s\n' "${CKPT_PATH}"
    )
}

colo_ckpt_path_of() {  # config_tag
    (
        set -euo pipefail
        MODEL_NAME=Qwen3-4B-Instruct-2507 DATASET_TAG=dapo-math-p10-90 PLACEMENT=colocated
        ADVANTAGE_ESTIMATOR=grpo ENTROPY_COEF=0.00 KL_LOSS_COEF=0.00
        EPS_CLIP=0.2 EPS_CLIP_HIGH=0.28 EPS_CLIP_C= RATIO_DENOMINATOR="${RATIO_DENOMINATOR}"
        IS_CORRECTION="${IS_CORRECTION}" TIS_CLIP="${TIS_CLIP}" TIS_CLIP_LOW="${TIS_CLIP_LOW}"
        MIS_PROFILE= USE_OPSM=0 OPSM_DELTA=1e-4
        ROLLOUT_BATCH_SIZE=192 N_SAMPLES_PER_PROMPT=16 GLOBAL_BATCH_SIZE="${GBS}"
        NUM_STEPS_PER_ROLLOUT=1 MAX_RESPONSE_LEN=32768 LR="${LR}"
        MAX_WEIGHT_STALENESS=0 PAUSE_GENERATION_MODE=none STALENESS_REFERENCE=completion
        TRAIN_SEED=1234 ROLLOUT_SEED="${ROLLOUT_SEED}" RUN_NAME=$1 CONFIG_TAG=$1
        source "${REPO_ROOT}/experiments/common/run_identity.sh" >/dev/null
        printf '%s\n' "${CKPT_PATH}"
    )
}

host_ckpt() { sed "s#^/ckpt/training#${TRAIN_CKPT_DIR}#" <<<"$1"; }

prepare_shared_output_permissions() {
    # env.sh already sets this, but keep it explicit at the submission boundary:
    # Slurm logs and checkpoint files created by every dependent job must be
    # readable by collaborators, while directories must remain traversable.
    umask 0022
    mkdir -p "${CKPT_ROOT}" "${TRAIN_CKPT_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
    chmod a+x "${WS}" "${CKPT_ROOT}"
    chmod a+rx "${TRAIN_CKPT_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

    # A default ACL survives into future subdirectories. umask 0022 remains the
    # portable fallback on filesystems or hosts without setfacl support.
    if command -v setfacl >/dev/null 2>&1; then
        setfacl -m d:u::rwx,d:g::rx,d:o::rx,d:m::rwx "${TRAIN_CKPT_DIR}" "${LOG_DIR}" ||
            echo "warning: could not set default ACL; relying on umask 0022" >&2
    fi
}

checkpoint_iteration() {  # host checkpoint path -> numeric iteration, or fail
    local host=$1 iteration iteration_dir
    [[ -f "${host}/latest_checkpointed_iteration.txt" ]] || return 1
    IFS= read -r iteration < "${host}/latest_checkpointed_iteration.txt"
    [[ "${iteration}" =~ ^[0-9]+$ ]] || return 1
    printf -v iteration_dir 'iter_%07d' "$(( 10#${iteration} ))"
    [[ -d "${host}/${iteration_dir}" ]] || return 1
    printf '%s\n' "$(( 10#${iteration} ))"
}

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

colo_host_ckpt_of() {  # name -> host path, or abort
    local cpath
    if ! cpath=$(colo_ckpt_path_of "$1"); then
        echo "run identity rejected $1; not submitting" >&2
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
    skipped=0; empty=0
    for f in "${LOG_DIR}"/s*-t*r*-*.log; do
        if ! grep -q "fully_async/queue_size" "${f}"; then
            empty=$(( empty + 1 ))
            continue
        fi
        b=$(basename "${f}" .log); point="${b%-*}"; jid="${b##*-}"
        [[ "${jid}" =~ ^[0-9]+$ ]] || continue
        if [[ -z "${newest_jid[${point}]:-}" ]] || (( jid > newest_jid[${point}] )); then
            [[ -z "${newest_jid[${point}]:-}" ]] || skipped=$(( skipped + 1 ))
            newest_jid[${point}]="${jid}"; newest_log[${point}]="${f}"
        else
            skipped=$(( skipped + 1 ))
        fi
    done
    async_logs=()
    for point in "${!newest_log[@]}"; do async_logs+=("${newest_log[${point}]}"); done
    if (( ${#async_logs[@]} > 0 )); then
        IFS=$'\n' async_logs=($(sort <<<"${async_logs[*]}")); unset IFS
    fi

    # The colocated arm has no fully-async queue metric. Include its newest
    # segment in the throughput comparison, but keep it out of the staleness
    # table below because staleness is structurally zero on this path.
    colocated_log=""
    if (( INCLUDE_COLOCATED == 1 )); then
        colocated_jid=0
        for f in "${LOG_DIR}/$(colo_name_of)-"*.log; do
            grep -q "'train/step':" "${f}" || { empty=$(( empty + 1 )); continue; }
            b=$(basename "${f}" .log); jid="${b##*-}"
            [[ "${jid}" =~ ^[0-9]+$ ]] || continue
            if (( jid > colocated_jid )); then
                [[ -z "${colocated_log}" ]] || skipped=$(( skipped + 1 ))
                colocated_jid=${jid}; colocated_log=${f}
            else
                skipped=$(( skipped + 1 ))
            fi
        done
    fi
    logs=("${async_logs[@]}")
    [[ -z "${colocated_log}" ]] || logs+=("${colocated_log}")
    (( ${#logs[@]} )) || { echo "no completed sweep log under ${LOG_DIR}"; exit 1; }
    (( skipped == 0 )) || echo "(${skipped} superseded log(s) skipped)"
    (( empty == 0 )) || echo "(${empty} dependency segment log(s) without training metrics skipped)"
    "${REPO_ROOT}/experiments/analyze_throughput.py" "${logs[@]}"
    if (( ${#async_logs[@]} > 0 )); then
        echo
        python3 - "${async_logs[@]}" <<'PY'
import ast, re, sys, pathlib

# The last rollout's values: the queue needs several rollouts to reach its depth,
# so an average over the run mixes the fill-up with the steady state.
KEYS = [
    ("staleness/total/mean", "total_mean"),
    ("staleness/pre_queue/mean", "pre_queue_mean"),
    ("staleness/in_queue/mean", "in_queue_mean"),
    ("staleness/bound/train/frac_at_bound", "at_bound"),
    ("staleness/bound/rollout/mean", "bound_offered_mean"),
    ("staleness/bound/rollout/max", "bound_offered_max"),
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
    fi
    exit 0
fi

checkpoint_state() {  # host checkpoint path
    local host=$1 iteration
    if iteration=$(checkpoint_iteration "${host}"); then
        if (( CLEAN_CHECKPOINTS == 1 )); then
            echo "will delete iter_${iteration} and start fresh"
        elif (( RESUME == 1 )); then
            echo "will resume iter_${iteration} in ${CHAIN_JOBS}-job chain"
        elif (( SUBMIT == 1 )); then
            echo "REFUSE existing iter_${iteration}; use --resume or --clean-checkpoints"
        else
            echo "has iter_${iteration}; --resume keeps, --clean-checkpoints deletes"
        fi
    elif [[ -e "${host}" ]]; then
        if (( CLEAN_CHECKPOINTS == 1 )); then
            echo "will delete invalid/incomplete checkpoint"
        else
            echo "invalid/incomplete; only --clean-checkpoints may replace it"
        fi
    elif (( RESUME == 1 )); then
        echo "REFUSE: no checkpoint to resume"
    else
        echo "clean; will start ${CHAIN_JOBS}-job dependency chain"
    fi
}

COLO_DP=$(( TOTAL_NODES * GPN / (TP * CP) ))
if (( INCLUDE_COLOCATED == 1 )); then
    (( (TOTAL_NODES * GPN) % (TP * CP) == 0 )) ||
        { echo "colocated: tp${TP}*cp${CP} does not divide $(( TOTAL_NODES * GPN )) GPUs" >&2; exit 1; }
    (( GBS % COLO_DP == 0 )) ||
        { echo "colocated: gbs ${GBS} not divisible by dp ${COLO_DP}" >&2; exit 1; }
fi

n_jobs=$(points | wc -l)
n_jobs=$(( n_jobs + INCLUDE_COLOCATED ))
if [[ -n "${APPEND_AFTER}" ]] && (( n_jobs != 1 )); then
    echo "--append-after requires exactly one selected arm; narrow with" \
         "--staleness/--ratio or --colocated-only" >&2
    exit 1
fi

printf 'lr %s, %s, %d nodes per job, %s wall, rseed %s\n' \
    "${LR}" "${IS_CORRECTION}" "${TOTAL_NODES}" "${WALL}" "${ROLLOUT_SEED}"
printf 'wandb project %s\n' "${WANDB_PROJECT}"
printf '%s rollouts, %s dependent %s jobs per arm; gbs %s, tp %s, cp %s.\n' \
    "${NUM_ROLLOUT}" "${CHAIN_JOBS}" "${WALL}" "${GBS}" "${TP}" "${CP}"
if [[ -n "${APPEND_AFTER}" ]]; then
    printf 'The new chain will append after Slurm job %s.\n' "${APPEND_AFTER}"
fi
printf 'Megatron checkpoints are retained every %s rollouts.\n' "${SAVE_RETAIN_INTERVAL}"
if (( SAVE_HF == 1 )); then
    printf 'HF checkpoints are retained every %s rollouts for offline eval.\n' "${HF_SAVE_INTERVAL}"
else
    printf 'HF checkpoint retention is disabled.\n'
fi
printf 'The bound is enforced from first prefill to training drain (%s).\n' "${STALENESS_REFERENCE}"
printf 's=N admits realized prefill staleness 0..N; s=0 is on-policy when each\n'
printf 'optimizer step is synced once and NUM_STEPS_PER_ROLLOUT=1. Recycling is\n'
printf 'part of the result; watch wasted_token_frac and mixed_version_frac.\n'
printf '\n'

printf '  %-3s %-3s %-3s %-5s %-5s %-9s %s\n' s T R dp gbs/dp place "checkpoint state"
while read -r s t r dp; do
    name=$(name_of "${s}" "${t}" "${r}")
    host=$(host_ckpt_of "${s}" "${name}")
    state=$(checkpoint_state "${host}")
    printf '  %-3s %-3s %-3s %-5s %-5s %-9s %s\n' \
        "${s}" "${t}" "${r}" "${dp}" "$(( GBS / dp ))" "async ${t}+${r}" "${state}"
done < <(points)
if (( INCLUDE_COLOCATED == 1 )); then
    name=$(colo_name_of)
    host=$(colo_host_ckpt_of "${name}")
    state=$(checkpoint_state "${host}")
    printf '  %-3s %-3s %-3s %-5s %-5s %-9s %s\n' \
        0 "${TOTAL_NODES}" 0 "${COLO_DP}" "$(( GBS / COLO_DP ))" colocated "${state}"
fi

printf '\n%d arms x %d dependent jobs x %d nodes x %s = %d node-hours at most\n' \
    "${n_jobs}" "${CHAIN_JOBS}" "${TOTAL_NODES}" "${WALL}" \
    "$(( n_jobs * CHAIN_JOBS * TOTAL_NODES * ${WALL%%:*} ))"
echo "one rollout seed: this orders the bounds; it does not separate two adjacent ratios."

if (( SUBMIT == 0 && RESUME == 0 )); then
    echo "re-run with --submit (fresh clean paths), --resume, or --submit --clean-checkpoints"
    exit 0
fi

prepare_shared_output_permissions

# A point already in the queue owns its checkpoint directory. Deleting it under a
# running job corrupts that run and produces a measurement of neither. Append
# mode is the one exception: it verifies that the supplied dependency belongs to
# the single selected arm, then starts the new chain only after that job exits.
if [[ -n "${APPEND_AFTER}" ]]; then
    if (( INCLUDE_COLOCATED == 1 )); then
        expected_append_name=$(colo_name_of)
    else
        read -r append_s append_t append_r append_dp < <(points)
        expected_append_name=$(name_of "${append_s}" "${append_t}" "${append_r}")
    fi
    append_record=$(squeue -h -j "${APPEND_AFTER}" -o '%j|%u' 2>/dev/null)
    IFS='|' read -r actual_append_name actual_append_user <<<"${append_record}"
    [[ "${actual_append_name}" == "${expected_append_name}" ]] || {
        echo "refusing: --append-after ${APPEND_AFTER} is '${actual_append_name:-not queued}'," \
             "expected '${expected_append_name}'" >&2
        exit 1
    }
    [[ "${actual_append_user}" == "${USER}" ]] || {
        echo "refusing: --append-after ${APPEND_AFTER} belongs to" \
             "'${actual_append_user:-unknown}', expected '${USER}'" >&2
        exit 1
    }
    append_children=$(
        squeue -h -u "${USER}" -o '%i|%E' |
            awk -F'|' -v id="${APPEND_AFTER}" \
                '$2 ~ ("(^|[^0-9])" id "([^0-9]|$)") { print $1 }'
    )
    [[ -z "${append_children}" ]] || {
        echo "refusing: --append-after ${APPEND_AFTER} is not the chain tail;" \
             "jobs ${append_children//$'\n'/ } already depend on it" >&2
        exit 1
    }
else
    while read -r s t r dp; do
        name=$(name_of "${s}" "${t}" "${r}")
        if [[ -n "$(squeue -h -n "${name}" -o '%i' 2>/dev/null)" ]]; then
            echo "refusing: ${name} is already queued or running as" \
                 "$(squeue -h -n "${name}" -o '%i' | tr '\n' ' ')" >&2
            echo "cancel it first, or narrow this submission with --staleness/--ratio" >&2
            exit 1
        fi
    done < <(points)
    if (( INCLUDE_COLOCATED == 1 )); then
        name=$(colo_name_of)
        if [[ -n "$(squeue -h -n "${name}" -o '%i' 2>/dev/null)" ]]; then
            echo "refusing: ${name} is already queued or running as" \
                 "$(squeue -h -n "${name}" -o '%i' | tr '\n' ' ')" >&2
            echo "cancel it first, or omit --include-colocated" >&2
            exit 1
        fi
    fi
fi

# Preflight is all-or-nothing. Validate every selected point before deleting a
# path or submitting a job, so the grid cannot become a mixed fresh/resumed run.
invalid=0
while read -r s t r dp; do
    name=$(name_of "${s}" "${t}" "${r}")
    host=$(host_ckpt_of "${s}" "${name}")
    if (( RESUME == 1 )); then
        if ! iteration=$(checkpoint_iteration "${host}"); then
            echo "refusing: ${name} has no valid resumable checkpoint at ${host}" >&2
            invalid=1
        fi
    elif [[ -e "${host}" ]] && (( CLEAN_CHECKPOINTS == 0 )); then
        echo "refusing: ${name} already has ${host}" >&2
        echo "use --resume to continue it or --submit --clean-checkpoints to replace it" >&2
        invalid=1
    fi
done < <(points)
if (( INCLUDE_COLOCATED == 1 )); then
    name=$(colo_name_of)
    host=$(colo_host_ckpt_of "${name}")
    if (( RESUME == 1 )); then
        if ! iteration=$(checkpoint_iteration "${host}"); then
            echo "refusing: ${name} has no valid resumable checkpoint at ${host}" >&2
            invalid=1
        fi
    elif [[ -e "${host}" ]] && (( CLEAN_CHECKPOINTS == 0 )); then
        echo "refusing: ${name} already has ${host}" >&2
        echo "use --resume to continue it or --submit --clean-checkpoints to replace it" >&2
        invalid=1
    fi
fi
(( invalid == 0 )) || exit 1

echo
# Deletion is explicit and happens only after every point passed preflight.
if (( CLEAN_CHECKPOINTS == 1 )); then
    while read -r s t r dp; do
        name=$(name_of "${s}" "${t}" "${r}")
        host=$(host_ckpt_of "${s}" "${name}")
        [[ -e "${host}" ]] || continue
        case "${host}" in
            "${TRAIN_CKPT_DIR}"/*) rm -rf -- "${host}"; echo "removed ${host}" ;;
            *) echo "refusing to delete outside ${TRAIN_CKPT_DIR}: ${host}" >&2; exit 1 ;;
        esac
    done < <(points)
    if (( INCLUDE_COLOCATED == 1 )); then
        name=$(colo_name_of)
        host=$(colo_host_ckpt_of "${name}")
        if [[ -e "${host}" ]]; then
            case "${host}" in
                "${TRAIN_CKPT_DIR}"/*) rm -rf -- "${host}"; echo "removed ${host}" ;;
                *) echo "refusing to delete outside ${TRAIN_CKPT_DIR}: ${host}" >&2; exit 1 ;;
            esac
        fi
    fi
fi

# Existing checkpoints may predate the shared umask. Repair them once before a
# resume; newly created files inherit umask/default ACL from the roots above.
if (( RESUME == 1 )); then
    while read -r s t r dp; do
        name=$(name_of "${s}" "${t}" "${r}")
        host=$(host_ckpt_of "${s}" "${name}")
        chmod -R a+rX "${host}"
    done < <(points)
    if (( INCLUDE_COLOCATED == 1 )); then
        name=$(colo_name_of)
        host=$(colo_host_ckpt_of "${name}")
        chmod -R a+rX "${host}"
    fi
fi

submit_async_chain() {  # staleness train_nodes rollout_nodes dp
    local s=$1 t=$2 r=$3 dp=$4 name host jid raw_jid k dependency_label
    local -a dependency=()
    [[ -z "${APPEND_AFTER}" ]] || dependency=("--dependency=afterany:${APPEND_AFTER}")

    name=$(name_of "${s}" "${t}" "${r}")
    host=$(host_ckpt_of "${s}" "${name}")
    for (( k = 1; k <= CHAIN_JOBS; k++ )); do
        raw_jid=$(sbatch --parsable "${dependency[@]}" \
            -A "${SLURM_ACCOUNT_NAME}" \
            --comment="${IDLE_EXEMPTION}" \
            --partition="${PARTITION}" \
            --job-name="${name}" \
            --nodes="${TOTAL_NODES}" --time="${WALL}" \
            --output="${LOG_DIR}/${name}-%j.log" \
            --export="ALL,WANDB_PROJECT=${WANDB_PROJECT},RUN_NAME=${name},CONFIG_TAG=${name},NUM_ROLLOUT=${NUM_ROLLOUT},SAVE_INTERVAL=${SAVE_INTERVAL},SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL},SAVE_HF=${SAVE_HF},HF_SAVE_INTERVAL=${HF_SAVE_INTERVAL},EVAL_INTERVAL=0,SKIP_EVAL_BEFORE_TRAIN=1,LR=${LR},MAX_WEIGHT_STALENESS=${s},STALENESS_REFERENCE=prefill,PAUSE_GENERATION_MODE=in_place,ACTOR_NUM_NODES=${t},ROLLOUT_NUM_GPUS=$(( r * GPN )),ROLLOUT_SEED=${ROLLOUT_SEED},IS_CORRECTION=${IS_CORRECTION},TIS_CLIP=${TIS_CLIP},TIS_CLIP_LOW=${TIS_CLIP_LOW},RATIO_DENOMINATOR=${RATIO_DENOMINATOR},FUSE_ONE_STEP_ACTOR_LOGPROBS=${FUSE_ONE_STEP_ACTOR_LOGPROBS},VERIFY_FUSED_ONE_STEP_ACTOR_LOGPROBS=0" \
            "${REPO_ROOT}/${ASYNC_RECIPE}")
        jid=${raw_jid%%;*}
        dependency_label=""
        if (( ${#dependency[@]} > 0 )); then
            dependency_label=" after ${dependency[0]##*:}"
        fi
        printf '%s  s=%-2s T=%s R=%s dp%-3s segment=%s/%s%s\n' \
            "${jid}" "${s}" "${t}" "${r}" "${dp}" "${k}" "${CHAIN_JOBS}" \
            "${dependency_label}"
        dependency=("--dependency=afterany:${jid}")
    done
    printf 'checkpoint: %s\n\n' "${host}"
}

submit_colocated_chain() {
    local name host jid raw_jid k dependency_label
    local -a dependency=()
    [[ -z "${APPEND_AFTER}" ]] || dependency=("--dependency=afterany:${APPEND_AFTER}")

    name=$(colo_name_of)
    host=$(colo_host_ckpt_of "${name}")
    for (( k = 1; k <= CHAIN_JOBS; k++ )); do
        # Override all async-only values explicitly. Because sbatch exports the
        # submitting environment, allowing one of them to leak into the
        # colocated recipe would make run_identity.sh reject the job.
        raw_jid=$(sbatch --parsable "${dependency[@]}" \
            -A "${SLURM_ACCOUNT_NAME}" \
            --comment="${IDLE_EXEMPTION}" \
            --partition="${PARTITION}" \
            --job-name="${name}" \
            --nodes="${TOTAL_NODES}" --time="${WALL}" \
            --output="${LOG_DIR}/${name}-%j.log" \
            --export="ALL,WANDB_PROJECT=${WANDB_PROJECT},RUN_NAME=${name},CONFIG_TAG=${name},NUM_ROLLOUT=${NUM_ROLLOUT},SAVE_INTERVAL=${SAVE_INTERVAL},SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL},SAVE_HF=${SAVE_HF},HF_SAVE_INTERVAL=${HF_SAVE_INTERVAL},EVAL_INTERVAL=0,SKIP_EVAL_BEFORE_TRAIN=1,LR=${LR},MAX_WEIGHT_STALENESS=0,STALENESS_REFERENCE=completion,PAUSE_GENERATION_MODE=none,ACTOR_NUM_NODES=${TOTAL_NODES},ROLLOUT_NUM_GPUS=0,ROLLOUT_SEED=${ROLLOUT_SEED},IS_CORRECTION=${IS_CORRECTION},TIS_CLIP=${TIS_CLIP},TIS_CLIP_LOW=${TIS_CLIP_LOW},RATIO_DENOMINATOR=${RATIO_DENOMINATOR},LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE},OBSERVE_TRAINING_ENTROPY=${OBSERVE_TRAINING_ENTROPY}" \
            "${REPO_ROOT}/${COLO_RECIPE}")
        jid=${raw_jid%%;*}
        dependency_label=""
        if (( ${#dependency[@]} > 0 )); then
            dependency_label=" after ${dependency[0]##*:}"
        fi
        printf '%s  s=0  T=%s R=0 dp%-3s segment=%s/%s%s\n' \
            "${jid}" "${TOTAL_NODES}" "${COLO_DP}" "${k}" "${CHAIN_JOBS}" \
            "${dependency_label}"
        dependency=("--dependency=afterany:${jid}")
    done
    printf 'checkpoint: %s\n\n' "${host}"
}

while read -r s t r dp; do
    submit_async_chain "${s}" "${t}" "${r}" "${dp}"
done < <(points)
if (( INCLUDE_COLOCATED == 1 )); then
    submit_colocated_chain
fi

echo
if (( INCLUDE_COLOCATED == 1 )); then
    echo "check with:  experiments/staleness_ratio_sweep.sh --check --include-colocated"
else
    echo "check with:  experiments/staleness_ratio_sweep.sh --check"
fi
