#!/bin/bash
#PBS -N eval-swebench-verified-hardened
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=08:00:00
#PBS -j oe

# Hardened-local SWE-bench Verified evaluation client. This job uses pinned
# official v2.0.13 parser/grading code, but its security path policy differs
# from run_evaluation and the resulting score is not leaderboard-comparable.
# Input rows must be bound to the exact live-admitted Harbor task tree.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || {
    echo "MILES_SWE_FIXED_EXPORTS must be 1" >&2
    exit 2
}
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || {
    echo "Unapproved submission environment reached SWE evaluation" >&2
    exit 2
}
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || {
    echo "tracked dotenv mask is missing or unsafe" >&2
    exit 2
}
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

: "${SWE_EVAL_TASK_ROWS:=/data/miles-swe/evaluation-input/swebench-verified-hardened-local.jsonl}"
: "${SWE_EVAL_TASK_SUMMARY:=/data/miles-swe/evaluation-input/swebench-verified-hardened-local.summary.json}"
: "${HARBOR_SWE_SERVER_URL:?set the Harbor /run server URL}"
: "${POLICY_BASE_URL:?set the OpenAI-compatible policy endpoint URL}"
: "${SERVED_MODEL:?set the served policy model name}"
: "${HARBOR_RUN_SECRET:?export the Harbor /run bearer secret before pbs_submit}"
(( ${#HARBOR_RUN_SECRET} >= 32 && ${#HARBOR_RUN_SECRET} <= 4096 )) && \
    [[ "$HARBOR_RUN_SECRET" != *$'\r'* && "$HARBOR_RUN_SECRET" != *$'\n'* ]] || {
    echo "HARBOR_RUN_SECRET must be 32-4096 characters without CR/LF" >&2
    exit 2
}
SWE_EVAL_OUTPUT_ROOT="${SWE_EVAL_OUTPUT_ROOT:-/data/miles-swe/eval/swebench-verified-hardened-local}"
SWE_EVAL_LIMIT="${SWE_EVAL_LIMIT:-}"
SWE_EVAL_CONCURRENCY="${SWE_EVAL_CONCURRENCY:-16}"
HARBOR_CLIENT_ID="pbs-eval-${MILES_JOB_ID:?}"
AGENT_SERVER_URL="${HARBOR_SWE_SERVER_URL}"
export SWE_EVAL_OUTPUT_ROOT
export AGENT_SERVER_URL HARBOR_CLIENT_ID HARBOR_RUN_SECRET
export SWE_EVAL_TASK_ROWS SWE_EVAL_TASK_SUMMARY

evaluation_step_pid=""
pretimeout_timer_pid=""
: "${SWE_EVAL_PRETIMEOUT_DELAY_SECONDS:=27900}"
[[ "${SWE_EVAL_PRETIMEOUT_DELAY_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "SWE_EVAL_PRETIMEOUT_DELAY_SECONDS must be a positive integer" >&2
    exit 2
}
drain_harbor_client() {
    PYTHONPATH="${REPO_ROOT}" PYTHONDONTWRITEBYTECODE=1 \
        python3 \
        "${REPO_ROOT}/examples/experimental/swe-agent-harbor-e2b/drain_harbor_client.py"
}
stop_evaluation_step() {
    if [[ -n "${evaluation_step_pid}" ]] && kill -0 "${evaluation_step_pid}" 2>/dev/null; then
        kill -TERM "${evaluation_step_pid}" 2>/dev/null || true
    fi
}
handle_pretimeout() {
    echo "scheduler termination signal received; draining Harbor evaluation trials" >&2
    drain_harbor_client || true
    stop_evaluation_step
}
cleanup_evaluation() {
    local status=$?
    trap - EXIT USR1 TERM
    if [[ -n "${pretimeout_timer_pid}" ]]; then
        kill "${pretimeout_timer_pid}" 2>/dev/null || true
    fi
    drain_harbor_client || true
    stop_evaluation_step
    if [[ -n "${evaluation_step_pid}" ]]; then
        wait "${evaluation_step_pid}" 2>/dev/null || true
    fi
    return "${status}"
}
trap handle_pretimeout USR1
trap 'handle_pretimeout; exit 143' TERM
trap cleanup_evaluation EXIT

# Keep the 15-minute Harbor drain window before the eight-hour PBS walltime.
( sleep "${SWE_EVAL_PRETIMEOUT_DELAY_SECONDS}"; kill -USR1 "$$" ) &
pretimeout_timer_pid=$!

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${SWE_CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=HARBOR_CLIENT_ID,HARBOR_RUN_SECRET,HARBOR_SWE_SERVER_URL,POLICY_BASE_URL,PYTHON_DOTENV_DISABLED,SERVED_MODEL,SWE_EVAL_CONCURRENCY,SWE_EVAL_LIMIT,SWE_EVAL_OUTPUT_ROOT,SWE_EVAL_TASK_ROWS,SWE_EVAL_TASK_SUMMARY,WANDB_DISABLED,WANDB_MODE \
    env -u BASH_ENV -u ENV bash --noprofile --norc -c '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles
run_dir="${SWE_EVAL_OUTPUT_ROOT}/${MILES_JOB_ID}"
mkdir -p "${run_dir}"
args=(
    --input "${SWE_EVAL_TASK_ROWS}"
    --input-summary "${SWE_EVAL_TASK_SUMMARY}"
    --output "${run_dir}/trials.jsonl"
    --summary "${run_dir}/summary.json"
    --server-url "${HARBOR_SWE_SERVER_URL}"
    --base-url "${POLICY_BASE_URL}"
    --model "${SERVED_MODEL}"
    --agent-name terminus-2
    --trials-per-task 1
    --concurrency "${SWE_EVAL_CONCURRENCY}"
    --max-response-tokens 16384
    --max-sequence-length 32768
    --temperature 0
    --top-p 1
    --request-timeout 13200
    --readiness-timeout 30
    --max-infrastructure-failures 0
)
[[ -z "${SWE_EVAL_LIMIT}" ]] || args+=(--limit "${SWE_EVAL_LIMIT}")
python3 -m experiments.src.evaluators.swe "${args[@]}"
' &
evaluation_step_pid=$!
set +e
wait "${evaluation_step_pid}"
evaluation_status=$?
set -e
evaluation_step_pid=""
exit "${evaluation_status}"
