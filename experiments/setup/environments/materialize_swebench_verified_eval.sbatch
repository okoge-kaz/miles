#!/bin/bash
#SBATCH --job-name=materialize-swebench-verified-hardened
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# Materialize only live-admitted SWE-bench Verified tasks, then bind gold-free
# rows to an immutable, hardened-local Harbor evaluator. Its path policy makes
# this score intentionally non-comparable to the official leaderboard harness.
# This producer never emits training data or reads a repository .env file.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_materialize_swebench_verified_eval.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || exit 2
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || exit 2
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
unset WANDB_API_KEY
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || exit 2
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${SWE_CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=SWE_EVAL_ALLOW_SUBSET,SWE_EVAL_CANDIDATE,SWE_EVAL_INPUT,SWE_EVAL_INPUT_SUMMARY,SWE_EVAL_MATERIALIZATION_EVIDENCE,SWE_EVAL_SEMANTIC_ADMISSION,SWE_EVAL_TASK_IDS,SWE_EVAL_TASK_MANIFEST,SWE_EVAL_TASK_OUTPUT,SWE_MATERIALIZE_DRY_RUN,SWE_MATERIALIZE_LIMIT \
    env -u BASH_ENV -u ENV bash --noprofile --norc -c '
set -euo pipefail
umask 077
cd /root/miles
export PYTHONPATH=/root/miles
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1

data_root=/data/miles-swe
private_root=${data_root}/private
candidate_default=${data_root}/candidates/not-admitted/swe-bench-verified-eval.not-admitted.jsonl
admitted_manifest_default=${private_root}/swe-bench-verified.admitted.tasks.private.jsonl
semantic_default=${private_root}/swe-bench-verified.admission.private.jsonl
tasks_default=${private_root}/harbor-tasks/swe-bench-verified
eval_root=${data_root}/evaluation-input
candidate=${SWE_EVAL_CANDIDATE:-${candidate_default}}
manifest=${SWE_EVAL_TASK_MANIFEST:-${admitted_manifest_default}}
semantic=${SWE_EVAL_SEMANTIC_ADMISSION:-${semantic_default}}
tasks=${SWE_EVAL_TASK_OUTPUT:-${tasks_default}}
evidence=${SWE_EVAL_MATERIALIZATION_EVIDENCE:-${private_root}/swe-bench-verified.materialization-evidence.private.jsonl}
output=${SWE_EVAL_INPUT:-${eval_root}/swebench-verified-hardened-local.jsonl}
summary=${SWE_EVAL_INPUT_SUMMARY:-${eval_root}/swebench-verified-hardened-local.summary.json}
task_ids=${SWE_EVAL_TASK_IDS:-${eval_root}/swebench-verified-hardened-local.task-ids.txt}
dry_run=${SWE_MATERIALIZE_DRY_RUN:-0}
[[ "${dry_run}" == 0 || "${dry_run}" == 1 ]] || {
    echo "SWE_MATERIALIZE_DRY_RUN must be 0 or 1" >&2
    exit 2
}
limit_args=()
if [[ -n "${SWE_MATERIALIZE_LIMIT:-}" ]]; then
    [[ "${SWE_MATERIALIZE_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
        echo "SWE_MATERIALIZE_LIMIT must be a positive integer" >&2
        exit 2
    }
    limit_args=(--limit "${SWE_MATERIALIZE_LIMIT}")
fi
if [[ "${dry_run}" == 0 && ${#limit_args[@]} -ne 0 && "${SWE_EVAL_ALLOW_SUBSET:-0}" != 1 ]]; then
    echo "production SWE_MATERIALIZE_LIMIT requires SWE_EVAL_ALLOW_SUBSET=1" >&2
    exit 2
fi

harness_args=(
    --swebench-harness-root
    /data/miles-swe/verifiers/swebench/SWE-bench
)
if [[ "${dry_run}" == 1 ]]; then
    raw_manifest=${private_root}/swe-bench-verified.tasks.private.jsonl
    dry_manifest=${SWE_EVAL_TASK_MANIFEST:-${raw_manifest}}
    dry_tasks=${SWE_EVAL_TASK_OUTPUT:-${private_root}/harbor-tasks-dry-run/swe-bench-verified}
    [[ -r "${dry_manifest}" ]] || {
        echo "missing Verified dry-run manifest: ${dry_manifest}" >&2
        exit 1
    }
    python3 -m experiments.src.environments.swe.materialize \
        --manifest "${dry_manifest}" \
        --output "${dry_tasks}" \
        --allow-mutable-images \
        --allow-unadmitted-swebench-verified-dry-run \
        "${limit_args[@]}" \
        "${harness_args[@]}"
    echo "inspection-only Verified task tree emitted; no eval input was finalized" >&2
    exit 0
fi

for required in "${candidate}" "${manifest}" "${semantic}"; do
    [[ -r "${required}" ]] || {
        echo "required production Verified input is missing: ${required}" >&2
        exit 1
    }
done
python3 -m experiments.src.environments.swe.materialize \
    --manifest "${manifest}" \
    --output "${tasks}" \
    --admission-evidence "${evidence}" \
    --swebench-verified-admission-manifest "${semantic}" \
    "${limit_args[@]}" \
    "${harness_args[@]}"

finalize_args=(
    --candidate "${candidate}"
    --manifest "${manifest}"
    --materialization-evidence "${evidence}"
    --semantic-admission-manifest "${semantic}"
    --tasks-dir "${tasks}"
    --output "${output}"
    --task-ids-output "${task_ids}"
    --summary "${summary}"
)
if [[ "${SWE_EVAL_ALLOW_SUBSET:-0}" == 1 ]]; then
    finalize_args+=(--allow-subset)
fi
python3 -m experiments.src.environments.swe.finalize_evaluation "${finalize_args[@]}"
'
