#!/bin/bash
#SBATCH --job-name=materialize-harbor-swe
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log
#SBATCH --export=NIL

# Materialize only live-admitted SWE tasks, emit exact task-tree evidence, then
# join that evidence into a separate admitted/*-train.jsonl. Inspection-only
# dry runs never emit production evidence or invoke the finalizer. No .env file
# is read by this script or its Python entry points.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER="$(id -un)"
export PATH USER
: "${MILES_SWE_FIXED_EXPORTS:?submit with submit_materialize_harbor_swe_tasks.sh}"
[[ "${MILES_SWE_FIXED_EXPORTS}" == 1 ]] || exit 2
[[ -z "${MILES_SWE_UNRELATED_ENV_SENTINEL+x}" ]] || exit 2
export WANDB_MODE=offline
export WANDB_DISABLED=true
export PYTHON_DOTENV_DISABLED=1
source "${REPO_ROOT}/experiments/env.sh"
DOTENV_MASK="${REPO_ROOT}/experiments/common/dotenv.disabled"
[[ -f "${DOTENV_MASK}" && ! -L "${DOTENV_MASK}" ]] || exit 2
SWE_CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${DOTENV_MASK}:/root/miles/.env:ro"
: "${SWE_TASKSET:?set SWE_TASKSET to an admitted SWE task-set name}"
export SWE_TASKSET

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${SWE_CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=SWE_ADMITTED_OUTPUT,SWE_ADMITTED_SUMMARY,SWE_CANDIDATE_DATASET,SWE_FINALIZE_ALLOW_SUBSET,SWE_MATERIALIZATION_EVIDENCE,SWE_MATERIALIZE_DRY_RUN,SWE_MATERIALIZE_LIMIT,SWE_R2E_ADMITTED_MANIFEST,SWE_SEMANTIC_ADMISSION_MANIFEST,SWE_TASKSET,SWE_TASK_IDS_OUTPUT,SWE_TASK_MANIFEST,SWE_TASK_OUTPUT \
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
candidate_root=${data_root}/candidates/not-admitted
admitted_root=${data_root}/admitted
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
if [[ "${dry_run}" == 0 && ${#limit_args[@]} -ne 0 && "${SWE_FINALIZE_ALLOW_SUBSET:-0}" != 1 ]]; then
    echo "production SWE_MATERIALIZE_LIMIT requires SWE_FINALIZE_ALLOW_SUBSET=1" >&2
    exit 2
fi

case "${SWE_TASKSET}" in
    swe-rebench-v2|swe-rebench-v2-filtered-verified|nemotron3-ultra-swe-rebench-v2)
        raw_manifest="${private_root}/${SWE_TASKSET}.tasks.private.jsonl"
        production_manifest="${private_root}/${SWE_TASKSET}.admitted.tasks.private.jsonl"
        semantic_manifest="${SWE_SEMANTIC_ADMISSION_MANIFEST:-${private_root}/${SWE_TASKSET}.admission.private.jsonl}"
        candidate="${SWE_CANDIDATE_DATASET:-${candidate_root}/${SWE_TASKSET}-train.not-admitted.jsonl}"
        environment_args=(
            --swe-rebench-log-parsers
            /data/miles-swe/verifiers/swe-rebench-v2/lib/agent/log_parsers.py
            --swe-rebench-constants
            /data/miles-swe/verifiers/swe-rebench-v2/lib/agent/swe_constants.py
            --swe-rebench-eval
            /data/miles-swe/verifiers/swe-rebench-v2/eval.py
        )
        semantic_args=(--swe-rebench-admission-manifest "${semantic_manifest}")
        dry_args=(--allow-unadmitted-swe-rebench-dry-run)
        dry_manifest_default="${raw_manifest}"
        ;;
    swe-gym|nemotron3-super-swe2-swe-gym|nemotron3-ultra-swe-gym)
        raw_manifest="${private_root}/${SWE_TASKSET}.tasks.private.jsonl"
        production_manifest="${private_root}/${SWE_TASKSET}.admitted.tasks.private.jsonl"
        semantic_manifest="${SWE_SEMANTIC_ADMISSION_MANIFEST:-${private_root}/${SWE_TASKSET}.admission.private.jsonl}"
        candidate="${SWE_CANDIDATE_DATASET:-${private_root}/unadmitted/${SWE_TASKSET}-train.gold-free.jsonl}"
        environment_args=(
            --swe-gym-harness-root
            /data/miles-swe/verifiers/swe-gym/SWE-Bench-Package
        )
        semantic_args=(--swe-gym-admission-manifest "${semantic_manifest}")
        dry_args=(--allow-unadmitted-swe-gym-dry-run)
        dry_manifest_default="${raw_manifest}"
        ;;
    r2e-gym-v1|nemotron3-super-swe2-r2e)
        raw_manifest="${private_root}/${SWE_TASKSET}.tasks.private.jsonl"
        production_manifest="${SWE_R2E_ADMITTED_MANIFEST:-${private_root}/${SWE_TASKSET}.admitted.tasks.private.jsonl}"
        semantic_manifest="${SWE_SEMANTIC_ADMISSION_MANIFEST:-${private_root}/${SWE_TASKSET}.admission.private.jsonl}"
        candidate="${SWE_CANDIDATE_DATASET:-${private_root}/unadmitted/${SWE_TASKSET}-train.gold-free.jsonl}"
        environment_args=(
            --r2e-execution-log-parser
            /data/miles-swe/verifiers/r2e-gym/execution_log_parser.py
        )
        semantic_args=(--r2e-admission-manifest "${semantic_manifest}")
        dry_args=(--allow-unadmitted-r2e-dry-run)
        # Raw R2E rows intentionally have oracle_patch=null. Only live admission
        # derives the gold-parent patch, so even inspection materialization must
        # consume the admitted private manifest rather than pretending raw rows
        # are materializable.
        dry_manifest_default="${production_manifest}"
        ;;
    *)
        echo "SWE_TASKSET=${SWE_TASKSET} has no production semantic materializer" >&2
        exit 2
        ;;
esac

if [[ "${dry_run}" == 1 ]]; then
    manifest="${SWE_TASK_MANIFEST:-${dry_manifest_default}}"
    [[ -r "${manifest}" ]] || {
        echo "dry materialization input is missing (R2E requires live-derived admitted rows): ${manifest}" >&2
        exit 1
    }
    output="${SWE_TASK_OUTPUT:-${private_root}/harbor-tasks-dry-run/${SWE_TASKSET}}"
    args=(
        --manifest "${manifest}"
        --output "${output}"
        --allow-mutable-images
        "${limit_args[@]}"
        "${dry_args[@]}"
        "${environment_args[@]}"
    )
    echo "WARNING: inspection-only SWE materialization; no production evidence/final dataset will be emitted" >&2
    python3 -m experiments.src.environments.swe.materialize "${args[@]}"
    exit 0
fi

manifest="${SWE_TASK_MANIFEST:-${production_manifest}}"
output="${SWE_TASK_OUTPUT:-${private_root}/harbor-tasks/${SWE_TASKSET}}"
evidence="${SWE_MATERIALIZATION_EVIDENCE:-${private_root}/${SWE_TASKSET}.materialization-evidence.private.jsonl}"
admitted_output="${SWE_ADMITTED_OUTPUT:-${admitted_root}/${SWE_TASKSET}-train.jsonl}"
admitted_summary="${SWE_ADMITTED_SUMMARY:-${admitted_root}/${SWE_TASKSET}-train.summary.json}"
task_ids_output="${SWE_TASK_IDS_OUTPUT:-${admitted_root}/${SWE_TASKSET}-train.task-ids.txt}"

for required in "${manifest}" "${semantic_manifest}" "${candidate}"; do
    [[ -r "${required}" ]] || {
        echo "required production SWE input is missing: ${required}" >&2
        exit 1
    }
done

materialize_args=(
    --manifest "${manifest}"
    --output "${output}"
    --admission-evidence "${evidence}"
    "${limit_args[@]}"
    "${environment_args[@]}"
    "${semantic_args[@]}"
)
python3 -m experiments.src.environments.swe.materialize "${materialize_args[@]}"

finalize_args=(
    --candidate "${candidate}"
    --manifest "${manifest}"
    --materialization-evidence "${evidence}"
    --semantic-admission-manifest "${semantic_manifest}"
    --tasks-dir "${output}"
    --output "${admitted_output}"
    --task-ids-output "${task_ids_output}"
    --summary "${admitted_summary}"
)
if [[ "${SWE_FINALIZE_ALLOW_SUBSET:-0}" == 1 ]]; then
    finalize_args+=(--allow-subset)
fi
python3 -m experiments.src.environments.swe.finalize_admitted "${finalize_args[@]}"
'
