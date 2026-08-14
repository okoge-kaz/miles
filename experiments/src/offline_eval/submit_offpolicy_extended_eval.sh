#!/bin/bash
# Evaluate every checkpoint in the fixed 75-checkpoint snapshot on OlympiadBench and AIME avg@32.

set -euo pipefail
shopt -s nullglob

SCRIPT_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${SCRIPT_REPO_ROOT}/experiments/env.sh"

SOURCE_ROOT="${SOURCE_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training/math/dapo-math-p10-90/Qwen3-4B-Instruct-2507/grpo-clip0.2-0.28-tis2.0/async/off-policy}"
SWEEP_NAME="${SWEEP_NAME:-hiso-offpolicy-20260813}"
MODE="${1:-check}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-01:30:00}"
HOST_RESULT_ROOT="${DATASET_DIR}/offline_eval/${SWEEP_NAME}"
HOST_OLYMPIAD="${DATASET_DIR}/olympiadbench/OE_TO_maths_en_COMP.jsonl"
HOST_OLYMPIAD_PYTHONPATH="${DATASET_DIR}/python/antlr4-python3-runtime-4.11.1"
CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-${SCRIPT_REPO_ROOT}/experiments/src/offline_eval/hiso_offpolicy_20260813_checkpoints.txt}"
CONTAINER_SOURCE=/source/off-policy
CONTAINER_RESULT_ROOT="/data/offline_eval/${SWEEP_NAME}"
OLYMPIAD_SPEC=olympiadbench:/data/olympiadbench/OE_TO_maths_en_COMP.jsonl
AIME_EXTRA_SPEC=aime24_extra16:/data/aime-2024/aime-2024.jsonl+aime25_extra16:/data/aime-2025/aime-2025.jsonl+aime26_extra16:/data/aime-2026/aime-2026.jsonl
AIME_FULL_SPEC=aime24:/data/aime-2024/aime-2024.jsonl+aime25:/data/aime-2025/aime-2025.jsonl+aime26:/data/aime-2026/aime-2026.jsonl
OLYMPIAD_RM=experiments.src.offline_eval.olympiadbench_reward.score_olympiadbench
OLYMPIAD_PYTHONPATH=/data/python/antlr4-python3-runtime-4.11.1
OLYMPIAD_SHA256=c3d747eff4ac633eb079e2fd8a27268376509b068632b63dda5af34bc4ba2870

[[ -f "${HOST_OLYMPIAD}" ]] || { echo "missing OlympiadBench dataset: ${HOST_OLYMPIAD}" >&2; exit 1; }
[[ $(wc -l < "${HOST_OLYMPIAD}") -eq 674 ]] || {
    echo "OlympiadBench dataset must contain exactly 674 rows" >&2
    exit 1
}
[[ $(sha256sum "${HOST_OLYMPIAD}" | awk '{print $1}') == "${OLYMPIAD_SHA256}" ]] || {
    echo "OlympiadBench dataset checksum does not match the validated conversion" >&2
    exit 1
}
[[ -r "${HOST_OLYMPIAD_PYTHONPATH}/antlr4/__init__.py" ]] || {
    echo "missing antlr4-python3-runtime 4.11.1 at ${HOST_OLYMPIAD_PYTHONPATH}" >&2
    exit 1
}
[[ -r "${CHECKPOINT_MANIFEST}" ]] || {
    echo "missing checkpoint manifest: ${CHECKPOINT_MANIFEST}" >&2
    exit 1
}

complete_file() {
    local path="$1"
    local rows="$2"
    [[ -f "${path}" ]] && [[ $(wc -l < "${path}") -eq "${rows}" ]]
}

checkpoint_readable() {
    local checkpoint="$1"
    local path
    local model_files=("${checkpoint}"/*.safetensors)
    [[ -r "${checkpoint}/config.json" ]] && [[ ${#model_files[@]} -gt 0 ]] || return 1
    for path in "${model_files[@]}" "${checkpoint}"/*.json; do
        [[ -r "${path}" ]] || return 1
    done
}

is_base_aime_complete() {
    local directory="$1"
    complete_file "${directory}/aime24.jsonl" 30 &&
        complete_file "${directory}/aime25.jsonl" 30 &&
        complete_file "${directory}/aime26.jsonl" 30
}

is_aime_extra_complete() {
    local directory="$1"
    complete_file "${directory}/aime24_extra16.jsonl" 30 &&
        complete_file "${directory}/aime25_extra16.jsonl" 30 &&
        complete_file "${directory}/aime26_extra16.jsonl" 30
}

is_aime_full32_complete() {
    local directory="$1"
    local year
    for year in 24 25 26; do
        complete_file "${directory}/aime${year}.jsonl" 30 || return 1
        [[ -f "${directory}/aime${year}.meta.json" ]] || return 1
        grep -Eq '"n_samples": 32([,[:space:]])' "${directory}/aime${year}.meta.json" || return 1
    done
}

is_aime32_complete() {
    local directory="$1"
    is_aime_full32_complete "${directory}" || (
        is_base_aime_complete "${directory}" && is_aime_extra_complete "${directory}"
    )
}

checkpoints=()
while IFS= read -r relative; do
    [[ -n "${relative}" ]] || continue
    checkpoint="${SOURCE_ROOT}/${relative}"
    [[ -f "${checkpoint}/config.json" ]] || {
        echo "manifest checkpoint is missing config.json: ${checkpoint}" >&2
        exit 1
    }
    checkpoints+=("${checkpoint}")
done < "${CHECKPOINT_MANIFEST}"
if [[ $(printf '%s\n' "${checkpoints[@]}" | sort -u | wc -l) -ne 75 ]]; then
    echo "checkpoint manifest contains duplicate entries" >&2
    exit 1
fi
[[ ${#checkpoints[@]} -eq 75 ]] || {
    echo "expected the fixed 75-checkpoint snapshot, found ${#checkpoints[@]}" >&2
    exit 1
}
printf 'selected all %d checkpoints in the fixed snapshot\n' "${#checkpoints[@]}" >&2

declare -A active_jobs=()

load_active_jobs() {
    local job_id job_name
    while IFS='|' read -r job_id job_name; do
        [[ -n "${job_name}" ]] && active_jobs["${job_name}"]="${job_id}"
    done < <(squeue -h -u "${USER}" -o '%i|%j')
}

submit_one() {
    local kind="$1"
    local checkpoint="$2"
    local relative setting step safe_tag out_dir job_id job_name
    relative="${checkpoint#"${SOURCE_ROOT}/"}"
    setting="${relative%/hf/*}"
    step="${relative##*/}"
    safe_tag="${setting//\//_}-step${step}"
    out_dir="${CONTAINER_RESULT_ROOT}/${setting}/step-${step}"

    if [[ "${kind}" == olympiad ]]; then
        job_name="ob-${safe_tag}"
        job_id="${active_jobs["${job_name}"]:-}"
        if [[ -n "${job_id}" ]]; then
            printf '%s\t%s\t%s\t%s\tqueued\n' "${job_id}" "${kind}" "${setting}" "${step}"
            return
        fi
        job_id=$(sbatch --parsable \
            -A "${SLURM_ACCOUNT_NAME}" \
            --partition=batch,batch_short \
            --time="${EVAL_TIME_LIMIT}" \
            --job-name="ob-${safe_tag}" \
            --export="ALL,CKPT=${CONTAINER_SOURCE}/${relative},TAG=${safe_tag},OUT_DIR=${out_dir},EXTRA_MOUNTS=${SOURCE_ROOT}:${CONTAINER_SOURCE},UNPAD_VOCAB=1,BENCHMARKS=${OLYMPIAD_SPEC},N_SAMPLES=8,RM_TYPE=olympiadbench,CUSTOM_RM_PATH=${OLYMPIAD_RM},EXTRA_PYTHONPATH=${OLYMPIAD_PYTHONPATH}" \
            experiments/src/offline_eval/run_eval.sbatch)
        active_jobs["${job_name}"]="${job_id}"
    else
        job_name="a32-${safe_tag}"
        job_id="${active_jobs["${job_name}"]:-}"
        if [[ -n "${job_id}" ]]; then
            printf '%s\t%s\t%s\t%s\tqueued\n' "${job_id}" "${kind}" "${setting}" "${step}"
            return
        fi
        if is_base_aime_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}"; then
            job_id=$(sbatch --parsable \
                -A "${SLURM_ACCOUNT_NAME}" \
                --partition=batch,batch_short \
                --time="${EVAL_TIME_LIMIT}" \
                --job-name="a32-${safe_tag}" \
                --export="ALL,CKPT=${CONTAINER_SOURCE}/${relative},TAG=${safe_tag},OUT_DIR=${out_dir},EXTRA_MOUNTS=${SOURCE_ROOT}:${CONTAINER_SOURCE},UNPAD_VOCAB=1,BENCHMARKS=${AIME_EXTRA_SPEC},N_SAMPLES=16,RM_TYPE=math" \
                experiments/src/offline_eval/run_eval.sbatch)
        else
            job_id=$(sbatch --parsable \
                -A "${SLURM_ACCOUNT_NAME}" \
                --partition=batch,batch_short \
                --time="${EVAL_TIME_LIMIT}" \
                --job-name="a32-${safe_tag}" \
                --export="ALL,CKPT=${CONTAINER_SOURCE}/${relative},TAG=${safe_tag},OUT_DIR=${out_dir},EXTRA_MOUNTS=${SOURCE_ROOT}:${CONTAINER_SOURCE},UNPAD_VOCAB=1,BENCHMARKS=${AIME_FULL_SPEC},N_SAMPLES=32,RM_TYPE=math" \
                experiments/src/offline_eval/run_eval.sbatch)
        fi
        active_jobs["${job_name}"]="${job_id}"
    fi
    printf '%s\t%s\t%s\t%s\tsubmitted\n' "${job_id}" "${kind}" "${setting}" "${step}"
}

is_complete() {
    local kind="$1"
    local directory="$2"
    if [[ "${kind}" == olympiad ]]; then
        complete_file "${directory}/olympiadbench.jsonl" 674
    else
        is_aime32_complete "${directory}"
    fi
}

run_probe() {
    local kind="$1"
    local checkpoint relative setting step directory
    load_active_jobs
    for checkpoint in "${checkpoints[@]}"; do
        if ! checkpoint_readable "${checkpoint}"; then
            printf 'SKIP unreadable checkpoint: %s\n' "${checkpoint}" >&2
            continue
        fi
        relative="${checkpoint#"${SOURCE_ROOT}/"}"
        setting="${relative%/hf/*}"
        step="${relative##*/}"
        directory="${HOST_RESULT_ROOT}/${setting}/step-${step}"
        if ! is_complete "${kind}" "${directory}"; then
            printf 'job_id\tkind\tsetting\tstep\tstatus\n'
            submit_one "${kind}" "${checkpoint}"
            return
        fi
    done
    echo "all ${kind} evaluations are already complete"
}

run_all() {
    local kind="$1"
    local checkpoint relative setting step directory
    load_active_jobs
    checkpoint="${checkpoints[0]}"
    relative="${checkpoint#"${SOURCE_ROOT}/"}"
    setting="${relative%/hf/*}"
    step="${relative##*/}"
    directory="${HOST_RESULT_ROOT}/${setting}/step-${step}"
    is_complete "${kind}" "${directory}" || {
        echo "refusing to fan out before the ${kind} probe completes; run '$0 probe-${kind}' first" >&2
        exit 1
    }
    printf 'job_id\tkind\tsetting\tstep\tstatus\n'
    for checkpoint in "${checkpoints[@]}"; do
        if ! checkpoint_readable "${checkpoint}"; then
            printf 'SKIP unreadable checkpoint: %s\n' "${checkpoint}" >&2
            continue
        fi
        relative="${checkpoint#"${SOURCE_ROOT}/"}"
        setting="${relative%/hf/*}"
        step="${relative##*/}"
        directory="${HOST_RESULT_ROOT}/${setting}/step-${step}"
        is_complete "${kind}" "${directory}" || submit_one "${kind}" "${checkpoint}"
    done
}

case "${MODE}" in
    check)
        printf 'setting\tstep\tcheckpoint\treadable\tolympiad\taime32\n'
        for checkpoint in "${checkpoints[@]}"; do
            relative="${checkpoint#"${SOURCE_ROOT}/"}"
            setting="${relative%/hf/*}"
            step="${relative##*/}"
            directory="${HOST_RESULT_ROOT}/${setting}/step-${step}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${setting}" "${step}" "${checkpoint}" \
                "$(checkpoint_readable "${checkpoint}" && echo yes || echo no)" \
                "$(is_complete olympiad "${directory}" && echo complete || echo pending)" \
                "$(is_complete aime32 "${directory}" && echo complete || echo pending)"
        done
        ;;
    probe-olympiad) run_probe olympiad ;;
    all-olympiad) run_all olympiad ;;
    probe-aime32) run_probe aime32 ;;
    all-aime32) run_all aime32 ;;
    *)
        echo "usage: $0 {check|probe-olympiad|all-olympiad|probe-aime32|all-aime32}" >&2
        exit 2
        ;;
esac
