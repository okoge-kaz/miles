#!/bin/bash
# Submit resumable DAPO-Math pass-rate measurements for the SFT Qwen3 4B, 8B,
# and 30B-A3B checkpoints. Each model gets one batch_long job by default. Set
# FILTER_ROUNDS>1 to chain retries; every retry resumes the same JSONL and exits
# immediately when the .complete marker already exists.
#
# Dry-run:
#   experiments/tools/difficulty_filter/dapo-math/submit_sft_models.sh
# Submit:
#   FILTER_ROUNDS=4 experiments/tools/difficulty_filter/dapo-math/submit_sft_models.sh --submit

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source experiments/env.sh

SUBMIT=0
[[ "${1:-}" == --submit ]] && SUBMIT=1
[[ $# -eq 0 || ( $# -eq 1 && "${1}" == --submit ) ]] || {
    echo "usage: $0 [--submit]" >&2
    exit 2
}

FILTER_PARTITION="${FILTER_PARTITION:-batch_long}"
FILTER_QOS="${FILTER_QOS:-normal}"
FILTER_WALL="${FILTER_WALL:-7-00:00:00}"
FILTER_ROUNDS="${FILTER_ROUNDS:-1}"
N_SAMPLES="${N_SAMPLES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-32768}"
PROMPT_HOST="${DATASET_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
MANIFEST=experiments/setup/manifests/sft_checkpoints.txt

[[ "${FILTER_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FILTER_ROUNDS must be a positive integer" >&2
    exit 2
}

mkdir -p "${OUTPUT_DIR}/download" "${OUTPUT_DIR}/difficulty"
dataset_dependency=""
if [[ ! -f "${PROMPT_HOST}" ]]; then
    if (( SUBMIT == 0 )); then
        echo "would download zhuzilin/dapo-math-17k -> ${PROMPT_HOST%/*}"
    else
        raw_dataset_job="$(sbatch --parsable \
            -A "${SLURM_ACCOUNT_NAME}" \
            -p "${CPU_PARTITION}" --qos="${INTERACTIVE_CPU_QOS}" \
            --job-name=ds-dapo-math-17k \
            --export=ALL,HF_REPO=zhuzilin/dapo-math-17k,LOCAL_NAME=dapo-math-17k \
            experiments/setup/download/download_dataset.sbatch)"
        dataset_job="${raw_dataset_job%%;*}"
        dataset_dependency="afterok:${dataset_job}"
        echo "submitted dataset job=${dataset_job}"
    fi
fi

manifest_path="${OUTPUT_DIR}/difficulty/dapo-sft-models.jobs"
if (( SUBMIT != 0 )); then
    printf 'SUBMITTED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${manifest_path}"
    printf 'DATASET_JOB=%q\n' "${dataset_job:-}" >> "${manifest_path}"
fi

while IFS='|' read -r model_name hf_root hf_model_name model_type; do
    model_name="$(echo "${model_name}" | xargs)"
    [[ -z "${model_name}" || "${model_name}" == \#* ]] && continue
    hf_root="$(echo "${hf_root}" | xargs)"
    hf_model_name="$(echo "${hf_model_name}" | xargs)"
    model_type="$(echo "${model_type}" | xargs)"
    [[ -f "${hf_root}/${hf_model_name}/config.json" ]] || {
        echo "missing ${hf_root}/${hf_model_name}/config.json" >&2
        exit 1
    }

    slug="$(echo "${model_name}" | tr '[:upper:]' '[:lower:]')"
    pass_rates="/data/difficulty/dapo-math-17k.${model_name}.n${N_SAMPLES}-len${MAX_NEW_TOKENS}-zero-trunc.passrate.jsonl"
    dataset_name="dapo-math-p10-90-${slug}"
    filtered_output="/data/${dataset_name}/${dataset_name}.jsonl"
    annotated_output="/data/${dataset_name}/dapo-math-17k.annotated.jsonl"
    exports="ALL,HF_CKPT_DIR=${hf_root},MODEL_NAME=${model_name},HF_MODEL_NAME=${hf_model_name},PROMPT_DATA=${PROMPT_DATA},OUTPUT=${pass_rates},N_SAMPLES=${N_SAMPLES},SAMPLES_PER_REQUEST=8,MAX_NEW_TOKENS=${MAX_NEW_TOKENS},MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH},CONCURRENCY=160,MAX_RUNNING_REQUESTS=80,RM_TYPE=deepscaler,ZERO_REWARD_ON_TRUNCATED=1,PREFLIGHT=strict,PASS_RATE_MIN=0.1,PASS_RATE_MAX=0.9,FILTERED_OUTPUT=${filtered_output},ANNOTATED_OUTPUT=${annotated_output}"

    if (( SUBMIT == 0 )); then
        printf 'would measure %-41s rounds=%s output=%s\n' \
            "${model_name}" "${FILTER_ROUNDS}" "${pass_rates}"
        continue
    fi

    previous_job=""
    for (( round = 1; round <= FILTER_ROUNDS; round++ )); do
        dependency=()
        if [[ -n "${previous_job}" ]]; then
            dependency=(--dependency="afterany:${previous_job}")
        elif [[ -n "${dataset_dependency}" ]]; then
            dependency=(--dependency="${dataset_dependency}")
        fi
        raw_job_id="$(sbatch --parsable \
            -A "${SLURM_ACCOUNT_NAME}" \
            -p "${FILTER_PARTITION}" --qos="${FILTER_QOS}" \
            --time="${FILTER_WALL}" \
            --job-name="pr-${slug}-r${round}" \
            "${dependency[@]}" \
            --export="${exports}" \
            experiments/tools/difficulty_filter/run_measure.sbatch)"
        previous_job="${raw_job_id%%;*}"
        printf 'submitted %-41s round=%s job=%s dependency=%s\n' \
            "${model_name}" "${round}" "${previous_job}" "${dependency[*]:-none}"
        printf '%s_ROUND_%s_JOB=%q\n' \
            "$(echo "${model_name}" | tr '[:lower:]-.' '[:upper:]__')" \
            "${round}" "${previous_job}" >> "${manifest_path}"
    done
done < "${MANIFEST}"

if (( SUBMIT == 0 )); then
    echo "dry-run only; pass --submit to enqueue the three measurements"
else
    echo "job manifest: ${manifest_path}"
fi
