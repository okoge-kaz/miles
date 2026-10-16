#!/bin/bash

# Sourced by <model-name>/<dataset>.sh; policy-specific defaults stay in that recipe.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MILES_REPO="${REPO_ROOT}"
source "${REPO_ROOT}/experiments/common/pbs.sh"
source "${REPO_ROOT}/experiments/common/job_env.sh"

if [[ -v FILTER_DP_SIZE || -v FILTER_SAMPLES_PER_REQUEST ]]; then
    echo 'Use FILTER_NUM_ENGINES; RL sampling uses DP1 engines and one response per request.' >&2
    exit 2
fi

mode="${1:-filter}"
(( $# == 0 )) || shift
case "${mode}" in
    smoke|filter|merge) ;;
    *) echo 'usage: <model>/<dataset>.sh [smoke|filter|merge] [--submit] [--shard-index N] [--no-merge] [--time HH:MM:SS] [--project GROUP]' >&2; exit 2 ;;
esac
submit=0
selected_shard=""
auto_merge=1
prepare_only=0
base_dependency=""
: "${FILTER_MAX_PARALLEL:=8}"
while (( $# )); do
    case "$1" in
        --submit) submit=1; shift ;;
        --no-merge) auto_merge=0; shift ;;
        --prepare-only) prepare_only=1; shift ;;
        --dependency)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            base_dependency="$2"; shift 2 ;;
        --shard-index)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            selected_shard="$2"; auto_merge=0; shift 2 ;;
        --time|--project)
            (( $# >= 2 )) || { echo "$1 needs a value" >&2; exit 2; }
            if [[ "$1" == --time ]]; then FILTER_WALLTIME="$2"; else export PBS_PROJECT="$2"; fi
            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "${FILTER_MAX_PARALLEL}" =~ ^[1-8]$ ]] || { echo 'FILTER_MAX_PARALLEL must be 1..8' >&2; exit 2; }
[[ -z "${base_dependency}" || "${base_dependency}" =~ ^afterok:[0-9]+(\.[A-Za-z0-9._-]+)?(:[0-9]+(\.[A-Za-z0-9._-]+)?)*$ ]] || {
    echo '--dependency must be afterok:<job ID>[:<job ID>...]' >&2; exit 2;
}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
[[ "${FILTER_HF_MODEL}" != /* && "/${FILTER_HF_MODEL}/" != */../* ]] || {
    echo 'FILTER_HF_MODEL must be relative to HF_CKPT_DIR, without ..' >&2; exit 2;
}
for path in "${FILTER_INPUT}" "${FILTER_MEASUREMENT_DIR}" "${FILTER_OUTPUT}"; do
    [[ "${path}" == /data/* && "${path}" != */../* && "${path}" != *[,:]* ]] || {
        echo 'Filter data paths must be under /data without .., commas or colons' >&2; exit 2;
    }
done
case "${FILTER_ZERO_REWARD_ON_TRUNCATED}" in
    1) truncation_arg=--zero-reward-on-truncated ;;
    0) truncation_arg=--no-zero-reward-on-truncated ;;
    *) echo 'FILTER_ZERO_REWARD_ON_TRUNCATED must be 0 or 1' >&2; exit 2 ;;
esac
config_args=(
    --model-path "/ckpt/hf/${FILTER_HF_MODEL}" --policy "${FILTER_POLICY}"
    --prompt-data "${FILTER_INPUT}" --measurement-dir "${FILTER_MEASUREMENT_DIR}"
    --filtered-output "${FILTER_OUTPUT}"
    --n-samples "${FILTER_N_SAMPLES}"
    --max-new-tokens "${FILTER_MAX_NEW_TOKENS}" --max-context-length "${FILTER_MAX_CONTEXT_LENGTH}"
    --temperature "${FILTER_TEMPERATURE}" --top-p "${FILTER_TOP_P}" --top-k "${FILTER_TOP_K}"
    --seed "${FILTER_SEED}" --correct-threshold "${FILTER_CORRECT_THRESHOLD}" "${truncation_arg}"
    --num-shards "${FILTER_NUM_SHARDS}" --tensor-parallel-size "${FILTER_TP_SIZE}"
    --num-engines "${FILTER_NUM_ENGINES}" --concurrency "${FILTER_CONCURRENCY}"
    --max-running-requests "${FILTER_MAX_RUNNING_REQUESTS}" --mem-fraction-static "${FILTER_MEM_FRACTION_STATIC}"
    --server-port "${FILTER_SERVER_PORT}" --startup-timeout "${FILTER_STARTUP_TIMEOUT}"
    --request-timeout "${FILTER_REQUEST_TIMEOUT}" --audit-prompts "${FILTER_AUDIT_PROMPTS}"
    --reward-backend "${FILTER_REWARD_BACKEND:-deepscaler}" --selection-rule "${FILTER_SELECTION_RULE:-mixed_correctness}"
    --generation-attempts "${FILTER_GENERATION_ATTEMPTS:-5}"
    --progress-every "${FILTER_PROGRESS_EVERY:-10}" --progress-seconds "${FILTER_PROGRESS_SECONDS:-60}"
    --code-concurrency "${FILTER_CODE_CONCURRENCY:-16}"
)
config_json="$(python3 -m experiments.src.difficulty_filtering.core.config "${config_args[@]}")"
if [[ -n "${selected_shard}" ]]; then
    [[ "${mode}" == filter && "${selected_shard}" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo '--shard-index requires filter mode and a nonnegative integer' >&2; exit 2;
    }
    (( selected_shard < FILTER_NUM_SHARDS )) || { echo 'Shard index is out of range' >&2; exit 2; }
fi
[[ "${mode}" == filter || "${auto_merge}" == 1 ]] || {
    echo '--no-merge and --shard-index are only valid in filter mode' >&2; exit 2;
}
(( FILTER_TP_SIZE * FILTER_NUM_ENGINES == PBS_GPU_GPUS_PER_NODE )) || {
    echo 'FILTER_TP_SIZE * FILTER_NUM_ENGINES must equal PBS_GPU_GPUS_PER_NODE' >&2; exit 2;
}
[[ "${FILTER_SMOKE_PROMPTS}" =~ ^[1-9][0-9]*$ ]] || { echo 'FILTER_SMOKE_PROMPTS must be positive' >&2; exit 2; }

run_name="${FILTER_RUN_NAME:-${FILTER_POLICY}-${FILTER_REWARD_BACKEND:-deepscaler}-${mode}-$(date +%Y%m%d-%H%M%S)-p$$}"
[[ "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid FILTER_RUN_NAME' >&2; exit 2; }
run_dir="${REPO_ROOT}/experiments/outputs/difficulty_filtering/${run_name}"
config_path="/root/miles/experiments/outputs/difficulty_filtering/${run_name}/config.json"
job_script="${REPO_ROOT}/experiments/scripts/difficulty_filtering/run.sh"
merge_script="${REPO_ROOT}/experiments/scripts/difficulty_filtering/merge.sh"

if (( submit )); then
    workspace="${MILES_WORKSPACE_ROOT:-/groups/gai51740/kazuki_fujii}"
    hf_root="${HF_CKPT_DIR:-${CHECKPOINT_ROOT:-${CKPT_ROOT:-${workspace}/checkpoints}}/hf}"
    data_root="${RL_DATASET_DIR:-${DATASET_DIR:-${DATASET_ROOT:-${workspace}/datasets}/rl}}"
    container_image="${CONTAINER_IMAGE:-${CONTAINER_DIR:-${workspace}/containers}/miles-v0.1.2.sif}"
    [[ -r "${hf_root}/${FILTER_HF_MODEL}/config.json" && -r "${data_root}/${FILTER_INPUT#/data/}" && -r "${container_image}" ]] || {
        echo 'Checkpoint, source JSONL, or container image is missing' >&2; exit 2;
    }
    container_image="$(readlink -f -- "${container_image}")"
    [[ ! -e "${run_dir}/config.json" && ! -e "${run_dir}/jobs.tsv" ]] || {
        echo 'Run name already submitted; use a new FILTER_RUN_NAME' >&2; exit 2;
    }
    mkdir -p "${run_dir}"
    validation_options=()
    [[ -z "${FILTER_INPUT_VALIDATION_FILE:-}" ]] || validation_options+=(--cache "${FILTER_INPUT_VALIDATION_FILE}")
    python3 -m experiments.src.difficulty_filtering.utils.preflight \
        --input "${data_root}/${FILTER_INPUT#/data/}" --backend "${FILTER_REWARD_BACKEND:-deepscaler}" \
        --output "${run_dir}/input_validation.json" "${validation_options[@]}"
    printf '%s\n' "${config_json}" > "${run_dir}/config.json"
    python3 -m experiments.src.difficulty_filtering.utils.snapshot "${run_dir}/source"
    python3 -c 'import json, sys; print(json.dumps(dict(image=sys.argv[1], walltime=sys.argv[2], smoke_prompts=int(sys.argv[3]))))' \
        "${container_image}" "${FILTER_WALLTIME}" "${FILTER_SMOKE_PROMPTS}" > "${run_dir}/launch.json"
    # PBS executes this copy, so editing the common script cannot change a running Bash reader.
    cp -- "${job_script}" "${run_dir}/run.sh"
    cp -- "${merge_script}" "${run_dir}/merge.sh"
    job_script="${run_dir}/run.sh"
    merge_script="${run_dir}/merge.sh"
fi
printf '%s\n' "${config_json}"
if (( prepare_only )); then
    (( submit )) || { echo '--prepare-only requires --submit to write the snapshot' >&2; exit 2; }
    exit 0
fi

submit_filter_job() {
    local action="$1" shard="$2" dependency="${3:-}"
    local profile=gpu walltime="${FILTER_WALLTIME}"
    local script="${job_script}"
    local -a options=()
    if [[ "${action}" == merge ]]; then profile=cpu; walltime=00:30:00; script="${merge_script}"; fi
    (( submit )) || options+=(--test-only)
    [[ -z "${dependency}" ]] || options+=(--dependency="${dependency}")
    pbs_submit --parsable --profile="${profile}" --nodes=1 --job-kind=prep \
        --time="${walltime}" --chdir="${REPO_ROOT}" --job-name="df-${action}-${shard}" \
        --output="${run_dir}/${action}-${shard}.ou" --export="$(miles_job_export_spec)" \
        "${options[@]}" "${script}" "${config_path}" "${action}" "${shard}" "${FILTER_SMOKE_PROMPTS}"
}

job_ids=()
if [[ "${mode}" != merge ]]; then
    first_shard=0
    shard_count="${FILTER_NUM_SHARDS}"
    [[ "${mode}" != smoke ]] || shard_count=1
    if [[ -n "${selected_shard}" ]]; then
        first_shard="${selected_shard}"
        shard_count=$(( selected_shard + 1 ))
    fi
    for (( shard = first_shard; shard < shard_count; shard++ )); do
        dependency="${base_dependency}"
        lane=$(( ${#job_ids[@]} - FILTER_MAX_PARALLEL ))
        if (( lane >= 0 )); then
            dependency="${base_dependency:+${base_dependency},}afterany:${job_ids[lane]}"
        fi
        result="$(submit_filter_job "${mode}" "${shard}" "${dependency}")"
        echo "${result}"
        if (( submit )); then
            job_ids+=("${result}")
            printf '%s\t%s\t%s\n' "${mode}" "${shard}" "${result}" >> "${run_dir}/jobs.tsv"
        else
            job_ids+=("$(( 999900 + shard ))")
        fi
    done
fi
if [[ "${mode}" == merge || ( "${mode}" == filter && "${auto_merge}" == 1 ) ]]; then
    dependency="${base_dependency}"
    if (( ${#job_ids[@]} )); then
        dependency="afterany:$(IFS=:; echo "${job_ids[*]}")"
    fi
    result="$(submit_filter_job merge 0 "${dependency}")"
    echo "${result}"
    (( ! submit )) || printf 'merge\t0\t%s\n' "${result}" >> "${run_dir}/jobs.tsv"
fi
if [[ "${mode}" == filter && "${auto_merge}" == 0 ]]; then
    echo 'Merge separately after every shard has completed: <model>/<dataset>.sh merge --submit'
fi
if (( submit )); then
    echo "Submission record: ${run_dir}/jobs.tsv"
else
    echo 'Preview only; dependency IDs 999900+ are placeholders. Add --submit to enqueue.'
fi
