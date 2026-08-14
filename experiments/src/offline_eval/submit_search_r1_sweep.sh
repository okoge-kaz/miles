#!/bin/bash
# Discover and submit Search-R1 HF checkpoints with the same probe-before-fanout
# discipline as the AIME offline-eval sweep.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

SOURCE_ROOT="${SOURCE_ROOT:-${TRAIN_CKPT_DIR}/search_r1/search-r1-nq-hotpotqa/Qwen3-4B-Instruct-2507}"
SWEEP_NAME="${SWEEP_NAME:-search-r1-$(date -u +%Y%m%d)}"
MODE="${1:-check}"
HOST_RESULT_ROOT="${DATASET_DIR}/offline_eval/search_r1/${SWEEP_NAME}"
CONTAINER_SOURCE=/source/search-r1
CONTAINER_RESULT_ROOT="/data/offline_eval/search_r1/${SWEEP_NAME}"

mapfile -t checkpoints < <(
    find "${SOURCE_ROOT}" -path '*/hf/*' -name config.json -printf '%h\n' | sort -V
)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "no HF checkpoints found under ${SOURCE_ROOT}" >&2
    exit 1
fi
printf 'discovered %d HF checkpoints\n' "${#checkpoints[@]}" >&2

unreadable=()
for checkpoint in "${checkpoints[@]}"; do
    while IFS= read -r shard; do
        [[ -r "${shard}" ]] || unreadable+=("${shard}")
    done < <(find "${checkpoint}" -maxdepth 1 -name '*.safetensors' -type f -print)
done
if [[ ${#unreadable[@]} -ne 0 ]]; then
    printf 'cannot read %d safetensors shards; first unreadable files:\n' "${#unreadable[@]}" >&2
    printf '  %s\n' "${unreadable[@]:0:5}" >&2
    exit 1
fi

is_complete() {
    local directory="$1"
    local name expected
    for name in nq hotpotqa triviaqa popqa 2wikimultihopqa musique; do
        expected=500
        [[ -f "${directory}/${name}.jsonl" ]] || return 1
        [[ $(wc -l < "${directory}/${name}.jsonl") -eq ${expected} ]] || return 1
    done
    name=bamboogle
    expected=125
    [[ -f "${directory}/${name}.jsonl" ]] || return 1
    [[ $(wc -l < "${directory}/${name}.jsonl") -eq ${expected} ]] || return 1
}

describe() {
    local checkpoint="$1"
    local relative setting step
    relative="${checkpoint#"${SOURCE_ROOT}/"}"
    setting="${relative%/hf/*}"
    step="${relative##*/}"
    printf '%s\t%s\t%s\n' "${setting}" "${step}" "${checkpoint}"
}

submit_one() {
    local checkpoint="$1"
    local relative setting step safe_tag job_hash out_dir queued_id job_id
    relative="${checkpoint#"${SOURCE_ROOT}/"}"
    setting="${relative%/hf/*}"
    step="${relative##*/}"
    safe_tag="${setting//\//_}-step${step}"
    job_hash=$(printf '%s' "${safe_tag}" | md5sum | cut -c1-8)
    out_dir="${CONTAINER_RESULT_ROOT}/${setting}/step-${step}"

    queued_id=$(squeue -h -n "sreval-${step}-${job_hash}" -o '%i' | head -1)
    if [[ -n "${queued_id}" ]]; then
        printf '%s\t%s\t%s\tqueued\n' "${queued_id}" "${setting}" "${step}"
        return
    fi

    job_id=$(sbatch --parsable \
        -A "${SLURM_ACCOUNT_NAME}" \
        --partition=batch,batch_short \
        --job-name="sreval-${step}-${job_hash}" \
        --export="ALL,CKPT=${CONTAINER_SOURCE}/${relative},TAG=${safe_tag},OUT_DIR=${out_dir},EXTRA_MOUNTS=${SOURCE_ROOT}:${CONTAINER_SOURCE},UNPAD_VOCAB=1" \
        experiments/src/offline_eval/run_search_r1_eval.sbatch)
    printf '%s\t%s\t%s\tsubmitted\n' "${job_id}" "${setting}" "${step}"
}

case "${MODE}" in
    check)
        printf 'setting\tstep\tcheckpoint\n'
        for checkpoint in "${checkpoints[@]}"; do
            describe "${checkpoint}"
        done
        ;;
    probe)
        printf 'job_id\tsetting\tstep\tstatus\n'
        for checkpoint in "${checkpoints[@]}"; do
            relative="${checkpoint#"${SOURCE_ROOT}/"}"
            setting="${relative%/hf/*}"
            step="${relative##*/}"
            if ! is_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}"; then
                submit_one "${checkpoint}"
                exit 0
            fi
        done
        echo "all checkpoints are already complete"
        ;;
    all)
        probe="${checkpoints[0]}"
        relative="${probe#"${SOURCE_ROOT}/"}"
        setting="${relative%/hf/*}"
        step="${relative##*/}"
        if ! is_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}"; then
            echo "refusing to fan out before the first checkpoint completes; run '$0 probe' first" >&2
            exit 1
        fi
        printf 'job_id\tsetting\tstep\tstatus\n'
        for checkpoint in "${checkpoints[@]}"; do
            relative="${checkpoint#"${SOURCE_ROOT}/"}"
            setting="${relative%/hf/*}"
            step="${relative##*/}"
            is_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}" || submit_one "${checkpoint}"
        done
        ;;
    *)
        echo "usage: $0 {check|probe|all}" >&2
        exit 2
        ;;
esac
