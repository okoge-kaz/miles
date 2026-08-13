#!/bin/bash
# Submit MATH-500 evaluation for the exact checkpoint snapshot evaluated on AIME.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

SOURCE_ROOT="${SOURCE_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training/math/dapo-math-p10-90/Qwen3-4B-Instruct-2507/grpo-clip0.2-0.28-tis2.0/async/off-policy}"
SWEEP_NAME="${SWEEP_NAME:-hiso-offpolicy-20260813}"
MODE="${1:-check}"
HOST_RESULT_ROOT="${DATASET_DIR}/offline_eval/${SWEEP_NAME}"
HOST_MATH500="${DATASET_DIR}/math-500/math-500.jsonl"
CONTAINER_SOURCE=/source/off-policy
CONTAINER_RESULT_ROOT="/data/offline_eval/${SWEEP_NAME}"
MATH500_SPEC=math500:/data/math-500/math-500.jsonl

[[ -f "${HOST_MATH500}" ]] || { echo "missing MATH-500 dataset: ${HOST_MATH500}" >&2; exit 1; }
[[ $(wc -l < "${HOST_MATH500}") -eq 500 ]] || {
    echo "MATH-500 dataset must contain exactly 500 rows" >&2
    exit 1
}

is_aime_complete() {
    local directory="$1"
    local year
    for year in 24 25 26; do
        [[ -f "${directory}/aime${year}.jsonl" ]] || return 1
        [[ $(wc -l < "${directory}/aime${year}.jsonl") -eq 30 ]] || return 1
    done
}

is_math500_complete() {
    local directory="$1"
    [[ -f "${directory}/math500.jsonl" ]] || return 1
    [[ $(wc -l < "${directory}/math500.jsonl") -eq 500 ]]
}

mapfile -t candidates < <(
    find "${SOURCE_ROOT}" -path '*/hf/*' -name config.json -printf '%h\n' | sort -V
)

checkpoints=()
for checkpoint in "${candidates[@]}"; do
    relative="${checkpoint#"${SOURCE_ROOT}/"}"
    setting="${relative%/hf/*}"
    step="${relative##*/}"
    if is_aime_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}"; then
        checkpoints+=("${checkpoint}")
    fi
done

if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "no checkpoints with complete AIME evaluation found" >&2
    exit 1
fi
printf 'selected %d previously evaluated HF checkpoints\n' "${#checkpoints[@]}" >&2

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
    local relative setting step safe_tag out_dir job_id
    relative="${checkpoint#"${SOURCE_ROOT}/"}"
    setting="${relative%/hf/*}"
    step="${relative##*/}"
    safe_tag="${setting//\//_}-step${step}"
    out_dir="${CONTAINER_RESULT_ROOT}/${setting}/step-${step}"

    job_id=$(squeue -h -n "m500-${safe_tag}" -o '%i' | head -1)
    if [[ -n "${job_id}" ]]; then
        printf '%s\t%s\t%s\tqueued\n' "${job_id}" "${setting}" "${step}"
        return
    fi

    job_id=$(sbatch --parsable \
        -A "${SLURM_ACCOUNT_NAME}" \
        --partition=batch,batch_short \
        --job-name="m500-${safe_tag}" \
        --export="ALL,CKPT=${CONTAINER_SOURCE}/${relative},TAG=${safe_tag},OUT_DIR=${out_dir},EXTRA_MOUNTS=${SOURCE_ROOT}:${CONTAINER_SOURCE},UNPAD_VOCAB=1,BENCHMARKS=${MATH500_SPEC},N_SAMPLES=16" \
        experiments/src/offline_eval/run_eval.sbatch)
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
        for checkpoint in "${checkpoints[@]}"; do
            relative="${checkpoint#"${SOURCE_ROOT}/"}"
            setting="${relative%/hf/*}"
            step="${relative##*/}"
            if ! is_math500_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}"; then
                printf 'job_id\tsetting\tstep\tstatus\n'
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
        if ! is_math500_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}"; then
            echo "refusing to fan out before the first checkpoint completes; run '$0 probe' first" >&2
            exit 1
        fi
        printf 'job_id\tsetting\tstep\tstatus\n'
        for checkpoint in "${checkpoints[@]}"; do
            relative="${checkpoint#"${SOURCE_ROOT}/"}"
            setting="${relative%/hf/*}"
            step="${relative##*/}"
            is_math500_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}" || submit_one "${checkpoint}"
        done
        ;;
    *)
        echo "usage: $0 {check|probe|all}" >&2
        exit 2
        ;;
esac
