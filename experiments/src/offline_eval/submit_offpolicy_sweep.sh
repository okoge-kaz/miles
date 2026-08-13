#!/bin/bash
# Submit the Hiso off-policy HF checkpoint sweep on non-interactive partitions.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

SOURCE_ROOT="${SOURCE_ROOT:-/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso/async-rl/checkpoints/training/math/dapo-math-p10-90/Qwen3-4B-Instruct-2507/grpo-clip0.2-0.28-tis2.0/async/off-policy}"
SWEEP_NAME="${SWEEP_NAME:-hiso-offpolicy-20260813}"
MODE="${1:-check}"
HOST_RESULT_ROOT="${DATASET_DIR}/offline_eval/${SWEEP_NAME}"
CONTAINER_SOURCE=/source/off-policy
CONTAINER_RESULT_ROOT="/data/offline_eval/${SWEEP_NAME}"

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
    echo "The checkpoint owner must grant read access before evaluation." >&2
    echo "Suggested owner-side fix:" >&2
    echo "  find '${SOURCE_ROOT}' -path '*/hf/*' -name '*.safetensors' -exec chmod a+r {} +" >&2
    exit 1
fi

is_complete() {
    local directory="$1"
    local year
    for year in 24 25 26; do
        [[ -f "${directory}/aime${year}.jsonl" ]] || return 1
        [[ $(wc -l < "${directory}/aime${year}.jsonl") -eq 30 ]] || return 1
    done
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
    local relative setting step safe_tag out_dir job_id
    relative="${checkpoint#"${SOURCE_ROOT}/"}"
    setting="${relative%/hf/*}"
    step="${relative##*/}"
    safe_tag="${setting//\//_}-step${step}"
    out_dir="${CONTAINER_RESULT_ROOT}/${setting}/step-${step}"

    job_id=$(squeue -h -n "oeval-${safe_tag}" -o '%i' | head -1)
    if [[ -n "${job_id}" ]]; then
        printf '%s\t%s\t%s\tqueued\n' "${job_id}" "${setting}" "${step}"
        return
    fi

    job_id=$(sbatch --parsable \
        -A "${SLURM_ACCOUNT_NAME}" \
        --partition=batch,batch_short \
        --job-name="oeval-${safe_tag}" \
        --export="ALL,CKPT=${CONTAINER_SOURCE}/${relative},TAG=${safe_tag},OUT_DIR=${out_dir},EXTRA_MOUNTS=${SOURCE_ROOT}:${CONTAINER_SOURCE},UNPAD_VOCAB=1" \
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
            if ! is_complete "${HOST_RESULT_ROOT}/${setting}/step-${step}"; then
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
