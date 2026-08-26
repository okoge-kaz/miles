#!/bin/bash
#SBATCH --job-name=q3-4b-reasoning-suite-score
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/reasoning_eval/%x-%j.log

# Score generations already produced by run-evaluation.sbatch. Generated code
# runs inside isolated user/network/PID/filesystem namespaces. The GPU-node
# seccomp profile rejects mounting a fresh procfs, so /proc is deliberately an
# empty directory inside the sandbox.

set -euo pipefail

REPO_ROOT="$(realpath -- "${SLURM_SUBMIT_DIR:?submit from the Miles repository root}")"
source "${REPO_ROOT}/experiments/env.sh"
: "${RESULT_ROOT:?set RESULT_ROOT to a completed generation directory}"

TASKS_TEXT="${TASKS:-aime24 aime25 aime26 math500 livecodebench gpqa_diamond gpqa_main gpqa_extended ifbench}"
TASKS_TEXT="${TASKS_TEXT//,/ }"
EVAL_MODE="${EVAL_MODE:-full}"
LCB_REPEATS="${LCB_REPEATS:-1}"
GPQA_REPEATS="${GPQA_REPEATS:-8}"
IFBENCH_REPEATS="${IFBENCH_REPEATS:-8}"
AIME_REPEATS="${AIME_REPEATS:-16}"
MATH500_REPEATS="${MATH500_REPEATS:-4}"
RESULT_ROOT="$(realpath -- "${RESULT_ROOT}")"
export WANDB_MODE="${WANDB_MODE:-offline}"
[[ "${WANDB_MODE}" == offline ]] || {
    echo "reasoning-eval validation requires WANDB_MODE=offline" >&2
    exit 2
}
[[ -f "${RESULT_ROOT}/evaluation-contract.env" ]] || {
    echo "missing evaluation contract: ${RESULT_ROOT}/evaluation-contract.env" >&2
    exit 2
}

[[ "${EVAL_MODE}" == smoke || "${EVAL_MODE}" == full ]] || {
    echo "EVAL_MODE must be smoke or full" >&2
    exit 2
}
mapfile -t REQUESTED_TASKS < <(tr ' ' '\n' <<< "${TASKS_TEXT}" | sed '/^$/d')
for task in "${REQUESTED_TASKS[@]}"; do
    case "${task}" in
        aime24|aime25|aime26)
            year="20${task#aime}"
            input="/data/aime-${year}/aime-${year}.jsonl"
            repeats="${AIME_REPEATS}"
            ;;
        math500)
            input=/data/math-500/math-500.jsonl
            repeats="${MATH500_REPEATS}"
            ;;
        livecodebench)
            input=/data/livecodebench-lite/livecodebench-release-v6-miles-eval.jsonl
            repeats="${LCB_REPEATS}"
            ;;
        gpqa_diamond|gpqa_main|gpqa_extended)
            split="${task#gpqa_}"
            input="/data/gpqa/gpqa-${split}-miles.jsonl"
            repeats="${GPQA_REPEATS}"
            ;;
        ifbench)
            input=/data/ifbench/IFBench_test_miles.jsonl
            repeats="${IFBENCH_REPEATS}"
            ;;
        *)
            echo "unsupported task: ${task}" >&2
            exit 3
            ;;
    esac
    [[ -f "${RESULT_ROOT}/${task}/candidates.jsonl" ]] || {
        echo "missing candidates: ${RESULT_ROOT}/${task}/candidates.jsonl" >&2
        exit 4
    }
    jq --exit-status --slurp '
        length > 0
        and all(.[];
            ((.response | type) == "string")
            and (
                ((.response | length) > 0)
                or (.generation_status == "empty_final_content")
            )
        )
    ' "${RESULT_ROOT}/${task}/candidates.jsonl" >/dev/null || {
        echo "candidates contain an unclassified empty final response: ${task}" >&2
        exit 4
    }
    limit_args=()
    [[ "${EVAL_MODE}" == full ]] || limit_args=(--limit 2)
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
        --container-image="${SQSH_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS},${RESULT_ROOT}:/results,/usr/bin/bwrap:/usr/local/bin/bwrap" \
        --container-writable --no-container-mount-home --export=ALL \
        bash -lc '
set -euo pipefail
export PYTHONPATH=/root/miles:/data/ifbench:/data/ifbench-deps
export NLTK_DATA=/data/ifbench-deps/nltk_data
export LCB_ALLOW_LOCAL_EXECUTION=1
export LCB_REPO_PATH=/data/livecodebench-code
export LCB_EVAL_PROCESSES=16
PYTHON_BIN="$(realpath -- "$(command -v python3)")"
exec unshare --user --map-root-user --net --fork \
    bwrap --die-with-parent --new-session --unshare-pid --unshare-ipc --unshare-uts \
    --dir /proc --dev /dev --tmpfs /tmp --dir /home \
    --ro-bind /usr /usr --ro-bind /usr/local /usr/local --ro-bind /bin /bin \
    --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
    --ro-bind /root/miles /root/miles --ro-bind /data /data --bind /results /results \
    --chdir /root/miles "${PYTHON_BIN}" experiments/tools/reasoning_eval/suite.py score \
    --task "$1" --input "$2" --candidates "/results/$1/candidates.jsonl" \
    --output "/results/$1/scores.jsonl" --summary "/results/$1/summary.json" \
    --repeats "$3" "${@:4}"
' _ "${task}" "${input}" "${repeats}" "${limit_args[@]}"
    [[ -s "${RESULT_ROOT}/${task}/provenance.env" ]] || {
        echo "missing task provenance: ${task}" >&2
        exit 5
    }
    (
        cd "${RESULT_ROOT}/${task}"
        sha256sum candidates.jsonl scores.jsonl summary.json provenance.env \
            > artifact-manifest.sha256
        sha256sum --check artifact-manifest.sha256 >/dev/null
    )
    touch "${RESULT_ROOT}/${task}/_SUCCESS"
done

echo "All requested domain scores completed under ${RESULT_ROOT}"
