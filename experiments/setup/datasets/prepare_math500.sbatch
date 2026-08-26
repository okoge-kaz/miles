#!/bin/bash
#SBATCH --job-name=prep-math500
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Convert the staged HuggingFaceH4/MATH-500 test split to the canonical Miles
# evaluation schema. The source solution is deliberately excluded from output.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export PYTHONPATH=/root/miles

raw_input="${MATH500_RAW_INPUT:-}"
if [[ -z "${raw_input}" ]]; then
    for source_root in /data/math-500 /data/HuggingFaceH4___math-500; do
        [[ -d "${source_root}" ]] || continue
        mapfile -t candidates < <(
            find "${source_root}" -type f \
                \( -name "*.arrow" -o -name "*.parquet" \) -print | sort
        )
        if (( ${#candidates[@]} > 0 )); then
            (( ${#candidates[@]} == 1 )) || {
                printf "expected one MATH-500 source under %s, found %d\n" \
                    "${source_root}" "${#candidates[@]}" >&2
                exit 2
            }
            raw_input="${candidates[0]}"
            break
        fi
    done
fi
[[ -f "${raw_input}" ]] || { echo "MATH-500 source not found" >&2; exit 3; }

mkdir -p /data/math-500
python3 -m experiments.src.datasets.nemotron.convert \
    --dataset math500 \
    --input "${raw_input}" \
    --output /data/math-500/math-500.jsonl \
    --summary /data/math-500/math-500.provenance.json
test "$(wc -l < /data/math-500/math-500.jsonl)" -eq 500
grep -F "\"verifier\":\"math\"" /data/math-500/math-500.jsonl >/dev/null
grep -F "\"eval_only\":true" /data/math-500/math-500.jsonl >/dev/null
'
