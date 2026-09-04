#!/bin/bash
#SBATCH --job-name=prep-nemotron-rl-math-v2
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu_datamover
#SBATCH --qos=cpu-datamover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=experiments/outputs/download/%x-%j.log

# Restore nvidia/Nemotron-RL-Math-v2 and convert it to the prompt JSONL the
# recipes read.
#
#   sbatch -A coreai_horizon_dilations \
#     experiments/setup/datasets/prepare_nemotron_rl_math.sbatch
#
# The released file masks every row that came from a public source: 3984 of 7732
# rows (2586 Skywork, 1398 DAPO-Math-17k) ship with empty `question` and
# `expected_answer` plus an `_hf_question_placeholder` pointing at the source row.
# The dataset's own `fill_placeholders.py` downloads those two datasets and fills
# them back in, so this step is a prerequisite, not an optimisation -- without it
# half the file is blank prompts.
#
# CPU only and network-bound; use the dedicated datamover partition and QoS.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

LOCAL_NAME="${LOCAL_NAME:-nemotron-rl-math-v2}"

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc "
set -euo pipefail
export HF_HOME=/root/.cache/huggingface
D=/data/${LOCAL_NAME}

# fill_placeholders.py has a uv shebang and reads/writes whole directories, so
# it gets its own masked/ and restored/ trees rather than being pointed at the
# repo copy.
rm -rf \${D}/masked \${D}/restored
mkdir -p \${D}/masked \${D}/restored
cp \${D}/data/train.jsonl \${D}/masked/train.jsonl

python3 \${D}/fill_placeholders.py --input-dir \${D}/masked --output-dir \${D}/restored

python3 - <<'PY'
import json, collections
p = '/data/${LOCAL_NAME}/restored/train.jsonl'
rows = [json.loads(l) for l in open(p) if l.strip()]
blank = sum(1 for r in rows if not r.get('question') or not r.get('expected_answer'))
print(f'restored rows={len(rows)} still_blank={blank}')
assert blank == 0, f'{blank} rows still blank after restore'
PY

echo '--- converting to miles prompt jsonl ---'
python3 /root/miles/experiments/setup/datasets/build_math_jsonl.py \
    --input \${D}/restored/train.jsonl \
    --output \${D}/nemotron-rl-math-v2-miles.jsonl \
    --dataset nemotron-rl-math-v2 \
    --problem-key question \
    --answer-key expected_answer \
    --no-instruction

wc -l \${D}/nemotron-rl-math-v2-miles.jsonl
head -1 \${D}/nemotron-rl-math-v2-miles.jsonl | cut -c1-400
"
