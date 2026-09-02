#!/bin/bash
#PBS -N prep-gpqa
#PBS -q R9920261300
#PBS -l select=1:ncpus=32:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

# Convert GPQA from either an already downloaded HF CSV payload or the existing
# user-owned NeMo Skills copy that was prepared from Idavidrein/gpqa. A fresh HF
# download remains gated and belongs in download_dataset.sbatch on data-mover
# QoS; this job is CPU conversion only.
#
#   pbs_submit --profile cpu experiments/setup/datasets/prepare_gpqa.sbatch
#
# Produces three splits under $DATASET_DIR/gpqa:
#   gpqa-diamond-miles.jsonl   198 questions, the split every model card reports
#   gpqa-main-miles.jsonl      448
#   gpqa-extended-miles.jsonl  546
#
# Eval only. 198 questions is far too few to train on, and the diamond split is
# the reported benchmark -- training on it would make the eval meaningless.

set -euo pipefail

REPO_ROOT="${PBS_O_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
source "${REPO_ROOT}/experiments/env.sh"

miles_srun \
    --image="${CONTAINER_IMAGE}" \
    --bind="${CONTAINER_MOUNTS}" \
    --writable-tmpfs \
    --no-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
cd /root/miles
export HF_HOME=/root/.cache/huggingface
export PYTHONPATH=/root/miles

GPQA=/data/gpqa
NEMO_SKILLS_GPQA=/data/evaluation/nemo-skills-26.03/gpqa
mkdir -p "${GPQA}/source"

for split in diamond main extended; do
    if [[ -s "${GPQA}/gpqa_${split}.csv" ]]; then
        input="${GPQA}/gpqa_${split}.csv"
    elif [[ -s "${NEMO_SKILLS_GPQA}/${split}.jsonl" ]]; then
        expected=0
        case "${split}" in
            diamond) expected=198 ;;
            main) expected=448 ;;
            extended) expected=546 ;;
        esac
        test "$(wc -l < "${NEMO_SKILLS_GPQA}/${split}.jsonl")" -eq "${expected}"
        staged="${GPQA}/source/nemo-skills-${split}.jsonl"
        cp "${NEMO_SKILLS_GPQA}/${split}.jsonl" "${staged}.partial"
        mv "${staged}.partial" "${staged}"
        input="${staged}"
    else
        echo "ERROR: no existing GPQA source for split ${split}." >&2
        echo "  Either stage gpqa_${split}.csv with download_dataset.sbatch after" >&2
        echo "  accepting the Idavidrein/gpqa terms, or provide the existing" >&2
        echo "  NeMo Skills asset at ${NEMO_SKILLS_GPQA}/${split}.jsonl." >&2
        exit 1
    fi
    python3 -m experiments.src.datasets.nemotron.convert \
        --dataset gpqa \
        --input "${input}" \
        --output "${GPQA}/gpqa-${split}-miles.jsonl" \
        --summary "${GPQA}/gpqa-${split}-conversion-summary.json"
done

python3 -m experiments.src.datasets.common.audit \
    --input "${GPQA}/gpqa-diamond-miles.jsonl" \
    --expected-rows 198 --require-verifiers gpqa --require-eval-only \
    --output "${GPQA}/gpqa-diamond-audit.json"
python3 -m experiments.src.datasets.common.audit \
    --input "${GPQA}/gpqa-main-miles.jsonl" \
    --expected-rows 448 --require-verifiers gpqa --require-eval-only \
    --output "${GPQA}/gpqa-main-audit.json"
python3 -m experiments.src.datasets.common.audit \
    --input "${GPQA}/gpqa-extended-miles.jsonl" \
    --expected-rows 546 --require-verifiers gpqa --require-eval-only \
    --output "${GPQA}/gpqa-extended-audit.json"

echo
echo "answer-letter balance:"
python3 - <<PY
import collections, itertools, json, sys

# A skew here means always answering the popular letter beats the chance floor,
# so the eval measures the converter rather than the model. This has already
# happened once: ordering by a 4-bit derived key put the correct answer on A
# 162/546 times, because sorting is stable and the correct option is built first.
# chi-square against uniform, df=3, p<0.01.
CRITICAL = 11.34
worst = 0.0
for split in ("diamond", "main", "extended"):
    labels = [json.loads(l)["label"] for l in open(f"/data/gpqa/gpqa-{split}-miles.jsonl")]
    counts = collections.Counter(labels)
    expected = len(labels) / 4
    chi2 = sum((counts[k] - expected) ** 2 / expected for k in "ABCD")
    worst = max(worst, chi2)
    print(f"  {split:9s} n={len(labels):4d}  {dict(sorted(counts.items()))}  chi2={chi2:.2f}")
if worst > CRITICAL:
    print(f"ERROR: answer letters are not uniform (chi2={worst:.2f} > {CRITICAL}); the shuffle is biased.")
    sys.exit(1)

# And the label has to survive the shuffle: point it back at the option text.
for split in ("diamond", "main", "extended"):
    for line in open(f"/data/gpqa/gpqa-{split}-miles.jsonl"):
        row = json.loads(line)
        m = row["metadata"]
        label = row["label"]
        if label not in m["valid_letters"] or len(m["choices"]) != 4:
            print("ERROR: malformed row in " + split + ": " + str(label))
            sys.exit(1)

# When the existing user-owned NeMo Skills copy is the source, prove that conversion
# preserves every source-row label and option ordering. zip_longest also
# turns an accidental row drop or duplicate into a hard failure.
for split in ("diamond", "main", "extended"):
    source_path = f"/data/gpqa/source/nemo-skills-{split}.jsonl"
    try:
        source_handle = open(source_path, encoding="utf-8")
    except FileNotFoundError:
        continue
    with source_handle, open(f"/data/gpqa/gpqa-{split}-miles.jsonl", encoding="utf-8") as output_handle:
        pairs = itertools.zip_longest(source_handle, output_handle)
        for index, (source_line, output_line) in enumerate(pairs, start=1):
            if source_line is None or output_line is None:
                raise SystemExit(f"ERROR: {split} source/output length mismatch at row {index}")
            source = json.loads(source_line)
            output = json.loads(output_line)
            source_choices = [str(source[letter]).strip() for letter in "ABCD"]
            source_label = str(source["expected_answer"]).strip().upper()
            source_problem = str(source["problem"]).strip()
            if output["label"] != source_label:
                raise SystemExit(f"ERROR: {split} label changed at row {index}")
            if output["metadata"]["choices"] != source_choices:
                raise SystemExit(f"ERROR: {split} choices changed at row {index}")
            if source_problem not in output["prompt"][0]["content"]:
                raise SystemExit(f"ERROR: {split} problem missing from prompt at row {index}")
print("balance and label/choice consistency: ok")
PY
'

echo
echo "GPQA evaluation config: experiments/configs/eval_gpqa.yaml"
