#!/bin/bash
#SBATCH --job-name=test-gpqa-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=experiments/outputs/validation/%x-%j.log

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
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

python3 -m pytest -q tests/fast/experiments/test_gpqa_eval_data.py
python3 - <<"PY"
import collections
import itertools
import json
from pathlib import Path

from miles.rollout.rm_hub.gpqa import compute_gpqa_reward


expected_records = {"diamond": 198, "main": 448, "extended": 546}
for split, expected in expected_records.items():
    data_path = Path(f"/data/gpqa/gpqa-{split}-miles.jsonl")
    source_path = Path(f"/data/gpqa/source/nemo-skills-{split}.jsonl")
    summary_path = Path(f"/data/gpqa/gpqa-{split}-conversion-summary.json")
    audit_path = Path(f"/data/gpqa/gpqa-{split}-audit.json")
    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == expected, (split, len(rows))
    counts = collections.Counter(row["label"] for row in rows)
    expected_per_letter = expected / 4
    chi2 = sum((counts[letter] - expected_per_letter) ** 2 / expected_per_letter for letter in "ABCD")
    assert chi2 <= 11.34, (split, counts, chi2)
    for index, row in enumerate(rows):
        assert set(row) == {"label", "metadata", "prompt"}, (split, index)
        assert row["label"] in "ABCD", (split, index)
        assert isinstance(row["prompt"], list) and len(row["prompt"]) == 1
        assert row["prompt"][0]["role"] == "user"
        metadata = row["metadata"]
        assert metadata["source"] == "gpqa"
        assert metadata["verifier"] == "gpqa"
        assert metadata["eval_only"] is True
        assert metadata["source_format"] == "nemo-skills-preprocessed"
        assert metadata["valid_letters"] == list("ABCD")
        assert len(metadata["choices"]) == 4

    source_rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for index, pair in enumerate(itertools.zip_longest(source_rows, rows)):
        source, output = pair
        assert source is not None and output is not None, (split, index)
        assert output["label"] == str(source["expected_answer"]).strip().upper()
        assert output["metadata"]["choices"] == [str(source[letter]).strip() for letter in "ABCD"]
        assert str(source["problem"]).strip() in output["prompt"][0]["content"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dataset"] == "gpqa"
    assert summary["inputs"] == [str(source_path)]
    assert summary["output"] == str(data_path)
    assert summary["kept"] == expected
    assert summary["skipped"] == 0
    assert summary["verifiers"] == {"gpqa": expected}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["input"] == str(data_path)
    assert audit["rows"] == expected
    assert audit["verifiers"] == {"gpqa": expected}
    assert audit["eval_only_required"] is True

diamond = json.loads(Path("/data/gpqa/gpqa-diamond-miles.jsonl").read_text().splitlines()[0])
label = diamond["label"]
wrong = next(letter for letter in "ABCD" if letter != label)
assert compute_gpqa_reward(f"Final answer: {label}", label, diamond["metadata"]) == 1.0
assert compute_gpqa_reward(f"Final answer: {wrong}", label, diamond["metadata"]) == 0.0
print("GPQA artifacts: diamond=198 main=448 extended=546, balanced and source-preserving")
print("GPQA scorer: correct=1 wrong=0 on a canonical diamond row")
PY
'
