#!/bin/bash
#SBATCH --job-name=test-lcb-eval
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=cpu
#SBATCH --qos=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
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

python3 -m pytest -q tests/fast/experiments/test_livecodebench_eval_data.py
python3 - <<"PY"
import json
from pathlib import Path

from experiments.src.evaluators.livecodebench import PINNED_REPO_COMMIT, decode_private_tests


data_path = Path("/data/livecodebench-lite/livecodebench-release-v6-miles-eval.jsonl")
summary_path = Path("/data/livecodebench-lite/miles-conversion-summary.json")
audit_path = Path("/data/livecodebench-lite/miles-audit.json")
rows = 0
question_ids = set()
with data_path.open(encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        row = json.loads(line)
        assert set(row) == {"label", "metadata", "prompt"}, line_number
        assert isinstance(row["prompt"], list) and len(row["prompt"]) == 1, line_number
        assert row["prompt"][0]["role"] == "user", line_number
        metadata = row["metadata"]
        assert metadata["source"] == "livecodebench", line_number
        assert metadata["verifier"] == "livecodebench", line_number
        assert metadata["eval_only"] is True, line_number
        assert row["label"] == metadata["question_id"], line_number
        assert row["label"] not in question_ids, row["label"]
        question_ids.add(row["label"])
        if line_number in {1, 528, 1055}:
            public = json.loads(metadata["public_test_cases"])
            private = decode_private_tests(metadata["private_test_cases"])
            assert isinstance(public, list) and public, line_number
            assert isinstance(private, list) and private, line_number
        rows += 1
assert rows == 1055, rows

summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["dataset"] == "livecodebench"
assert summary["kept"] == 1055
assert summary["skipped"] == 0
assert summary["output"] == str(data_path)
assert summary["verifiers"] == {"livecodebench": 1055}
assert len(summary["inputs"]) == 6
assert all(Path(path).is_file() for path in summary["inputs"])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
assert audit["input"] == str(data_path)
assert audit["rows"] == 1055
assert audit["verifiers"] == {"livecodebench": 1055}
assert audit["eval_only_required"] is True
print(f"LiveCodeBench artifact: rows={rows} unique_ids={len(question_ids)}")
print(f"Pinned official evaluator commit: {PINNED_REPO_COMMIT}")
PY
'

srun \
    --container-image="${SQSH_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS},/usr/bin/bwrap:/usr/local/bin/bwrap" \
    --container-writable \
    --no-container-mount-home \
    --export=ALL \
    bash -lc '
set -euo pipefail
export PYTHONPATH=/root/miles
PYTHON_BIN="$(realpath -- "$(command -v python3)")"
exec unshare --user --map-root-user --net --fork \
    bwrap --die-with-parent --new-session --unshare-pid --unshare-ipc --unshare-uts \
    --dir /proc --dev /dev --tmpfs /tmp --dir /home \
    --ro-bind /usr /usr --ro-bind /usr/local /usr/local --ro-bind /bin /bin \
    --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
    --ro-bind /root/miles /root/miles --ro-bind /data /data \
    --chdir /root/miles \
    --setenv HOME /home --setenv PYTHONPATH /root/miles \
    --setenv LCB_ALLOW_LOCAL_EXECUTION 1 \
    --setenv LCB_REPO_PATH /data/livecodebench-code \
    --setenv LCB_EVAL_PROCESSES 2 --setenv LCB_TEST_TIMEOUT 3 \
    "${PYTHON_BIN}" - <<"PY"
import asyncio
import json

from experiments.src.evaluators.livecodebench import livecodebench_reward
from miles.utils.types import Sample


metadata = {
    "eval_only": True,
    "verifier": "livecodebench",
    "public_test_cases": json.dumps([{"input": "1 2\n", "output": "3\n", "testtype": "stdin"}]),
    "private_test_cases": json.dumps([{"input": "4 5\n", "output": "9\n", "testtype": "stdin"}]),
    "lcb_metadata": json.dumps({"func_name": None}),
}
correct = Sample(
    response="```python\na, b = map(int, input().split())\nprint(a + b)\n```",
    metadata=metadata,
)
wrong = Sample(response="```python\nprint(0)\n```", metadata=metadata)
network_only = Sample(
    response=(
        "```python\nimport socket\nsocket.setdefaulttimeout(0.2)\n"
        "try:\n socket.create_connection((\"1.1.1.1\", 53))\n print(3)\n"
        "except OSError:\n print(0)\n```"
    ),
    metadata=metadata,
)
rewards = asyncio.run(livecodebench_reward(None, [correct, wrong, network_only]))
assert rewards == [1.0, 0.0, 0.0], rewards
print("Official LiveCodeBench scorer: correct=1 wrong=0 network-only=0")
PY
'
