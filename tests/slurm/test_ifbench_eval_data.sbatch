#!/bin/bash
#SBATCH --job-name=test-ifbench-eval
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

python3 -m pytest -q tests/fast/experiments/test_ifbench_eval_data.py
python3 - <<"PY"
import hashlib
import json
import subprocess
from pathlib import Path

from experiments.src.datasets.nemotron.adapters import adapt_ifbench
from miles.rollout.rm_hub.ifbench import PINNED_DEPS_MARKER, PINNED_IFBENCH_COMMIT


repo_path = Path("/data/ifbench")
deps_path = Path("/data/ifbench-deps")
raw_path = repo_path / "data/IFBench_test.jsonl"
data_path = repo_path / "IFBench_test_miles.jsonl"
summary_path = repo_path / "IFBench_test_miles.summary.json"
expected_hashes = {
    raw_path: "a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85",
    data_path: "36c302f83ddfec7216cad6fc1a59293fe2f6ae81ace444ff1208654b40cc4f23",
    summary_path: "1b734fdc965eb9c67641a85ae1da2720bd7543c30a55e093ab331a534084bb78",
}
for path, expected in expected_hashes.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (path, actual)

revision = subprocess.run(
    ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
tracked_status = subprocess.run(
    ["git", "-C", str(repo_path), "status", "--porcelain", "--untracked-files=no"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert revision == PINNED_IFBENCH_COMMIT, revision
assert not tracked_status, tracked_status
assert (deps_path / PINNED_DEPS_MARKER).is_file()
assert (deps_path / "nltk_data").is_dir()

with raw_path.open(encoding="utf-8") as stream:
    raw_rows = [json.loads(line) for line in stream if line.strip()]
with data_path.open(encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
assert len(raw_rows) == len(rows) == 300
record_ids = set()
constraint_ids = set()
constraint_slots = 0
for line_number, (raw, row) in enumerate(zip(raw_rows, rows, strict=True), start=1):
    assert row == adapt_ifbench(raw), line_number
    assert set(row) == {"label", "metadata", "prompt"}, line_number
    assert row["prompt"] == [{"role": "user", "content": raw["prompt"].strip()}], line_number
    metadata = row["metadata"]
    assert metadata["source"] == "ifbench", line_number
    assert metadata["verifier"] == "ifbench", line_number
    assert metadata["eval_only"] is True, line_number
    assert len(metadata["instruction_id_list"]) == len(metadata["kwargs"]), line_number
    assert 1 <= len(metadata["instruction_id_list"]) <= 2, line_number
    assert metadata["record_id"] not in record_ids, metadata["record_id"]
    record_ids.add(metadata["record_id"])
    constraint_ids.update(metadata["instruction_id_list"])
    constraint_slots += len(metadata["instruction_id_list"])
assert len(record_ids) == 300
assert len(constraint_ids) == 58
assert constraint_slots == 344

summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary == {
    "dataset": "ifbench",
    "inputs": [str(raw_path)],
    "kept": 300,
    "output": str(data_path),
    "skipped": 0,
    "verifiers": {"ifbench": 300},
}
print(
    "IFBench artifact: "
    f"rows={len(rows)} unique_ids={len(record_ids)} "
    f"constraint_ids={len(constraint_ids)} constraint_slots={constraint_slots}"
)
print(f"Pinned official evaluator commit: {revision}")
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
PYTHON_BIN="$(realpath -- "$(command -v python3)")"
exec unshare --user --map-root-user --net --fork \
    bwrap --die-with-parent --new-session --unshare-pid --unshare-ipc --unshare-uts \
    --dir /proc --dev /dev --tmpfs /tmp --dir /home \
    --ro-bind /usr /usr --ro-bind /usr/local /usr/local --ro-bind /bin /bin \
    --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
    --ro-bind /root/miles /root/miles --ro-bind /data /data \
    --chdir /root/miles \
    --setenv HOME /home \
    --setenv PYTHONPATH /root/miles \
    --setenv IFBENCH_REPO_PATH /data/ifbench \
    --setenv IFBENCH_DEPS_PATH /data/ifbench-deps \
    --setenv NLTK_DATA /data/ifbench-deps/nltk_data \
    "${PYTHON_BIN}" - <<"PY"
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.rm_hub import async_rm
from miles.rollout.rm_hub.ifbench import _build_input_example
from miles.utils.types import Sample


data_path = Path("/data/ifbench/IFBench_test_miles.jsonl")
with data_path.open(encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
for line_number, row in enumerate(rows, start=1):
    example = _build_input_example(row["metadata"])
    assert example is not None, line_number
    assert example.prompt == row["metadata"]["prompt_text"], line_number

metadata = {
    "rm_type": "ifbench",
    "verifier": "ifbench",
    "eval_only": True,
    "instruction_id_list": ["keywords:existence"],
    "prompt_text": "Include cat and dog.",
    "kwargs": [{"keywords": ["cat", "dog"]}],
    "record_id": 0,
}
args = SimpleNamespace(custom_rm_path=None, rm_type=None, zero_reward_on_truncated=False)
correct = Sample(response="A cat and a dog.", label=None, metadata=metadata)
wrong = Sample(response="A cat.", label=None, metadata=metadata)


async def score():
    return await asyncio.gather(async_rm(args, correct), async_rm(args, wrong))


rewards = asyncio.run(score())
assert rewards == [1.0, 0.0], rewards
print("Official IFBench scorer: correct=1 wrong=0 network=isolated")
PY
'
