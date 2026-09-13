import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.download.checkpoints import nemotron_3_5_lightning_30b_a3b as checkpoint
from scripts.download.datasets import common

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "module_name,name,revision",
    [
        (
            "scripts.download.datasets.training.dapo_math_17k",
            "dapo-math-17k",
            "2e65612930298bde4c5d58fd97b3f23a483aaff9",
        ),
        ("scripts.download.datasets.eval.aime_2024", "aime-2024", "1c625e328db94ec7ef7ff169016b097c468d60b9"),
    ],
)
def test_dataset_entrypoints_download_only_the_selected_dataset(monkeypatch, tmp_path, module_name, name, revision):
    module = importlib.import_module(module_name)
    payload = json.dumps({"prompt": [{"role": "user", "content": "Return zero."}], "label": "0"}) + "\n"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        destination = kwargs["local_dir"]
        destination.mkdir(parents=True)
        (destination / f"{name}.jsonl").write_text(payload)

    monkeypatch.setattr(common, "snapshot_download", download)
    monkeypatch.setattr(sys, "argv", [module_name, "--dataset-root", str(tmp_path)])
    module.main()

    assert calls == [
        dict(repo_id=f"zhuzilin/{name}", repo_type="dataset", revision=revision, local_dir=tmp_path / name)
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [name]
    assert (tmp_path / name / f"{name}.jsonl").read_text() == payload
    manifest = json.loads((tmp_path / name / "staging_manifest.json").read_text())
    assert manifest["revision"] == revision
    assert manifest["rows"] == 1
    assert manifest["difficulty_filtering"] is False


@pytest.mark.parametrize("policy_only", [False, True])
def test_checkpoint_entrypoint_has_no_dataset_dependency(monkeypatch, tmp_path, policy_only):
    calls = []
    monkeypatch.setattr(checkpoint, "stage_model", lambda root: calls.append(("download", root)))
    monkeypatch.setattr(checkpoint, "prepare_policy_view", lambda root: calls.append(("policy", root)))
    argv = ["checkpoint", "--hf-root", str(tmp_path)] + (["--policy-only"] if policy_only else [])
    monkeypatch.setattr(sys, "argv", argv)
    checkpoint.main()
    assert calls == ([] if policy_only else [("download", tmp_path)]) + [("policy", tmp_path)]


def test_cpu_wrapper_preserves_module_arguments_without_invoking_slurm(tmp_path):
    repo = tmp_path / "checkout"
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments/env.sh").write_text("export SQSH_IMAGE=image.sqsh\nexport CONTAINER_MOUNTS=source:target\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    recorder = binary / "srun"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['LIGHTNING_ARGV_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    recorder.chmod(0o755)
    capture = tmp_path / "argv.json"
    module = "scripts.download.datasets.eval.aime_2024"
    destination = "/data/a directory with spaces"
    env = dict(
        os.environ,
        MILES_REPO=str(repo),
        LIGHTNING_ARGV_CAPTURE=str(capture),
        PATH=f"{binary}:{os.environ['PATH']}",
    )
    subprocess.run(
        ["bash", str(REPO / "experiments/scripts/download.sbatch"), module, "--dataset-root", destination],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    argv = json.loads(capture.read_text())
    assert argv[-4:] == ["bash", module, "--dataset-root", destination]
    assert 'python3 -u -m "$@"' in argv[-5]
