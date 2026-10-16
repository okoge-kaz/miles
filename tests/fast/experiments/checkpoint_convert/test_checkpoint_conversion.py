import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CONVERSION_ROOT = REPO_ROOT / "experiments/scripts/checkpoint_convert"


def _environment(tmp_path, **overrides):
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "WANDB_MODE": "disabled",
        "MILES_WORKSPACE_ROOT": str(tmp_path / "workspace"),
        "HF_MODEL_NAME": "test-source",
        "MODEL_NAME": "test-model",
        "MODEL_PROFILE": "qwen3-4B",
        **overrides,
    }


def _stub(path, body):
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def _run_conversion(direction, environment):
    return subprocess.run(
        ["bash", str(CONVERSION_ROOT / direction / "convert.sh")],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("action", ["convert", "hf_mcore", "mcore_hf"])
def test_conversion_preview_exports_configuration_without_submitting(tmp_path, action):
    result = subprocess.run(
        ["bash", "experiments/setup.sh", action],
        cwd=REPO_ROOT,
        env=_environment(
            tmp_path,
            PBS_QSUB_BIN="must-not-run",
            CONVERT_INPUT_DIR="/ckpt/training/example",
            CONVERT_OUTPUT_DIR="/ckpt/hf/example-export",
            CONVERT_ORIGIN_HF_DIR="/ckpt/hf/test-source",
            CONVERT_CHUNK_SIZE="1024",
            CONVERT_VOCAB_SIZE="151936",
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "must-not-run -q" in result.stdout
    for name in (
        "CONVERT_INPUT_DIR",
        "CONVERT_OUTPUT_DIR",
        "CONVERT_ORIGIN_HF_DIR",
        "CONVERT_CHUNK_SIZE",
        "CONVERT_VOCAB_SIZE",
    ):
        assert name in result.stdout


@pytest.mark.parametrize("direction", ["hf_mcore", "mcore_hf"])
def test_allocation_forwards_conversion_paths_without_splitting_spaces(tmp_path, direction):
    singularity = tmp_path / "singularity"
    capture = tmp_path / "container-args.json"
    _stub(
        singularity,
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['FAKE_CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n",
    )
    image = tmp_path / "image.sif"
    image.touch()
    result = subprocess.run(
        ["bash", str(CONVERSION_ROOT / direction / "run.sh")],
        cwd=tmp_path,
        env=_environment(
            tmp_path,
            SINGULARITY_BIN=str(singularity),
            CONTAINER_IMAGE=str(image),
            FAKE_CAPTURE=str(capture),
            CONVERT_INPUT_DIR="/ckpt/training/run with spaces",
            CONVERT_OUTPUT_DIR="/ckpt/hf/export with spaces",
            CONVERT_ORIGIN_HF_DIR="/ckpt/hf/test-source",
            CONVERT_CHUNK_SIZE="1024",
            CONVERT_VOCAB_SIZE="151936",
        ),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    arguments = json.loads(capture.read_text())
    assert "CONVERT_INPUT_DIR=/ckpt/training/run with spaces" in arguments
    assert "CONVERT_OUTPUT_DIR=/ckpt/hf/export with spaces" in arguments
    if direction == "mcore_hf":
        assert "CONVERT_ORIGIN_HF_DIR=/ckpt/hf/test-source" in arguments
        assert "CONVERT_CHUNK_SIZE=1024" in arguments
        assert "CONVERT_VOCAB_SIZE=151936" in arguments


def test_hf_conversion_forwards_input_and_skips_completed_release(tmp_path):
    source = tmp_path / "hf source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    destination = tmp_path / "megatron/model"
    capture = tmp_path / "torchrun.json"
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    _stub(
        binary_dir / "torchrun",
        """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['FAKE_CAPTURE']).write_text(json.dumps({
    'args': args,
    'gpus': os.environ['CUDA_VISIBLE_DEVICES'],
    'connections': os.environ['CUDA_DEVICE_MAX_CONNECTIONS'],
}))
destination = Path(args[args.index('--save') + 1])
destination.mkdir(parents=True)
(destination / 'latest_checkpointed_iteration.txt').write_text('release')
""",
    )
    environment = _environment(
        tmp_path,
        PATH=f"{binary_dir}:{os.environ['PATH']}",
        CONVERT_INPUT_DIR=str(source),
        CONVERT_OUTPUT_DIR=str(destination),
        GPUS_PER_NODE="2",
        CUDA_DEVICE_MAX_CONNECTIONS="8",
        FAKE_CAPTURE=str(capture),
    )
    result = _run_conversion("hf_mcore", environment)
    assert result.returncode == 0, result.stderr
    recorded = json.loads(capture.read_text())
    arguments = recorded["args"]
    assert arguments[arguments.index("--nproc-per-node") + 1] == "2"
    assert arguments[arguments.index("--hf-checkpoint") + 1] == str(source)
    assert recorded["gpus"] == "0,1"
    assert recorded["connections"] == "1"
    capture.unlink()
    result = _run_conversion("hf_mcore", environment)
    assert result.returncode == 0 and "already exists" in result.stdout
    assert not capture.exists()
    (destination / "latest_checkpointed_iteration.txt").write_text("100")
    result = _run_conversion("hf_mcore", environment)
    assert result.returncode == 2 and "not a release" in result.stderr
    assert not capture.exists()


def _preset_config(tmp_path, name, **overrides):
    environment = _environment(tmp_path)
    for variable in ("MODEL_NAME", "MODEL_PROFILE", "HF_MODEL_NAME"):
        environment.pop(variable)
    environment.update(overrides)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; python3 -c "import json, os; print(json.dumps(dict(os.environ)))"',
            "_",
            str(CONVERSION_ROOT / "models" / f"{name}.sh"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("direction", ["hf_mcore", "mcore_hf"])
def test_shared_custom_sft_preset_is_forwarded_and_can_be_loaded_on_compute(tmp_path, direction):
    capture = tmp_path / "qsub.json"
    qsub = tmp_path / "qsub"
    _stub(
        qsub,
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(os.environ['FAKE_CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}))\n"
        "print('123.pbs1')\n",
    )
    environment = _environment(tmp_path)
    for variable in ("MODEL_NAME", "MODEL_PROFILE", "HF_MODEL_NAME"):
        environment.pop(variable)
    environment.update(
        PBS_QSUB_BIN=str(qsub),
        FAKE_CAPTURE=str(capture),
        MODEL_VARIANT="sft",
        HF_SFT_REF=f"{tmp_path}/workspace/checkpoints/hf/custom-lightning/iter_0000042",
    )
    result = subprocess.run(
        ["bash", "experiments/setup.sh", direction, "nemotron3_5_lightning", "--submit"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    submitted = json.loads(capture.read_text())
    arguments = submitted["args"]
    exported_names = arguments[arguments.index("-v") + 1].split(",")
    assert "HF_SFT_REF" in exported_names and "CONVERT_MODEL_PRESET" in exported_names
    compute = {}
    for item in exported_names:
        name, separator, value = item.partition("=")
        compute[name] = value if separator else submitted["env"][name]
    compute["PATH"] = os.environ["PATH"]
    config = _preset_config(tmp_path, "nemotron3_5_lightning", **compute)
    assert config["HF_MODEL_NAME"] == "custom-lightning/iter_0000042"


def _reverse_fixture(tmp_path, input_kind):
    root = tmp_path / "megatron"
    root.mkdir()
    if input_kind == "direct":
        iteration = root
    else:
        tracker, directory = {"release": ("release", "release"), "iteration": ("42\n", "iter_0000042")}[input_kind]
        (root / "latest_checkpointed_iteration.txt").write_text(tracker)
        iteration = root / directory
        iteration.mkdir()
    (iteration / ".metadata").write_text("fake-metadata")
    origin = tmp_path / "original hf"
    origin.mkdir()
    (origin / "config.json").write_text("{}")
    (origin / "tokenizer.json").write_text("{}")
    destination = tmp_path / "exports/hf"
    capture = tmp_path / "reverse.json"
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    _stub(
        binary_dir / "python3",
        """import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['FAKE_CAPTURE']).write_text(json.dumps(args))
output = Path(args[args.index('--output-dir') + 1])
output.mkdir()
(output / 'model.safetensors.index.json').write_text('{"weight_map":{"tensor":"model.safetensors"}}')
(output / 'model.safetensors').write_text('fake-weights')
if os.environ.get('FAKE_FAILURE') == 'converter':
    sys.exit(12)
origin = Path(args[args.index('--origin-hf-dir') + 1])
for asset in ('config.json', 'tokenizer.json'):
    shutil.copy2(origin / asset, output / asset)
if os.environ.get('FAKE_FAILURE') == 'output-race':
    destination = Path(os.environ['CONVERT_OUTPUT_DIR'])
    destination.mkdir()
    (destination / 'keep').write_text('existing-output')
""",
    )
    environment = _environment(
        tmp_path,
        PATH=f"{binary_dir}:{os.environ['PATH']}",
        CONVERT_INPUT_DIR=str(root),
        CONVERT_OUTPUT_DIR=str(destination),
        CONVERT_ORIGIN_HF_DIR=str(origin),
        CONVERT_VOCAB_SIZE="151936",
        CONVERT_CHUNK_SIZE="1024",
        FAKE_CAPTURE=str(capture),
    )
    return environment, iteration, destination, capture


@pytest.mark.parametrize("input_kind", ["release", "iteration", "direct"])
def test_reverse_conversion_resolves_input_and_publishes_export(tmp_path, input_kind):
    environment, iteration, destination, capture = _reverse_fixture(tmp_path, input_kind)
    result = _run_conversion("mcore_hf", environment)
    assert result.returncode == 0, result.stderr
    arguments = json.loads(capture.read_text())
    assert arguments[arguments.index("--input-dir") + 1] == str(iteration)
    assert arguments[arguments.index("--vocab-size") + 1] == "151936"
    assert arguments[arguments.index("--chunk-size") + 1] == "1024"
    assert (destination / "config.json").read_text() == "{}"
    assert (destination / "tokenizer.json").exists() and (destination / "model.safetensors").exists()
    assert not list(destination.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("failure,exit_code", [("converter", 12), ("output-race", 2)])
def test_reverse_export_failure_cleans_temporary_output_and_preserves_existing_destination(
    tmp_path, failure, exit_code
):
    environment, _, destination, _ = _reverse_fixture(tmp_path, "release")
    result = _run_conversion("mcore_hf", {**environment, "FAKE_FAILURE": failure})
    assert result.returncode == exit_code, result.stderr
    if failure == "output-race":
        assert (destination / "keep").read_text() == "existing-output"
        assert not (destination / "config.json").exists()
    else:
        assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp.*"))


def test_reverse_conversion_rejects_existing_output_and_invalid_tracker(tmp_path):
    environment, _, destination, capture = _reverse_fixture(tmp_path, "release")
    destination.mkdir(parents=True)
    (destination / "keep").write_text("keep")
    result = _run_conversion("mcore_hf", environment)
    assert result.returncode == 2 and "already exists" in result.stderr
    assert (destination / "keep").read_text() == "keep" and not capture.exists()
    tracker = Path(environment["CONVERT_INPUT_DIR"]) / "latest_checkpointed_iteration.txt"
    tracker.write_text("../outside")
    result = _run_conversion("mcore_hf", environment)
    assert result.returncode == 2 and "invalid checkpoint tracker" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("direction", ["hf_mcore", "mcore_hf"])
def test_bridge_conversion_forwards_direction_and_requested_topology(tmp_path, direction):
    environment, iteration, destination, capture = _reverse_fixture(tmp_path, "release")
    binary_dir = tmp_path / "bridge-bin"
    binary_dir.mkdir()
    _stub(
        binary_dir / "torchrun",
        """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ['FAKE_CAPTURE']).write_text(json.dumps(args))
destination = Path(args[args.index('--save') + 1])
destination.mkdir(parents=True)
if 'hf_mcore' in args:
    (destination / 'latest_checkpointed_iteration.txt').write_text('release')
else:
    (destination / 'config.json').write_text('{}')
    (destination / 'model.safetensors').write_text('fake-bridge-weights')
""",
    )
    environment.update(
        PATH=f"{binary_dir}:{os.environ['PATH']}",
        MODEL_PROFILE="nemotron-3-nano-30b-a3b",
        CONVERT_BACKEND="bridge",
        CONVERT_TENSOR_PARALLEL_SIZE="1",
        CONVERT_PIPELINE_PARALLEL_SIZE="2",
        CONVERT_EXPERT_PARALLEL_SIZE="4",
        GPUS_PER_NODE="8",
    )
    for name in ("CONVERT_CHUNK_SIZE", "CONVERT_VOCAB_SIZE"):
        environment.pop(name)
    if direction == "hf_mcore":
        environment["CONVERT_INPUT_DIR"] = environment["CONVERT_ORIGIN_HF_DIR"]
    result = _run_conversion(direction, environment)
    assert result.returncode == 0, result.stderr
    arguments = json.loads(capture.read_text())
    assert direction in arguments
    for flag, value in {
        "--nproc-per-node": "8",
        "--tensor-model-parallel-size": "1",
        "--pipeline-model-parallel-size": "2",
        "--expert-model-parallel-size": "4",
        "--megatron-to-hf-mode": "bridge",
    }.items():
        assert arguments[arguments.index(flag) + 1] == value
    if direction == "mcore_hf":
        assert arguments[arguments.index("--load") + 1] == str(iteration)
        assert (destination / "model.safetensors").read_text() == "fake-bridge-weights"
