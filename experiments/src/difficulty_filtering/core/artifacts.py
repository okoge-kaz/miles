import fcntl
import hashlib
import importlib.metadata
import json
import os
from contextlib import contextmanager
from pathlib import Path

from experiments.src.difficulty_filtering.core.records import validate_record


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another job is writing {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def checkpoint_hashes(model_path):
    model_path = Path(model_path)
    paths = sorted(path for path in model_path.iterdir() if path.suffix in {".json", ".jinja", ".txt", ".safetensors"})
    if not (model_path / "config.json").is_file() or not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError("Expected a local Hugging Face checkpoint with safetensors weights")
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        for name in set(json.loads(index.read_text())["weight_map"].values()):
            if not (model_path / name).is_file():
                raise ValueError(f"Missing checkpoint shard: {name}")
    return {path.name: sha256_file(path) for path in paths}


def build_manifest(config):
    root = Path(__file__).resolve().parents[4]
    paths = list(Path(__file__).resolve().parents[1].rglob("*.py")) + [
        root / name
        for name in (
            "miles/rollout/rm_hub/__init__.py",
            "miles/rollout/rm_hub/deepscaler.py",
            "miles/rollout/rm_hub/math_utils.py",
            "experiments/src/training/truncation.py",
            "miles/utils/types.py",
            "release-lock.json",
        )
    ]
    # The optional reward-worker extension is absent from some Miles checkouts.
    reward_worker = root / "miles/rollout/rm_hub/worker.py"
    if reward_worker.is_file():
        paths.append(reward_worker)
    gym_provenance = None
    if config.reward_backend != "deepscaler":
        from experiments.src.nemo_gym.runtime.serve import check_install

        gym_provenance = check_install(Path("/opt/nemo-gym"))
        paths.extend(
            path
            for path in (root / "experiments/src/nemo_gym").rglob("*")
            if path.suffix in {".py", ".json", ".txt", ".patch"}
        )
    print("Hashing checkpoint, input, and measurement source...", flush=True)
    return {
        "schema_version": 2,
        "backend": "sglang",
        "rm_type": config.reward_backend,
        "nemo_gym": gym_provenance,
        "config": config.measurement_settings(),
        "input_sha256": sha256_file(config.prompt_data),
        "checkpoint_sha256": checkpoint_hashes(config.model_path),
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in sorted(paths)},
        "packages": {name: importlib.metadata.version(name) for name in ("sglang", "transformers", "torch")},
    }


def ensure_manifest(path, expected):
    if Path(path).exists():
        if json.loads(Path(path).read_text()) != expected:
            raise ValueError("Measurement provenance changed; choose a new measurement directory")
    else:
        write_json(path, expected)


def repair_trailing_record(path):
    """Only an incomplete final write may be repaired; internal corruption is fatal."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as stream:
        content = stream.read()
        if content.endswith(b"\n"):
            return
        boundary = content.rfind(b"\n") + 1
        try:
            json.loads(content[boundary:])
        except (ValueError, UnicodeDecodeError):
            stream.truncate(boundary)
        else:
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")


def read_records(path, rows, config, allowed_indices):
    records = {}
    if not Path(path).exists():
        return records
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            index = record["index"]
            if type(index) is not int or index not in allowed_indices or index in records:
                raise ValueError(f"Invalid or duplicate source index in {path}: {index}")
            validate_record(record, rows[index], config)
            records[index] = record
    return records
