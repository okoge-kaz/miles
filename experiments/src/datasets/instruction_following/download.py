import argparse
import hashlib
import json
from pathlib import Path

IF_TRAIN_DATASET = "allenai/IF_multi_constraints_upto5"
IF_TRAIN_REVISION = "2e3a77407b7fce69f95b248d64a884e3ae1c2423"


def stage_dataset(destination: Path) -> None:
    # Conversion imports the source identity without needing the download client.
    from huggingface_hub import snapshot_download

    repository, revision = IF_TRAIN_DATASET, IF_TRAIN_REVISION
    manifest_path = destination / "miles-source.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["repository"] != repository or manifest["revision"] != revision:
            raise ValueError(f"Refusing to mix source revisions in {destination}")
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        allow_patterns=["README.md", "data/*.parquet"],
        max_workers=2,
    )
    files = {}
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path == manifest_path:
            continue
        with path.open("rb") as stream:
            files[str(path.relative_to(destination))] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"repository": repository, "revision": revision, "sha256": files}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps({"dataset": repository, "revision": revision, "files": len(files)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned IF multi-constraint training data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_dataset(args.output)


if __name__ == "__main__":
    main()
