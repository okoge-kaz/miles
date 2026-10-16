import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from experiments.src.datasets.common.io import read_rows
from experiments.src.datasets.common.parallel import map_rows


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert(source: Path, destination: Path, adapter, *, num_proc=1, chunk_size=128) -> dict:
    manifest_path = destination.with_suffix(".manifest.json")
    source_digest = sha256(source)
    if destination.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("conversion_schema") != 2 or manifest.get("adapter", adapter.__name__) != adapter.__name__:
            raise ValueError(f"Reconvert the legacy data to a new output path: {destination}")
        if manifest["source_sha256"] != source_digest or manifest["sha256"] != sha256(destination):
            raise ValueError(f"Existing conversion does not match its source/manifest: {destination}")
        return manifest
    counts = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with (
            os.fdopen(descriptor, "w") as stream,
            map_rows(read_rows([source]), adapter, num_proc=num_proc, chunk_size=chunk_size) as rows,
        ):
            for row in rows:
                counts[row["metadata"].get("source_dataset", row["metadata"]["source"])] += 1
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest = {
            "source": str(source),
            "source_sha256": source_digest,
            "rows": sum(counts.values()),
            "source_counts": dict(counts),
            "sha256": sha256(Path(temporary)),
            "adapter": adapter.__name__,
            "conversion_schema": 2,
        }
        Path(temporary).replace(destination)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        Path(temporary).unlink(missing_ok=True)
