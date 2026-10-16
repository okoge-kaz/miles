import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from experiments.src.difficulty_filtering.core.artifacts import file_lock, read_records, sha256_file, write_json
from experiments.src.difficulty_filtering.core.config import load_config
from experiments.src.difficulty_filtering.core.records import keep_record, read_prompts, shard_indices, summarize


def merge_measurements(config, rows):
    records, manifests = {}, []
    measurement = None
    for shard_index in range(config.num_shards):
        directory = Path(config.measurement_dir) / f"part-{shard_index:03d}"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assigned = shard_indices(len(rows), config.num_shards, shard_index)
        if (manifest["shard_index"], manifest["start"], manifest["stop"]) != (
            shard_index,
            assigned.start,
            assigned.stop,
        ):
            raise ValueError("Shard range mismatch")
        current = manifest["measurement"]
        if current["config"] != config.measurement_settings():
            raise ValueError("Recipe differs from measured settings")
        if measurement is not None and current != measurement:
            raise ValueError("Shards used different checkpoints, code, or settings")
        measurement = current
        part = read_records(directory / "pass_rates.jsonl", rows, config, assigned)
        if set(part) != set(assigned):
            raise ValueError(f"Incomplete shard {shard_index}: {len(part)}/{len(assigned)} groups")
        records.update(part)
        manifests.append(
            {
                "shard_index": shard_index,
                "manifest_sha256": sha256_file(manifest_path),
                "pass_rates_sha256": sha256_file(directory / "pass_rates.jsonl"),
            }
        )
    if measurement["input_sha256"] != sha256_file(config.prompt_data):
        raise ValueError("Input content differs from measured source")
    return [records[index] for index in range(len(rows))], measurement, manifests


def apply_filter(config):
    output = Path(config.filtered_output)
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    with ExitStack() as stack:
        stack.enter_context(file_lock(output.with_suffix(output.suffix + ".lock")))
        for shard_index in range(config.num_shards):
            stack.enter_context(file_lock(Path(config.measurement_dir) / f"part-{shard_index:03d}" / ".lock"))
        rows = read_prompts(config.prompt_data)
        records, measurement, manifests = merge_measurements(config, rows)
        selected = [record["index"] for record in records if keep_record(record)]
        summary = {
            "selection": config.selection_rule,
            "measurement": measurement,
            "shards": manifests,
            **summarize(records, config.n_samples),
            "kept_indices": selected,
        }
        if not selected:
            raise ValueError("No selected groups; inspect reward and response audits")
        temporary = output.with_suffix(output.suffix + ".tmp")
        # Preserve the original rows, including all source metadata.
        selected_set = set(selected)
        with Path(config.prompt_data).open("rb") as source, temporary.open("wb") as target:
            for index, line in enumerate(source):
                if index in selected_set:
                    target.write(line if line.endswith(b"\n") else line + b"\n")
        summary["output_sha256"] = sha256_file(temporary)
        if output.exists() or metadata_path.exists():
            if (
                output.exists()
                and metadata_path.exists()
                and sha256_file(output) == summary["output_sha256"]
                and json.loads(metadata_path.read_text()) == summary
            ):
                temporary.unlink()
                print(f"Already filtered: {output}")
                return summary
            temporary.unlink()
            raise FileExistsError(f"Refusing to replace existing output: {output}")
        temporary.replace(output)
        write_json(metadata_path, summary)
    print(f"Kept {len(selected)}/{len(rows)} prompts: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Merge complete measurements and keep mixed-correctness groups")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    apply_filter(load_config(args.config))


if __name__ == "__main__":
    main()
