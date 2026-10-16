import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.src.difficulty_filtering.core.artifacts import write_json
from experiments.src.difficulty_filtering.core.records import summarize


class Progress:
    def __init__(self, config, records, target, directory):
        self.config = config
        self.records = dict(records)
        self.initial = len(records)
        self.initial_tokens = sum(sum(record["completion_tokens"]) for record in records.values())
        self.target = target
        self.directory = directory
        self.started = time.monotonic()

    def emit(self):
        elapsed = time.monotonic() - self.started
        result = summarize(self.records.values(), self.config.n_samples)
        rate = (len(self.records) - self.initial) / max(1, elapsed)
        result.update(
            target=self.target,
            updated_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=elapsed,
            groups_per_second=rate,
            response_tokens_per_second=(result["response_tokens"] - self.initial_tokens) / max(1, elapsed),
            eta_seconds=(self.target - len(self.records)) / rate if rate > 0 else None,
        )
        write_json(self.directory / "progress.json", result)
        with (self.directory / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print("FILTER_PROGRESS " + json.dumps(result), flush=True)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(self.config.progress_seconds)
            self.emit()


def inspect_measurement(directory):
    """Read only complete journal lines while workers are appending, including legacy math runs."""
    records, shards, expected_measurement = [], [], None
    for part in sorted(directory.glob("part-*")):
        manifest_path = part / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if expected_measurement is not None and manifest["measurement"] != expected_measurement:
            raise ValueError(f"Inconsistent measurement manifests in {directory}")
        expected_measurement = manifest["measurement"]
        journal = part / "pass_rates.jsonl"
        current = []
        if journal.exists():
            with journal.open() as stream:
                for line in stream:
                    if line.endswith("\n"):
                        current.append(json.loads(line))
        target = manifest["stop"] - manifest["start"]
        shards.append({"part": part.name, "done": len(current), "target": target})
        records.extend(current)
    config = (expected_measurement or {}).get("config", {})
    return {
        "measurement_dir": str(directory),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **summarize(records, config.get("n_samples", 16)),
        "shards": shards,
        "expected_shards": config.get("num_shards"),
        "target_in_started_shards": sum(shard["target"] for shard in shards),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read live filtering metrics without GPU allocation or journal mutation"
    )
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds; 0 prints once")
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch must be nonnegative")
    while True:
        for directory in args.directories:
            print(json.dumps(inspect_measurement(directory), ensure_ascii=False, indent=2), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
