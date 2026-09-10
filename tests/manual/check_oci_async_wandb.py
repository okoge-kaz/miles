"""Read retained W&B offline histories without uploading or accessing credentials."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def check_history(root, run_id, expected_steps):
    paths = sorted(root.glob(f"offline-run-*-{run_id}/run-{run_id}.wandb"))
    assert paths, f"no offline history for {run_id}"
    merged = defaultdict(dict)
    file_records = {}
    for path in paths:
        reader = DataStore()
        reader.open_for_scan(str(path))
        count = 0
        try:
            while (data := reader.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if not record.HasField("history"):
                    continue
                count += 1
                row = {
                    item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in record.history.item
                }
                for axis in ("train/step", "rollout/step", "eval/step"):
                    if axis in row:
                        merged[(axis, row[axis])].update(row)
        finally:
            reader.close()
        file_records[str(path)] = count
    for step in expected_steps:
        rollout = merged[("rollout/step", step)]
        trainer = merged[("train/step", step)]
        for key in (
            "fully_async/train_weight_version",
            "staleness/total/mean",
            "staleness/token_lag/exact/covered_response_token_frac",
            "queue/consumption/wall_wait_seconds/mean",
            "throughput/window_seconds",
        ):
            assert key in rollout, (run_id, step, key)
        for key in ("train/loss", "sample_staleness/s_0/consumed_sequence_mass"):
            assert key in trainer, (run_id, step, key)
        assert any(key.startswith("policy_lag/") for key in trainer), (run_id, step, "policy_lag")
    return {
        "run_id": run_id,
        "passed": True,
        "expected_steps": expected_steps,
        "file_history_records": file_records,
        "logged_axes": {
            axis: sorted(step for observed_axis, step in merged if axis == observed_axis)
            for axis in ("train/step", "rollout/step", "eval/step")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("wandb"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(check_history(args.root, args.run_id, args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
