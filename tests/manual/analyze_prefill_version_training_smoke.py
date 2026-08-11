#!/usr/bin/env python3
"""Extract policy-version evidence from an end-to-end training smoke log."""

import argparse
import ast
import json
from pathlib import Path


def parse_payload(line: str, marker: str) -> tuple[int, dict] | None:
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1]
    index_text, values_text = payload.split(": ", 1)
    return int(index_text), ast.literal_eval(values_text)


def select(values: dict, keys: tuple[str, ...]) -> dict:
    return {key: values[key] for key in keys if key in values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--ray-log-root", type=Path)
    args = parser.parse_args()

    report: dict[str, object] = {"rollout_metrics": {}, "train_rollouts": {}, "train_steps": {}}
    for line in args.log.read_text(errors="replace").splitlines():
        if parsed := parse_payload(line, "metrics.py:79 - perf "):
            index, values = parsed
            report["rollout_metrics"][index] = select(
                values,
                (
                    "rollout/fully_async/current_weight_version",
                    "staleness/mixed_version_frac/rollout",
                    "staleness/mixed_version_frac/train",
                    "staleness/total/mean",
                    "staleness/total/max",
                    "staleness/pre_queue/mean",
                    "staleness/in_queue/mean",
                    "rollout/first_prefill_weight_version/mean",
                    "rollout/first_prefill_weight_version/min",
                    "rollout/first_prefill_weight_version/max",
                    "rollout/weight_version/mixed_version_ratio",
                    "perf/rollout_time",
                ),
            )
        elif parsed := parse_payload(line, "log_utils.py:66 - rollout "):
            index, values = parsed
            report["train_rollouts"][index] = select(
                values,
                (
                    "rollout/rewards",
                    "rollout/advantages",
                    "rollout/first_prefill_weight_versions",
                    "rollout/min_forward_weight_versions",
                    "rollout/max_forward_weight_versions",
                    "rollout/last_forward_weight_versions",
                ),
            )
        elif parsed := parse_payload(line, "log_utils.py:473 - step "):
            index, values = parsed
            report["train_steps"][index] = select(
                values,
                (
                    "train/loss",
                    "train/pg_loss",
                    "train/grad_norm",
                    "train/step",
                ),
            )

    if args.ray_log_root:
        texts = []
        for path in args.ray_log_root.rglob("*"):
            if (
                path.is_symlink()
                or not path.name.startswith("worker-")
                or path.suffix not in {".out", ".err"}
            ):
                continue
            try:
                if path.is_file():
                    texts.append(path.read_text(errors="replace"))
            except OSError:
                continue
        combined = "\n".join(texts)
        report["sglang_http"] = {
            "begin_weight_update_200": combined.count('"POST /begin_weight_update HTTP/1.1" 200'),
            "weight_bucket_200": combined.count('"POST /update_weights_from_distributed HTTP/1.1" 200'),
            "end_weight_update_200": combined.count('"POST /end_weight_update HTTP/1.1" 200'),
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
