#!/usr/bin/env python3
"""Microbenchmark the primitive-only fully-async queue lifecycle recorder."""

import argparse
import io
import json
import statistics
import time

import torch

from miles.rollout.queue_telemetry import _QueueLifecycleRecorder, group_reward_values
from miles.utils.types import Sample


def make_groups(num_groups: int, samples_per_group: int) -> list[list[Sample]]:
    groups = []
    for group_index in range(num_groups):
        group = []
        for sample_offset in range(samples_per_group):
            response_length = 256 + ((group_index * 97 + sample_offset * 53) % 7936)
            group.append(
                Sample(
                    group_index=group_index,
                    index=group_index * samples_per_group + sample_offset,
                    response_length=response_length,
                    reward=float((group_index + sample_offset) % 2),
                    status=Sample.Status.COMPLETED,
                    weight_versions=["3"],
                    first_prefill_weight_versions=[2],
                    min_forward_weight_versions=[2],
                    max_forward_weight_versions=[3],
                    last_forward_weight_versions=[3],
                )
            )
        groups.append(group)
    return groups


def record_batch(groups: list[list[Sample]], *, enabled: bool) -> dict | None:
    recorder = _QueueLifecycleRecorder(enabled=enabled)
    for sequence, group in enumerate(groups):
        record = recorder.begin_attempt(group, submission_version=2)
        rewards = group_reward_values(group, reward_key=None) if record is not None else None
        recorder.group_ready(record, group, ready_version=3, reward_values=rewards)
        recorder.enqueued(
            group,
            queue_put_version=3,
            depth_before=sequence,
            depth_after=sequence + 1,
        )
        recorder.dequeued(group, depth_after_observed=len(groups) - sequence - 1)
        recorder.finish(
            group,
            disposition="trained",
            decision_version=3,
            rollout_id=0,
            reference_version=2,
            bound_staleness=1,
        )
    return recorder.take_metadata(policy="queue-max", capacity_groups=1000)


def median_elapsed_ms(fn, iterations: int) -> float:
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        elapsed.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=192)
    parser.add_argument("--samples-per-group", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    groups = make_groups(args.groups, args.samples_per_group)
    disabled_ms = median_elapsed_ms(lambda: record_batch(groups, enabled=False), args.iterations)
    enabled_ms = median_elapsed_ms(lambda: record_batch(groups, enabled=True), args.iterations)
    metadata = record_batch(groups, enabled=True)

    def serialize() -> None:
        buffer = io.BytesIO()
        torch.save(metadata, buffer)

    serialization_ms = median_elapsed_ms(serialize, args.iterations)
    buffer = io.BytesIO()
    torch.save(metadata, buffer)

    print(
        json.dumps(
            {
                "groups": args.groups,
                "samples": args.groups * args.samples_per_group,
                "disabled_path_ms": disabled_ms,
                "enabled_path_ms": enabled_ms,
                "incremental_recording_ms": enabled_ms - disabled_ms,
                "metadata_torch_save_ms": serialization_ms,
                "metadata_bytes": buffer.tell(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
