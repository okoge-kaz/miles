#!/usr/bin/env python3
"""Microbenchmark SGLang policy-version telemetry on its CPU hot paths."""

import json
import statistics
import time
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats


class DecodeMode:
    @staticmethod
    def is_extend() -> bool:
        return False


def make_batch(batch_size: int) -> SimpleNamespace:
    reqs = []
    for _ in range(batch_size):
        stats = SimpleNamespace(
            first_prefill_weight_version=10,
            min_forward_weight_version=10,
            max_forward_weight_version=10,
            last_forward_weight_version=10,
        )
        reqs.append(SimpleNamespace(time_stats=stats))
    return SimpleNamespace(forward_mode=DecodeMode(), reqs=reqs, forward_weight_version=-1)


def measure_ns(callable_, iterations: int) -> int:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        callable_()
    return time.perf_counter_ns() - start


def noop() -> None:
    return None


def main() -> None:
    scheduler_off = SimpleNamespace(
        applied_weight_version=11,
        server_args=SimpleNamespace(enable_response_weight_version_segments=False),
    )
    scheduler_on = SimpleNamespace(
        applied_weight_version=11,
        server_args=SimpleNamespace(enable_response_weight_version_segments=True),
    )
    scheduler_results = []

    for batch_size in (1, 8, 32, 128, 512):
        batch_off = make_batch(batch_size)
        batch_on = make_batch(batch_size)

        def stamp_off() -> None:
            Scheduler.stamp_forward_weight_version(scheduler_off, batch_off)

        def stamp_on() -> None:
            Scheduler.stamp_forward_weight_version(scheduler_on, batch_on)

        iterations = max(5_000, 2_000_000 // batch_size)

        stamp_off()
        stamp_on()
        samples = []
        for _ in range(7):
            off_ns = measure_ns(stamp_off, iterations)
            on_ns = measure_ns(stamp_on, iterations)
            samples.append((on_ns - off_ns) / iterations)

        median_batch_ns = statistics.median(samples)
        scheduler_results.append(
            {
                "batch_size": batch_size,
                "iterations": iterations,
                "median_incremental_ns_per_forward_batch": round(median_batch_ns, 1),
                "median_incremental_ns_per_request": round(median_batch_ns / batch_size, 1),
            }
        )

    segment_results = []
    for same_version_run in (1, 32, 512):
        stats = SimpleNamespace(response_weight_version_segments=None)
        position = [0]

        def record_segment() -> None:
            start = position[0]
            position[0] += 1
            SchedulerReqTimeStats.record_response_weight_version_segment(
                stats,
                response_start=start,
                response_end=start + 1,
                weight_version=start // same_version_run,
            )

        iterations = 200_000
        record_segment()
        samples = []
        for _ in range(7):
            baseline_ns = measure_ns(noop, iterations)
            record_ns = measure_ns(record_segment, iterations)
            samples.append((record_ns - baseline_ns) / iterations)
        segment_results.append(
            {
                "tokens_per_weight_version": same_version_run,
                "iterations": iterations,
                "median_incremental_ns_per_generated_token": round(
                    statistics.median(samples),
                    1,
                ),
                "segments_retained": len(stats.response_weight_version_segments),
            }
        )

    print(
        json.dumps(
            {
                "scheduler_feature_on_minus_off": scheduler_results,
                "response_segment_recording_minus_noop": segment_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
