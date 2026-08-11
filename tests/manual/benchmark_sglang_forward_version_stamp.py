#!/usr/bin/env python3
"""Microbenchmark the scheduler-side policy-version stamp on the CPU hot path."""

import json
import statistics
import time
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler


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
    return SimpleNamespace(forward_mode=DecodeMode(), reqs=reqs)


def measure_ns(callable_, iterations: int) -> int:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        callable_()
    return time.perf_counter_ns() - start


def noop() -> None:
    return None


def main() -> None:
    scheduler = SimpleNamespace(applied_weight_version=11)
    results = []

    for batch_size in (1, 8, 32, 128, 512):
        batch = make_batch(batch_size)

        def stamp() -> None:
            Scheduler.stamp_forward_weight_version(scheduler, batch)

        iterations = max(5_000, 2_000_000 // batch_size)

        stamp()
        samples = []
        for _ in range(7):
            baseline_ns = measure_ns(noop, iterations)
            stamp_ns = measure_ns(stamp, iterations)
            samples.append(max(0, stamp_ns - baseline_ns) / iterations)

        median_batch_ns = statistics.median(samples)
        results.append(
            {
                "batch_size": batch_size,
                "iterations": iterations,
                "median_ns_per_forward_batch": round(median_batch_ns, 1),
                "median_ns_per_request": round(median_batch_ns / batch_size, 1),
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
