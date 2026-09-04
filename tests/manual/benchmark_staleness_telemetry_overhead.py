#!/usr/bin/env python3
"""Measure CPU overhead of the always-on fully-async telemetry helpers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

from miles.rollout.recycle_compute_metrics import (
    ADMITTED_TOKENS_KEY,
    BOUND_REFERENCE_VERSION_KEY,
    DRAIN_TIME_KEY,
    DRAIN_VERSION_KEY,
    GENERATED_TOKENS_KEY,
    GROUP_GENERATION_COMPLETE_TIME_KEY,
    GROUP_GENERATION_COMPLETE_VERSION_KEY,
    GROUP_READY_TIME_KEY,
    GROUP_READY_VERSION_KEY,
    LIFECYCLE_EXACT_KEY,
    QUEUE_PUT_TIME_KEY,
    SAMPLE_GENERATION_COMPLETE_TIME_KEY,
    SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
    SAMPLE_REFERENCE_VERSION_KEY,
    TRAIN_VERSION_KEY,
    TRAIN_WEIGHT_VERSION_METRIC,
    TRAJECTORY_START_TIME_KEY,
    TRAJECTORY_START_VERSION_KEY,
    add_selection_population,
    finalize_useful_rollout_metrics,
    prequeue_phase_metrics,
    selection_population_metrics,
    waste_vector,
)
from miles.utils.types import Sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3072)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=31)
    return parser.parse_args()


def _samples(count: int, group_size: int) -> list[Sample]:
    samples = []
    for index in range(count):
        response_length = 256 + index % 1792
        sample = Sample(
            group_index=index // group_size,
            index=index,
            response_length=response_length,
            loss_mask=[1] * (response_length - 1) + [0],
            reward=float(index % 2),
            non_generation_time=0.001 * (index % 7),
        )
        sample.prefix_cache_info.cached_tokens = 128
        sample.prefix_cache_info.total_prompt_tokens = 512
        start = 100 + index % 3
        sample_complete = start + index % 4
        group_complete = sample_complete + index % 2
        ready = group_complete + index % 3
        sample.metadata.update(
            {
                TRAJECTORY_START_VERSION_KEY: start,
                SAMPLE_GENERATION_COMPLETE_VERSION_KEY: sample_complete,
                GROUP_GENERATION_COMPLETE_VERSION_KEY: group_complete,
                GROUP_READY_VERSION_KEY: ready,
                SAMPLE_REFERENCE_VERSION_KEY: start,
                BOUND_REFERENCE_VERSION_KEY: start,
                DRAIN_VERSION_KEY: ready + 1,
                TRAIN_VERSION_KEY: ready + 2,
                TRAJECTORY_START_TIME_KEY: float(start),
                SAMPLE_GENERATION_COMPLETE_TIME_KEY: float(sample_complete),
                GROUP_GENERATION_COMPLETE_TIME_KEY: float(group_complete),
                GROUP_READY_TIME_KEY: float(ready),
                QUEUE_PUT_TIME_KEY: float(ready),
                DRAIN_TIME_KEY: float(ready + 1),
                LIFECYCLE_EXACT_KEY: True,
            }
        )
        samples.append(sample)
    return samples


def _measure(call: Callable[[], None], repetitions: int) -> dict[str, float]:
    call()
    elapsed_ms = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        call()
        elapsed_ms.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(elapsed_ms),
        "p90_ms": sorted(elapsed_ms)[int(0.9 * (len(elapsed_ms) - 1))],
    }


def _set_exact_segments(samples: list[Sample], *, segments_per_sample: int) -> None:
    for sample in samples:
        if segments_per_sample == 0:
            sample.response_weight_version_segments = []
        elif segments_per_sample == 1:
            sample.response_weight_version_segments = [[[0, sample.response_length, 109]]]
        elif segments_per_sample == 2:
            split = sample.response_length // 2
            sample.response_weight_version_segments = [[[0, split, 108], [split, sample.response_length, 109]]]
        else:
            raise ValueError(f"Unsupported segment count: {segments_per_sample}")


def main() -> None:
    args = _parse_args()
    if args.samples <= 0 or args.group_size <= 0 or args.repetitions <= 0:
        raise ValueError("sample, group, and repetition counts must be positive")
    samples = _samples(args.samples, args.group_size)
    generated_tokens = sum(sample.response_length for sample in samples)

    def population_pass() -> None:
        populations: dict[str, dict[str, list[float]]] = {}
        add_selection_population(populations, population_name="generated", samples=samples)
        add_selection_population(populations, population_name="consumed", samples=samples)
        selection_population_metrics(populations)

    def finalization_pass() -> None:
        metrics = {
            GENERATED_TOKENS_KEY: generated_tokens,
            ADMITTED_TOKENS_KEY: generated_tokens,
            "rollout/fully_async/aborted_tokens": 0,
            "rollout/fully_async/stale_tokens": 0,
            "rollout/fully_async/dynamic_filter_tokens": 0,
            TRAIN_WEIGHT_VERSION_METRIC: 110,
        }
        finalize_useful_rollout_metrics(samples, metrics, has_custom_converter=False)

    group = samples[: args.group_size]
    _set_exact_segments(samples, segments_per_sample=0)
    no_exact = _measure(finalization_pass, args.repetitions)
    _set_exact_segments(samples, segments_per_sample=1)
    exact_single = _measure(finalization_pass, args.repetitions)
    _set_exact_segments(samples, segments_per_sample=2)
    exact_mixed = _measure(finalization_pass, args.repetitions)

    for sample in samples:
        sample.loss_mask = [1] * sample.response_length
    _set_exact_segments(samples, segments_per_sample=0)
    all_loss_no_exact = _measure(finalization_pass, args.repetitions)
    _set_exact_segments(samples, segments_per_sample=1)
    all_loss_exact_single = _measure(finalization_pass, args.repetitions)

    results = {
        "configuration": {
            "samples": args.samples,
            "group_size": args.group_size,
            "generated_tokens": generated_tokens,
            "repetitions": args.repetitions,
        },
        "generated_and_consumed_population_pass": _measure(population_pass, args.repetitions),
        "batch_finalization_no_exact_segments": no_exact,
        "batch_finalization_exact_single_segment_masked_tail": exact_single,
        "batch_finalization_exact_two_segments_masked_tail": exact_mixed,
        "batch_finalization_all_loss_tokens_no_exact_segments": all_loss_no_exact,
        "batch_finalization_all_loss_tokens_exact_single_segment": all_loss_exact_single,
        "exact_segment_increment": {
            "single_segment_masked_tail_median_ms": exact_single["median_ms"] - no_exact["median_ms"],
            "two_segments_masked_tail_median_ms": exact_mixed["median_ms"] - no_exact["median_ms"],
            "single_segment_all_loss_tokens_median_ms": (
                all_loss_exact_single["median_ms"] - all_loss_no_exact["median_ms"]
            ),
        },
        "prequeue_phase_pass": _measure(lambda: prequeue_phase_metrics(samples), args.repetitions),
        "discard_waste_vector_one_group": _measure(lambda: waste_vector(group), args.repetitions),
    }
    for name, value in results.items():
        if isinstance(value, dict) and "median_ms" in value:
            denominator = args.group_size if name == "discard_waste_vector_one_group" else args.samples
            value["median_ns_per_sample"] = value["median_ms"] * 1e6 / denominator
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
