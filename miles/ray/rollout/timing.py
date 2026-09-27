"""Optional rollout wall timers; thread CPU applies only to synchronous phases."""

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def measure_rollout_phase(metrics: dict[str, float] | None, name: str, *, thread_cpu: bool = True) -> Iterator[None]:
    if metrics is None:
        yield
        return

    started = time.perf_counter()
    cpu_started = time.thread_time() if thread_cpu else None
    try:
        yield
    finally:
        metrics[f"perf/rollout_{name}_time"] = time.perf_counter() - started
        if cpu_started is not None:
            metrics[f"perf/rollout_{name}_thread_cpu_time"] = time.thread_time() - cpu_started
