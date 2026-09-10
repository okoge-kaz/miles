from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainResultWithTiming:
    """Opt-in train result carrying one worker-local wake-up timer component.

    ``local_wake_up_time`` is measured inside one worker and excludes group
    refresh and Ray dispatch time. Groups may take the maximum worker-local
    duration as an explicitly named component, but it is still not an
    end-to-end wall time. The existing primary-worker ``perf/wake_up_time``
    metric remains independent of this opt-in result.
    """

    result: Any
    local_wake_up_time: float


def unwrap_train_results_with_timing(
    results: Sequence[TrainResultWithTiming],
) -> tuple[list[Any], float]:
    """Restore normal worker results and return their maximum local wake-up time."""
    if not results:
        raise ValueError("Timed train results cannot be empty")
    if not all(isinstance(result, TrainResultWithTiming) for result in results):
        raise TypeError("Every timed train worker must return TrainResultWithTiming")

    return [result.result for result in results], max(result.local_wake_up_time for result in results)
