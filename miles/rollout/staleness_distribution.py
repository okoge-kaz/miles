"""Shared scalar summaries for integer weight-staleness populations."""

from collections.abc import Sequence

import numpy as np

# Values past this fixed-resolution tail use one overflow bucket so metric
# cardinality remains bounded.
STALENESS_HISTOGRAM_MAX = 32


def staleness_distribution_metrics(values: Sequence[int]) -> dict[str, float]:
    """Reduce a non-empty integer lag population to fixed-cardinality scalars."""
    if not values:
        raise ValueError("Cannot summarize an empty staleness population")

    array = np.asarray(values, dtype=float)
    metrics = {
        "mean": float(array.mean()),
        "variance": float(array.var()),
        "std": float(array.std()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "frac_zero": float((array <= 0).mean()),
        "num_groups": float(array.size),
    }
    for level in range(STALENESS_HISTOGRAM_MAX + 1):
        metrics[f"count_{level}"] = float((array == level).sum())
    metrics[f"count_ge_{STALENESS_HISTOGRAM_MAX + 1}"] = float(
        (array > STALENESS_HISTOGRAM_MAX).sum()
    )
    return metrics
