"""Schema-safe adapters for executable software-engineering tasks."""

from experiments.src.datasets.swe.schema import (
    NormalizedSWETask,
    normalize_swe_row,
)

__all__ = ["NormalizedSWETask", "normalize_swe_row"]
