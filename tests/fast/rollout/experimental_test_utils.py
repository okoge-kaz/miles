"""Typed policy versions for the migrated experiment fixtures."""

from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall


def make_weight_versions(versions):
    return [
        WeightVersionsPerCall([WeightVersionSpan(version=str(version), abs_start=i, abs_end=i + 1)])
        for i, version in enumerate(versions)
    ]
