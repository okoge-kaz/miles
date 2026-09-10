"""Shared arm-grid configuration for reasoning-evaluation tools."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningEvalGrid:
    """One staleness/node-ratio cohort consumed by evaluation and plotting."""

    staleness_levels: tuple[int, ...]
    node_ratios: tuple[tuple[int, int], ...]
    include_colocated: bool

    @property
    def async_arms(self) -> tuple[str, ...]:
        return tuple(
            f"s{staleness}-t{trainer_nodes}r{rollout_nodes}"
            for staleness in self.staleness_levels
            for trainer_nodes, rollout_nodes in self.node_ratios
        )

    @property
    def all_arms(self) -> tuple[str, ...]:
        if self.include_colocated:
            return (*self.async_arms, "s0-colocated")
        return self.async_arms


def _positive_integers(value: str, *, name: str) -> tuple[int, ...]:
    fields = value.split()
    if not fields or any(not field.isdigit() or int(field) < 1 for field in fields):
        raise ValueError(f"{name} must contain positive integers")
    values = tuple(int(field) for field in fields)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _node_ratios(value: str) -> tuple[tuple[int, int], ...]:
    ratios: list[tuple[int, int]] = []
    for field in value.split():
        trainer_text, separator, rollout_text = field.partition(":")
        if (
            separator != ":"
            or not trainer_text.isdigit()
            or not rollout_text.isdigit()
            or int(trainer_text) < 1
            or int(rollout_text) < 1
        ):
            raise ValueError("RATIOS must contain positive T:R pairs")
        ratios.append((int(trainer_text), int(rollout_text)))
    if not ratios or len(set(ratios)) != len(ratios):
        raise ValueError("RATIOS must contain unique T:R pairs")
    return tuple(ratios)


def reasoning_eval_grid_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ReasoningEvalGrid:
    """Read the cohort contract shared with the shell launchers."""
    values = os.environ if environ is None else environ
    include_colocated_text = values.get("INCLUDE_COLOCATED", "1")
    if include_colocated_text not in {"0", "1"}:
        raise ValueError("INCLUDE_COLOCATED must be 0 or 1")
    return ReasoningEvalGrid(
        staleness_levels=_positive_integers(
            values.get("STALENESS_LEVELS", "1 2 4 8"),
            name="STALENESS_LEVELS",
        ),
        node_ratios=_node_ratios(values.get("RATIOS", "1:7 2:6 3:5 4:4")),
        include_colocated=include_colocated_text == "1",
    )
