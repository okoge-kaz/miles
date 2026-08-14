"""Low-cost wall-clock counters for the fully asynchronous pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable


class FullyAsyncPipelineTelemetry:
    """Accumulate event-loop counters and emit non-overlapping rate windows."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        now = clock()
        self._queue_depth = 0
        self._queue_depth_integral = 0.0
        self._queue_depth_updated_at = now
        self._active_groups = 0
        self._max_active_groups = 0
        self._rollout_idle_capacity_integral = 0.0
        self._active_groups_updated_at = now
        self._generated_tokens = 0
        self._generated_groups = 0
        self._completed_training_batches = 0
        self._accepted_tokens = 0
        self._accepted_tokens_known_batches = 0
        self._optimizer_updates = 0
        self._trainer_starvation_seconds = 0.0
        self._rollout_backpressure_seconds = 0.0
        self._snapshot_at = now
        self._snapshot_counters = self._counters()

    def _advance_queue_integral(self, now: float) -> None:
        elapsed = max(0.0, now - self._queue_depth_updated_at)
        self._queue_depth_integral += elapsed * self._queue_depth
        self._queue_depth_updated_at = now

    def set_queue_depth(self, depth: int) -> None:
        if depth < 0:
            raise ValueError(f"Queue depth cannot be negative: {depth}")
        now = self._clock()
        self._advance_queue_integral(now)
        self._queue_depth = depth

    def _advance_active_integral(self, now: float) -> None:
        elapsed = max(0.0, now - self._active_groups_updated_at)
        if self._max_active_groups > 0:
            idle_fraction = 1.0 - self._active_groups / self._max_active_groups
            self._rollout_idle_capacity_integral += elapsed * max(0.0, idle_fraction)
        self._active_groups_updated_at = now

    def set_active_groups(self, active_groups: int, max_active_groups: int) -> None:
        if not 0 <= active_groups <= max_active_groups:
            raise ValueError(
                f"Active rollout groups must be within capacity: active={active_groups}, max={max_active_groups}"
            )
        now = self._clock()
        self._advance_active_integral(now)
        self._active_groups = active_groups
        self._max_active_groups = max_active_groups

    def add_generated_group(self, response_tokens: int) -> None:
        self._generated_groups += 1
        self._generated_tokens += max(0, int(response_tokens))

    def add_trained_batch(self, *, accepted_tokens: int | None, optimizer_updates: int) -> None:
        """Record work only after the actor reports a successful train call."""
        self._completed_training_batches += 1
        self._optimizer_updates += max(0, int(optimizer_updates))
        if accepted_tokens is not None:
            self._accepted_tokens_known_batches += 1
            self._accepted_tokens += max(0, int(accepted_tokens))

    def add_trainer_starvation(self, elapsed_seconds: float) -> None:
        self._trainer_starvation_seconds += max(0.0, elapsed_seconds)

    def add_rollout_backpressure(self, elapsed_seconds: float) -> None:
        self._rollout_backpressure_seconds += max(0.0, elapsed_seconds)

    def reset_window(self) -> None:
        """Start rate accounting when the worker, rather than its owner, starts."""
        now = self._clock()
        self._advance_queue_integral(now)
        self._advance_active_integral(now)
        self._snapshot_at = now
        self._snapshot_counters = self._counters()

    def _counters(self) -> dict[str, float]:
        return {
            "generated_tokens": float(self._generated_tokens),
            "generated_groups": float(self._generated_groups),
            "completed_training_batches": float(self._completed_training_batches),
            "accepted_tokens": float(self._accepted_tokens),
            "accepted_tokens_known_batches": float(self._accepted_tokens_known_batches),
            "optimizer_updates": float(self._optimizer_updates),
            "trainer_starvation_seconds": self._trainer_starvation_seconds,
            "rollout_backpressure_seconds": self._rollout_backpressure_seconds,
            "queue_depth_integral": self._queue_depth_integral,
            "rollout_idle_capacity_integral": self._rollout_idle_capacity_integral,
        }

    def snapshot(self, *, active_groups: int, max_active_groups: int) -> dict[str, float]:
        """Return deltas since the previous snapshot and advance the window."""
        self.set_active_groups(active_groups, max_active_groups)
        now = self._clock()
        self._advance_queue_integral(now)
        current = self._counters()
        elapsed = max(0.0, now - self._snapshot_at)
        deltas = {key: current[key] - self._snapshot_counters[key] for key in current}
        completed_batches = deltas["completed_training_batches"]
        result = {
            "window_seconds": elapsed,
            "generated_tokens": deltas["generated_tokens"],
            "generated_groups": deltas["generated_groups"],
            "completed_training_batches": completed_batches,
            "accepted_tokens": deltas["accepted_tokens"],
            "accepted_tokens_available": float(deltas["accepted_tokens_known_batches"] == completed_batches),
            "optimizer_updates": deltas["optimizer_updates"],
            "trainer_starvation_seconds": deltas["trainer_starvation_seconds"],
            "rollout_backpressure_seconds": deltas["rollout_backpressure_seconds"],
            "rollout_idle_capacity_seconds": deltas["rollout_idle_capacity_integral"],
            "queue_depth_time_mean": (
                deltas["queue_depth_integral"] / elapsed if elapsed > 0.0 else float(self._queue_depth)
            ),
            "queue_depth_current": float(self._queue_depth),
            "active_groups": float(active_groups),
            "max_active_groups": float(max_active_groups),
            "active_group_capacity_fraction": (active_groups / max_active_groups if max_active_groups > 0 else 0.0),
            "active_group_capacity_time_mean": (
                1.0 - deltas["rollout_idle_capacity_integral"] / elapsed if elapsed > 0.0 else 0.0
            ),
        }
        self._snapshot_at = now
        self._snapshot_counters = current
        return result
