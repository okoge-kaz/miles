import asyncio
import copy
import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING

from miles.rollout.filter_hub.common_filters import group_weight_version_stats
from miles.rollout.replay_data_buffer_checkpoint import (
    commit_replay_data_buffer_checkpoint,
    iter_replay_data_buffer_parts,
    load_replay_data_buffer_auxiliary,
    load_replay_data_buffer_source_state,
    optional_replay_data_buffer_header,
    save_replay_data_buffer_checkpoint,
)
from miles.utils import async_utils

if TYPE_CHECKING:
    from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn

logger = logging.getLogger(__name__)


async def _next_part(parts):
    read = asyncio.create_task(asyncio.to_thread(next, parts, None))
    try:
        return await asyncio.shield(read)
    except asyncio.CancelledError:
        # A running disk-read thread cannot be cancelled; join this one bounded part before closing its iterator.
        await read
        raise


def _configuration_fingerprint(args) -> str:
    keys = (
        "n_samples_per_prompt",
        "rollout_batch_size",
        "async_data_buffer_order",
        "async_staleness_metric",
        "max_weight_staleness",
        "async_unused_samples_handler",
        "dynamic_sampling_filter_path",
        "custom_staleness_function_path",
        "rollout_sample_filter_path",
        "custom_generate_function_path",
        "rm_type",
        "custom_rm_path",
        "reward_key",
        "rollout_temperature",
        "rollout_top_p",
        "rollout_top_k",
        "rollout_max_response_len",
        "rollout_max_context_len",
    )
    return hashlib.sha256(
        json.dumps({key: getattr(args, key, None) for key in keys}, sort_keys=True).encode()
    ).hexdigest()


def _snapshot_entries(owner):
    entries = (*owner._output.snapshot_entries(), *owner._pending_ready.values())
    if getattr(owner.args, "async_data_buffer_order", "fifo") == "oldest-policy-first":

        def priority(entry):
            version = group_weight_version_stats(entry.group).oldest_version
            return version is None, version if version is not None else 0

        entries = tuple(sorted(entries, key=priority))
    return entries


class ReplayDataBuffer:
    """Checkpoint lifecycle for ready data and unfinished prompts in the built-in DataBuffer."""

    def __init__(self, owner: "FullyAsyncRolloutFn"):
        self._owner = owner
        self._args = owner.args
        self._fingerprint = _configuration_fingerprint(owner.args)
        self._restore_task: asyncio.Task | None = None
        self._restore_metrics = {}
        self.mode = getattr(owner.args, "replay_data_buffer_mode", "rollout")
        self.inflight = None
        if self.mode == "inflight":
            # The generator/router dependencies are only needed by inflight mode.
            from miles.rollout.replay_data_buffer_inflight import InflightReplay

            self.inflight = InflightReplay(owner)
        for method in ("state_dict", "load_state_dict", "add_samples"):
            if not callable(getattr(owner.data_source, method, None)):
                raise ValueError(f"ReplayDataBuffer data source must implement {method}()")

    def load(self, rollout_id: int | None) -> None:
        if self._args.load is None or rollout_id is None or rollout_id < 0:
            return
        header = optional_replay_data_buffer_header(self._args.load, rollout_id)
        if header is None:
            return
        if header.get("mode", "rollout") != self.mode:
            raise ValueError("ReplayDataBuffer mode differs from the checkpoint")
        if header["configuration_fingerprint"] != self._fingerprint:
            raise ValueError("ReplayDataBuffer sampling/filter/ordering settings differ from the checkpoint")
        state = load_replay_data_buffer_source_state(self._args.load, rollout_id, header)
        self._owner.data_source.load_state_dict(state)
        self._owner.on_weight_version(header["served_version"])
        async_utils.run(self._start_restore(rollout_id, header))

    async def _start_restore(self, rollout_id: int, header: dict) -> None:
        started = time.monotonic()
        self._owner._ensure_buffer()
        parts = iter_replay_data_buffer_parts(self._args.load, rollout_id, header)
        asynchronous = getattr(self._args, "replay_data_buffer_async_restore", False)
        if asynchronous:
            if self._owner._output._capacity < self._args.rollout_batch_size:
                raise ValueError("Async ReplayDataBuffer restore needs capacity for one training batch")
            first = await _next_part(parts)
            for entry in first or []:
                await self._owner._output.restore_entry(entry)
            self._restore_metrics = {
                "rollout/replay_data_buffer/first_part_seconds": time.monotonic() - started,
                "rollout/replay_data_buffer/restored_first_part_groups": float(len(first or [])),
                "rollout/replay_data_buffer/checkpoint_groups": float(header["entry_count"]),
            }
            self._restore_task = asyncio.create_task(
                self._restore_remaining(parts, rollout_id, header, wait_for_capacity=True)
            )
            logger.info(
                "ReplayDataBuffer first part ready: %s; remaining parts restore in background", self._restore_metrics
            )
        else:
            self._restore_task = asyncio.create_task(
                self._restore_remaining(parts, rollout_id, header, wait_for_capacity=False)
            )
            await self._restore_task
            self._restore_metrics = {
                "rollout/replay_data_buffer/full_restore_seconds": time.monotonic() - started,
                "rollout/replay_data_buffer/checkpoint_groups": float(header["entry_count"]),
            }

    async def _restore_remaining(self, parts, rollout_id: int, header: dict, *, wait_for_capacity: bool) -> None:
        try:
            while (part := await _next_part(parts)) is not None:
                for entry in part:
                    await self._owner._output.restore_entry(entry, wait_for_capacity=wait_for_capacity)
            auxiliary = await asyncio.to_thread(load_replay_data_buffer_auxiliary, self._args.load, rollout_id, header)
            if self.inflight is not None:
                self.inflight.pending.extend(auxiliary["inflight_entries"])
            pending = auxiliary["pending_prompts"]
            self._owner.data_source.add_samples(pending)
            logger.info("ReplayDataBuffer restore fully verified; regenerated pending prompt groups=%d", len(pending))
        finally:
            parts.close()

    async def before_production(self) -> None:
        if self._restore_task is not None:
            await self._restore_task

    def check_restore(self) -> None:
        if self._restore_task is not None and self._restore_task.done():
            self._restore_task.result()

    def pop_metrics(self) -> dict:
        metrics, self._restore_metrics = self._restore_metrics, {}
        return metrics

    def save(self, rollout_id: int) -> None:
        async_utils.run(self._save(rollout_id))

    async def _save(self, rollout_id: int) -> None:
        owner = self._owner
        if owner._draining:
            raise RuntimeError("ReplayDataBuffer snapshot requires a completed training batch, not a prefetched drain")
        owner._ensure_buffer()
        if self._restore_task is not None and not self._restore_task.done():
            # A save must include the unconsumed tail. Complete the loader without
            # waiting for a consumer that is currently blocked in this save.
            await owner._output.request_restore_completion()
            await self._restore_task
            owner._output._complete_restore_for_checkpoint = False
        self.check_restore()
        owner._producer_resumed.clear()
        try:
            if self.inflight is not None:
                await self.inflight.capture()
            entries = _snapshot_entries(owner)
            pending = tuple(copy.deepcopy(group) for group in getattr(owner.data_source, "buffer", ()))
            active = tuple(owner._original_prompts[task] for task in owner._active if task not in owner._pending_ready)
            state = copy.deepcopy(owner.data_source.state_dict())
            version = owner._weight_version
            if version is None:
                raise ValueError("Cannot save ReplayDataBuffer before an engine weight version was published")
            # Cursor and ready/active ownership are captured on the producer loop
            # without an intervening await. The legacy cursor file uses that cut too.
            owner.data_source.save(rollout_id)
            await asyncio.to_thread(
                save_replay_data_buffer_checkpoint,
                self._args.save,
                rollout_id,
                entries=entries,
                pending_prompts=(*pending, *active),
                source_state=state,
                served_version=version,
                configuration_fingerprint=self._fingerprint,
                groups_per_part=self._args.rollout_batch_size,
                mode=self.mode,
                inflight_entries=tuple(self.inflight.pending) if self.inflight is not None else (),
            )
            commit_replay_data_buffer_checkpoint(self._args.save, rollout_id)
        finally:
            if self.inflight is not None:
                self.inflight.resume()
            owner._producer_resumed.set()

    async def close(self) -> None:
        if self._restore_task is not None and not self._restore_task.done():
            self._restore_task.cancel()
            await asyncio.gather(self._restore_task, return_exceptions=True)
