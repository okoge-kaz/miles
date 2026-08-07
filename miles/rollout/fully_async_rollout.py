"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
worker's output queue. Rollout production and training consumption run in parallel,
so per-iteration wall time moves from ``rollout_time + train_time`` toward
``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which also requires the class-based
rollout API (``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1``).

Evaluation is not served by this function; ``--fully-async`` therefore points
``--eval-function-path`` at the standard inference rollout unless it is set
explicitly.
"""

import asyncio
import logging
import time
from collections.abc import Iterator

import httpx
import numpy as np

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnInput, RolloutFnOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils.http_utils import get
from miles.utils.misc import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

OUTPUT_QUEUE_MAX_GROUPS = 1000
NO_PROGRESS_WARN_SECS = 30.0
WEIGHT_VERSION_QUERY_TIMEOUT_SECS = 2.0
# Realized lag is a small integer; anything past this goes in one overflow bucket
# so the metric count stays bounded no matter how far behind a run drifts.
STALENESS_HISTOGRAM_MAX = 8

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]


def _iter_samples(group: Group) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _first_sample(group: Group) -> Sample:
    return group[0][0] if isinstance(group[0], list) else group[0]


def group_oldest_weight_version(group: Group) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [v for s in _iter_samples(group) if (v := s.oldest_weight_version) is not None]
    return min(versions) if versions else None


# Which name carries weight_version depends on the router, so it is discovered
# rather than assumed. Checked against sglang 0.5.17:
#
#   engine (uvicorn)  /model_info is canonical; /get_model_info delegates to it
#                     with a deprecation notice; /get_weight_version raises
#                     (http_server.py:727-733), so it can never answer
#   sglang_router     has its own /model_info route, which 404s; /get_model_info
#                     is not a route, falls through to the engine, and answers
#   MilesRouter       proxies everything (router.py:71), so both reach the engine
#
# Canonical name first: it is the one that survives, and it is right for
# --use-miles-router and for any router that proxies. The fallback covers
# sglang_router as it stands. A wrong first guess costs one 404 round trip, once,
# because the name that answers is remembered.
WEIGHT_VERSION_ENDPOINTS = ("/model_info", "/get_model_info")


class _CachedWeightVersion:
    """Throttled query of the current engine weight version via the router."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query = float("-inf")
        self._failures = 0
        self._endpoint: str | None = None

    async def get(self, args) -> int | None:
        # Throttles failures too: the drain queries once per group, and an unreachable
        # router would otherwise cost every one of them the full timeout.
        if (time.monotonic() - self._last_query) < self._ttl:
            return self._value
        base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        # Remembered name first, then the rest: preferring it costs nothing, and
        # keeping the others behind it means a router replaced under a resumed run
        # recovers within the same query instead of after another TTL.
        candidates = WEIGHT_VERSION_ENDPOINTS
        if self._endpoint:
            candidates = (self._endpoint, *(e for e in candidates if e != self._endpoint))
        last_error: Exception | None = None
        try:
            for endpoint in candidates:
                try:
                    data = await asyncio.wait_for(
                        get(f"{base}{endpoint}"), timeout=WEIGHT_VERSION_QUERY_TIMEOUT_SECS
                    )
                    self._value = int(data["weight_version"])
                    self._endpoint = endpoint
                    self._failures = 0
                    return self._value
                except (httpx.HTTPStatusError, KeyError, TypeError, ValueError) as e:
                    # The endpoint answered but is not the right one: a 404 for a name
                    # this build does not serve, or a payload with no usable
                    # weight_version. Both mean "try the next name".
                    last_error = e
                except (httpx.HTTPError, asyncio.TimeoutError) as e:
                    # The router itself is unreachable, so every other name is too.
                    # Probing them would multiply the timeout by the endpoint count on
                    # a path that already runs once per group.
                    last_error = e
                    break
            self._failures += 1
            self._warn_on_failure(f"{base}{{{','.join(candidates)}}}", last_error)
        finally:
            # Stamped on completion, so a router slower than the TTL still gets throttled.
            self._last_query = time.monotonic()
        return self._value

    def _warn_on_failure(self, url: str, error: Exception) -> None:
        """Loud while no version has ever been read, quiet once one has.

        Until the first success there is no reference version, so
        ``--max-weight-staleness`` silently enforces nothing and no staleness metric
        is emitted -- a run can look like a staleness arm while training fully
        on-policy. That is worth a warning every time. After a success the cached
        value keeps the filter working across a blip, so back off to avoid a warning
        per group.
        """
        if self._value is None:
            logger.warning(
                f"Cannot read the engine weight version from {url} ({error!r}). "
                "--max-weight-staleness cannot be enforced and no staleness metric will be "
                "emitted until this succeeds."
            )
        elif self._failures & (self._failures - 1) == 0:  # 1, 2, 4, 8, ... consecutive
            logger.warning(f"Weight version query failed {self._failures}x, using cached value: {error!r}")


def group_response_tokens(group: Group) -> int:
    """Response tokens generated for a group.

    Call it *before* recycling: ``Sample.reset_for_retry`` clears ``tokens`` and
    ``response`` (``types.py:236``), so a count taken afterwards is always zero.
    """
    return sum(sample.response_length for sample in _iter_samples(group))


def _staleness_metrics(values: list[int], bound: int | None) -> dict[str, float]:
    """P(L) reduced to bounded scalars: the logger takes scalars, not histograms.

    Percentiles rather than a mean alone because the tail is the quantity of
    interest -- a mean of 0.4 with a p99 of 12 and a mean of 0.4 with a p99 of 1
    are different training regimes. ``frac_at_bound`` says whether the configured
    cap is binding at all; a plateau in a results table is otherwise easy to read
    as insensitivity to staleness when the run was simply never stale.
    """
    array = np.asarray(values, dtype=float)
    metrics = {
        "avg_staleness": float(array.mean()),
        "max_staleness": float(array.max()),
        "staleness_p50": float(np.percentile(array, 50)),
        "staleness_p90": float(np.percentile(array, 90)),
        "staleness_p99": float(np.percentile(array, 99)),
        "staleness_frac_zero": float((array <= 0).mean()),
        "staleness_num_groups": float(array.size),
    }
    if bound is not None:
        metrics["staleness_frac_at_bound"] = float((array >= bound).mean())

    # The full histogram, not just moments. Realized lag is a small integer, so
    # P(L) fits in a handful of scalars, and the shape is the result: percentiles
    # cannot say whether lag 4 happened twice or two hundred times, and that is
    # what decides whether a bound of 4 is a real constraint or a formality.
    # Counts are observations, not distinct groups -- a recycled group is counted
    # again when it is regenerated, which is the honest denominator for "how often
    # did the pipeline produce a sample this stale".
    for level in range(STALENESS_HISTOGRAM_MAX + 1):
        metrics[f"staleness_count_{level}"] = float((array == level).sum())
    metrics[f"staleness_count_ge_{STALENESS_HISTOGRAM_MAX + 1}"] = float(
        (array > STALENESS_HISTOGRAM_MAX).sum()
    )
    return metrics


class FullyAsyncRolloutFn:
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Groups whose samples were aborted (e.g. by a
    weight update pausing generation) or whose weights are older than
    ``--max-weight-staleness`` are recycled back into the data source.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self._dynamic_filter = load_function(input.args.dynamic_sampling_filter_path)
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._weight_version = _CachedWeightVersion()
        self._worker: asyncio.Task | None = None
        self._output: asyncio.Queue[tuple[list[Sample], Group]] | None = None

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            raise ValueError(
                "FullyAsyncRolloutFn does not serve eval; set --eval-function-path to "
                "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"
            )
        if self._worker is None:
            self._output = asyncio.Queue(maxsize=OUTPUT_QUEUE_MAX_GROUPS)
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input.rollout_id)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> asyncio.Task:
        [prompt_group] = self.data_source.get_samples(1)
        return asyncio.create_task(self._generate_group(prompt_group))

    async def _generate_group(self, prompt_group: list[Sample]) -> tuple[list[Sample], Group]:
        """Return the submitted prompt group next to its result.

        A retry has to resubmit the prompt group: a generate function may expand one
        trajectory into several samples, and ``generate_and_rm_group`` does not accept
        that shape back.
        """
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
        )
        return prompt_group, result

    async def _worker_loop(self):
        active: set[asyncio.Task] = set()
        while True:
            while len(active) < self._max_in_flight_groups():
                active.add(self._submit_one_group())
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                # Blocks when the queue is full: training lagging behind rollout
                # production pauses submission instead of growing the queue unboundedly.
                await self._output.put(task.result())

    # -------------------------- consumer --------------------------

    async def _next_group(self) -> tuple[list[Sample], Group]:
        queue_get = asyncio.create_task(self._output.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(
                    f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s (queued: {self._output.qsize()})"
                )
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, rollout_id: int) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        aborted_groups_recycled = 0
        stale_groups_recycled = 0
        # Two populations, because they answer different questions and only one of
        # them is the study's variable. ``offered`` is every group the pipeline
        # handed over, including those the bound then sent back -- that is the
        # *natural* lag of this node ratio. ``trained`` is what survived into the
        # batch, and is what the loss actually saw. They diverge exactly where the
        # bound bites, which is where a reader is most likely to be misled.
        trained_staleness: list[int] = []
        offered_staleness: list[int] = []
        current_version: int | None = None
        # Generation that was produced and then thrown away. Counted in tokens, not
        # groups, because that is the unit a sample-efficiency claim is made in, and
        # because the three ways to waste generation cost wildly different amounts.
        aborted_tokens = 0
        stale_tokens = 0
        filtered_tokens = 0
        metric_gatherer = MetricGatherer()
        do_print = True

        while len(data) < target_data_size:
            prompt_group, group = await self._next_group()
            assert len(group) == args.n_samples_per_prompt

            # A weight update paused generation mid-group: return it for re-sampling.
            if any(s.status == Sample.Status.ABORTED for s in _iter_samples(group)):
                aborted_tokens += group_response_tokens(group)
                self._recycle(prompt_group)
                aborted_groups_recycled += 1
                continue

            group_staleness: int | None = None
            if args.max_weight_staleness is not None:
                oldest = group_oldest_weight_version(group)
                current = await self._weight_version.get(args)
                if current is not None:
                    current_version = current
                if oldest is not None and current is not None:
                    staleness = current - oldest
                    group_staleness = staleness
                    offered_staleness.append(staleness)
                    if staleness > args.max_weight_staleness:
                        stale_tokens += group_response_tokens(group)
                        self._recycle(prompt_group)
                        stale_groups_recycled += 1
                        logger.info(
                            f"Recycled stale group (oldest_version={oldest}, current={current}, "
                            f"staleness={staleness} > max={args.max_weight_staleness})"
                        )
                        continue

            filter_output = call_dynamic_filter(self._dynamic_filter, args, group)
            if not filter_output.keep:
                # Dropped, not recycled: no usable gradient signal.
                filtered_tokens += group_response_tokens(group)
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue

            if do_print:
                sample = _first_sample(group)
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, "
                    f"label: {sample.label}, reward: {sample.reward}"
                )
                do_print = False

            if group_staleness is not None:
                trained_staleness.append(group_staleness)
            data.append(group)

        sample = _first_sample(data[-1])
        logger.info(
            f"Finish rollout: {[str(sample.prompt) + sample.response]}, "
            f"label: {sample.label}, reward: {sample.reward}"
        )

        data.sort(key=lambda group: _first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        kept_tokens = sum(group_response_tokens(group) for group in data)
        wasted_tokens = aborted_tokens + stale_tokens + filtered_tokens
        metrics = {
            "rollout/fully_async/queue_size": self._output.qsize(),
            "rollout/fully_async/aborted_groups_recycled": aborted_groups_recycled,
            "rollout/fully_async/stale_groups_recycled": stale_groups_recycled,
            "rollout/fully_async/aborted_tokens": aborted_tokens,
            "rollout/fully_async/stale_tokens": stale_tokens,
            "rollout/fully_async/dynamic_filter_tokens": filtered_tokens,
            "rollout/fully_async/kept_tokens": kept_tokens,
            "rollout/fully_async/wasted_token_frac": (
                wasted_tokens / (wasted_tokens + kept_tokens) if wasted_tokens + kept_tokens else 0.0
            ),
            **metric_gatherer.collect(),
        }
        if current_version is not None:
            # Logged next to the staleness itself: staleness is a difference against
            # this version, and without it a missing staleness metric is impossible to
            # tell apart from a router that never answered.
            metrics["rollout/fully_async/current_weight_version"] = current_version
        # The unprefixed names are the lag the loss saw. The offered distribution
        # keeps its own prefix rather than the short name it used to own: a chart
        # of "staleness" against a bound is read as what was trained on.
        if trained_staleness:
            metrics |= {
                f"rollout/fully_async/{name}": value
                for name, value in _staleness_metrics(trained_staleness, args.max_weight_staleness).items()
            }
        if offered_staleness:
            metrics |= {
                f"rollout/fully_async/offered/{name}": value
                for name, value in _staleness_metrics(offered_staleness, args.max_weight_staleness).items()
            }

        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])
