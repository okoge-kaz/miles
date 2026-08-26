from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from collections import deque
from contextlib import suppress
from dataclasses import replace

import httpx
import pytest

import miles.rollout.fully_async_rollout as fully_async
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalInput, RolloutFnTrainInput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

N_SAMPLES_PER_PROMPT = 2


class FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.aborted = False


class FakeDataSource:
    """Serves scripted groups first, then manufactures completed groups forever."""

    def __init__(self, scripted=None):
        self.scripted = deque(scripted or [])
        self.next_group_index = 1000
        self.recycled = []
        self.num_get_calls = 0

    def get_samples(self, num_samples):
        assert num_samples == 1
        self.num_get_calls += 1
        if self.scripted:
            return [self.scripted.popleft()]
        self.next_group_index += 1
        return [make_group(self.next_group_index)]

    def add_samples(self, groups):
        self.recycled.extend(groups)


def make_group(
    group_index: int,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_versions: list[str] | None = None,
) -> list[Sample]:
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1,
            status=status,
            weight_versions=list(weight_versions or []),
        )
        for i in range(N_SAMPLES_PER_PROMPT)
    ]


def make_args(**overrides) -> Namespace:
    defaults = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_weight_staleness=None,
        async_max_concurrent_samples=None,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        staleness_reference="completion",
        save_debug_rollout_data=None,
        fully_async_queue_type="queue-recycle",
        fully_async_queue_factor=1,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def make_fn(monkeypatch, args, data_source, generate=None):
    async def default_generate(state, group, sampling_params, evaluation=False):
        await asyncio.sleep(0)
        return group

    monkeypatch.setattr(fully_async, "GenerateState", FakeGenerateState)
    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate or default_generate)
    return fully_async.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))


async def test_drain_collects_batch_sorted_with_metrics(monkeypatch):
    args = make_args(rollout_batch_size=3)
    fn = make_fn(monkeypatch, args, FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 3
    indices = [group[0].index for group in output.samples]
    assert indices == sorted(indices)
    assert all(len(group) == N_SAMPLES_PER_PROMPT for group in output.samples)
    assert output.metrics["rollout/fully_async/aborted_groups_recycled"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0

    # The worker persists across calls; a second drain works on the same instance.
    output2 = await fn(RolloutFnTrainInput(rollout_id=1))
    assert len(output2.samples) == 3


async def test_queue_lifecycle_dump_is_primitive_and_opt_in(monkeypatch):
    group = make_group(1, weight_versions=["3"])
    group[0].response_length = 2
    group[1].response_length = 4
    args = make_args(rollout_batch_size=1, save_debug_rollout_data="/tmp/rollout_{rollout_id}.pt")
    fn = make_fn(monkeypatch, args, FakeDataSource(scripted=[group]))
    fn.commit_applied_weight_version(3)

    output = await fn(RolloutFnTrainInput(rollout_id=7, updates_before_train=1))

    assert output.debug_metadata["schema_version"] == 2
    assert output.debug_metadata["policy"] == "queue-recycle"
    assert output.debug_metadata["bound_staleness_semantics"] == "scheduled_train_version_minus_reference"
    [record] = output.debug_metadata["records"]
    assert record["disposition"] == "trained"
    assert record["rollout_id"] == 7
    assert record["response_lengths"] == [2, 4]
    assert record["reward_values"] == [1.0, 1.0]
    assert record["completion_version_min"] == 3
    assert record["ready_version"] == 3
    assert record["queue_depth_before_enqueue"] == 0
    assert record["queue_depth_after_enqueue"] == 1
    assert record["decision_version"] == 3
    assert record["train_version"] == 4
    assert record["bound_staleness"] == 1
    assert not any(isinstance(value, Sample) for value in record.values())
    [compute_record] = output.debug_metadata["recycle_compute"]["records"]
    assert compute_record["versions"]["drain"] == 3
    assert compute_record["versions"]["train"] == 4
    assert compute_record["in_queue_staleness"] == [1, 1]

    no_dump = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())
    no_dump_output = await no_dump(RolloutFnTrainInput(rollout_id=0))
    assert no_dump_output.debug_metadata is None


def test_scheduled_train_version_is_fixed_for_the_whole_drain():
    progress = fully_async._DrainProgress(rollout_id=3, updates_before_train=1)

    assert fully_async._scheduled_train_version(progress, dequeue_version=5) == 6
    assert fully_async._scheduled_train_version(progress, dequeue_version=6) == 6
    with pytest.raises(RuntimeError, match="exceeds scheduled train version 6"):
        fully_async._scheduled_train_version(progress, dequeue_version=7)


async def test_eval_raises(monkeypatch):
    fn = make_fn(monkeypatch, make_args(), FakeDataSource())
    with pytest.raises(ValueError, match="does not serve eval"):
        await fn(RolloutFnEvalInput(rollout_id=0))
    assert fn._worker is None


async def test_aborted_group_recycled(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    data_source = FakeDataSource(scripted=[aborted])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [aborted]
    # reset_for_retry cleared generated outputs so the prompt can be re-sampled
    assert all(sample.response == "" and sample.weight_versions == [] for sample in aborted)
    assert output.samples[0][0].group_index != 1
    assert output.metrics["rollout/fully_async/aborted_groups_recycled"] == 1


async def test_stale_group_recycled(monkeypatch):
    stale = make_group(1, weight_versions=["5"])
    stale[0].response_length = 1
    stale[1].response_length = 3
    data_source = FakeDataSource(scripted=[stale])
    data_source_fresh_versions = ["10"]

    original_make = data_source.get_samples

    def get_samples_with_fresh_versions(num_samples):
        groups = original_make(num_samples)
        for group in groups:
            for sample in group:
                if not sample.weight_versions:
                    sample.weight_versions = list(data_source_fresh_versions)
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)

    class FakeWeightVersion:
        async def get(self, args):
            return 10

    fn._weight_version = FakeWeightVersion()
    fn.commit_applied_weight_version(10)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [stale]
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 1
    # The absolute scheduled train version is logged alongside all relative gaps.
    assert output.metrics["fully_async/train_weight_version"] == 10

    # Queue selection is group-valued. The stale group's slowest sample had
    # length 3, while the replacement group has two length-1 samples.
    assert output.metrics["queue/selection/stale_recycled/sample_length/mean"] == pytest.approx(2.0)
    assert output.metrics["queue/selection/stale_recycled/group_max_length/max"] == 3
    assert output.metrics["queue/selection/trained/sample_length/mean"] == pytest.approx(1.0)
    assert output.metrics["queue/selection/offered/sample_length/count"] == 4
    assert output.metrics["queue/selection/offered/sample_length/sum"] == 6
    assert output.metrics["queue/selection/generated/sample_length/count"] >= 4
    assert output.metrics["queue/selection/aborted_recycled/group_max_length/count"] == 0

    # `staleness/total/` is the accepted training population. The group at 5
    # exceeded the bound of 2 and was recycled, so only the fresh group at 10 --
    # lag 0 -- was trained on.
    assert output.metrics["staleness/total/max"] == 0
    assert output.metrics["staleness/total/num_groups"] == 1
    assert output.metrics["staleness/total/frac_zero"] == pytest.approx(1.0)
    assert output.metrics["staleness/total/count_0"] == 1
    assert output.metrics["staleness/total/count_5"] == 0

    # `staleness/rollout/` keeps both: it is the natural lag of this node ratio,
    # and the gap between the two is what recycling cost.
    assert output.metrics["staleness/rollout/max"] == 5
    assert output.metrics["staleness/rollout/num_groups"] == 2
    assert output.metrics["staleness/rollout/p50"] == pytest.approx(2.5)
    assert output.metrics["staleness/rollout/frac_zero"] == pytest.approx(0.5)

    # Tokens counted before reset_for_retry cleared them.
    stale_response_tokens = 4
    assert output.metrics["rollout/fully_async/stale_tokens"] == stale_response_tokens
    assert output.metrics["rollout/fully_async/recycle_reason/stale_during_reward_finalize/groups"] == 1
    assert (
        output.metrics["rollout/fully_async/waste/stale_during_reward_finalize/decode_tokens"] == stale_response_tokens
    )
    assert output.metrics["selection_bias/generated/samples"] >= 2 * N_SAMPLES_PER_PROMPT
    assert output.metrics["selection_bias/recycled/samples"] == N_SAMPLES_PER_PROMPT
    assert output.metrics["selection_bias/admitted/samples"] == N_SAMPLES_PER_PROMPT
    assert output.metrics[fully_async.GENERATED_TOKENS_KEY] == stale_response_tokens + N_SAMPLES_PER_PROMPT
    assert output.metrics[fully_async.ADMITTED_TOKENS_KEY] == N_SAMPLES_PER_PROMPT

    # Named for the cause, under the section that owns it, so the bound's cost does
    # not have to be reconstructed by differencing group counts -- which would fold
    # in the dynamic-filter drops.
    assert output.metrics["staleness/bound_exceeded_groups"] == 1
    assert output.metrics["staleness/bound_exceeded_samples"] == N_SAMPLES_PER_PROMPT
    assert output.metrics["staleness/bound_exceeded_sample_frac"] == pytest.approx(0.5)
    assert output.metrics["staleness/bound_exceeded_tokens"] == 4

    # The recycled group was regenerated, so its retry counter advanced. The group
    # that trained this step was never recycled.
    assert all(sample.retry_count == 1 for sample in stale)
    assert output.metrics["staleness/retry_count_max"] == 0
    assert output.metrics["staleness/retry_frac_nonzero"] == pytest.approx(0.0)


def test_retry_count_survives_the_reset_that_increments_it():
    """`reset_for_retry` wipes the generated output; the counter is not output.

    It counts calls to that method, so resetting it there would pin it at zero --
    and the failure is silent, because every other field it touches *should* be
    cleared.
    """
    sample = Sample(prompt="p", tokens=[1, 2, 3], response="r", response_length=3, retry_count=2)
    sample.reset_for_retry()

    assert sample.retry_count == 2
    assert sample.tokens == [] and sample.response == "" and sample.weight_versions == []

    # It also has to survive the dump round trip, which is where the offline
    # analysis reads it from.
    assert Sample.from_dict(sample.to_dict()).retry_count == 2


async def test_trained_staleness_excludes_dynamic_filter_drops(monkeypatch):
    """A group the filter drops never reached the loss, so it is not trained lag.

    The bound is the obvious way a group leaves the batch; the dynamic filter is
    the one that is easy to forget, because it runs *after* the staleness check
    and drops rather than recycles.
    """
    stale_but_allowed = make_group(1, weight_versions=["9"])
    data_source = FakeDataSource(scripted=[stale_but_allowed])

    original_make = data_source.get_samples

    def get_samples_with_fresh_versions(num_samples):
        groups = original_make(num_samples)
        for group in groups:
            for sample in group:
                if not sample.weight_versions:
                    sample.weight_versions = ["10"]
        return groups

    data_source.get_samples = get_samples_with_fresh_versions

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=4), data_source)

    class FakeWeightVersion:
        async def get(self, args):
            return 10

    fn._weight_version = FakeWeightVersion()
    fn.commit_applied_weight_version(10)

    # Drop the scripted group -- lag 1, comfortably inside the bound of 4 -- and
    # keep the fresh one that replaces it.
    def reject_the_stale_one(args, group, **kwargs):
        keep = group[0].group_index != 1
        return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")

    fn._dynamic_filter = reject_the_stale_one

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    # It was offered at lag 1 and the bound did not stop it...
    assert output.metrics["staleness/rollout/count_1"] == 1
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    # ...but the filter dropped it, so the loss only ever saw the lag-0 group.
    assert output.metrics["staleness/total/count_1"] == 0
    assert output.metrics["staleness/total/count_0"] == 1
    assert output.metrics["staleness/total/num_groups"] == 1


async def test_wasted_token_accounting(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    data_source = FakeDataSource(scripted=[aborted])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    # response_length is 1 per sample in make_group, so the counts are sample counts.
    assert output.metrics["rollout/fully_async/aborted_tokens"] == N_SAMPLES_PER_PROMPT
    assert output.metrics["rollout/fully_async/kept_tokens"] == N_SAMPLES_PER_PROMPT
    assert output.metrics["rollout/fully_async/wasted_token_frac"] == pytest.approx(0.5)


async def test_generated_token_denominator_precedes_post_generation_sample_filter(monkeypatch):
    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1),
        FakeDataSource(),
    )

    def trim_all_responses(args, data):
        del args
        for group in data:
            for sample in group:
                sample.response = ""
                sample.response_length = 0
                sample.loss_mask = []

    fn._sample_filter = trim_all_responses
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert output.metrics[fully_async.GENERATED_TOKENS_KEY] == N_SAMPLES_PER_PROMPT
    assert output.metrics[fully_async.ADMITTED_TOKENS_KEY] == N_SAMPLES_PER_PROMPT
    assert output.metrics["rollout/fully_async/kept_tokens"] == 0


def test_staleness_histogram_reports_the_shape_not_just_moments():
    """Percentiles cannot distinguish a rare tail from a common one.

    ``staleness_p99`` alone cannot say whether lag 4 happened twice or two
    hundred times.
    """
    cap = fully_async.STALENESS_HISTOGRAM_MAX
    overflow = f"count_ge_{cap + 1}"
    values = [0] * 90 + [1] * 8 + [4, 12, cap + 3]
    m = fully_async._staleness_metrics(values)

    assert m["count_0"] == 90
    assert m["count_1"] == 8
    assert m["count_4"] == 1
    # 12 is resolved rather than swallowed: the cap is above it precisely so the
    # tail of an unbounded `staleness/total` stays readable.
    assert m["count_12"] == 1
    assert m[overflow] == 1
    assert sum(m[f"count_{i}"] for i in range(cap + 1)) + m[overflow] == len(values)
    # The moments still agree with the histogram.
    assert m["max"] == cap + 3
    assert m["frac_zero"] == 90 / len(values)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    assert m["variance"] == pytest.approx(variance)
    assert m["std"] == pytest.approx(variance**0.5)


def test_shared_staleness_reducer_preserves_fully_async_legacy_api():
    from miles.rollout.staleness_distribution import staleness_distribution_metrics

    values = [0, 1, 4, fully_async.STALENESS_HISTOGRAM_MAX + 1]

    assert fully_async._staleness_metrics(values) == staleness_distribution_metrics(values)


async def test_weight_version_endpoint_is_discovered_not_assumed(monkeypatch):
    """A 404 on one endpoint name must fall through to the next.

    SGLang renamed this endpoint twice, and a router that does not serve a given
    name answers 404 rather than erroring. Assuming one name made the cap silently
    unenforceable on a build that serves another: job 15204795 logged 285 failures
    against /model_info while /get_model_info returned weight_version fine.
    """
    seen = []

    async def only_get_model_info(url, *args, **kwargs):
        seen.append(url)
        if url.endswith("/get_model_info"):
            return {"weight_version": "4"}
        raise httpx.HTTPStatusError("404", request=None, response=None)

    monkeypatch.setattr(fully_async, "get", only_get_model_info)
    cache = fully_async._CachedWeightVersion()
    args = make_args(max_weight_staleness=2)

    assert await cache.get(args) == 4
    assert any(u.endswith("/get_model_info") for u in seen)

    # The canonical name is tried first even though it is the one that 404s here.
    assert seen[0].endswith("/model_info")

    # The working name is remembered, so later queries do not re-probe the others.
    cache._last_query = float("-inf")
    seen.clear()
    assert await cache.get(args) == 4
    assert [u.rsplit("/", 1)[-1] for u in seen] == ["get_model_info"]


@pytest.mark.asyncio
async def test_remembered_endpoint_is_dropped_when_it_stops_answering(monkeypatch):
    """A router swapped under a resumed run must not pin the cache to a dead name."""
    serving = {"name": "/get_model_info"}

    async def only_current(url, *args, **kwargs):
        if url.endswith(serving["name"]):
            return {"weight_version": "7"}
        raise httpx.HTTPStatusError("404", request=None, response=None)

    monkeypatch.setattr(fully_async, "get", only_current)
    cache = fully_async._CachedWeightVersion(ttl=0.0)
    args = make_args(max_weight_staleness=2)
    assert await cache.get(args) == 7
    assert cache._endpoint == "/get_model_info"

    serving["name"] = "/model_info"  # router replaced
    assert await cache.get(args) == 7
    assert cache._endpoint == "/model_info"


async def test_unreachable_router_only_disables_submission_diagnostics(monkeypatch, caplog):
    """The HTTP cache is diagnostic-only; tracker-based drain control remains live."""

    async def unreachable(url, *args, **kwargs):
        raise httpx.ConnectError("no router")

    monkeypatch.setattr(fully_async, "get", unreachable)
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), FakeDataSource())

    with caplog.at_level("WARNING"):
        output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert "submission_weight_version diagnostics will be absent" in caplog.text
    assert "staleness/rollout/mean" not in output.metrics
    assert output.metrics["fully_async/train_weight_version"] == 0


async def test_malformed_model_info_does_not_kill_the_drain(monkeypatch):
    """A /model_info payload without weight_version used to raise KeyError out of
    the drain; it is a configuration failure and must degrade, not crash."""

    async def no_version(url, *args, **kwargs):
        return {}

    monkeypatch.setattr(fully_async, "get", no_version)
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), FakeDataSource())

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 1


async def test_worker_error_propagates(monkeypatch):
    async def failing_generate(state, group, sampling_params, evaluation=False):
        raise RuntimeError("generation exploded")

    fn = make_fn(monkeypatch, make_args(), FakeDataSource(), generate=failing_generate)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_worker_bounds_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False):
        await release.wait()
        return group

    data_source = FakeDataSource()
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 2  # in-flight bound, not more

    release.set()
    output = await drain
    assert len(output.samples) == 2


async def test_async_max_concurrent_samples_caps_in_flight_groups(monkeypatch):
    release = asyncio.Event()

    async def blocking_generate(state, group, sampling_params, evaluation=False):
        await release.wait()
        return group

    data_source = FakeDataSource()
    # 3 samples // 2 per group -> 1 group in flight, below rollout_batch_size
    args = make_args(rollout_batch_size=4, async_max_concurrent_samples=3)
    fn = make_fn(monkeypatch, args, data_source, generate=blocking_generate)

    drain = asyncio.create_task(fn(RolloutFnTrainInput(rollout_id=0)))
    await asyncio.sleep(0.05)
    assert data_source.num_get_calls == 1

    release.set()
    output = await drain
    assert len(output.samples) == 4


async def test_queue_recycle_preserves_completion_fifo_and_backpressure(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())
    fn._output = asyncio.Queue()
    fn._output_slots = asyncio.Semaphore(fully_async.OUTPUT_QUEUE_MAX_GROUPS)
    never = asyncio.Event()
    fn._worker = asyncio.create_task(never.wait())

    first, second = make_group(2), make_group(1)
    await fn._enqueue_completed_group((first, first))
    await fn._enqueue_completed_group((second, second))

    selected = [await fn._next_group(), await fn._next_group()]
    assert [item[1][0].group_index for item in selected] == [2, 1]
    assert fn._output_slots._value == fully_async.OUTPUT_QUEUE_MAX_GROUPS

    fn._worker.cancel()
    with suppress(asyncio.CancelledError):
        await fn._worker


async def test_queue_drop_evicts_oldest_and_keeps_completion_fifo(monkeypatch):
    args = make_args(
        rollout_batch_size=2,
        fully_async_queue_type="queue-drop",
        fully_async_queue_factor=1,
        max_weight_staleness=None,
        save_debug_rollout_data="/tmp/rollout_{rollout_id}.pt",
    )
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._policy_output = deque()
    fn._policy_output_ready = asyncio.Event()

    groups = [make_group(group_index) for group_index in (1, 2, 3)]
    for group in groups:
        record = fn._queue_lifecycle.begin_attempt(group, submission_version=0)
        fn._queue_lifecycle.group_ready(record, group, ready_version=0)
        await fn._enqueue_completed_group((group, group))

    assert [item[1][0].group_index for item in fn._policy_output] == [2, 3]
    assert fn._queue_evicted_groups == 1
    assert fn._queue_evicted_tokens == N_SAMPLES_PER_PROMPT

    selected = await fn._take_policy_groups(2)
    assert [item[0][1][0].group_index for item in selected] == [2, 3]
    assert [depth for _, depth in selected] == [1, 0]

    metadata = fn._queue_lifecycle.take_metadata(policy="queue-drop", capacity_groups=2)
    [evicted] = metadata["records"]
    assert evicted["group_index"] == 1
    assert evicted["disposition"] == "queue_evicted"
    assert evicted["bound_staleness"] is None


async def test_queue_max_waits_for_full_batch_then_takes_oldest(monkeypatch):
    args = make_args(
        rollout_batch_size=2,
        fully_async_queue_type="queue-max",
        max_weight_staleness=2,
        staleness_reference="prefill",
    )
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._policy_output = deque()
    fn._policy_output_ready = asyncio.Event()
    fn._output_slots = asyncio.Semaphore(fully_async.OUTPUT_QUEUE_MAX_GROUPS)
    never = asyncio.Event()
    fn._worker = asyncio.create_task(never.wait())

    first, second = make_group(1), make_group(2)
    await fn._enqueue_completed_group((first, first))
    take = asyncio.create_task(fn._take_policy_groups(2))
    await asyncio.sleep(0)
    assert not take.done()

    await fn._enqueue_completed_group((second, second))
    selected = await take
    assert [item[0][1][0].group_index for item in selected] == [1, 2]
    assert fn._output_slots._value == fully_async.OUTPUT_QUEUE_MAX_GROUPS

    fn._worker.cancel()
    with suppress(asyncio.CancelledError):
        await fn._worker


async def test_policy_queue_worker_failure_beats_queued_groups(monkeypatch):
    args = make_args(
        rollout_batch_size=1,
        fully_async_queue_type="queue-drop",
        max_weight_staleness=None,
    )
    fn = make_fn(monkeypatch, args, FakeDataSource())
    group = make_group(1)
    fn._policy_output = deque([(group, group)])

    async def fail():
        raise RuntimeError("generation exploded")

    fn._worker = asyncio.create_task(fail())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn._take_policy_groups(1)


def test_queue_max_safety_capacity_cannot_deadlock_a_large_batch(monkeypatch):
    args = make_args(
        rollout_batch_size=fully_async.OUTPUT_QUEUE_MAX_GROUPS + 1,
        fully_async_queue_type="queue-max",
        max_weight_staleness=2,
        staleness_reference="prefill",
    )
    fn = make_fn(monkeypatch, args, FakeDataSource())

    assert fn._queue_capacity_groups() == args.rollout_batch_size


async def test_queue_max_at_one_accepts_one_step_and_drops_two_step_without_recycling(monkeypatch):
    stale = make_group(1)
    allowed = make_group(2)
    stale[0].response_length = 5
    stale[1].response_length = 7
    data_source = FakeDataSource(scripted=[stale, allowed])

    async def generate_with_prefill_version(state, group, sampling_params, evaluation=False):
        group_index = group[0].group_index
        await asyncio.sleep(0.001 if group_index <= 2 else 0.01)
        first_prefill = 1 if group_index == 1 else (2 if group_index == 2 else 3)
        _stamp_prefill_provenance(
            group,
            first=first_prefill,
            minimum=first_prefill,
            maximum=3,
            last=3,
        )
        for sample in group:
            sample.weight_versions = ["3"]
        return group

    args = make_args(
        rollout_batch_size=2,
        fully_async_queue_type="queue-max",
        max_weight_staleness=1,
        staleness_reference="prefill",
    )
    fn = make_fn(monkeypatch, args, data_source, generate=generate_with_prefill_version)
    fn._weight_version = StubWeightVersion(3)
    fn.commit_applied_weight_version(3)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == []
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_dropped"] == 1
    assert output.metrics["rollout/fully_async/age_cutoff_tokens"] == 12
    assert output.metrics["queue/selection/age_cutoff_dropped/sample_length/mean"] == 6
    assert output.metrics["staleness/bound_exceeded_groups"] == 1
    assert output.metrics["staleness/total/count_1"] == 1
    assert output.metrics["staleness/total/max"] == 1
    assert stale not in output.samples
    assert allowed in output.samples


async def test_queue_max_at_zero_rejects_one_step_and_accepts_on_policy(monkeypatch):
    one_step_stale = make_group(1)
    on_policy = make_group(2)
    data_source = FakeDataSource(scripted=[one_step_stale, on_policy])

    async def generate_with_prefill_version(state, group, sampling_params, evaluation=False):
        first_prefill = 2 if group[0].group_index == 1 else 3
        _stamp_prefill_provenance(group, first=first_prefill, maximum=3, last=3)
        return group

    args = make_args(
        rollout_batch_size=1,
        fully_async_queue_type="queue-max",
        max_weight_staleness=0,
        staleness_reference="prefill",
    )
    fn = make_fn(monkeypatch, args, data_source, generate=generate_with_prefill_version)
    fn._weight_version = StubWeightVersion(3)
    fn.commit_applied_weight_version(3)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert output.metrics["rollout/fully_async/stale_groups_dropped"] == 1
    assert output.metrics["staleness/rollout/count_1"] == 1
    assert output.metrics["staleness/total/count_0"] == 1
    assert output.metrics["staleness/total/max"] == 0
    assert one_step_stale not in output.samples
    assert on_policy in output.samples


async def test_worker_failure_beats_queued_groups(monkeypatch):
    """A dead worker fails the step even when it left completed groups behind."""
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), FakeDataSource())

    async def boom():
        raise RuntimeError("generation exploded")

    fn._output = asyncio.Queue(maxsize=fully_async.OUTPUT_QUEUE_MAX_GROUPS)
    group = make_group(1)
    await fn._output.put((group, group))
    fn._worker = asyncio.create_task(boom())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="generation exploded"):
        await fn(RolloutFnTrainInput(rollout_id=0))


async def test_nested_group_recycles_the_flat_prompt_group(monkeypatch):
    """A generate function may expand one trajectory into several samples; the retry
    must resubmit the flat prompt group the data source handed out."""
    prompt_group = make_group(1)
    data_source = FakeDataSource(scripted=[prompt_group])
    submitted = []

    async def multi_sample_generate(state, group, sampling_params, evaluation=False):
        assert all(isinstance(sample, Sample) for sample in group), "resubmitted a nested group"
        submitted.append(group)
        if len(submitted) > 1:
            return group
        expanded = []
        for sample in group:
            aborted = replace(sample, status=Sample.Status.ABORTED)
            expanded.append([aborted, replace(sample)])
        return expanded

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source, generate=multi_sample_generate)
    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [prompt_group]
    assert all(isinstance(sample, Sample) for sample in data_source.recycled[0])
    assert len(submitted) > 1
    assert len(output.samples) == 1


async def test_dynamic_filter_drops_group_without_recycling(monkeypatch):
    rejected = make_group(1)
    data_source = FakeDataSource(scripted=[rejected])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    def reject_group_1(args, group, **kwargs):
        keep = group[0].group_index != 1
        return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")

    fn._dynamic_filter = reject_group_1

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 1
    assert output.samples[0][0].group_index != 1
    # Unlike a recycle, a filtered group is not returned to the data source for re-sampling.
    assert data_source.recycled == []
    assert output.metrics["rollout/dynamic_filter/drop_rejected"] == 1


async def test_sample_filter_marks_samples_without_shrinking_the_batch(monkeypatch):
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())

    def mark_first_of_each_group(args, data):
        for group in data:
            group[0].remove_sample = True

    fn._sample_filter = mark_first_of_each_group

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 2
    assert [sample.remove_sample for sample in output.samples[0]] == [True, False]


async def test_weight_version_throttles_failed_queries(monkeypatch):
    """A drain queries once per group, so an unreachable router must not cost one timeout each."""
    calls = []

    async def unreachable_router(url):
        calls.append(url)
        raise httpx.ConnectError("router down")

    monkeypatch.setattr(fully_async, "get", unreachable_router)
    args = make_args()

    # One probe, not one per candidate endpoint: an unreachable router means every
    # name is unreachable, so discovery stops at the first connection error.
    throttled = fully_async._CachedWeightVersion(ttl=60.0)
    assert await throttled.get(args) is None
    assert await throttled.get(args) is None
    assert len(calls) == 1

    calls.clear()
    expired = fully_async._CachedWeightVersion(ttl=0.0)
    assert await expired.get(args) is None
    assert await expired.get(args) is None
    assert len(calls) == 2


class StubWeightVersion:
    """A settable current version, so a test can move it mid-generation."""

    def __init__(self, value: int | None):
        self.value = value
        self.calls = 0

    async def get(self, args):
        self.calls += 1
        return self.value


async def test_prefill_pre_queue_staleness_records_updates_crossed_during_generation(monkeypatch):
    """Prefill provenance exposes the span hidden by completion metadata."""
    version = StubWeightVersion(4)

    async def generate_crossing_an_update(state, group, sampling_params, evaluation=False):
        await asyncio.sleep(0)
        version.value = 6
        fn.commit_applied_weight_version(6)
        _stamp_prefill_provenance(group, first=4, minimum=4, maximum=6, last=6)
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, max_weight_staleness=3, staleness_reference="prefill"),
        FakeDataSource(),
        generate=generate_crossing_an_update,
    )
    fn._weight_version = version
    fn.commit_applied_weight_version(4)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert output.metrics["staleness/total/max"] == 2
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0

    assert output.metrics["staleness/pre_queue/max"] == 2
    assert output.metrics["staleness/in_queue/max"] == 0
    assert output.metrics["staleness/total/max"] == 2
    assert output.metrics["staleness/total/num_groups"] == 1


async def test_straggler_is_charged_to_pre_queue_not_in_queue(monkeypatch):
    """A group is one request per sample joined by `asyncio.gather`, so it enters
    the queue when its *slowest* sample lands. Keying the split on the group's
    oldest sample would charge that straggler's crossing to in-queue staleness --
    inverting the two components in exactly the case the decomposition exists for.
    """
    version = StubWeightVersion(1)

    async def generate_with_a_straggler(state, group, sampling_params, evaluation=False):
        await asyncio.sleep(0)
        # First sample lands under v1; the update arrives; the straggler lands under v2.
        group[0].weight_versions = ["1"]
        version.value = 2
        fn.commit_applied_weight_version(2)
        for sample in group[1:]:
            sample.weight_versions = ["2"]
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, max_weight_staleness=2),
        FakeDataSource(),
        generate=generate_with_a_straggler,
    )
    fn._weight_version = version
    fn.commit_applied_weight_version(1)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    # Queue entry is the straggler at v2, so the update it crossed is generation.
    assert output.metrics["staleness/pre_queue/max"] == 1
    assert output.metrics["staleness/in_queue/max"] == 0
    assert output.metrics["staleness/total/max"] == 1
    # `total` keys on the group's oldest sample, so it reads 1 here -- that is
    # `in_queue` plus the group's internal spread, and is why it is not `in_queue`.
    assert output.metrics["staleness/total/max"] == 1


async def test_components_sum_to_the_total(monkeypatch):
    """`total = pre_queue + in_queue` has to hold per group, or the decomposition
    is two unrelated numbers rather than a split of one."""
    version = StubWeightVersion(2)

    async def generate_then_let_the_queue_age(state, group, sampling_params, evaluation=False):
        await asyncio.sleep(0)
        # max(), because the worker keeps submitting: a later group must not walk
        # the version backwards and make the assertion depend on task scheduling.
        version.value = max(version.value, 4)  # two updates during generation
        fn.commit_applied_weight_version(4)
        for sample in group:
            sample.weight_versions = ["4"]
        return group

    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, max_weight_staleness=None, staleness_reference="submission"),
        FakeDataSource(),
        generate=generate_then_let_the_queue_age,
    )
    fn._weight_version = version
    fn.commit_applied_weight_version(2)
    original_next_group = fn._next_group

    async def next_group_after_queue_ages():
        item = await original_next_group()
        version.value = max(version.value, 7)
        fn.commit_applied_weight_version(7)
        return item

    fn._next_group = next_group_after_queue_ages

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    pre, inq, total = (
        output.metrics["staleness/pre_queue/mean"],
        output.metrics["staleness/in_queue/mean"],
        output.metrics["staleness/total/mean"],
    )
    assert (pre, inq, total) == (2.0, 3.0, 5.0)
    assert pre + inq == total
    # With no filtering, the offered and accepted populations agree.
    assert output.metrics["staleness/rollout/max"] == output.metrics["staleness/total/max"]


async def test_staleness_components_absent_when_the_router_never_answers(monkeypatch):
    """No reference version means no stamp; the drain must degrade, not invent a zero.

    A zero here would be the worst outcome: it reads as "nothing crossed" exactly
    when the instrument is broken.
    """
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=2), FakeDataSource())
    fn._weight_version = StubWeightVersion(None)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert len(output.samples) == 2
    for family in ("rollout", "pre_queue", "in_queue", "total"):
        assert not [k for k in output.metrics if k.startswith(f"staleness/{family}/")]


async def test_submission_stamp_is_refreshed_when_a_group_is_recycled(monkeypatch):
    """`reset_for_retry` keeps `metadata` (types.py:240-249), so a recycled group
    carries its old stamp. The regenerated attempt must be measured against the
    version *it* started under, not the one the discarded attempt did."""
    stale = make_group(1, weight_versions=["5"])
    version = StubWeightVersion(10)
    data_source = FakeDataSource(scripted=[stale])

    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), data_source)
    fn._weight_version = version
    fn.commit_applied_weight_version(10)

    await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [stale]
    for sample in stale:
        assert sample.metadata[fully_async.SUBMISSION_VERSION_KEY] == 10


async def test_concurrent_submissions_share_one_weight_version_query(monkeypatch):
    """The first fill starts a whole batch of groups in one pass, and each one now
    asks for the current version before generating. Without single-flight that is
    one /model_info request per group against the router, at once."""
    calls = []

    async def slow_router(url, *args, **kwargs):
        calls.append(url)
        await asyncio.sleep(0.01)
        return {"weight_version": "4"}

    monkeypatch.setattr(fully_async, "get", slow_router)
    cache = fully_async._CachedWeightVersion(ttl=60.0)
    args = make_args()

    results = await asyncio.gather(*(cache.get(args) for _ in range(32)))

    assert results == [4] * 32
    assert len(calls) == 1


async def _run_with_reference(monkeypatch, reference: str, bound: int):
    """One group submitted under v1 that finishes under v2, drained at v2.

    `total` is 1 and what the completion reference tests is 0, so the two
    references disagree at the strict queue-recycle bound of 1 -- which is the
    point of the option.
    """
    version = StubWeightVersion(1)

    async def generate_crossing_an_update(state, group, sampling_params, evaluation=False):
        await asyncio.sleep(0)
        version.value = max(version.value, 2)
        fn.commit_applied_weight_version(2)
        for sample in group:
            sample.weight_versions = ["2"]
        return group

    data_source = FakeDataSource()
    fn = make_fn(
        monkeypatch,
        make_args(rollout_batch_size=1, max_weight_staleness=bound, staleness_reference=reference),
        data_source,
        generate=generate_crossing_an_update,
    )
    fn._weight_version = version
    fn.commit_applied_weight_version(1)
    return await fn(RolloutFnTrainInput(rollout_id=0)), data_source


async def test_completion_reference_does_not_see_the_generation_span(monkeypatch):
    """The default. A group that crossed an update while generating passes a bound
    of 1, because the gap is measured from the version it finished under."""
    output, data_source = await _run_with_reference(monkeypatch, "completion", bound=1)

    assert output.metrics["staleness/total/max"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    assert data_source.recycled == []
    assert output.metrics["staleness/bound_reference_is_submission"] == 0.0


async def test_submission_reference_bounds_the_whole_off_policy_distance(monkeypatch):
    """The option. The same group is rejected at the same bound, because the gap is
    now measured from the version it was submitted under -- i.e. `staleness/total`."""
    output, data_source = await _run_with_reference(monkeypatch, "submission", bound=1)

    assert output.metrics["rollout/fully_async/stale_groups_recycled"] >= 1
    assert data_source.recycled, "the group that crossed an update must be recycled"
    assert output.metrics["staleness/bound_reference_is_submission"] == 1.0

    # The offered population saw a group whose whole off-policy distance was 1, and
    # the bound rejected it; what survived into the batch crossed nothing. That gap
    # between the populations is the bound doing its job on `total` -- under the
    # completion reference the same group passes (see the test above).
    assert output.metrics["staleness/rollout/max"] == 1
    assert output.metrics["staleness/total/max"] == 0


def _stamp_prefill_provenance(
    group: list[Sample],
    *,
    first: int,
    minimum: int | None = None,
    maximum: int | None = None,
    last: int | None = None,
) -> None:
    minimum = first if minimum is None else minimum
    maximum = first if maximum is None else maximum
    last = maximum if last is None else last
    for sample in group:
        sample.first_prefill_weight_versions = [first]
        sample.min_forward_weight_versions = [minimum]
        sample.max_forward_weight_versions = [maximum]
        sample.last_forward_weight_versions = [last]
        sample.response_weight_versions = [str(last)]
        sample.weight_versions = [str(last)]


async def test_prefill_reference_ignores_update_while_waiting_for_prefill(monkeypatch):
    """submission=10, prefill=11, drain=11 has zero realized staleness."""

    class SubmissionVersion:
        async def get(self, args):
            return 10

    async def generate_after_queue_update(state, group, sampling_params, evaluation=False):
        fn.commit_applied_weight_version(11)
        _stamp_prefill_provenance(group, first=11)
        return group

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            max_weight_staleness=1,
            staleness_reference="prefill",
        ),
        FakeDataSource(),
        generate=generate_after_queue_update,
    )
    fn._weight_version = SubmissionVersion()
    fn.commit_applied_weight_version(10)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert output.metrics["staleness/total/max"] == 0
    assert output.metrics["staleness/pre_queue/max"] == 0
    assert output.metrics["staleness/in_queue/max"] == 0
    assert output.metrics["staleness/total/max"] == 0
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    sample = output.samples[0][0]
    assert sample.metadata[fully_async.SUBMISSION_VERSION_KEY] == 10
    assert sample.metadata[fully_async.GROUP_READY_VERSION_KEY] == 11
    assert sample.metadata[fully_async.QUEUE_PUT_VERSION_KEY] == 11
    assert sample.metadata[fully_async.DRAIN_VERSION_KEY] == 11
    assert sample.metadata[fully_async.TRAIN_VERSION_KEY] == 11


async def test_prefill_reference_detects_mixed_generation_and_enforces_bound(monkeypatch):
    """prefill=10, mixed decode=11, drain=11 reaches max=1 and is recycled."""
    scripted = make_group(1)
    data_source = FakeDataSource(scripted=[scripted])

    async def generate_mixed_then_fresh(state, group, sampling_params, evaluation=False):
        if group[0].group_index == 1:
            _stamp_prefill_provenance(group, first=10, minimum=10, maximum=11, last=11)
            fn.commit_applied_weight_version(11)
        else:
            _stamp_prefill_provenance(group, first=11)
        return group

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            max_weight_staleness=1,
            staleness_reference="prefill",
        ),
        data_source,
        generate=generate_mixed_then_fresh,
    )
    fn._weight_version = StubWeightVersion(10)
    fn.commit_applied_weight_version(10)

    output = await fn(RolloutFnTrainInput(rollout_id=0, updates_before_train=1))

    assert data_source.recycled == [scripted]
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 1
    assert output.metrics["staleness/rollout/count_2"] == 1
    assert output.metrics["staleness/total/count_1"] == 1
    assert output.metrics["staleness/mixed_version_frac/rollout"] == pytest.approx(0.5)
    assert output.metrics["staleness/mixed_version_frac/train"] == 0.0


async def test_prefill_reference_trains_one_step_mixed_generation_at_max_two(monkeypatch):
    async def generate_mixed(state, group, sampling_params, evaluation=False):
        _stamp_prefill_provenance(group, first=10, minimum=10, maximum=11, last=11)
        fn.commit_applied_weight_version(11)
        return group

    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            max_weight_staleness=2,
            staleness_reference="prefill",
        ),
        FakeDataSource(),
        generate=generate_mixed,
    )
    fn._weight_version = StubWeightVersion(10)
    fn.commit_applied_weight_version(10)

    output = await fn(RolloutFnTrainInput(rollout_id=0, updates_before_train=1))

    assert output.metrics["staleness/rollout/max"] == 2
    assert output.metrics["staleness/pre_queue/max"] == 1
    assert output.metrics["staleness/in_queue/max"] == 1
    assert output.metrics["staleness/total/max"] == 2
    assert output.metrics["fully_async/train_weight_version"] == 12
    assert output.samples[0][0].metadata[fully_async.TRAIN_VERSION_KEY] == 12
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    assert output.metrics["staleness/mixed_version_frac/train"] == 1.0


async def test_prefill_reference_fails_fast_without_scheduler_metadata(monkeypatch):
    fn = make_fn(
        monkeypatch,
        make_args(
            rollout_batch_size=1,
            max_weight_staleness=1,
            staleness_reference="prefill",
        ),
        FakeDataSource(),
    )
    fn._weight_version = StubWeightVersion(0)

    with pytest.raises(RuntimeError, match="patched SGLang image"):
        await fn(RolloutFnTrainInput(rollout_id=0))


def test_applied_weight_version_tracker_rejects_rollback():
    tracker = fully_async.AppliedWeightVersionTracker(initial_version=10)
    tracker.commit(11)
    assert tracker.current() == 11
    with pytest.raises(ValueError, match="cannot move backwards"):
        tracker.commit(10)


async def test_decomposition_uses_the_selected_bound_reference(monkeypatch):
    """Every mode reports `total` from the same endpoints its bound enforces."""
    completion, _ = await _run_with_reference(monkeypatch, "completion", bound=4)
    submission, _ = await _run_with_reference(monkeypatch, "submission", bound=4)

    assert completion.metrics["staleness/pre_queue/max"] == 0
    assert completion.metrics["staleness/in_queue/max"] == 0
    assert completion.metrics["staleness/total/max"] == 0

    assert submission.metrics["staleness/pre_queue/max"] == 1
    assert submission.metrics["staleness/in_queue/max"] == 0
    assert submission.metrics["staleness/total/max"] == 1
