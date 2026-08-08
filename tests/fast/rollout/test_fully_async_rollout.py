from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace
from collections import deque
from dataclasses import replace

import httpx
import pytest
from types import SimpleNamespace

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

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert data_source.recycled == [stale]
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 1
    # The reference version the staleness is a difference against: without it, a
    # missing staleness metric cannot be told apart from a router that never answered.
    assert output.metrics["rollout/fully_async/current_weight_version"] == 10

    # The unprefixed metrics are the lag the loss saw. The group at 5 exceeded the
    # bound of 2 and was recycled, so only the fresh group at 10 -- lag 0 -- was
    # trained on. A reader plotting "max_staleness" against a bound of 2 must never
    # see a 5 there.
    assert output.metrics["staleness/max"] == 0
    # Upstream's own key keeps upstream's meaning: the offered lag, before the
    # bound check. Two miles runs must not plot different quantities under it.
    assert output.metrics["rollout/fully_async/max_staleness"] == 5
    assert output.metrics["staleness/num_groups"] == 1
    assert output.metrics["staleness/frac_zero"] == pytest.approx(1.0)
    assert output.metrics["staleness/count_0"] == 1
    assert output.metrics["staleness/count_5"] == 0

    # The offered distribution keeps both: it is the natural lag of this node
    # ratio, and the gap between the two is what recycling cost.
    assert output.metrics["staleness/offered/max"] == 5
    assert output.metrics["staleness/offered/num_groups"] == 2
    assert output.metrics["staleness/offered/p50"] == pytest.approx(2.5)
    assert output.metrics["staleness/offered/frac_zero"] == pytest.approx(0.5)
    assert output.metrics["staleness/offered/frac_at_bound"] == pytest.approx(0.5)

    # Tokens counted before reset_for_retry cleared them.
    assert output.metrics["rollout/fully_async/stale_tokens"] == N_SAMPLES_PER_PROMPT

    # Named for the cause, under the section that owns it, so the bound's cost does
    # not have to be reconstructed by differencing group counts -- which would fold
    # in the dynamic-filter drops.
    assert output.metrics["staleness/bound_exceeded_groups"] == 1
    assert output.metrics["staleness/bound_exceeded_tokens"] == N_SAMPLES_PER_PROMPT

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

    # Drop the scripted group -- lag 1, comfortably inside the bound of 4 -- and
    # keep the fresh one that replaces it.
    def reject_the_stale_one(args, group, **kwargs):
        keep = group[0].group_index != 1
        return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")

    fn._dynamic_filter = reject_the_stale_one

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    # It was offered at lag 1 and the bound did not stop it...
    assert output.metrics["staleness/offered/count_1"] == 1
    assert output.metrics["rollout/fully_async/stale_groups_recycled"] == 0
    # ...but the filter dropped it, so the loss only ever saw the lag-0 group.
    assert output.metrics["staleness/count_1"] == 0
    assert output.metrics["staleness/count_0"] == 1
    assert output.metrics["staleness/num_groups"] == 1


async def test_wasted_token_accounting(monkeypatch):
    aborted = make_group(1, status=Sample.Status.ABORTED)
    data_source = FakeDataSource(scripted=[aborted])
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1), data_source)

    output = await fn(RolloutFnTrainInput(rollout_id=0))

    # response_length is 1 per sample in make_group, so the counts are sample counts.
    assert output.metrics["rollout/fully_async/aborted_tokens"] == N_SAMPLES_PER_PROMPT
    assert output.metrics["rollout/fully_async/kept_tokens"] == N_SAMPLES_PER_PROMPT
    assert output.metrics["rollout/fully_async/wasted_token_frac"] == pytest.approx(0.5)


def test_staleness_histogram_reports_the_shape_not_just_moments():
    """Percentiles cannot distinguish a rare tail from a common one.

    A bound is only a real constraint if samples actually reach it, and
    ``staleness_p99`` alone cannot say whether lag 4 happened twice or two
    hundred times.
    """
    values = [0] * 90 + [1] * 8 + [4, 12]
    m = fully_async._staleness_metrics(values, bound=2)

    assert m["count_0"] == 90
    assert m["count_1"] == 8
    assert m["count_4"] == 1
    assert m["count_ge_9"] == 1  # 12 lands in the overflow bucket
    assert sum(m[f"count_{i}"] for i in range(9)) + m["count_ge_9"] == len(values)
    # The moments still agree with the histogram.
    assert m["max"] == 12
    assert m["frac_zero"] == 0.9


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


async def test_unreachable_router_disables_the_cap_loudly(monkeypatch, caplog):
    """A router that never answers must not fail silently.

    ``_CachedWeightVersion.get`` returning None skips the staleness branch entirely,
    so ``--max-weight-staleness`` enforces nothing and no staleness metric is emitted.
    A run then looks like a staleness arm while training fully on-policy.
    """

    async def unreachable(url, *args, **kwargs):
        raise httpx.ConnectError("no router")

    monkeypatch.setattr(fully_async, "get", unreachable)
    fn = make_fn(monkeypatch, make_args(rollout_batch_size=1, max_weight_staleness=2), FakeDataSource())

    with caplog.at_level("WARNING"):
        output = await fn(RolloutFnTrainInput(rollout_id=0))

    assert "--max-weight-staleness cannot be enforced" in caplog.text
    assert "rollout/fully_async/avg_staleness" not in output.metrics
    assert "rollout/fully_async/current_weight_version" not in output.metrics


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
