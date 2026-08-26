from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.ray.actor_group import RayTrainGroup as V1RayTrainGroup
from miles.ray.train import group as train_group_module
from miles.ray.train.group import RayTrainGroup, _unwrap_cell_train_results_with_timing
from miles.ray.train.types import TrainResultWithTiming


def _timed(result, seconds: float) -> TrainResultWithTiming:
    return TrainResultWithTiming(result=result, local_wake_up_time=seconds)


def test_cell_timing_uses_maximum_across_workers_and_surviving_cells():
    cell_error = RuntimeError("cell failed")

    results, wake_up_time = _unwrap_cell_train_results_with_timing(
        [
            [_timed(TrainStepOutcome.NORMAL, 1.0), _timed(TrainStepOutcome.NORMAL, 4.0)],
            cell_error,
            [_timed(TrainStepOutcome.NORMAL, 3.0)],
        ]
    )

    assert results == [
        [TrainStepOutcome.NORMAL, TrainStepOutcome.NORMAL],
        cell_error,
        [TrainStepOutcome.NORMAL],
    ]
    assert wake_up_time == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_v2_timing_uses_successful_retry_only(monkeypatch):
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace()
    group._test_action_executor = MagicMock()
    group._allocate_witness_info = MagicMock(return_value=None)
    group._refresh_cells = AsyncMock()
    group._log_step_end_event = MagicMock()
    cell = SimpleNamespace(cell_index=0)
    attempt_results = [
        [[_timed(TrainStepOutcome.DISCARDED_SHOULD_RETRY, 9.0), _timed(TrainStepOutcome.NORMAL, 7.0)]],
        [[_timed(TrainStepOutcome.NORMAL, 3.0), _timed(TrainStepOutcome.NORMAL, 5.0)]],
    ]
    dispatched_kwargs = []

    async def execute_all_alive_and_catch(_method_name, **kwargs):
        dispatched_kwargs.append(kwargs)
        return [cell], attempt_results.pop(0)

    async def retry_immediately(fn, **_kwargs):
        with pytest.raises(ValueError, match="DISCARDED_SHOULD_RETRY"):
            await fn(0)
        await fn(1)

    group._execute_all_alive_and_catch = execute_all_alive_and_catch
    monkeypatch.setattr(train_group_module.event_analyzer, "run_analysis_from_args", lambda _args: None)
    monkeypatch.setattr(train_group_module, "retry", retry_immediately)

    result = await group.train(
        rollout_id=6,
        rollout_data_pack={"data_ref": "data", "sample_indices": [0]},
        collect_wake_up_time=True,
    )

    assert result == TrainResultWithTiming(result=None, local_wake_up_time=5.0)
    assert [kwargs["attempt"] for kwargs in dispatched_kwargs] == [0, 1]
    assert all(kwargs["collect_wake_up_time"] is True for kwargs in dispatched_kwargs)


@pytest.mark.asyncio
async def test_v1_timing_uses_worker_max_and_default_dispatch_is_unchanged():
    group = V1RayTrainGroup.__new__(V1RayTrainGroup)
    group._broadcast = AsyncMock(
        return_value=[_timed(TrainStepOutcome.NORMAL, 2.0), _timed(TrainStepOutcome.NORMAL, 6.0)]
    )

    result = await group.train(3, {"data_ref": "data"}, collect_wake_up_time=True)

    assert result == TrainResultWithTiming(
        result=[TrainStepOutcome.NORMAL, TrainStepOutcome.NORMAL],
        local_wake_up_time=6.0,
    )
    group._broadcast.assert_awaited_once_with(
        "train",
        3,
        "data",
        witness_info=None,
        attempt=0,
        collect_wake_up_time=True,
    )

    group._broadcast.reset_mock(return_value=True)
    group._broadcast.return_value = [TrainStepOutcome.NORMAL]

    assert await group.train(4, {"data_ref": "data"}) == [TrainStepOutcome.NORMAL]
    group._broadcast.assert_awaited_once_with(
        "train",
        4,
        "data",
        witness_info=None,
        attempt=0,
    )
