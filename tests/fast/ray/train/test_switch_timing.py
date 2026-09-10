from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.ray.train import group as train_group_module
from miles.ray.train.group import TrainerController, _unwrap_cell_train_results_with_timing
from miles.ray.train.types import TrainResultWithTiming


def _timed(result, seconds: float) -> TrainResultWithTiming:
    return TrainResultWithTiming(result=result, local_wake_up_time=seconds)


def test_cell_timing_uses_maximum_across_workers_and_surviving_cells():
    cell_error = RuntimeError("cell failed")

    results, wake_up_time = _unwrap_cell_train_results_with_timing(
        [
            [
                _timed(TrainStepOutput(TrainStepOutcome.NORMAL), 1.0),
                _timed(TrainStepOutput(TrainStepOutcome.NORMAL), 4.0),
            ],
            cell_error,
            [_timed(TrainStepOutput(TrainStepOutcome.NORMAL), 3.0)],
        ]
    )

    assert results == [
        [TrainStepOutput(TrainStepOutcome.NORMAL), TrainStepOutput(TrainStepOutcome.NORMAL)],
        cell_error,
        [TrainStepOutput(TrainStepOutcome.NORMAL)],
    ]
    assert wake_up_time == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_v2_timing_uses_successful_retry_only(monkeypatch):
    group = TrainerController.__new__(TrainerController)
    group.args = SimpleNamespace()
    group._test_action_executor = SimpleNamespace(run_after_step=AsyncMock())
    group._allocate_witness_info = MagicMock(return_value=None)
    group._refresh_cells = AsyncMock()
    group._log_step_end_event = MagicMock()
    cell = SimpleNamespace(cell_index=0, is_alive=True, train=AsyncMock())
    group._cells_by_id = {"test": cell}
    attempt_results = [
        [
            [
                _timed(TrainStepOutput(TrainStepOutcome.DISCARDED_SHOULD_RETRY), 9.0),
                _timed(TrainStepOutput(TrainStepOutcome.NORMAL), 7.0),
            ]
        ],
        [
            [
                _timed(TrainStepOutput(TrainStepOutcome.NORMAL), 3.0),
                _timed(TrainStepOutput(TrainStepOutcome.NORMAL), 5.0),
            ]
        ],
    ]
    dispatched_kwargs = []

    async def gather(compute_coroutine, **kwargs):
        await compute_coroutine(cell)
        dispatched_kwargs.append(cell.train.call_args.kwargs)
        return [cell], attempt_results.pop(0)

    async def retry_immediately(fn, **_kwargs):
        with pytest.raises(ValueError, match="DISCARDED_SHOULD_RETRY"):
            await fn(0)
        return await fn(1)

    group._gather_all_alive_and_catch = gather
    monkeypatch.setattr(train_group_module.event_analyzer, "run_analysis_from_args", lambda _args: None)
    monkeypatch.setattr(train_group_module, "retry", retry_immediately)

    result = await group.train(
        rollout_id=6,
        rollout_data_pack={"data_ref": "data", "sample_indices": [0]},
        collect_wake_up_time=True,
    )

    assert result == TrainResultWithTiming(
        result=[TrainStepOutput(TrainStepOutcome.NORMAL)] * 2, local_wake_up_time=5.0
    )
    assert [kwargs["attempt"] for kwargs in dispatched_kwargs] == [0, 1]
    assert all(kwargs["collect_wake_up_time"] is True for kwargs in dispatched_kwargs)


@pytest.mark.asyncio
async def test_v2_switch_worker_results_are_only_forwarded_when_collection_is_enabled():
    group = TrainerController.__new__(TrainerController)
    group.args = SimpleNamespace(log_colocate_switch_metrics=False)
    cell = SimpleNamespace(health_checker=MagicMock())
    group._cells_by_id = {"test": cell}
    group._health_checker_activeness = MagicMock()
    group._execute_all_alive_and_catch = AsyncMock(return_value=([cell], [{"bytes": 10}]))

    assert await group.offload() is None

    group.args.log_colocate_switch_metrics = True
    assert await group.offload() == [{"bytes": 10}]
