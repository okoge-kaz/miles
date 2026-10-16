"""CPU checkpoint orchestration; isolate driver imports from CUDA/Ray startup."""

import ast
import asyncio
import logging
import os
from argparse import Namespace
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.utils.async_utils import eager_create_task
from miles.utils.misc import checkpoint_artifacts_due, should_run_periodic_action
from tests.fast.fixtures.driver_fakes import FakeInferenceController, FakeRolloutExecutor, FakeTrainingModel

ROOT = Path(__file__).resolve().parents[4]


def function(path, name, namespace):
    module = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    node.decorator_list = []
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "hf_interval,expected",
    [(None, [(3, True, True), (5, True, True)]), (2, [(1, False, True), (3, True, True), (5, True, True)])],
)
@pytest.mark.parametrize("driver", ["train", "train_async"])
async def test_driver_saves_cursor_only_with_distributed_checkpoint(driver, hf_interval, expected):
    events = []
    model = FakeTrainingModel(events, "actor")
    executor = FakeRolloutExecutor(events)
    controller = FakeInferenceController(events)
    args = function(f"tests/fast/test_{driver}.py", "_make_args", {"SimpleNamespace": SimpleNamespace})(
        num_rollout=6, save_interval=4, hf_save_interval=hf_interval, save_hf="hf/{rollout_id}"
    )
    noops = (
        "configure_logger",
        "maybe_start_periodic_pyspy_dump",
        "launch_worker_manager",
        "init_tracking",
        "maybe_start_mini_ft_controller",
        "remove_rollout_data_refs",
        "validate_async_off_policy_correction",
    )
    namespace = {name: Mock() for name in noops}
    namespace.update(
        asyncio=asyncio,
        monotonic=monotonic,
        os=os,
        logger=logging.getLogger(__name__),
        MainProcessIdentity=lambda: None,
        object_store=SimpleNamespace(init_instance=Mock()),
        create_rollout_components=AsyncMock(return_value=(controller, executor, None)),
        create_training_models=AsyncMock(return_value=(model, None)),
        update_weights=AsyncMock(),
        EvalDispatcher=lambda *a: SimpleNamespace(drain=AsyncMock(), dispatch=AsyncMock()),
        eager_create_task=eager_create_task,
        should_run_periodic_action=should_run_periodic_action,
        checkpoint_artifacts_due=checkpoint_artifacts_due,
        lora_rollout_enabled=lambda _: False,
    )
    await function(f"{driver}.py", "train", namespace)(args)
    assert model.save_artifacts == expected
    assert [e for e in events if e.startswith("executor_save:")] == ["executor_save:3", "executor_save:5"]


def test_external_epoch_and_final_save_triggers():
    assert checkpoint_artifacts_due(0, save_interval=10, hf_save_interval=2, external_save=True) == (True, False)
    assert checkpoint_artifacts_due(0, save_interval=10, hf_save_interval=None, external_save=True) == (True, True)
    assert checkpoint_artifacts_due(2, save_interval=10, hf_save_interval=2, num_rollout_per_epoch=3) == (True, True)
    assert checkpoint_artifacts_due(4, save_interval=10, hf_save_interval=2, num_rollout=5) == (True, True)
    assert checkpoint_artifacts_due(1, save_interval=None, hf_save_interval=2) == (False, True)


@pytest.mark.parametrize("write_dist,write_hf", [(False, True), (True, False), (True, True)])
def test_megatron_artifacts_are_independently_selected(write_dist, write_hf):
    namespace = {"save": Mock(), "save_hf_model": Mock()}
    save_model = function("miles/backends/megatron_utils/actor.py", "save_model", namespace)
    worker = SimpleNamespace(
        args=Namespace(
            debug_rollout_only=False,
            async_save=False,
            save_hf="hf/{rollout_id}",
            custom_megatron_post_save_hook_path=None,
        ),
        _heartbeat=Mock(),
        model=object(),
        optimizer=object(),
        opt_param_scheduler=object(),
        snapshot_publisher=object(),
        role="actor",
    )
    save_model(worker, 2, write_dist=write_dist, write_hf=write_hf)
    assert namespace["save"].call_count == int(write_dist)
    assert namespace["save_hf_model"].call_count == int(write_hf)


def test_repetition_off_never_calls_detector_and_preserves_other_metrics(monkeypatch):
    import miles.ray.rollout.metrics as metrics
    from miles.utils.types import Sample

    args = Namespace(
        reward_key=None, advantage_estimator="ppo", sglang_speculative_algorithm=None, log_reward_category=None
    )
    samples = [Sample(group_index=0, prompt="p", tokens=[1, 2], response="x", response_length=1, reward=1)]
    detector = Mock(return_value=True)
    monkeypatch.setattr(metrics, "has_repetition", detector)
    enabled = metrics._compute_metrics_from_samples(args, samples)
    detector.assert_called_once_with("x")
    assert enabled.pop("repetition_frac") == 1
    args.log_rollout_repetition = False
    detector.side_effect = AssertionError("compression must not run")
    disabled = metrics._compute_metrics_from_samples(args, samples)
    assert "repetition_frac" not in disabled and enabled == disabled
