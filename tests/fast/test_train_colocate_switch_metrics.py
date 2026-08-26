from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])

import asyncio
import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import patch

import pytest

from miles.ray.train.types import TrainResultWithTiming

# Import the sync orchestration loop without loading GPU-backed placement code.
_placement_group_stub = ModuleType("miles.ray.placement_group")
_placement_group_stub.create_placement_groups = None
_placement_group_stub.create_rollout_manager = None
_placement_group_stub.create_training_models = None
with patch.dict(sys.modules, {"miles.ray.placement_group": _placement_group_stub}):
    import train


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        result = self._fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return asyncio.create_task(result)

        async def completed():
            return result

        return asyncio.create_task(completed())


class _RolloutManager:
    def __init__(self, clock: _Clock, events: list) -> None:
        self._clock = clock
        self._events = events
        self.generate = _RemoteMethod(self._generate)
        self.offload = _RemoteMethod(self._offload)
        self.onload_weights = _RemoteMethod(self._onload_weights)
        self.onload_kv = _RemoteMethod(self._onload_kv)
        self.save = _RemoteMethod(self._save)
        self.eval = _RemoteMethod(lambda _rollout_id: None)
        self.dispose = _RemoteMethod(lambda: self._events.append("dispose"))

    def _generate(self, rollout_id: int):
        self._events.append(("generate", rollout_id))
        self._clock.advance(100.0)
        return {"data_ref": object(), "sample_indices": [rollout_id]}

    def _offload(self, *, tags):
        self._events.append(("rollout_offload", tuple(tags)))
        self._clock.advance(2.0)

    def _onload_weights(self):
        self._events.append("rollout_onload_weights")
        self._clock.advance(5.0)

    def _onload_kv(self):
        self._events.append("rollout_onload_kv")
        self._clock.advance(7.0)

    def _save(self, rollout_id: int):
        self._events.append(("rollout_save", rollout_id))
        self._clock.advance(25.0)


class _ActorModel:
    def __init__(self, clock: _Clock, events: list) -> None:
        self._clock = clock
        self._events = events

    async def train(self, rollout_id, _rollout_data_pack, **kwargs):
        self._events.append(("train", rollout_id, kwargs))
        self._clock.advance(50.0)
        result = f"trained-{rollout_id}"
        if kwargs.get("collect_wake_up_time"):
            return TrainResultWithTiming(result=result, local_wake_up_time=3.0)
        return result

    async def offload(self):
        self._events.append("train_offload")
        self._clock.advance(4.0)

    async def clear_memory(self):
        raise AssertionError("offload_train=True must use offload()")

    async def update_weights(self, rollout_id=None):
        self._events.append(("update_weights", rollout_id))
        self._clock.advance(6.0)

    async def save_model(self, rollout_id, force_sync=False, *, write_dist=True, write_hf=True):
        self._events.append(("model_save", rollout_id, force_sync, write_dist, write_hf))
        self._clock.advance(100.0)


def _args() -> Namespace:
    return Namespace(
        fully_async=False,
        control_server_port=None,
        ft_components=[],
        offload_rollout=True,
        offload_rollout_level=["kv_cache", "weight"],
        offload_train=True,
        check_weight_update_equal=False,
        num_rollout=2,
        start_rollout_id=0,
        eval_interval=None,
        skip_eval_before_train=True,
        use_critic=False,
        save_trigger_sentinel=None,
        save_interval=1,
        hf_save_interval=None,
        debug_exit_after_rollout=None,
        colocate=True,
        wandb_always_use_train_step=False,
    )


@pytest.mark.asyncio
async def test_colocate_switch_metrics_are_current_step_and_exclude_bootstrap_train_and_save(monkeypatch):
    clock = _Clock()
    events = []
    logged = []
    manager = _RolloutManager(clock, events)
    actor = _ActorModel(clock, events)

    monkeypatch.setattr(train.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(train, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train, "create_rollout_manager", lambda _args, _pg: (manager, 1))

    async def create_models(_args, _pgs, _manager):
        return actor, None

    monkeypatch.setattr(train, "create_training_models", create_models)
    monkeypatch.setattr(train.object_store, "init_instance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train, "remove_rollout_data_refs", lambda *_args: None)
    monkeypatch.setattr(train, "checkpoint_artifacts_due", lambda *_args, **_kwargs: (True, False))
    monkeypatch.setattr(train, "should_run_periodic_action", lambda *_args: False)
    monkeypatch.setattr(
        train,
        "log_tracking",
        lambda _args, metrics, *, step_key: logged.append((dict(metrics), step_key)),
    )

    await train.train(_args())

    assert [metrics["rollout/step"] for metrics, _step_key in logged] == [0, 1]
    assert [step_key for _metrics, step_key in logged] == ["rollout/step", "rollout/step"]
    for metrics, _step_key in logged:
        assert metrics["perf/colocate/rollout_offload_block_time"] == pytest.approx(2.0)
        assert metrics["perf/colocate/rollout_to_train_active_time"] == pytest.approx(5.0)
        assert metrics["perf/colocate/train_to_rollout_block_time"] == pytest.approx(22.0)
        assert metrics["perf/colocate/switch_total_active_time"] == pytest.approx(27.0)

    # Bootstrap onload/update costs and the 125 seconds of model/rollout save work
    # are deliberately outside the per-step switch measurements.
    assert events.count(("update_weights", None)) == 1
    assert events.count(("update_weights", 0)) == 1
    assert events.count(("update_weights", 1)) == 1
    assert events[-1] == "dispose"


@pytest.mark.asyncio
async def test_train_model_does_not_add_timing_kwarg_when_collection_is_disabled():
    calls = []

    class Model:
        async def train(self, rollout_id, rollout_data_pack, **kwargs):
            calls.append((rollout_id, rollout_data_pack, kwargs))
            return "raw-result"

    result, wake_up_time = await train._train_model(Model(), 4, {"data_ref": object()})

    assert result == "raw-result"
    assert wake_up_time == 0.0
    assert calls[0][2] == {}
