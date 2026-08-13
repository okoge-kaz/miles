from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import asyncio
import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import patch

import pytest

# Import the orchestration loop without importing the real placement module.
# The latter initializes SGLang/Megatron/Transformer Engine at module import and
# requires libcuda even though these tests use only pure-Python fakes.
_placement_group_stub = ModuleType("miles.ray.placement_group")
_placement_group_stub.create_placement_groups = None
_placement_group_stub.create_rollout_manager = None
_placement_group_stub.create_training_models = None
with patch.dict(sys.modules, {"miles.ray.placement_group": _placement_group_stub}):
    import train_async


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
    def __init__(self, events):
        self.events = events
        self.generate = _RemoteMethod(self._generate)
        self.acknowledge_trained_batch = _RemoteMethod(self._acknowledge)
        self.save = _RemoteMethod(self._save)
        self.mark_checkpoint_published = _RemoteMethod(self._mark_published)
        self.eval = _RemoteMethod(lambda rollout_id: None)
        self.dispose = _RemoteMethod(self._dispose)

    async def _generate(self, rollout_id):
        await asyncio.sleep(0)
        self.events.append(("generate", rollout_id))
        return {
            "sample_indices": [rollout_id],
            "data_ref": object(),
            "fully_async_batch_token": f"token-{rollout_id}",
        }

    def _acknowledge(self, rollout_id, token):
        self.events.append(("ack", rollout_id, token))

    def _save(self, rollout_id):
        self.events.append(("sidecar", rollout_id))

    def _mark_published(self, rollout_id):
        self.events.append(("prune", rollout_id))

    def _dispose(self):
        self.events.append(("dispose",))


class _ActorModel:
    def __init__(self, events, *, fail_train=False):
        self.events = events
        self.fail_train = fail_train

    async def update_weights(self, rollout_id=None):
        self.events.append(("update_weights", rollout_id))

    async def train(self, rollout_id, rollout_data, external_data=None):
        self.events.append(("train", rollout_id))
        if self.fail_train:
            raise RuntimeError("synthetic trainer failure")

    async def save_model(self, rollout_id, force_sync=False, *, write_dist=True, write_hf=True):
        self.events.append(("model", rollout_id, force_sync, write_dist, write_hf))


def _args(**overrides):
    values = {
        "colocate": False,
        "fully_async_rollout_checkpoint": True,
        "control_server_port": None,
        "ft_components": [],
        "use_critic": False,
        "offload_train": False,
        "check_weight_update_equal": False,
        "eval_interval": None,
        "start_rollout_id": 0,
        "skip_eval_before_train": True,
        "num_rollout": 2,
        "save_trigger_sentinel": None,
        "save_interval": 1,
        "hf_save_interval": None,
        "update_weights_interval": 1,
        "debug_exit_after_rollout": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _patch_runtime(monkeypatch, manager, actor):
    monkeypatch.setattr(train_async, "validate_async_off_policy_correction", lambda args: None)
    monkeypatch.setattr(train_async, "configure_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_async, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_async.object_store, "init_instance", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_async, "create_rollout_manager", lambda args, pg: (manager, None))

    async def create_models(args, pgs, rollout_manager):
        return actor, None

    monkeypatch.setattr(train_async, "create_training_models", create_models)
    monkeypatch.setattr(train_async, "maybe_start_mini_ft_controller", lambda args: None)
    monkeypatch.setattr(train_async, "remove_rollout_data_refs", lambda args, data: None)
    monkeypatch.setattr(
        train_async,
        "checkpoint_artifacts_due",
        lambda rollout_id, **kwargs: (True, False),
    )
    monkeypatch.setattr(train_async, "should_run_periodic_action", lambda *args, **kwargs: False)


async def test_full_replay_checkpoint_commit_order(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events)
    _patch_runtime(monkeypatch, manager, actor)

    await train_async.train(_args())

    assert events.index(("train", 0)) < events.index(("ack", 0, "token-0"))
    assert events.index(("generate", 1)) < events.index(("sidecar", 0))
    assert events.index(("sidecar", 0)) < events.index(("model", 0, True, True, False))
    assert events.index(("model", 0, True, True, False)) < events.index(("prune", 0))


async def test_full_replay_does_not_ack_failed_training(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events, fail_train=True)
    _patch_runtime(monkeypatch, manager, actor)

    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        await train_async.train(_args(num_rollout=1))

    assert not any(event[0] == "ack" for event in events)
    assert not any(event[0] == "sidecar" for event in events)


async def test_legacy_checkpoint_order_and_async_save_semantics_are_unchanged(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events)
    _patch_runtime(monkeypatch, manager, actor)

    await train_async.train(
        _args(
            fully_async_rollout_checkpoint=False,
            debug_exit_after_rollout=1,
        )
    )

    assert not any(event[0] == "ack" for event in events)
    model_event = ("model", 0, False, True, False)
    assert model_event in events
    assert events.index(model_event) < events.index(("sidecar", 0))
