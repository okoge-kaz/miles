from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import asyncio
import json
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
    def __init__(self, events, checkpoint_metrics=None):
        self.events = events
        self.checkpoint_metrics = checkpoint_metrics
        self.generate = _RemoteMethod(self._generate)
        self.record_batch_consumption = _RemoteMethod(self._record_batch_consumption)
        self.record_batch_trained = _RemoteMethod(self._record_batch_trained)
        self.acknowledge_trained_batch = _RemoteMethod(self._acknowledge)
        self.save = _RemoteMethod(self._save)
        self.mark_replay_buffer_committed = _RemoteMethod(self._mark_committed)
        self.eval = _RemoteMethod(lambda rollout_id: None)
        self.dispose = _RemoteMethod(self._dispose)

    async def _generate(self, rollout_id, *, updates_before_train):
        await asyncio.sleep(0)
        self.events.append(("generate", rollout_id, updates_before_train))
        return {
            "sample_indices": [rollout_id],
            "data_ref": object(),
            "replay_buffer_batch_token": f"token-{rollout_id}",
        }

    async def _record_batch_consumption(self, rollout_id):
        await asyncio.sleep(0)
        self.events.append(("consume", rollout_id))

    async def _record_batch_trained(self, rollout_id, *, actor_trained):
        await asyncio.sleep(0)
        self.events.append(("trained_telemetry", rollout_id, actor_trained))

    def _acknowledge(self, rollout_id, token):
        self.events.append(("ack", rollout_id, token))

    def _save(self, rollout_id):
        self.events.append(("rollout_save", rollout_id))
        return self.checkpoint_metrics

    def _mark_committed(self, rollout_id):
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
        "fully_async": False,
        "fully_async_queue_type": "queue-recycle",
        "use_replay_buffer": True,
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
        "debug_fail_after_rollout": None,
        "debug_failure_marker": None,
        "debug_failure_min_outstanding_groups": 0,
        "debug_failure_min_completed_groups": 0,
        "debug_failure_min_inflight_groups": 0,
        "debug_failure_min_inflight_tokens": 0,
        "debug_failure_min_regenerate_groups": 0,
        "save": None,
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


def test_updates_before_training_rollout_tracks_real_weight_pushes():
    args = _args(update_weights_interval=2)

    assert train_async._updates_before_training_rollout(args, 1) == 0
    assert train_async._updates_before_training_rollout(args, 2) == 1
    assert train_async._updates_before_training_rollout(_args(debug_skip_weight_update=True), 1) == 0


async def test_replay_buffer_commit_order(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events)
    _patch_runtime(monkeypatch, manager, actor)

    await train_async.train(_args())

    assert events.index(("train", 0)) < events.index(("ack", 0, "token-0"))
    assert events.index(("generate", 1, 1)) < events.index(("rollout_save", 0))
    assert events.index(("rollout_save", 0)) < events.index(("model", 0, True, True, False))
    assert events.index(("model", 0, True, True, False)) < events.index(("prune", 0))


async def test_replay_buffer_does_not_ack_failed_training(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events, fail_train=True)
    _patch_runtime(monkeypatch, manager, actor)

    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        await train_async.train(_args(num_rollout=1))

    assert not any(event[0] == "ack" for event in events)
    assert not any(event[0] == "rollout_save" for event in events)


async def test_legacy_checkpoint_order_and_async_save_semantics_are_unchanged(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events)
    _patch_runtime(monkeypatch, manager, actor)

    await train_async.train(
        _args(
            use_replay_buffer=False,
            debug_exit_after_rollout=1,
        )
    )

    assert not any(event[0] == "ack" for event in events)
    model_event = ("model", 0, False, True, False)
    assert model_event in events
    assert events.index(model_event) < events.index(("rollout_save", 0))


async def test_debug_failure_requires_committed_replay_work_and_writes_marker(monkeypatch, tmp_path):
    events = []
    metrics = {
        "resume/benchmark/checkpoint/outstanding_groups": 4.0,
        "resume/benchmark/checkpoint/completed_groups_reused": 3.0,
        "resume/benchmark/checkpoint/partial_groups_continued": 1.0,
        "resume/benchmark/checkpoint/partial_response_tokens_continued": 128.0,
        "resume/benchmark/checkpoint/groups_to_regenerate": 0.0,
    }
    manager = _RolloutManager(events, checkpoint_metrics=metrics)
    actor = _ActorModel(events)
    _patch_runtime(monkeypatch, manager, actor)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("0\n", encoding="utf-8")
    marker = checkpoint / "intentional-failure.json"

    class InjectedFailure(Exception):
        pass

    monkeypatch.setattr(
        train_async,
        "_terminate_for_debug_failure",
        lambda: (_ for _ in ()).throw(InjectedFailure()),
    )
    with pytest.raises(InjectedFailure):
        await train_async.train(
            _args(
                num_rollout=2,
                debug_fail_after_rollout=1,
                debug_failure_marker=str(marker),
                debug_failure_min_outstanding_groups=3,
                debug_failure_min_completed_groups=3,
                debug_failure_min_inflight_groups=1,
                debug_failure_min_inflight_tokens=64,
                save=str(checkpoint),
            )
        )

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["event"] == "intentional_whole_job_failure"
    assert payload["rollout_id"] == 0
    assert payload["checkpoint"]["tracker_iteration"] == 0
    assert events.index(("model", 0, True, True, False)) < events.index(("prune", 0))
    assert not any(event[0] == "update_weights" and event[1] == 0 for event in events)


async def test_queue_recycle_prefetches_next_batch_before_its_weight_update(monkeypatch):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events)
    _patch_runtime(monkeypatch, manager, actor)

    await train_async.train(
        _args(
            fully_async=True,
            use_replay_buffer=False,
        )
    )

    assert events.index(("generate", 0, 0)) < events.index(("generate", 1, 1))
    assert events.index(("generate", 1, 1)) < events.index(("consume", 0))
    assert events.index(("consume", 0)) < events.index(("train", 0))
    assert events.index(("train", 0)) < events.index(("trained_telemetry", 0, True))
    assert events.index(("generate", 1, 1)) < events.index(("update_weights", 0))
    assert events.index(("update_weights", 0)) < events.index(("consume", 1))
    assert events.index(("consume", 1)) < events.index(("train", 1))


@pytest.mark.parametrize("policy", ["queue-max", "queue-drop"])
async def test_selection_policies_defer_next_batch_until_after_weight_update(monkeypatch, policy):
    events = []
    manager = _RolloutManager(events)
    actor = _ActorModel(events)
    _patch_runtime(monkeypatch, manager, actor)

    await train_async.train(
        _args(
            fully_async=True,
            fully_async_queue_type=policy,
            use_replay_buffer=False,
        )
    )

    assert events.index(("consume", 0)) < events.index(("train", 0))
    assert events.index(("update_weights", 0)) < events.index(("generate", 1, 0))
