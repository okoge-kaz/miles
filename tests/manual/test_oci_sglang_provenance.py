"""Real-image unit checks for the opt-in OCI scheduler provenance overlay."""

import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import BeginWeightUpdateReqInput, EndWeightUpdateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats

pytestmark = pytest.mark.skipif(
    not hasattr(Scheduler, "stamp_forward_weight_version"),
    reason="Requires the opt-in OCI SGLang provenance overlay; skips are not qualification.",
)


def test_forward_provenance_uses_committed_serving_version(monkeypatch):
    serving = SimpleNamespace(weight_version="10")
    monkeypatch.setattr(scheduler_module, "get_serving", lambda: serving)
    scheduler = Scheduler.__new__(Scheduler)
    stats = SchedulerReqTimeStats()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        reqs=[SimpleNamespace(time_stats=stats, beam_group=None)],
    )
    scheduler.stamp_forward_weight_version(batch)
    assert batch.forward_weight_version == 10
    serving.weight_version = "11"
    batch.forward_mode = SimpleNamespace(is_extend=lambda: False)
    scheduler.stamp_forward_weight_version(batch)
    assert stats.convert_to_policy_version_meta_info() == {
        "first_prefill_weight_version": 10,
        "min_forward_weight_version": 10,
        "max_forward_weight_version": 11,
        "last_forward_weight_version": 11,
    }
    assert batch.forward_weight_version == 11


def test_token_segments_merge_clip_and_survive_pickle_without_metrics():
    stats = SchedulerReqTimeStats(first_prefill_weight_version=10)
    stats.enable_metrics = False
    for start, end, version in [(0, 1, 10), (1, 3, 10), (3, 5, 11)]:
        stats.record_response_weight_version_segment(response_start=start, response_end=end, weight_version=version)
    restored = pickle.loads(pickle.dumps(stats))
    assert restored.convert_to_policy_version_meta_info(4)["response_weight_version_segments"] == [
        [0, 3, 10],
        [3, 4, 11],
    ]


def test_disabled_provenance_keeps_empty_serialization():
    stats = SchedulerReqTimeStats()
    stats.enable_metrics = False
    assert stats.__getstate__() == {}


@pytest.mark.parametrize("failed_finalize", [False, True])
def test_session_commits_through_native_version_tracker(monkeypatch, failed_finalize):
    monkeypatch.setattr("torch.distributed.barrier", lambda **kwargs: None)
    runner = MagicMock()
    scheduler = MagicMock()
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(iter_runners=lambda: [("", runner)]),
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=MagicMock(),
        flush_cache=MagicMock(return_value=True),
        is_fully_idle=MagicMock(return_value=True),
        scheduler=scheduler,
    )
    assert manager.begin_weight_update(BeginWeightUpdateReqInput(weight_version="11")).success
    scheduler.record_weight_version_change.assert_not_called()
    if failed_finalize:
        runner.end_weight_update.side_effect = RuntimeError("finalize failed")
        with pytest.raises(RuntimeError, match="finalize failed"):
            manager.end_weight_update(EndWeightUpdateReqInput())
        scheduler.record_weight_version_change.assert_not_called()
    else:
        assert manager.end_weight_update(EndWeightUpdateReqInput()).success
        scheduler.record_weight_version_change.assert_called_once_with(new_version="11")
