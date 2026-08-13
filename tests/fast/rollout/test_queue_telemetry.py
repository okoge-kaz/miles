import copy

from miles.rollout.queue_telemetry import _QueueLifecycleRecorder, _ResponseLengthMetrics
from miles.utils.types import Sample


def _group(group_index: int, lengths: tuple[int, ...]) -> list[Sample]:
    group = []
    for offset, length in enumerate(lengths):
        sample = Sample(group_index=group_index, index=group_index * 10 + offset)
        sample.response_length = length
        sample.weight_versions = ["2"]
        sample.first_prefill_weight_versions = [1]
        group.append(sample)
    return group


def test_response_length_metrics_checkpoint_roundtrip_preserves_accumulators():
    original = _ResponseLengthMetrics(populations=("generated", "queue_evicted"))
    original.record("generated", _group(1, (2, 5)))

    restored = _ResponseLengthMetrics(populations=("generated", "queue_evicted"))
    restored.restore_checkpoint_state(original.checkpoint_state())
    restored.record("queue_evicted", _group(2, (3, 7)))

    metrics = restored.collect()
    assert metrics["queue/selection/generated/sample_length/count"] == 2
    assert metrics["queue/selection/generated/sample_length/sum"] == 7
    assert metrics["queue/selection/generated/group_max_length/max"] == 5
    assert metrics["queue/selection/queue_evicted/sample_length/count"] == 2
    assert metrics["queue/selection/queue_evicted/sample_length/sum"] == 10


def test_queue_lifecycle_checkpoint_roundtrip_uses_stable_group_identity():
    group = _group(7, (3, 4))
    original = _QueueLifecycleRecorder(enabled=True)
    attempt = original.begin_attempt(group, submission_version=1)
    original.group_ready(attempt, group, ready_version=2)
    original.enqueued(group, queue_put_version=2, depth_before=0, depth_after=1)

    restored = _QueueLifecycleRecorder(enabled=True)
    restored.restore_checkpoint_state(original.checkpoint_state())
    decoded_group = copy.deepcopy(group)
    restored.dequeued(decoded_group, depth_after_observed=0)
    restored.finish(
        decoded_group,
        disposition="trained",
        decision_version=3,
        rollout_id=11,
        reference_version=1,
        bound_staleness=2,
    )

    metadata = restored.take_metadata(policy="queue-recycle", capacity_groups=1000)
    [record] = metadata["records"]
    assert record["attempt_id"] == 0
    assert record["submission_seq"] == 0
    assert record["enqueue_seq"] == 0
    assert record["dequeue_seq"] == 0
    assert record["sample_indices"] == [70, 71]
    assert record["disposition"] == "trained"
    assert record["bound_staleness"] == 2

    next_attempt = restored.begin_attempt(_group(8, (1, 1)), submission_version=3)
    assert next_attempt["attempt_id"] == 1
    assert next_attempt["submission_seq"] == 1


def test_queue_lifecycle_restore_records_checkpoint_promoted_admission():
    group = _group(9, (2, 6))
    original = _QueueLifecycleRecorder(enabled=True)
    attempt = original.begin_attempt(group, submission_version=1)
    original.group_ready(attempt, group, ready_version=2)

    restored = _QueueLifecycleRecorder(enabled=True)
    restored.restore_checkpoint_state(original.checkpoint_state())
    restored.restore_queue_admission(
        copy.deepcopy(group),
        queue_put_version=2,
        depth_before=4,
        depth_after=5,
    )
    restored.finish(
        copy.deepcopy(group),
        disposition="trained",
        decision_version=2,
        rollout_id=12,
    )

    [record] = restored.take_metadata(policy="queue-recycle", capacity_groups=1000)["records"]
    assert record["enqueue_seq"] == 0
    assert record["queue_put_version"] == 2
    assert record["queue_depth_before_enqueue"] == 4
    assert record["queue_depth_after_enqueue"] == 5
