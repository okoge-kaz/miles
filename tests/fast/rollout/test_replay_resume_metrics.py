from miles.rollout.replay_resume_metrics import checkpoint_resume_metrics, replay_load_metrics


def _replay_state(replay_type: str) -> dict:
    return {
        "replay_buffer_type": replay_type,
        "data_source": {"sample_group_index": 36},
        "snapshot_counts": {
            "pending_groups": 6,
            "inflight_groups": 2 if replay_type == "inflight" else 0,
            "inflight_response_tokens": 700 if replay_type == "inflight" else 0,
        },
        "ready_items": [{}, {}],
        "drain_progress": [{"data": [{"group": 1}]}],
        "prepared_batches": [{"group_ids": [31]}],
        "regeneration_group_ids": [34, 35] if replay_type == "rollout" else [],
    }


def test_no_replay_reports_all_cursor_ahead_samples_as_lost() -> None:
    metrics = checkpoint_resume_metrics(
        rollout_id=9,
        rollout_batch_size=3,
        n_samples_per_prompt=16,
        data_source_state={"sample_group_index": 36},
        replay_state=None,
    )

    assert metrics["resume/benchmark/checkpoint/outstanding_groups"] == 6
    assert metrics["resume/benchmark/checkpoint/lost_samples"] == 96
    assert metrics["resume/benchmark/checkpoint/carried_samples"] == 0
    assert metrics["resume/benchmark/checkpoint/sample_conservation_fraction"] == 0


def test_rollout_replay_preserves_completed_work_and_regenerates_active_groups() -> None:
    state = _replay_state("rollout")
    metrics = checkpoint_resume_metrics(
        rollout_id=9,
        rollout_batch_size=3,
        n_samples_per_prompt=16,
        data_source_state=state["data_source"],
        replay_state=state,
    )

    assert metrics["resume/benchmark/checkpoint/carried_samples"] == 96
    assert metrics["resume/benchmark/checkpoint/lost_samples"] == 0
    assert metrics["resume/benchmark/checkpoint/completed_groups_reused"] == 4
    assert metrics["resume/benchmark/checkpoint/groups_to_regenerate"] == 2
    assert metrics["resume/benchmark/checkpoint/partial_groups_continued"] == 0


def test_inflight_replay_counts_partial_groups_and_tokens_as_preserved() -> None:
    state = _replay_state("inflight")
    metrics = checkpoint_resume_metrics(
        rollout_id=9,
        rollout_batch_size=3,
        n_samples_per_prompt=16,
        data_source_state=state["data_source"],
        replay_state=state,
    )

    assert metrics["resume/benchmark/checkpoint/carried_samples"] == 96
    assert metrics["resume/benchmark/checkpoint/partial_groups_continued"] == 2
    assert metrics["resume/benchmark/checkpoint/partial_response_tokens_continued"] == 700
    assert metrics["resume/benchmark/checkpoint/generated_work_group_fraction"] == 1


def test_replay_load_metrics_distinguish_no_replay() -> None:
    metrics = replay_load_metrics(
        replay_type=None,
        read_seconds=0.25,
        restore_seconds=0.0,
        total_seconds=0.25,
    )

    assert metrics["resume/benchmark/load/total_seconds"] == 0.25
    assert metrics["resume/benchmark/mode/no_replay"] == 1
    assert metrics["resume/benchmark/mode/inflight"] == 0
