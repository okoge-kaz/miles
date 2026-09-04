from argparse import Namespace

from miles.rollout.data_source import PARTIAL_ROLLOUT_BUFFER_KEY, RolloutDataSourceWithBuffer
from miles.rollout.partial_rollout_telemetry import (
    TOKEN_BOUNDARY_LEDGER_KEY,
    collect_partial_rollout_staleness_metrics,
)
from miles.utils.types import Sample


def _make_source(**overrides) -> RolloutDataSourceWithBuffer:
    values = {
        "partial_rollout": True,
        "fully_async": False,
        "use_replay_buffer": False,
        "n_samples_per_prompt": 2,
        "rollout_global_dataset": False,
        "rollout_shuffle": False,
    }
    values.update(overrides)
    source = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
    source.args = Namespace(**values)
    source.buffer = []
    source.sample_offset = 5
    source.epoch_id = 2
    source.sample_group_index = 8
    source.sample_index = 16
    source.metadata = {"marker": "kept"}
    return source


def test_sync_partial_buffer_round_trips_through_data_source_checkpoint() -> None:
    source = _make_source()
    source.buffer = [
        [
            Sample(
                index=3,
                response="old",
                response_length=2,
                metadata={"start_rollout_id": 7, TOKEN_BOUNDARY_LEDGER_KEY: [[2, 7]]},
            ),
            Sample(index=4, status=Sample.Status.ABORTED),
        ]
    ]

    state = source.checkpoint_state()
    source.buffer[0][0].response = "mutated after snapshot"

    restored = _make_source()
    restored.restore_checkpoint_state(state)

    assert restored.sample_offset == 5
    assert restored.metadata == {"marker": "kept"}
    assert len(restored.buffer) == 1
    assert restored.buffer[0][0].response == "old"
    assert restored.buffer[0][0].metadata["start_rollout_id"] == 7
    assert restored.buffer[0][1].status == Sample.Status.ABORTED

    restored.buffer[0][0].response_length = 3
    metrics = collect_partial_rollout_staleness_metrics(restored.buffer, rollout_id=8)
    assert metrics["staleness/total/max"] == 1.0
    assert metrics["staleness/partial_rollout/carried_prefix_tokens"] == 2.0
    assert metrics["staleness/partial_rollout/current_suffix_tokens"] == 1.0


def test_fully_async_does_not_duplicate_partial_buffer_in_data_source_state() -> None:
    source = _make_source(fully_async=True)
    source.buffer = [[Sample(), Sample()]]

    state = source.checkpoint_state()

    assert PARTIAL_ROLLOUT_BUFFER_KEY not in state


def test_replay_buffer_mode_does_not_duplicate_partial_buffer_in_data_source_state() -> None:
    source = _make_source(use_replay_buffer=True)
    source.buffer = [[Sample(), Sample()]]

    state = source.checkpoint_state()

    assert PARTIAL_ROLLOUT_BUFFER_KEY not in state
