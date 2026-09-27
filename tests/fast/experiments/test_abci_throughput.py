from pathlib import Path

import pytest
from experiments.tools.analyze_abci_throughput import read_windows, summarize


def _window(seconds, tokens):
    return {
        "throughput/window_seconds": seconds,
        "throughput/optimizer_updates": 1,
        "throughput/completed_training_batches": 1,
        "throughput/generated_tokens": tokens,
        "throughput/window_accepted_loss_tokens_available": 1,
        "throughput/window_accepted_loss_tokens": tokens / 2,
        "queue/trainer_starvation_seconds": 0,
        "queue/rollout_backpressure_seconds": 0,
    }


def test_unequal_windows_use_total_time_instead_of_average_rates():
    result = summarize([_window(1, 100), _window(9, 9000)])
    assert result["generated_tokens_per_second"] == 910
    assert result["useful_tokens_per_second"] == 455
    assert result["seconds_per_update"] == 5


def test_only_complete_twenty_update_runs_are_accepted(tmp_path: Path):
    path = tmp_path / "job.log"
    lines = [f"rollout pipeline throughput {step}: {_window(2, 10)}\n" for step in range(20)]
    path.write_text("".join(lines[:19]))
    with pytest.raises(ValueError, match="expected steps"):
        read_windows(path)
    path.write_text("".join(lines + [lines[0]]))
    assert len(read_windows(path)) == 20
    with path.open("a") as stream:
        stream.write(f"rollout pipeline throughput 0: {_window(3, 10)}\n")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        read_windows(path)


def test_missing_accepted_counter_is_not_reported_as_zero_throughput(tmp_path: Path):
    path = tmp_path / "job.log"
    rows = [_window(2, 10) for _ in range(20)]
    rows[5]["throughput/window_accepted_loss_tokens_available"] = 0
    path.write_text("".join(f"rollout pipeline throughput {step}: {row}\n" for step, row in enumerate(rows)))
    with pytest.raises(ValueError, match="accepted token counter missing"):
        read_windows(path)
