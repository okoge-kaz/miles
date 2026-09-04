from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from experiments.tools.training_analysis import summarize_dump
from experiments.tools.training_analysis.summarize_log import summarize
from miles.utils.types import Sample


def test_log_summary_measures_reward_change_and_staleness(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "sglang_enable_response_weight_version_segments .. True\n"
        "rollout 0: {'rollout/rewards': -0.1, 'rollout/raw_reward': 0.25, "
        "'rollout/truncated': 0.1, 'rollout/sample_staleness': 0.0}\n"
        "step 0: {'train/optimizer_step_applied': 1.0}\n"
        "ft cls=actor fn=update_weights phase=end ok=true elapsed_s=1.0\n"
        "rollout 1: {'rollout/rewards': 0.1, 'rollout/raw_reward': 0.75, "
        "'rollout/truncated': 0.0, 'rollout/sample_staleness': 2.0}\n",
        encoding="utf-8",
    )

    summary = summarize(log)

    assert summary["reward"]["last_minus_first"] == 0.5
    assert summary["reward"]["mean"] == 0.5
    assert summary["reward"]["normalized_mean"] == 0.0
    assert summary["reward"]["source"] == "raw_reward"
    assert summary["optimizer_steps_applied"] == 1
    assert summary["sample_staleness"]["within_requested_bound_4"] is True
    assert summary["response_weight_version_segments"] is True


def test_dump_summary_groups_environments_and_checks_segments(monkeypatch, tmp_path: Path) -> None:
    code_sample = Sample(
        index=1,
        tokens=[1, 2, 3],
        response_length=3,
        reward=1.0,
        status=Sample.Status.COMPLETED,
        metadata={
            "verifier": "python_code",
            "sample_staleness_reference_weight_version": 2,
            "train_weight_version": 4,
        },
        response_weight_version_segments=[[[0, 1, 2], [1, 3, 3]]],
    )
    math_sample = Sample(
        index=2,
        tokens=[1, 2],
        response_length=2,
        reward=0.0,
        status=Sample.Status.TRUNCATED,
        metadata={
            "verifier": "math",
            "sample_staleness_reference_weight_version": 3,
            "train_weight_version": 4,
        },
    )

    class FakeReader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def rollout_ids(self):
            return SimpleNamespace(train=[0])

        def load_joined(self, rollout_id: int):
            assert rollout_id == 0
            train_rows = {
                1: SimpleNamespace(raw_reward=1.0),
                2: SimpleNamespace(raw_reward=0.0),
            }
            return SimpleNamespace(samples=[code_sample, math_sample], train_rows=train_rows)

    monkeypatch.setattr(summarize_dump, "DumpReader", FakeReader)
    summary = summarize_dump.summarize(tmp_path)

    assert summary["by_domain"]["code"]["reward_mean"] == 1.0
    assert summary["by_domain"]["math"]["truncated_fraction"] == 1.0
    assert summary["sample_staleness"]["max"] == 2.0
    assert summary["sample_staleness"]["within_requested_bound"] is True
    segments = summary["response_weight_version_segments"]
    assert segments["exact_coverage_fraction"] == 1.0
    assert segments["exact_policy_token_coverage_fraction"] == 1.0
    assert segments["mixed_version_samples"] == 1


def test_dump_summary_distinguishes_observations_from_policy_tokens(monkeypatch, tmp_path: Path) -> None:
    sample = Sample(
        index=1,
        tokens=[1, 2, 3],
        response_length=3,
        reward=1.0,
        loss_mask=[1, 0, 1],
        status=Sample.Status.COMPLETED,
        metadata={
            "verifier": "tau_bench_environment",
            "tau_done": True,
            "tau_turns": 2,
            "tau_user_length_truncations": 1,
        },
        response_weight_version_segments=[[[0, 2, 1]]],
    )

    class FakeReader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def rollout_ids(self):
            return SimpleNamespace(train=[0])

        def load_joined(self, rollout_id: int):
            assert rollout_id == 0
            return SimpleNamespace(samples=[sample], train_rows={})

    monkeypatch.setattr(summarize_dump, "DumpReader", FakeReader)
    summary = summarize_dump.summarize(tmp_path)

    segments = summary["response_weight_version_segments"]
    assert segments["exact_coverage_fraction"] == 0.0
    assert segments["exact_policy_token_coverage_fraction"] == 1.0
    assert segments["covered_response_token_fraction"] == 2 / 3
    assert segments["covered_policy_token_fraction"] == 1.0
    tau = summary["tau_bench_environment"]
    assert tau["done_fraction"] == 1.0
    assert tau["turns_mean"] == 2.0
    assert tau["user_length_truncation_samples"] == 1
    assert tau["user_length_truncation_events"] == 1.0
