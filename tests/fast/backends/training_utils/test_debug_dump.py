from argparse import Namespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state

from miles.backends.training_utils import debug_dump
from miles.backends.training_utils.log_utils import aggregate_train_losses
from miles.backends.training_utils.loss import _pack_logging_values

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def test_logging_values_are_packed_without_retaining_autograd() -> None:
    metric = torch.tensor(2.5, requires_grad=True)
    packed = _pack_logging_values(3, {"metric": metric}, device=torch.device("cpu"))
    torch.testing.assert_close(packed["values"], torch.tensor([3.0, 2.5]))
    assert not packed["values"].requires_grad
    assert "diagnostic_values" not in packed


def test_logging_values_preserve_float64_additive_parts() -> None:
    packed = _pack_logging_values(
        1,
        {
            "legacy_metric": torch.tensor(0.25, dtype=torch.float32),
            "_staleness_gradient_part/sequence_ess_sum_w/0": torch.tensor(1.0e100, dtype=torch.float64),
            "_staleness_gradient_part/token_count/0": torch.tensor(3.0, dtype=torch.float32),
        },
        device=torch.device("cpu"),
    )

    assert packed["keys"] == ["legacy_metric"]
    assert packed["values"].dtype == torch.float32
    assert packed["diagnostic_keys"] == [
        "_staleness_gradient_part/sequence_ess_sum_w/0",
        "_staleness_gradient_part/token_count/0",
    ]
    assert packed["diagnostic_values"].dtype == torch.float64
    assert torch.isfinite(packed["diagnostic_values"]).all()


def test_float64_additive_parts_reduce_without_promoting_historical_metrics(monkeypatch) -> None:
    make_parallel_state()
    monkeypatch.setattr(
        "miles.backends.training_utils.log_utils.MultiPGUtil.all_reduce",
        lambda *_args, **_kwargs: None,
    )
    packed = _pack_logging_values(
        2,
        {
            "legacy_metric": torch.tensor(4.0, dtype=torch.float32),
            "_staleness_gradient_part/high_dynamic_range/0": torch.tensor(1.0e100, dtype=torch.float64),
        },
        device=torch.device("cpu"),
    )

    metrics = aggregate_train_losses([packed])

    assert metrics["legacy_metric"] == 2.0
    assert metrics["_staleness_gradient_part/high_dynamic_range/0"] == pytest.approx(5.0e99)


def test_policy_loss_debug_records_joinable_final_sample_diagnostics(tmp_path, monkeypatch) -> None:
    make_parallel_state()
    monkeypatch.setattr(debug_dump, "_POLICY_LOSS_DUMP_COUNTER", 0)
    args = Namespace(
        dump_details=str(tmp_path),
        dump_policy_loss_debug=True,
        qkv_format="thd",
    )
    debug_dump.maybe_dump_policy_loss_debug(
        args=args,
        batch={
            "total_lengths": [3],
            "response_lengths": [2],
            "sample_indices": [17],
            "sample_group_indices": [8],
            "generation_attempt_numbers": [2],
            "training_steps": [5],
            "optimizer_step_ids": [9],
            "sample_staleness": [3],
        },
        train_log_probs=[torch.tensor([0.1, 0.2])],
        old_log_probs=[torch.tensor([0.0, 0.0])],
        rollout_log_probs=[torch.tensor([-0.1, -0.2])],
        advantages=[torch.tensor([1.0, 1.0])],
        local_loss_masks=[torch.ones(2)],
        ppo_kl=torch.tensor([-0.1, -0.2]),
        pg_loss=torch.tensor([1.0, 2.0]),
        final_pg_loss=torch.tensor([1.0, 0.0]),
        final_loss_masks=[torch.tensor([1.0, 0.0])],
        ppo_clipfrac=torch.tensor([1.0, 0.0]),
        policy_log_ratio=torch.tensor([0.2, 0.4]),
        tis_metrics={"tis_clipfrac": torch.tensor([0.0, 1.0])},
    )

    payload = torch.load(tmp_path / "policy_loss_debug" / "rank_0_call_0.pt", weights_only=False)
    [sample] = payload["samples"]
    assert sample["sample_index"] == 17
    assert sample["sample_group_index"] == 8
    assert sample["generation_attempt_id"] == "8:2"
    assert sample["training_step"] == 5
    assert sample["optimizer_step_id"] == 9
    assert sample["sample_staleness"] == 3
    assert sample["ppo_clip_fraction_local"] == 0.5
    assert sample["mask_fraction_local"] == 0.5
    assert sample["response_token_count_local"] == 2
    assert sample["pre_loss_token_count_local"] == 2
    assert sample["final_loss_token_count_local"] == 1
    assert sample["ppo_clip_count_local"] == 1
    assert sample["importance_clip_count_local"] == 1
    assert sample["absolute_pg_contribution_local"] == pytest.approx(1.0)
    assert sample["sequence_policy_rollout_log_ratio_local"] == pytest.approx(0.2)
    torch.testing.assert_close(sample["importance_clip_indicator"], torch.tensor([0.0, 1.0]))
