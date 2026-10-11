import math
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.loss_hub.policy_lag_metrics import (
    collect_policy_lag_parts,
    finalize_policy_lag_metrics,
)


def _parts(current, behavior, masks, categories):
    return collect_policy_lag_parts(
        [torch.tensor(x, requires_grad=True) for x in current],
        [torch.tensor(x) for x in behavior],
        [torch.tensor(x) for x in masks],
        categories=categories,
        category_vocabulary=["math", "code"],
        truncated=[0] * len(categories),
    )


def test_signed_delta_global_pooling_masks_and_empty_category():
    first = _parts([[-1.0, -2.0]], [[-2.0, -1.0]], [[1, 0]], ["math"])
    second = _parts([[-4.0, -3.0, -2.0]], [[-2.0, -1.0, -4.0]], [[1, 1, 1]], ["code"])
    assert first.keys() == second.keys()
    assert not any(v.requires_grad for v in first.values())
    pooled = {key: float(first[key] + second[key]) for key in first}
    metrics = finalize_policy_lag_metrics(pooled)
    p = "policy_lag/total/loss_enabled"
    assert metrics[f"{p}/token_count"] == 4
    assert metrics[f"{p}/delta_mean"] == pytest.approx(-0.25)
    assert metrics[f"{p}/delta_abs_mean"] == pytest.approx(1.75)
    assert metrics[f"{p}/delta_sq_mean"] == pytest.approx(3.25)
    assert metrics["policy_lag/category/math/loss_enabled/delta_mean"] == 1
    assert metrics["policy_lag/category/code/loss_enabled/delta_mean"] == pytest.approx(-2 / 3)
    assert not any(k.startswith("_policy") for k in metrics)


def test_nonfinite_response_invalidates_actual_population_not_masked_loss():
    parts = _parts([[-1.0, float("nan")]], [[-2.0, -1.0]], [[1, 0]], ["math"])
    metrics = finalize_policy_lag_metrics({k: float(v) for k, v in parts.items()})
    assert math.isnan(metrics["policy_lag/total/all_response/delta_mean"])
    assert metrics["policy_lag/total/all_response/nonfinite_token_count"] == 1
    assert metrics["policy_lag/total/loss_enabled/delta_mean"] == 1
    assert "policy_lag/category/code/loss_enabled/delta_mean" not in metrics


def test_cp_partitioned_moments_match_unsplit_tokens():
    whole = _parts([[-1.0, -2.0, -4.0, -3.0]], [[-2.0, -1.0, -2.0, -4.0]], [[1, 0, 1, 1]], ["math"])
    left = _parts([[-1.0, -4.0]], [[-2.0, -2.0]], [[1, 1]], ["math"])
    right = _parts([[-2.0, -3.0]], [[-1.0, -4.0]], [[0, 1]], ["math"])
    for key, value in whole.items():
        assert torch.allclose(value, left[key] + right[key])


@pytest.mark.parametrize("num_rollouts", [None, 2])
def test_training_reduction_logs_pooled_token_moments_without_sample_normalization(monkeypatch, num_rollouts):
    parts = [
        _parts([[-1.0, -2.0]], [[-2.0, -1.0]], [[1, 0]], ["math"]),
        _parts([[-4.0, -3.0, -2.0]], [[-2.0, -1.0, -4.0]], [[1, 1, 1]], ["code"]),
    ]
    keys = ["loss", *parts[0]]
    microbatches = [
        {"keys": keys, "values": torch.tensor([1.0, 2.0, *p.values()], dtype=torch.float64)} for p in parts
    ]
    state = SimpleNamespace(cp=SimpleNamespace(size=2), effective_dp_cp=SimpleNamespace(groups_inner_to_outer=[0]))
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: state)
    # A second replica has identical token populations. Counts must double;
    # sample/CP normalization belongs only to the training losses.
    monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", lambda values, *a, **kw: values.mul_(2))
    captured = []
    monkeypatch.setattr(log_utils.tracking, "log", lambda args, values, **kw: captured.append(values))

    reduced = log_utils.aggregate_train_losses(microbatches, num_rollouts=num_rollouts)
    output = log_utils.log_train_step(
        SimpleNamespace(), reduced, grad_norm=1.0, rollout_id=2, step_id=1, num_steps_per_rollout=3, should_log=True
    )

    prefix = "train/policy_lag/total/loss_enabled"
    assert output[f"{prefix}/token_count"] == 8
    assert output[f"{prefix}/delta_mean"] == pytest.approx(-0.25)
    assert output[f"{prefix}/delta_sq_mean"] == pytest.approx(3.25)
    assert output["train/policy_lag/category/math/loss_enabled/delta_mean"] == 1
    assert output["train/step"] == 7
    assert captured == [output]
    assert not any("_policy_lag_part" in key for key in output)
