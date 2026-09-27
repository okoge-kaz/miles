from argparse import Namespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.log_utils import aggregate_train_losses
from miles.backends.training_utils.loss import _pack_logging_values
from miles.backends.training_utils.loss_hub import losses as loss_utils
from miles.backends.training_utils.loss_hub.policy_lag_metrics import (
    POLICY_LAG_PART_PREFIX,
    finalize_policy_lag_parts,
)
from miles.backends.training_utils.loss_hub.tis_population_metrics import (
    TIS_POPULATION_PART_PREFIX,
    compute_tis_abs_population_parts,
    finalize_tis_abs_population_parts,
)

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def _args(*, calculate_per_token_loss: bool) -> Namespace:
    return Namespace(
        calculate_per_token_loss=calculate_per_token_loss,
        qkv_format="thd",
    )


def _finalize(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    return finalize_tis_abs_population_parts({key: float(value) for key, value in parts.items()})


def test_per_token_metrics_recover_masked_truncated_population(monkeypatch) -> None:
    make_parallel_state()
    monkeypatch.setattr(
        "miles.backends.training_utils.log_utils.MultiPGUtil.all_reduce",
        lambda *_args, **_kwargs: None,
    )
    parts = compute_tis_abs_population_parts(
        args=_args(calculate_per_token_loss=True),
        batch={
            "total_lengths": [3, 4],
            "response_lengths": [2, 3],
            "loss_masks": [torch.zeros(2), torch.tensor([1.0, 0.0, 1.0])],
            "truncated": [1, 0],
        },
        tis_abs=torch.tensor([0.2, 0.6, 0.1, 0.3, 0.5]),
    )
    packed = _pack_logging_values(3, {"loss": torch.tensor(2.0), **parts}, device=torch.device("cpu"))

    metrics = aggregate_train_losses([packed])

    assert packed["keys"] == ["loss"]
    assert all(key.startswith(TIS_POPULATION_PART_PREFIX) for key in packed["diagnostic_keys"])
    assert metrics["tis_abs/active"] == pytest.approx(0.2)
    assert metrics["tis_abs/all_response"] == pytest.approx(0.34)
    assert metrics["tis_abs/truncated"] == pytest.approx(0.4)
    assert metrics["tis_abs/non_truncated"] == pytest.approx(0.3)
    assert metrics["tis_abs_approx_p95_capped_1e3/active"] == pytest.approx(0.5)
    assert metrics["tis_abs_approx_p95_capped_1e3/truncated"] == pytest.approx(1.0)
    assert metrics["tis_abs_approx_p99_capped_1e3/non_truncated"] == pytest.approx(0.5)
    assert not any(key.startswith(TIS_POPULATION_PART_PREFIX) for key in metrics)
    assert all(not value.requires_grad for value in parts.values())


def test_sample_mean_metrics_match_loss_reducer_semantics() -> None:
    make_parallel_state()
    parts = compute_tis_abs_population_parts(
        args=_args(calculate_per_token_loss=False),
        batch={
            "total_lengths": [3, 4],
            "response_lengths": [2, 3],
            "loss_masks": [torch.tensor([1.0, 0.0]), torch.ones(3)],
            "truncated": [1, 0],
        },
        tis_abs=torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0]),
    )

    metrics = _finalize(parts)

    assert metrics["tis_abs/active"] == pytest.approx(0.5)
    assert metrics["tis_abs/all_response"] == pytest.approx(0.75)
    assert metrics["tis_abs/truncated"] == pytest.approx(0.5)
    assert metrics["tis_abs/non_truncated"] == pytest.approx(1.0)


def test_missing_truncated_metadata_preserves_legacy_metrics() -> None:
    make_parallel_state()

    parts = compute_tis_abs_population_parts(
        args=_args(calculate_per_token_loss=True),
        batch={
            "total_lengths": [2],
            "response_lengths": [1],
            "loss_masks": [torch.ones(1)],
        },
        tis_abs=torch.ones(1),
    )

    assert parts == {}


def test_rejects_invalid_truncated_metadata() -> None:
    make_parallel_state()

    with pytest.raises(ValueError, match="must be 0 or 1"):
        compute_tis_abs_population_parts(
            args=_args(calculate_per_token_loss=True),
            batch={
                "total_lengths": [2],
                "response_lengths": [1],
                "loss_masks": [torch.ones(1)],
                "truncated": [2],
            },
            tis_abs=torch.ones(1),
        )


def test_excluded_infinite_values_do_not_poison_other_population() -> None:
    make_parallel_state()
    parts = compute_tis_abs_population_parts(
        args=_args(calculate_per_token_loss=True),
        batch={
            "total_lengths": [2, 2],
            "response_lengths": [1, 1],
            "loss_masks": [torch.zeros(1), torch.ones(1)],
            "truncated": [1, 0],
        },
        tis_abs=torch.tensor([torch.inf, 0.25]),
    )

    metrics = _finalize(parts)

    assert metrics["tis_abs/truncated"] == torch.inf
    assert metrics["tis_abs/non_truncated"] == pytest.approx(0.25)


def test_policy_loss_emits_pre_target_population_parts(monkeypatch) -> None:
    make_parallel_state()
    old_log_probs = [torch.tensor([0.0]), torch.log(torch.tensor([1.25]))]
    rollout_log_probs = [-torch.log(torch.tensor([2.0])), torch.tensor([0.0])]
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *unused_args, **unused_kwargs: {"log_probs": [value.clone() for value in old_log_probs]},
    )
    _, reported = loss_utils.policy_loss_function(
        make_args(
            use_tis=True,
            calculate_per_token_loss=True,
            entropy_coef=0.0,
            kl_coef=0.0,
            observe_training_entropy=False,
            log_policy_lag_metrics=True,
            zero_loss_on_truncated=True,
        ),
        {
            "advantages": [torch.ones(1), torch.ones(1)],
            "log_probs": old_log_probs,
            "rollout_log_probs": rollout_log_probs,
            "unconcat_tokens": [torch.tensor([1, 2]), torch.tensor([3, 4])],
            "response_lengths": [1, 1],
            "total_lengths": [2, 2],
            "loss_masks": [torch.zeros(1), torch.ones(1)],
            "policy_lag_initial_loss_masks": [torch.ones(1), torch.ones(1)],
            "truncated": [1, 0],
        },
        logits=torch.zeros((1, 4, 8)),
        sum_of_sample_mean=lambda value: value.sum(),
    )

    metrics = _finalize({key: value for key, value in reported.items() if key.startswith(TIS_POPULATION_PART_PREFIX)})
    policy_lag = finalize_policy_lag_parts(
        {key: float(value) for key, value in reported.items() if key.startswith(POLICY_LAG_PART_PREFIX)}
    )

    assert metrics["tis_abs/active"] == pytest.approx(0.125)
    assert metrics["tis_abs/all_response"] == pytest.approx(0.625)
    assert metrics["tis_abs/truncated"] == pytest.approx(1.0)
    assert metrics["tis_abs/non_truncated"] == pytest.approx(0.25)
    expected_rms = torch.sqrt((torch.log(torch.tensor(2.0)).square() + torch.log(torch.tensor(1.25)).square()) / 2)
    expected_pre_sensitivity_rms = torch.sqrt(
        (1.5 * torch.log(torch.tensor(2.0)).square() + 1.25 * torch.log(torch.tensor(1.25)).square()) / 2.75
    )
    assert policy_lag["policy_lag/all_response/delta_rms"] == pytest.approx(float(expected_rms))
    assert policy_lag["policy_lag/all_response/loss_sensitivity_weighted_delta_rms_pre"] == pytest.approx(
        float(expected_pre_sensitivity_rms)
    )
    assert policy_lag["policy_lag/all_response/loss_sensitivity_weighted_delta_rms_post"] == pytest.approx(
        float(torch.log(torch.tensor(1.25)))
    )
    assert policy_lag["policy_lag/all_response/loss_sensitivity_retained_fraction"] == pytest.approx(1.25 / 2.75)
