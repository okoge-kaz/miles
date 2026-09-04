import pytest
from tests.ci.ci_register import register_cpu_ci
from tests.fast.ray.rollout.conftest import make_args, make_sample

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.recycle_compute_metrics import SAMPLE_REFERENCE_VERSION_KEY, TRAIN_VERSION_KEY

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def test_staleness_aware_loss_carries_staleness_without_diagnostic_logging() -> None:
    args = make_args(
        rewards_normalization=False,
        log_sample_staleness_metrics=False,
        use_staleness_aware_loss=True,
    )
    sample = make_sample()
    sample.metadata[SAMPLE_REFERENCE_VERSION_KEY] = 3
    sample.metadata[TRAIN_VERSION_KEY] = 8

    out = convert_samples_to_train_data(
        args,
        [sample],
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )

    assert out["sample_staleness"] == [5]


def test_staleness_aware_loss_rejects_incomplete_staleness_provenance() -> None:
    args = make_args(
        rewards_normalization=False,
        log_sample_staleness_metrics=False,
        use_staleness_aware_loss=True,
    )
    sample = make_sample()
    sample.metadata[TRAIN_VERSION_KEY] = 8

    with pytest.raises(RuntimeError, match="requires complete per-sample training-staleness provenance"):
        convert_samples_to_train_data(
            args,
            [sample],
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
