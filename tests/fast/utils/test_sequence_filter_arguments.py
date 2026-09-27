from argparse import Namespace

import pytest

from miles.utils.arguments import validate_train_rollout_logprob_sequence_filter


def _args(**overrides):
    return Namespace(
        **{
            "use_train_rollout_logprob_sequence_filter": True,
            "train_rollout_logprob_sequence_filter_threshold": 2.0,
            "train_backend": "megatron",
            "fuse_one_step_actor_logprobs": True,
            "calculate_per_token_loss": True,
            "loss_type": "policy_loss",
            "custom_pg_loss_reducer_function_path": None,
            "custom_tis_function_path": None,
            **overrides,
        }
    )


@pytest.mark.parametrize("threshold", [0.5, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="finite and >= 1"):
        validate_train_rollout_logprob_sequence_filter(
            _args(train_rollout_logprob_sequence_filter_threshold=threshold)
        )


@pytest.mark.parametrize(
    "override",
    [
        {"train_backend": "fsdp"},
        {"fuse_one_step_actor_logprobs": False},
        {"calculate_per_token_loss": False},
        {"loss_type": "value_loss"},
        {"multi_lora": True},
        {"indep_dp": True},
        {"enable_mtp_training": True},
        {"custom_tis_function_path": "custom.correction"},
        {"custom_pg_loss_reducer_function_path": "custom.reducer"},
    ],
)
def test_unsupported_loss_normalization_is_rejected(override):
    with pytest.raises(ValueError, match="sequence-filter requires"):
        validate_train_rollout_logprob_sequence_filter(_args(**override))


def test_recipe_supported_and_disabled_flag_has_no_new_requirements():
    validate_train_rollout_logprob_sequence_filter(_args())
    validate_train_rollout_logprob_sequence_filter(Namespace())
