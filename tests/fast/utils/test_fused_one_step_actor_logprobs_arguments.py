from types import SimpleNamespace

import pytest

from miles.utils.arguments import (
    validate_fused_one_step_actor_logprobs,
    validate_fused_one_step_actor_logprobs_runtime,
)


def _args(**overrides) -> SimpleNamespace:
    defaults = dict(
        fuse_one_step_actor_logprobs=True,
        verify_fused_one_step_actor_logprobs=False,
        train_backend="megatron",
        num_steps_per_rollout=1,
        rollout_batch_size=192,
        n_samples_per_prompt=16,
        global_batch_size=3072,
        advantage_estimator="grpo",
        kl_coef=0.0,
        use_rollout_logprobs=False,
        keep_old_actor=False,
        use_opd=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        use_indexer_replay=False,
        use_rollout_indexer_replay=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_backend": "fsdp"}, "Megatron"),
        ({"num_steps_per_rollout": 2}, "num-steps-per-rollout 1"),
        ({"advantage_estimator": "gspo"}, "advantage-estimator grpo"),
        ({"kl_coef": 0.1}, "kl-coef 0"),
        ({"use_rollout_logprobs": True}, "use-rollout-logprobs disabled"),
        ({"keep_old_actor": True}, "keep-old-actor disabled"),
        ({"use_opd": True}, "use-opd disabled"),
        ({"attention_dropout": 0.1}, "attention-dropout 0"),
        ({"hidden_dropout": 0.1}, "hidden-dropout 0"),
    ],
)
def test_static_fused_guard_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_fused_one_step_actor_logprobs(_args(**overrides))


@pytest.mark.parametrize(
    "flag",
    [
        "use_routing_replay",
        "use_rollout_routing_replay",
        "use_indexer_replay",
        "use_rollout_indexer_replay",
    ],
)
def test_static_fused_guard_rejects_replay_that_depends_on_legacy_forward(flag):
    with pytest.raises(ValueError, match="routing/indexer replay"):
        validate_fused_one_step_actor_logprobs(_args(**{flag: True}))


def test_verify_requires_fused_mode():
    with pytest.raises(ValueError, match="requires --fuse-one-step-actor-logprobs"):
        validate_fused_one_step_actor_logprobs(
            _args(
                fuse_one_step_actor_logprobs=False,
                verify_fused_one_step_actor_logprobs=True,
            )
        )


def test_valid_static_configuration_passes():
    validate_fused_one_step_actor_logprobs(_args())


def test_static_guard_accepts_one_step_batch_shape_when_option_is_unspecified():
    validate_fused_one_step_actor_logprobs(_args(num_steps_per_rollout=None))


def test_static_guard_checks_implied_step_count_when_option_is_unspecified():
    with pytest.raises(ValueError, match="one-step batch shape"):
        validate_fused_one_step_actor_logprobs(
            _args(
                num_steps_per_rollout=None,
                global_batch_size=1536,
            )
        )


def test_runtime_guard_uses_effective_step_count_not_microbatch_count():
    args = _args()
    validate_fused_one_step_actor_logprobs_runtime(args, [8])

    with pytest.raises(RuntimeError, match="exactly one optimizer step"):
        validate_fused_one_step_actor_logprobs_runtime(args, [4, 4])


def test_runtime_guard_is_noop_when_feature_is_disabled():
    validate_fused_one_step_actor_logprobs_runtime(
        _args(fuse_one_step_actor_logprobs=False),
        [4, 4],
    )
