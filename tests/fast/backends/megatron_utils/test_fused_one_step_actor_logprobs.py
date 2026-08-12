from types import SimpleNamespace

import pytest

from miles.utils.arguments import should_run_actor_logprob_forward


@pytest.mark.parametrize(
    ("fused", "verify", "use_rollout", "mismatch", "expected"),
    [
        (True, False, False, False, False),
        (True, True, False, False, True),
        (False, False, False, False, True),
        (False, False, True, False, False),
        (False, False, True, True, True),
    ],
)
def test_actor_logprob_preforward_condition(fused, verify, use_rollout, mismatch, expected):
    args = SimpleNamespace(
        fuse_one_step_actor_logprobs=fused,
        verify_fused_one_step_actor_logprobs=verify,
        use_rollout_logprobs=use_rollout,
        get_mismatch_metrics=mismatch,
    )

    assert should_run_actor_logprob_forward(args) is expected
