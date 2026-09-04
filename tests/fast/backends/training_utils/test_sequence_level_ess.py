"""The sequence-level ESS, and what distinguishes it from the token-level one.

`train/rollout_token_level_ess` takes Kish's ratio over the tokens *within* a
sequence; `train/rollout_sequence_level_ess` takes it over the sequences in the
batch, which is the population VCPO (arXiv:2602.17616 eq. 4) uses. The two do
not measure the same thing, and the case that separates them is the one this
study is in: a per-token mismatch that is uniform along the sequence.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from miles.backends.training_utils.loss_hub.math_utils import (
    compute_ess_ratio_contribution,
    compute_sequence_level_ess_parts,
)
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


@pytest.fixture(autouse=True)
def _single_rank():
    """Both helpers read the global ParallelState to size the CP group.

    Nothing here is distributed -- CP size 1 is the whole point, since these tests
    are about the arithmetic, not the reduction.
    """
    trivial = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial,
            intra_dp_cp=trivial,
            cp=trivial,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
        )
    )


def _parts(log_ratio_per_sample: list[list[float]]) -> tuple[float, float, float]:
    lengths = [len(s) for s in log_ratio_per_sample]
    flat = torch.tensor([v for s in log_ratio_per_sample for v in s], dtype=torch.float32)
    masks = [torch.ones(n, dtype=torch.float32) for n in lengths]
    sum_w, sum_w2, n_seq = compute_sequence_level_ess_parts(
        log_ratio=flat,
        loss_masks=masks,
        total_lengths=lengths,
        response_lengths=lengths,
        qkv_format="thd",
        max_seq_lens=None,
    )
    return float(sum_w), float(sum_w2), float(n_seq)


def _rho(log_ratio_per_sample: list[list[float]]) -> float:
    a, b, c = _parts(log_ratio_per_sample)
    return (a * a) / (b * c)


def test_equal_sequence_weights_give_rho_one():
    """B identical sequences are B effective samples, whatever the common value."""
    assert _rho([[0.0] * 4, [0.0] * 4, [0.0] * 4]) == pytest.approx(1.0)
    # Shift invariance: rho depends on the spread of log w, never on its level.
    # This is why clamping log w is not a safe way to bound exp -- a clamp that
    # saturates erases the spread and reports a collapsed batch as healthy.
    assert _rho([[-5.0] * 4, [-5.0] * 4, [-5.0] * 4]) == pytest.approx(1.0)


def test_one_dominant_sequence_collapses_rho_towards_one_over_b():
    """When a single sequence carries the mass, ESS is 1 and rho is 1/B."""
    rho = _rho([[20.0], [0.0], [0.0], [0.0]])
    assert rho == pytest.approx(0.25, abs=0.01)


def test_uniform_per_token_mismatch_separates_the_two_populations():
    """The measured regime: a constant per-token offset, sequences of unequal length.

    Token-level ESS cannot see it -- inside a sequence every weight is the same,
    so Kish's ratio is 1. Sequence-level ESS does, because the sequence weight is
    the *sum* of the offset over the sequence and length varies by two orders of
    magnitude. Values are the ones measured on the running arms: a per-token
    log-ratio of -5.4e-04 against a 347..32677-token length distribution.
    """
    offset = -5.4341e-04
    lengths = [347, 6386, 10814, 32630, 32677]
    per_sample = [[offset] * n for n in lengths]

    sequence_level = _rho(per_sample)

    lengths_t = lengths
    flat = torch.tensor([v for s in per_sample for v in s], dtype=torch.float32)
    masks = [torch.ones(n, dtype=torch.float32) for n in lengths_t]
    token_level = float(
        compute_ess_ratio_contribution(
            ppo_kl=-flat,  # the helper takes ppo_kl = -log_ratio
            loss_masks=masks,
            total_lengths=lengths_t,
            response_lengths=lengths_t,
            qkv_format="thd",
            max_seq_lens=None,
            calculate_per_token_loss=False,
        )
    ) / len(lengths_t)

    assert token_level == pytest.approx(1.0, abs=1e-6)
    assert sequence_level == pytest.approx(0.216, abs=0.01)
    assert sequence_level < 0.3, "sequence-level ESS must react where token-level cannot"


def test_long_sequences_do_not_underflow():
    """float64 keeps a 32k sequence representable; float32 exp would not.

    At |log w| = 88 a float32 exp is already inf/0. The measured range at 32k is
    17.8, but a mismatch an order of magnitude larger is not exotic, and the
    failure would be silent -- inf/inf is nan, and 0/0 is nan.
    """
    rho = _rho([[-1.0] * 200, [-0.5] * 200, [0.0] * 200])  # |log w| up to 200
    assert math.isfinite(rho)
    assert 0.0 < rho <= 1.0
