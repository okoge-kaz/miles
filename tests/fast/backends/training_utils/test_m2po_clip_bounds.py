"""M2PO's clip range, derived from a second-moment budget (arXiv:2510.01161).

The bound is not a hyperparameter: it is the widest clip range whose *harmful*
tokens -- the ones the clip would act on -- keep mean (log pi_behav - log
pi_theta)^2 under the budget. These tests pin the three properties that make it
that rather than a static clip: it is inert when the batch is already inside the
budget, it tightens as the batch drifts, and it never counts a token the clip
would not have touched.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from miles.backends.training_utils.loss_hub.math_utils import compute_m2po_clip_bounds

MINI_LOW, MINI_HIGH = 0.3, 0.5


def _bounds(deltas, advantages, budget=0.04, mask=None):
    d = torch.tensor(deltas, dtype=torch.float32)
    a = torch.tensor(advantages, dtype=torch.float32)
    m = torch.ones_like(d) if mask is None else torch.tensor(mask, dtype=torch.float32)
    return compute_m2po_clip_bounds(d, a, m, budget, MINI_LOW, MINI_HIGH)


def test_on_policy_batch_leaves_the_floors_untouched():
    """delta = 0 everywhere: nothing is harmful, so the budget cannot bind."""
    low, high, before, after = _bounds([0.0] * 8, [1.0] * 4 + [-1.0] * 4)
    assert (low, high) == (MINI_LOW, MINI_HIGH)
    assert before == 0.0 and after == 0.0


def test_batch_inside_the_budget_is_not_clipped_below_the_floors():
    """Harmful tokens exist but their mean delta^2 is under budget."""
    # delta = -0.1 with A > 0 gives ratio = exp(0.1) > 1, so these are harmful.
    low, high, before, after = _bounds([-0.1] * 6, [1.0] * 6)
    assert before == pytest.approx(0.01)
    assert after == before, "an unbinding budget must not move the second moment"
    assert (low, high) == (MINI_LOW, MINI_HIGH)


def test_over_budget_batch_tightens_to_within_the_budget():
    """A batch above the budget comes back under it, and the bounds agree on tau.

    The fixture is chosen so the derived range is outside the floors, which is
    the only regime where the derived value is observable at all.
    """
    deltas = [-0.6] * 5 + [-3.0]  # one outlier carries the excess
    low, high, before, after = _bounds(deltas, [1.0] * 6, budget=0.4)
    assert before == pytest.approx(1.8), "the fixture must actually exceed the budget"
    assert after <= 0.4 < before

    # tau caps |delta|, so the two epsilons are exp(-tau) and exp(+tau) around 1.
    tau = math.log1p(high)
    assert tau == pytest.approx(0.6, rel=1e-5)
    assert low == pytest.approx(1.0 - math.exp(-tau), rel=1e-5)
    assert (low, high) > (MINI_LOW, MINI_HIGH), "floors must not be active here"


def test_the_floors_can_only_loosen_the_range():
    """miniclip is a floor on the epsilon, so a tiny tau widens back to it.

    This is the reference implementation's behaviour and it is worth pinning:
    with the paper's 0.3 / 0.5 the range never goes below [0.7, 1.5], which is
    already wider than the DAPO clip the rest of this study runs.
    """
    low, high, _, after = _bounds([-0.05] * 5 + [-1.2], [1.0] * 6, budget=0.04)
    assert after < 0.04
    assert (low, high) == (MINI_LOW, MINI_HIGH)


def test_a_tighter_budget_gives_a_tighter_range():
    deltas = [-0.4, -0.5, -0.6, -0.7, -0.8, -1.5]
    _, high_loose, _, _ = _bounds(deltas, [1.0] * 6, budget=0.5)
    _, high_tight, _, _ = _bounds(deltas, [1.0] * 6, budget=0.05)
    assert high_tight <= high_loose


def test_only_harmful_tokens_enter_the_budget():
    """A large delta the clip would never act on must not tighten the range.

    A > 0 with ratio < 1, and A < 0 with ratio > 1, are the two quadrants PPO's
    max() leaves alone. Budgeting them would let a token that cannot affect the
    update shrink the range for every token that can.
    """
    # A > 0 with delta > 0 => ratio = exp(-delta) < 1 => not harmful.
    harmless_only = _bounds([2.0] * 6, [1.0] * 6)
    assert harmless_only[:2] == (MINI_LOW, MINI_HIGH)
    assert harmless_only[2] == 0.0

    # A < 0 with delta > 0 => ratio < 1 => harmful, and the same magnitude bites.
    harmful = _bounds([2.0] * 6, [-1.0] * 6)
    assert harmful[2] == pytest.approx(4.0)


def test_masked_tokens_are_excluded():
    """Padding must not enter the second moment."""
    deltas = [-1.5, -1.5, -0.01, -0.01]
    both = _bounds(deltas, [1.0] * 4)
    kept = _bounds(deltas, [1.0] * 4, mask=[0.0, 0.0, 1.0, 1.0])
    assert both[2] > kept[2]
    assert kept[:2] == (MINI_LOW, MINI_HIGH), "the unmasked tail is inside the budget"
