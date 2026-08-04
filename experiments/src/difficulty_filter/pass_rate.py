"""Pass rate as the difficulty coordinate, shared by the offline and online filters.

Difficulty is a property of the (prompt, policy, sampling-params) triple, not of
the prompt alone. Measured on this cluster, DAPO-Math-17K leaves Qwen3-4B
(thinking) with 21.6 of every 32 groups at zero variance and a flat 0.84 reward
over 46 steps, while the same file is a reasonable curriculum for a weaker
policy. So nothing here scores a prompt in the abstract: every number is the
pass rate some specific policy achieved under the sampling parameters the
training run will use.

Why the pass rate is the right coordinate for GRPO specifically: with binary
rewards the group advantage is proportional to the group standard deviation,
which is exactly sqrt(p * (1 - p)) for pass rate p. Selecting a pass-rate window
therefore selects an advantage-magnitude window directly, with no reference to
the reward scale.
"""

import math
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_CORRECT_THRESHOLD",
    "DEFAULT_PASS_RATE_MAX",
    "DEFAULT_PASS_RATE_MIN",
    "PassRateRecord",
    "group_std_from_pass_rate",
    "pass_rate_from_rewards",
    "pass_rate_in_window",
    "resolve_pass_rate_window",
]

# The open interval (0, 1) is what `check_reward_nonzero_std` already keeps.
# The tighter default below additionally drops the near-degenerate groups: at
# n_samples_per_prompt=8, pass rates 1/8 and 7/8 survive a zero-std test but
# carry only sqrt(0.125 * 0.875) = 0.33 of the maximum advantage magnitude while
# costing a full group of generation.
DEFAULT_PASS_RATE_MIN = 0.2
DEFAULT_PASS_RATE_MAX = 0.8

# Rule-based verifiers in miles/rollout/rm_hub return 0/1, but custom reward
# functions may return a continuous score, so "correct" is a threshold rather
# than an equality test.
DEFAULT_CORRECT_THRESHOLD = 0.5


def pass_rate_from_rewards(rewards, correct_threshold: float = DEFAULT_CORRECT_THRESHOLD) -> float:
    """Fraction of samples in a group that count as correct.

    An empty group has no evidence either way; it is reported as 0.0 and is
    rejected by any window whose lower bound is above 0.
    """
    if not rewards:
        return 0.0
    return sum(1 for r in rewards if r >= correct_threshold) / len(rewards)


def group_std_from_pass_rate(pass_rate: float) -> float:
    """Population std of a binary-reward group with this pass rate.

    The quantity GRPO normalizes the advantage by, so it says directly how much
    gradient signal a group of this difficulty can carry.
    """
    return math.sqrt(max(0.0, pass_rate * (1.0 - pass_rate)))


def pass_rate_in_window(pass_rate: float, minimum: float, maximum: float) -> bool:
    """Inclusive window test. Kept as one function so offline selection and the
    online dynamic filter cannot drift apart."""
    return minimum <= pass_rate <= maximum


def resolve_pass_rate_window(args=None) -> tuple[float, float]:
    """The (min, max) window, preferring explicit args over the defaults.

    `experiments/` cannot add entries to the miles argument parser, so the
    online filter reads the window off `args` when a caller has set the
    attributes and otherwise falls back to the module defaults. Keeping the
    lookup here means `check_pass_rate_window` has no configuration logic of its
    own.
    """
    minimum = getattr(args, "pass_rate_min", None)
    maximum = getattr(args, "pass_rate_max", None)
    minimum = DEFAULT_PASS_RATE_MIN if minimum is None else float(minimum)
    maximum = DEFAULT_PASS_RATE_MAX if maximum is None else float(maximum)
    if minimum > maximum:
        raise ValueError(f"empty pass-rate window: min={minimum} > max={maximum}")
    return minimum, maximum


@dataclass
class PassRateRecord:
    """One prompt's measurement. Serialized one per line by `measure_pass_rate`.

    `truncated_frac` is not decoration: a truncated sample scores 0 under every
    rule-based verifier, so measuring with a shorter budget than training uses
    makes prompts look harder than they are. Recording it keeps that bias
    auditable instead of silent.
    """

    index: int
    pass_rate: float
    n_correct: int
    n_samples: int
    response_len_mean: float
    truncated_frac: float
    label: str | None = None
    rewards: list[float] = field(default_factory=list)

    @property
    def group_std(self) -> float:
        return group_std_from_pass_rate(self.pass_rate)


def select_by_pass_rate(records, minimum: float, maximum: float):
    """The kept subset, in input order."""
    return [r for r in records if pass_rate_in_window(r.pass_rate, minimum, maximum)]


def summarize(records, minimum: float, maximum: float) -> dict:
    """Counts and rates a filtering run should report before it writes anything."""
    total = len(records)
    if total == 0:
        return {"total": 0}
    all_wrong = sum(1 for r in records if r.pass_rate == 0.0)
    all_correct = sum(1 for r in records if r.pass_rate == 1.0)
    kept = len(select_by_pass_rate(records, minimum, maximum))
    nonzero_std = total - all_wrong - all_correct
    return {
        "total": total,
        "all_wrong": all_wrong,
        "all_correct": all_correct,
        "zero_std_frac": (all_wrong + all_correct) / total,
        "nonzero_std": nonzero_std,
        "kept": kept,
        "kept_frac": kept / total,
        "mean_pass_rate": sum(r.pass_rate for r in records) / total,
        "mean_group_std": sum(r.group_std for r in records) / total,
        "mean_truncated_frac": sum(r.truncated_frac for r in records) / total,
    }
