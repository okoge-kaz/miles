"""Dynamic sampling filters selecting on group difficulty.

Mounted the same way as the built-ins, via `--dynamic-sampling-filter-path`:

    --dynamic-sampling-filter-path \
        experiments.src.difficulty_filter.filters.check_pass_rate_window

Relationship to `miles.rollout.filter_hub.dynamic_sampling_filters`:
`check_reward_nonzero_std` is the limiting case of `check_pass_rate_window` with
the window (0, 1) open — it keeps every group that is not unanimous. The window
form additionally drops the near-degenerate groups, which a std test cannot
distinguish from informative ones.

These run online and pay for the generation before deciding. When the same
prompts are visited repeatedly, measuring pass rates once offline
(`measure_pass_rate.py`) and filtering the dataset up front
(`apply_filter.py`) is strictly cheaper: on this cluster the online filter
raised mean rollout time from 253 s to 761 s, a 3x cost for the same signal.
Prefer the offline path for a fixed prompt set, and this one when the policy has
moved far enough that the offline measurement is stale.
"""

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

from experiments.src.difficulty_filter.pass_rate import (
    DEFAULT_CORRECT_THRESHOLD,
    pass_rate_from_rewards,
    pass_rate_in_window,
    resolve_pass_rate_window,
)

__all__ = ["check_pass_rate_window"]


def _flatten_samples(samples: list[Sample | list[Sample]]):
    """Flatten a group whose elements are `Sample` or `list[Sample]` (generate-function dependent).

    Same shape as the built-in helper; multi-turn generate functions return a
    list per group element, single-turn ones return a bare Sample.
    """
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def check_pass_rate_window(args, samples: list[Sample | list[Sample]], **kwargs):
    """Keep a group only if its pass rate lands inside the configured window.

    The window comes from `args.pass_rate_min` / `args.pass_rate_max` when set,
    otherwise from the module defaults (0.2, 0.8).
    """
    minimum, maximum = resolve_pass_rate_window(args)
    correct_threshold = getattr(args, "pass_rate_correct_threshold", None)
    correct_threshold = DEFAULT_CORRECT_THRESHOLD if correct_threshold is None else float(correct_threshold)

    rewards = [sample.get_reward_value(args) for sample in _flatten_samples(samples)]
    pass_rate = pass_rate_from_rewards(rewards, correct_threshold)
    keep = pass_rate_in_window(pass_rate, minimum, maximum)

    # Bucketing the reason keeps the metric cardinality bounded: the gatherer in
    # base_types.MetricGatherer emits one counter per distinct reason string, so
    # a raw float would create a new series per group.
    if keep:
        return DynamicFilterOutput(keep=True)
    reason = "pass_rate_all_wrong" if pass_rate == 0.0 else (
        "pass_rate_all_correct" if pass_rate == 1.0 else
        ("pass_rate_too_low" if pass_rate < minimum else "pass_rate_too_high")
    )
    return DynamicFilterOutput(keep=False, reason=reason)
