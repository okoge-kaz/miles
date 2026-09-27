from typing_extensions import deprecated

from miles.rollout.filter_hub.common_filters import apply_aborted_filter, apply_reward_nonzero_std_filter
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_no_aborted", "check_no_aborted_and_reward_nonzero_std"]


@deprecated("Use miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter", category=None)
def check_reward_nonzero_std(args, samples: list[Sample | list[Sample]], **kwargs):
    return apply_reward_nonzero_std_filter(args, samples, **kwargs)


@deprecated("Use miles.rollout.filter_hub.common_filters.apply_aborted_filter", category=None)
def check_no_aborted(args, samples: list[Sample | list[Sample]], **kwargs):
    return apply_aborted_filter(args, samples, **kwargs)


def check_no_aborted_and_reward_nonzero_std(args, samples: list[Sample | list[Sample]], **kwargs):
    """Reject infrastructure failures before inspecting numeric rewards."""
    status_result = check_no_aborted(args, samples, **kwargs)
    if not status_result.keep:
        return status_result
    return check_reward_nonzero_std(args, samples, **kwargs)
