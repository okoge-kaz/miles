import asyncio
from copy import copy
from dataclasses import replace


async def _score_builtin(args, sample, **kwargs):
    # Reward dependencies are needed in the training image, not CLI previews.
    from miles.rollout.rm_hub import async_rm

    builtin_args = copy(args)
    builtin_args.custom_rm_path = None
    if sample.reward_spec is not None and (sample.reward_spec.custom_rm_path or "").startswith(
        "experiments.src.training.truncation."
    ):
        sample = replace(sample, reward_spec=replace(sample.reward_spec, custom_rm_path=None))
    return await async_rm(builtin_args, sample, **kwargs)


async def _score(args, sample, *, mode: str, **kwargs):
    if isinstance(sample, list):
        return await asyncio.gather(*[_score(args, item, mode=mode, **kwargs) for item in sample])
    truncated = sample.status == sample.Status.TRUNCATED
    if truncated and mode == "zero_reward":
        return 0.0
    if truncated and mode == "zero_loss":
        sample.loss_mask = [0] * sample.response_length
    return await _score_builtin(args, sample, **kwargs)


async def zero_reward(args, sample, **kwargs):
    return await _score(args, sample, mode="zero_reward", **kwargs)


async def zero_loss(args, sample, **kwargs):
    return await _score(args, sample, mode="zero_loss", **kwargs)
