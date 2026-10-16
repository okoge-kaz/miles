import asyncio
import hashlib
import logging
import math
import random

import aiohttp

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward as get_deepscaler_rule_based_reward
from miles.rollout.rm_hub.deepscaler import get_gemma_math_reward as get_gemma_math_reward
from miles.rollout.rm_hub.f1 import f1_score as f1_score
from miles.rollout.rm_hub.gpqa import compute_gpqa_reward as compute_gpqa_reward
from miles.rollout.rm_hub.math_dapo_utils import (  # noqa: F401 - compatibility export
    compute_score as compute_score_dapo,
)
from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from miles.rollout.rm_hub.math_utils import grade_answer_verl as grade_answer_verl
from miles.rollout.rm_hub.worker import BUILTIN_REWARD_TYPES, score_reward
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    max_retries = getattr(args, "rm_max_retries", 1)
    request_timeout = getattr(args, "rm_request_timeout", 30.0)
    if max_retries < 0 or not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("Remote reward requires nonnegative retries and a positive finite timeout")
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(max_retries + 1):
            http_status = None
            try:
                async with session.post(args.rm_url, json=payload) as resp:
                    http_status = resp.status
                    resp.raise_for_status()
                    result = await resp.json()
                if isinstance(sample.metadata, dict):
                    sample.metadata.pop("remote_reward_error", None)
                return result
            except (aiohttp.ClientError, TimeoutError, ValueError) as error:
                if attempt == max_retries:
                    details = {"attempts": attempt + 1, "http_status": http_status, "error": type(error).__name__}
                    sample.metadata = {**(sample.metadata or {}), "remote_reward_error": details}
                    logger.warning(
                        "Remote reward sample excluded: group=%s sample=%s attempts=%s http_status=%s error=%s",
                        sample.group_index,
                        sample.index,
                        attempt + 1,
                        http_status,
                        type(error).__name__,
                    )
                    return None
            await asyncio.sleep(min(0.25 * 2**attempt, 2))


def _resolve_reward_config(args, sample: Sample) -> tuple[str | None, str]:
    # Spec fields win when set; unset/empty fields fall back to sample metadata and process-wide args.
    spec = sample.reward_spec
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    custom_rm_path = (spec.custom_rm_path if spec is not None else None) or getattr(args, "custom_rm_path", None)
    rm_type = (
        (spec.rm_type if spec is not None else None) or metadata.get("rm_type") or getattr(args, "rm_type", None) or ""
    ).strip()
    return custom_rm_path, rm_type


async def async_rm(args, sample: Sample, **kwargs):
    custom_rm_path, rm_type = _resolve_reward_config(args, sample)

    if custom_rm_path is not None:
        rm_function = load_function(custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    response = sample.response
    label = sample.label
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type in BUILTIN_REWARD_TYPES:
        return await score_reward(
            rm_type, response, label, metadata=metadata if rm_type in {"gpqa", "ifbench"} else None
        )
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type == "deterministic_random":
        content = str(sample.tokens) + response
        content_hash = hashlib.sha256(content.encode()).digest()
        return int(content_hash[0]) % 2
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    inplace_set_reward_field: bool = False,
    **kwargs,
) -> list[int | float] | None:
    if inplace_set_reward_field:
        rewards = await batched_async_rm(args, samples, **kwargs)
        for sample, reward in zip(samples, rewards, strict=True):
            assert (
                sample.reward is None
            ), f"Overriding sample.reward from {sample.reward} to {reward}, is this intended?"
            sample.reward = reward
        return None

    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
