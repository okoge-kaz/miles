"""Shared routing contract for restricted reward sets."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

RewardResult = float | list[float]
RewardHandler = Callable[..., Awaitable[RewardResult]]


def _sample_verifier(sample: Any, verifier_keys: tuple[str, ...]) -> str:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    return str(next((metadata.get(key) for key in verifier_keys if metadata.get(key)), "")).strip()


def _validate_verifiers(
    samples: list[Any],
    handlers: Mapping[str, RewardHandler],
    reward_set: str,
    verifier_keys: tuple[str, ...],
) -> list[str]:
    verifiers = [_sample_verifier(sample, verifier_keys) for sample in samples]
    rejected = sorted(set(verifiers).difference(handlers))
    if rejected:
        allowed = ", ".join(sorted(handlers))
        rejected_text = ", ".join(repr(verifier or "<missing>") for verifier in rejected)
        raise ValueError(f"{reward_set} rejects verifier(s) {rejected_text}; allowed verifier(s): {allowed}")
    return verifiers


def _normalize_batch_result(result: RewardResult, expected_size: int, verifier: str) -> list[float]:
    if not isinstance(result, list):
        raise TypeError(f"reward handler for {verifier!r} returned a scalar for a batch")
    if len(result) != expected_size:
        raise ValueError(
            f"reward handler for {verifier!r} returned {len(result)} rewards for {expected_size} samples"
        )
    return [float(reward) for reward in result]


async def dispatch_restricted_reward(
    args: Any,
    sample_or_samples: Any,
    *,
    handlers: Mapping[str, RewardHandler],
    reward_set: str,
    verifier_keys: tuple[str, ...] = ("verifier",),
    **kwargs: Any,
) -> RewardResult:
    """Dispatch only explicitly allowed verifier ids, preserving batch order."""

    is_batch = isinstance(sample_or_samples, list)
    samples = sample_or_samples if is_batch else [sample_or_samples]
    if not samples:
        return []
    verifiers = _validate_verifiers(samples, handlers, reward_set, verifier_keys)
    if not is_batch:
        result = await handlers[verifiers[0]](args, samples[0], **kwargs)
        if isinstance(result, list):
            raise TypeError(f"reward handler for {verifiers[0]!r} returned a batch for a scalar sample")
        return float(result)

    grouped: dict[str, tuple[list[int], list[Any]]] = {}
    for index, (verifier, sample) in enumerate(zip(verifiers, samples, strict=True)):
        indices, verifier_samples = grouped.setdefault(verifier, ([], []))
        indices.append(index)
        verifier_samples.append(sample)

    verifier_names = list(grouped)
    results = await asyncio.gather(
        *(
            handlers[verifier](args, grouped[verifier][1], **kwargs)
            for verifier in verifier_names
        )
    )
    rewards: list[float | None] = [None] * len(samples)
    for verifier, result in zip(verifier_names, results, strict=True):
        indices, verifier_samples = grouped[verifier]
        verifier_rewards = _normalize_batch_result(result, len(verifier_samples), verifier)
        for index, reward in zip(indices, verifier_rewards, strict=True):
            rewards[index] = reward
    if any(reward is None for reward in rewards):
        raise RuntimeError(f"{reward_set} did not score every sample")
    return [float(reward) for reward in rewards]


def synchronous_handler(scorer: Callable[[Any], float]) -> RewardHandler:
    """Adapt a deterministic scalar scorer to the Miles scalar/batch contract."""

    async def reward(args: Any, sample_or_samples: Any, **kwargs: Any) -> RewardResult:
        if isinstance(sample_or_samples, list):
            return [float(scorer(sample)) for sample in sample_or_samples]
        return float(scorer(sample_or_samples))

    return reward
