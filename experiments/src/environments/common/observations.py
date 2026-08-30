"""Append externally produced observations to a generated trajectory."""

from __future__ import annotations

from typing import Any

from miles.rollout.generate_utils.tool_call_utils import tokenize_tool_responses
from miles.utils.types import Sample


def append_loss_masked_tokens(
    sample: Sample,
    tokenizer: Any,
    token_ids: list[int],
    max_response_len: int,
) -> bool:
    """Append an observation without assigning policy loss to its tokens."""

    remaining = max_response_len - sample.response_length
    if len(token_ids) > remaining:
        sample.status = Sample.Status.TRUNCATED
        return False
    sample.tokens.extend(token_ids)
    sample.response += tokenizer.decode(token_ids)
    sample.response_length += len(token_ids)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask.extend([0] * len(token_ids))
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs.extend([0.0] * len(token_ids))
    return True


def append_tool_observation(
    sample: Sample,
    tokenizer: Any,
    tool_message: dict[str, Any],
    max_response_len: int,
) -> bool:
    """Append one loss-masked tool result without exceeding the rollout budget."""

    token_ids = tokenize_tool_responses([tool_message], tokenizer=tokenizer)
    return append_loss_masked_tokens(sample, tokenizer, token_ids, max_response_len)
