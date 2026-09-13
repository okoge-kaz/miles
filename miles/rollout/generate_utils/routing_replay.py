"""Preserve routing decisions for tokens generated before an interrupted request."""

import numpy as np


def merge_routing_prefix(
    previous: np.ndarray | None,
    current: np.ndarray | None,
    *,
    prefix_tokens: int,
    prefix_response_tokens: int,
    new_tokens: int,
    num_layers: int,
    topk: int,
) -> np.ndarray | None:
    """Keep old token decisions and append only the newly generated suffix.

    SGLang returns routing for the full re-prefilled prefix. Those new decisions
    must not replace decisions paired with already persisted rollout logprobs.
    There are T-1 routing rows: the last token has not been forwarded yet, so its
    routing row belongs to the continuation, not to the saved prefix.
    """
    prefix_rows = max(0, prefix_tokens - 1)
    if prefix_response_tokens:
        _validate_routes(previous, (prefix_rows, num_layers, topk), "persisted prefix")
    if not new_tokens:
        # Aborting a queued request can return no routing payload at all.
        return previous
    _validate_routes(current, (prefix_tokens + new_tokens - 1, num_layers, topk), "generation response")
    if not prefix_response_tokens:
        return current
    return np.concatenate((previous, current[prefix_rows:]), axis=0)


def _validate_routes(routes: np.ndarray | None, shape: tuple[int, int, int], source: str) -> None:
    if routes is None or routes.shape != shape or routes.dtype != np.int32:
        actual = None if routes is None else (routes.shape, str(routes.dtype))
        raise ValueError(
            f"Rollout routing replay requires {source} routing with shape {shape} and int32, got {actual}"
        )
