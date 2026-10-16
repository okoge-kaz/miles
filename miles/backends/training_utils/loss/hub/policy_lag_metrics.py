import math
from collections.abc import Mapping, Sequence

import torch

from miles.rollout.staleness_metrics import category_key

POLICY_LAG_PART_PREFIX = "_policy_lag_part/"


def collect_policy_lag_parts(
    train_log_probs: Sequence[torch.Tensor],
    rollout_log_probs: Sequence[torch.Tensor],
    local_loss_masks: Sequence[torch.Tensor],
    *,
    categories: Sequence[str],
    category_vocabulary: Sequence[str],
    truncated: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    if not len(train_log_probs) == len(rollout_log_probs) == len(local_loss_masks) == len(categories):
        raise ValueError("Policy-lag sample/logprob/mask/category counts differ")
    if len(set(category_vocabulary)) != len(category_vocabulary) or set(categories) - set(category_vocabulary):
        raise ValueError("Policy-lag categories must belong to the batch-global vocabulary")
    if truncated is not None and (len(truncated) != len(categories) or any(v not in (0, 1) for v in truncated)):
        raise ValueError("Policy-lag truncated indicators must be response-aligned 0/1 values")
    if not train_log_probs:
        raise ValueError("Policy-lag collector needs at least one sample (its local response may be empty)")
    for current, behavior, mask in zip(train_log_probs, rollout_log_probs, local_loss_masks, strict=True):
        if current.shape != behavior.shape or current.shape != mask.shape:
            raise ValueError("Policy-lag tensors must align to the actual local response tokens")
    delta = torch.cat(
        [
            current.detach().double() - behavior.detach().double()
            for current, behavior in zip(train_log_probs, rollout_log_probs, strict=True)
        ]
    )
    lengths = torch.tensor([p.numel() for p in train_log_probs], device=delta.device)
    vocabulary = {category: i for i, category in enumerate(category_vocabulary)}
    row_categories = torch.tensor([vocabulary[c] for c in categories], device=delta.device)
    token_categories = torch.repeat_interleave(row_categories, lengths, output_size=delta.numel())
    populations = {
        "all_response": torch.ones_like(delta, dtype=torch.bool),
        "loss_enabled": torch.cat(local_loss_masks).bool(),
    }
    if truncated is not None:
        rows = torch.tensor(truncated, device=delta.device, dtype=torch.bool)
        populations["truncated"] = torch.repeat_interleave(rows, lengths, output_size=delta.numel())
        populations["non_truncated"] = ~populations["truncated"]
    finite = torch.isfinite(delta)
    clean = torch.where(finite, delta, torch.zeros_like(delta))
    base = torch.stack(
        (torch.ones_like(delta), finite.double(), (~finite).double(), clean, clean.abs(), clean.square())
    )
    names = ("token_count", "valid_token_count", "nonfinite_token_count", "delta_sum", "delta_abs_sum", "delta_sq_sum")
    result = {}
    for population, mask in populations.items():
        values = torch.where(mask.unsqueeze(0), base, torch.zeros_like(base))
        grouped = delta.new_zeros((len(names), len(category_vocabulary)))
        grouped.scatter_add_(1, token_categories.expand(len(names), -1), values)
        totals = values.sum(dim=1)
        scopes = [
            ("total", totals),
            *((f"category/{category_key(c)}", grouped[:, i]) for i, c in enumerate(category_vocabulary)),
        ]
        for scope, parts in scopes:
            result.update(
                {f"{POLICY_LAG_PART_PREFIX}{scope}/{population}/{name}": parts[i] for i, name in enumerate(names)}
            )
    return result


def finalize_policy_lag_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    result = {key: value for key, value in metrics.items() if not key.startswith(POLICY_LAG_PART_PREFIX)}
    scopes = sorted({key.rsplit("/", 1)[0] for key in metrics if key.startswith(POLICY_LAG_PART_PREFIX)})
    for scope in scopes:

        def part(key, scope=scope):
            return float(metrics[f"{scope}/{key}"])

        prefix = "policy_lag/" + scope.removeprefix(POLICY_LAG_PART_PREFIX)
        count, valid, invalid = part("token_count"), part("valid_token_count"), part("nonfinite_token_count")
        result[f"{prefix}/token_count"] = count
        result[f"{prefix}/nonfinite_token_count"] = invalid
        if not count:
            continue
        result[f"{prefix}/finite_coverage"] = valid / count
        if invalid:
            result.update(
                {
                    f"{prefix}/{name}": math.nan
                    for name in ("delta_mean", "delta_abs_mean", "delta_sq_mean", "delta_std")
                }
            )
            continue
        mean = part("delta_sum") / valid
        square = part("delta_sq_sum") / valid
        result.update(
            {
                f"{prefix}/delta_mean": mean,
                f"{prefix}/delta_abs_mean": part("delta_abs_sum") / valid,
                f"{prefix}/delta_sq_mean": square,
                f"{prefix}/delta_std": math.sqrt(max(0.0, square - mean * mean)),
            }
        )
    return result
