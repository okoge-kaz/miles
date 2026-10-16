import math
from collections.abc import Mapping

import torch

TIS_PART_PREFIX = "_tis_part/"


@torch.no_grad()
def collect_tis_parts(
    log_ratio: torch.Tensor,
    post_weights: torch.Tensor,
    loss_masks: torch.Tensor,
    *,
    clip_low: float,
    clip_high: float,
) -> dict[str, torch.Tensor]:
    """Collect token-weighted moments and extrema without changing the loss.

    The population is the original, CP-local loss-enabled tokens, including
    zero advantages. Keep raw ratios in float64 for diagnostics; post_weights
    are the actual weights used by the loss. No token arrays leave the device.
    """
    if log_ratio.shape != post_weights.shape or log_ratio.shape != loss_masks.shape:
        raise ValueError("TIS metric tensors must align to local response tokens")
    active = loss_masks.bool()
    known = active & ~torch.isnan(log_ratio)
    clipped = log_ratio > math.log(clip_high)
    if clip_low > 0:
        clipped = clipped | (log_ratio < math.log(clip_low))
    parts = {
        "token_count": active.double().sum(),
        "clip_known_count": known.double().sum(),
        "clip_count": (known & clipped).double().sum(),
    }
    for stage, values in (("pre", log_ratio.double().exp()), ("post", post_weights.double())):
        finite = active & torch.isfinite(values)
        clean = torch.where(finite, values, 0.0)
        parts.update(
            {
                f"{stage}/nonfinite_count": (active & ~finite).double().sum(),
                f"{stage}/sum": clean.sum(),
                f"{stage}/sq_sum": clean.square().sum(),
                f"{stage}/min": (
                    torch.where(finite, values, torch.inf).amin() if values.numel() else values.new_tensor(torch.inf)
                ),
                f"{stage}/max": (
                    torch.where(finite, values, -torch.inf).amax() if values.numel() else values.new_tensor(-torch.inf)
                ),
            }
        )
    return {f"{TIS_PART_PREFIX}{key}": value for key, value in parts.items()}


def finalize_tis_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    result = {key: value for key, value in metrics.items() if not key.startswith(TIS_PART_PREFIX)}
    if f"{TIS_PART_PREFIX}token_count" not in metrics:
        return result

    def part(key: str) -> float:
        return float(metrics[f"{TIS_PART_PREFIX}{key}"])

    count = part("token_count")
    result["tis/token_count"] = count
    for stage in ("pre", "post"):
        prefix = f"tis/{stage}"
        invalid = part(f"{stage}/nonfinite_count")
        result[f"{prefix}/nonfinite_token_count"] = invalid
        if not count:
            continue
        result[f"{prefix}/finite_coverage"] = (count - invalid) / count
        mean = part(f"{stage}/sum") / count
        variance = part(f"{stage}/sq_sum") / count - mean * mean
        result.update(
            {
                f"{prefix}/{name}": math.nan if invalid else value
                for name, value in (
                    ("mean", mean),
                    ("std", math.sqrt(max(variance, 0.0))),
                    ("min", part(f"{stage}/min")),
                    ("max", part(f"{stage}/max")),
                )
            }
        )
    if count:
        known = part("clip_known_count")
        result["tis/clip_fraction"] = part("clip_count") / count if known == count else math.nan
        result["tis/clip_coverage"] = known / count
    return result
