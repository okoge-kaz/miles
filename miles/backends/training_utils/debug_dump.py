from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_local_response_loss_masks
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

_POLICY_LOSS_DUMP_COUNTER = 0


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def _batch_row(batch: RolloutBatch, key: str, index: int):
    rows = batch.get(key)
    if rows is None:
        return None
    value = rows[index]
    return value.item() if hasattr(value, "item") else value


def _split_optional(tensor: torch.Tensor | None, lengths: list[int]) -> list[torch.Tensor] | None:
    return list(tensor.split(lengths)) if tensor is not None else None


def _importance_clip_indicator(
    tis_metrics: dict[str, torch.Tensor] | None,
    template: torch.Tensor | None,
) -> torch.Tensor | None:
    if template is None:
        return None
    indicator = torch.zeros_like(template, dtype=torch.bool)
    for name, value in (tis_metrics or {}).items():
        if (
            any(token in name for token in ("clip", "truncate"))
            and isinstance(value, torch.Tensor)
            and value.shape == template.shape
        ):
            indicator |= value.detach() != 0
    return indicator.float()


def _add_sample_identity(sample: dict, batch: RolloutBatch, index: int) -> None:
    output_names = {
        "sample_indices": "sample_index",
        "sample_group_indices": "sample_group_index",
        "generation_attempt_numbers": "generation_attempt_number",
        "training_steps": "training_step",
        "optimizer_step_ids": "optimizer_step_id",
        "sample_staleness": "sample_staleness",
    }
    for field, output_name in output_names.items():
        value = _batch_row(batch, field, index)
        if value is not None:
            sample[output_name] = value
    group_index = sample.get("sample_group_index")
    attempt = sample.get("generation_attempt_number")
    if group_index is not None and attempt is not None:
        sample["generation_attempt_id"] = f"{group_index}:{attempt}"


def _add_final_sample_diagnostics(
    sample: dict,
    *,
    pre_mask: torch.Tensor,
    final_mask: torch.Tensor,
    ppo_clip: torch.Tensor,
    importance_clip: torch.Tensor,
    policy_log_ratio: torch.Tensor,
    final_pg_loss: torch.Tensor,
    objective_denominator: float,
) -> None:
    pre_mask = _to_cpu(pre_mask)
    final_mask = _to_cpu(final_mask)
    ppo_clip = _to_cpu(ppo_clip)
    importance_clip = _to_cpu(importance_clip)
    policy_log_ratio = _to_cpu(policy_log_ratio)
    final_pg_loss = _to_cpu(final_pg_loss)
    pre_tokens = float(pre_mask.sum())
    response_tokens = pre_mask.numel()
    sample.update(
        {
            "final_local_loss_mask": final_mask,
            "ppo_clip_indicator": ppo_clip,
            "importance_clip_indicator": importance_clip,
            "policy_rollout_log_ratio": policy_log_ratio,
            "final_pg_loss": final_pg_loss,
            "response_token_count_local": float(response_tokens),
            "pre_loss_token_count_local": pre_tokens,
            "final_loss_token_count_local": float(final_mask.sum()),
            "ppo_clip_count_local": float((ppo_clip * pre_mask).sum()),
            "importance_clip_count_local": float((importance_clip * pre_mask).sum()),
            "absolute_pg_contribution_local": float((final_pg_loss.abs() * final_mask).sum())
            / max(objective_denominator, 1.0),
            "ppo_clip_fraction_local": (float((ppo_clip * pre_mask).sum()) / pre_tokens if pre_tokens > 0 else 0.0),
            "mask_fraction_local": (1.0 - float(final_mask.sum()) / response_tokens if response_tokens > 0 else 0.0),
            "sequence_policy_rollout_log_ratio_local": float((policy_log_ratio * final_mask).sum()),
        }
    )


def maybe_dump_policy_loss_debug(
    *,
    args: Namespace,
    batch: RolloutBatch,
    train_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor] | None,
    advantages: list[torch.Tensor],
    local_loss_masks: list[torch.Tensor],
    ppo_kl: torch.Tensor,
    pg_loss: torch.Tensor,
    final_pg_loss: torch.Tensor | None = None,
    final_loss_masks: list[torch.Tensor] | None = None,
    ppo_clipfrac: torch.Tensor | None = None,
    policy_log_ratio: torch.Tensor | None = None,
    tis_metrics: dict[str, torch.Tensor] | None = None,
) -> None:
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir is None or not getattr(args, "dump_policy_loss_debug", True):
        return

    global _POLICY_LOSS_DUMP_COUNTER
    counter = _POLICY_LOSS_DUMP_COUNTER
    _POLICY_LOSS_DUMP_COUNTER += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    path = Path(dump_dir) / "policy_loss_debug" / f"rank_{rank}_call_{counter}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    lengths = [tensor.numel() for tensor in local_loss_masks]
    final_local_masks = (
        get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            final_loss_masks,
            args.qkv_format,
            batch.get("max_seq_lens", None),
        )
        if final_loss_masks is not None
        else local_loss_masks
    )
    ppo_clip_rows = _split_optional(ppo_clipfrac, lengths)
    policy_ratio_rows = _split_optional(policy_log_ratio, lengths)
    final_pg_rows = _split_optional(final_pg_loss, lengths)
    importance_clip_rows = _split_optional(
        _importance_clip_indicator(tis_metrics, final_pg_loss),
        lengths,
    )
    samples = []
    for index, train_lp in enumerate(train_log_probs):
        sample = {
            "index": index,
            "total_length": batch["total_lengths"][index],
            "response_length": batch["response_lengths"][index],
            "train_log_probs": _to_cpu(train_lp),
            "old_log_probs": _to_cpu(old_log_probs[index]),
            "advantages": _to_cpu(advantages[index]),
            "local_loss_mask": _to_cpu(local_loss_masks[index]),
        }
        _add_sample_identity(sample, batch, index)
        if rollout_log_probs is not None:
            sample["rollout_log_probs"] = _to_cpu(rollout_log_probs[index])
            if train_lp.shape == rollout_log_probs[index].shape:
                sample["train_rollout_abs_diff"] = _to_cpu((train_lp - rollout_log_probs[index]).abs())
        if all(rows is not None for rows in (ppo_clip_rows, policy_ratio_rows, final_pg_rows, importance_clip_rows)):
            _add_final_sample_diagnostics(
                sample,
                pre_mask=local_loss_masks[index],
                final_mask=final_local_masks[index],
                ppo_clip=ppo_clip_rows[index],
                importance_clip=importance_clip_rows[index],
                policy_log_ratio=policy_ratio_rows[index],
                final_pg_loss=final_pg_rows[index],
                objective_denominator=(
                    1.0
                    if getattr(args, "calculate_per_token_loss", False)
                    else float(torch.as_tensor(final_loss_masks[index]).sum())
                ),
            )
        samples.append(sample)

    torch.save(
        {
            "rank": rank,
            "parallel": {
                "tp_rank": get_parallel_state().tp.rank,
                "cp_rank": get_parallel_state().cp.rank,
                "pp_rank": get_parallel_state().pp.rank,
                "effective_dp_rank": get_parallel_state().effective_dp.rank,
            },
            "call": counter,
            "samples": samples,
            "ppo_kl": _to_cpu(ppo_kl),
            "pg_loss": _to_cpu(pg_loss),
            "finite": {
                "ppo_kl": torch.isfinite(ppo_kl).all().item(),
                "pg_loss": torch.isfinite(pg_loss).all().item(),
                "train_log_probs": all(torch.isfinite(t).all().item() for t in train_log_probs),
                "old_log_probs": all(torch.isfinite(t).all().item() for t in old_log_probs),
                "advantages": all(torch.isfinite(t).all().item() for t in advantages),
                "final_pg_loss": final_pg_loss is None or torch.isfinite(final_pg_loss).all().item(),
            },
        },
        path,
    )
