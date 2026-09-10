from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils import log_utils


def test_true_on_policy_rollout_logprob_dtype_follows_training_precision():
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=True, fp16=False)) is torch.bfloat16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=True, bf16=False, fp16=True)) is torch.float16
    )
    assert (
        data_utils._rollout_logprob_dtype(Namespace(true_on_policy_mode=False, bf16=True, fp16=False)) is torch.float32
    )


def test_true_on_policy_log_checker_passes_when_values_and_dtype_match(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )

    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda metric_name, args, rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
        "log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
        "rollout_log_probs": [torch.tensor([-13.25, -13.5], dtype=torch.bfloat16)],
    }

    log_utils.log_rollout_data(
        1,
        Namespace(
            ci_test=True,
            ci_disable_logprobs_checker=False,
            true_on_policy_mode=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert captured["log_dict"]["log_probs"] == captured["log_dict"]["rollout_log_probs"]


def test_rollout_logger_skips_sampling_and_policy_masks_and_accepts_bool_metrics(monkeypatch):
    captured = {}
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0),
        cp=SimpleNamespace(size=1),
        is_pp_last_stage=True,
    )
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        log_utils,
        "gather_log_data",
        lambda _metric_name, _args, _rollout_id, log_dict: captured.setdefault("log_dict", log_dict),
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 0], dtype=torch.int32)],
        "rollout_sampling_mask_ids": [[1, 7]],
        "rollout_sampling_mask_offsets": [[0, 2]],
        "policy_lag_initial_loss_masks": [torch.tensor([True, True])],
        "bool_metric": [torch.tensor([True, False])],
    }

    log_utils.log_rollout_data(
        1,
        SimpleNamespace(
            ci_test=False,
            ci_disable_logprobs_checker=True,
            qkv_format="thd",
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert "rollout_sampling_mask_ids" not in captured["log_dict"]
    assert "rollout_sampling_mask_offsets" not in captured["log_dict"]
    assert "policy_lag_initial_loss_masks" not in captured["log_dict"]
    assert captured["log_dict"]["bool_metric"] == (0.5, 1)
