import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.training_utils.loss.hub import losses, score_centering_loss
from miles.backends.training_utils.loss.hub.tis_metrics import collect_tis_parts, finalize_tis_metrics
from miles.backends.training_utils.metrics import log_utils
from miles.utils.async_buffer_arguments import validate_tis_metrics_arguments


def _parts(ratios, masks=None, low=0.0):
    ratio = torch.tensor(ratios, dtype=torch.float64, requires_grad=True)
    mask = torch.ones_like(ratio) if masks is None else torch.tensor(masks)
    return collect_tis_parts(ratio.log(), ratio.clamp(low, 2), mask, clip_low=low, clip_high=2)


def _final(parts):
    return finalize_tis_metrics({key: float(value) for key, value in parts.items()})


def _assert_stats(metrics, pre, post, clipped):
    for stage, values in (("pre", pre), ("post", post)):
        values = torch.tensor(values, dtype=torch.float64)
        for stat, expected in (
            ("mean", values.mean()),
            ("std", values.std(correction=0)),
            ("min", values.min()),
            ("max", values.max()),
        ):
            assert metrics[f"tis/{stage}/{stat}"] == pytest.approx(float(expected), abs=1e-7)
    assert metrics["tis/token_count"] == len(pre)
    assert metrics["tis/clip_fraction"] == pytest.approx(clipped)


def test_pre_post_moments_extrema_masking_and_lower_bound():
    parts = _parts([0.25, 1, 2, 4, 8, math.nan], [1, 0, 1, 1, 1, 0], low=0.5)
    assert all(not value.requires_grad for value in parts.values())
    _assert_stats(_final(parts), [0.25, 2, 4, 8], [0.5, 2, 2, 2], 3 / 4)


def test_float32_clip_boundary_is_not_counted_as_clipped():
    ratio = torch.tensor([2.0]).log()
    metrics = _final(collect_tis_parts(ratio, ratio.exp(), torch.ones(1), clip_low=0, clip_high=2))
    assert metrics["tis/clip_fraction"] == 0


@pytest.mark.parametrize("empty", [True, False])
def test_empty_local_populations_have_mergeable_extrema_and_no_reported_mean(empty):
    parts = _parts([] if empty else [math.nan], [] if empty else [0])
    metrics = _final(parts)
    assert metrics["tis/token_count"] == 0
    assert "tis/pre/mean" not in metrics and "tis/post/max" not in metrics
    assert parts["_tis_part/pre/min"] == math.inf
    assert parts["_tis_part/post/max"] == -math.inf


def test_nonfinite_raw_ratio_does_not_hide_finite_post_weights():
    log_ratio = torch.tensor([1000.0, 0.0])
    metrics = _final(collect_tis_parts(log_ratio, torch.tensor([2.0, 1.0]), torch.ones(2), clip_low=0, clip_high=2))
    assert metrics["tis/pre/nonfinite_token_count"] == 1
    assert metrics["tis/pre/finite_coverage"] == 0.5
    assert math.isnan(metrics["tis/pre/mean"])
    assert metrics["tis/post/mean"] == 1.5
    assert metrics["tis/post/std"] == 0.5
    assert metrics["tis/clip_fraction"] == 0.5


def test_unknown_ratio_reports_clip_coverage():
    metrics = _final(_parts([math.nan, 1]))
    assert metrics["tis/clip_coverage"] == 0.5
    assert math.isnan(metrics["tis/clip_fraction"])


@pytest.mark.parametrize("num_rollouts", [None, 3])
def test_microbatch_and_distributed_reduction_uses_global_extrema_and_counts(monkeypatch, num_rollouts):
    local = [_parts([0.25, 1]), _parts([4]), _parts([])]
    remote = _parts([0.5, 2, 8, 16])
    keys = ["loss", *local[0]]
    microbatches = [
        {"keys": keys, "values": torch.tensor([1, 2, *parts.values()], dtype=torch.float64)} for parts in local
    ]
    remote_values = torch.tensor([1, 3, *remote.values()], dtype=torch.float64)
    indices = [i + 1 for i, key in enumerate(keys) if key.startswith("_tis_part/") and key.endswith(("/min", "/max"))]
    signs = torch.tensor([-1 if keys[i - 1].endswith("/min") else 1 for i in indices])
    remote_extrema = remote_values[indices] * signs
    remote_values[indices] = 0
    state = SimpleNamespace(cp=SimpleNamespace(size=2), effective_dp_cp=SimpleNamespace(groups_inner_to_outer=[0]))
    monkeypatch.setattr(log_utils, "get_parallel_state", lambda: state)
    operations = []

    def reduce(values, groups, op):
        operations.append(op)
        if op == dist.ReduceOp.SUM:
            values.add_(remote_values)
        else:
            assert op == dist.ReduceOp.MAX and values.numel() == 4
            values.copy_(torch.maximum(values, remote_extrema))

    monkeypatch.setattr(log_utils.MultiPGUtil, "all_reduce", reduce)
    reduced = log_utils.aggregate_train_losses(microbatches, num_rollouts=num_rollouts)
    output = log_utils.log_train_step(SimpleNamespace(trainer_model_id=None), reduced, 0.0, 0, 0, 1, should_log=False)
    metrics = {key.removeprefix("train/"): value for key, value in output.items()}
    _assert_stats(metrics, [0.25, 1, 4, 0.5, 2, 8, 16], [0.25, 1, 2, 0.5, 2, 2, 2], 3 / 7)
    assert operations == [dist.ReduceOp.SUM, dist.ReduceOp.MAX]
    assert not any("_tis_part" in key for key in output)


@pytest.mark.parametrize("variant", ["tis", "tis_sc"])
def test_both_loss_paths_log_the_same_population_without_changing_gradients(monkeypatch, variant):
    make_parallel_state()
    ratios = torch.tensor([0.25, 2, 4, 8])
    train = torch.full((4,), -3.0)
    batch = {
        "total_lengths": [5],
        "response_lengths": [4],
        "unconcat_tokens": [torch.tensor([0, 1, 1, 1, 1])],
        "loss_masks": [torch.tensor([1, 1, 0, 1])],
        "advantages": [torch.tensor([1.0, 0, 1, -1])],
        "rollout_log_probs": [train - ratios.log()],
        "rollout_topk_token_ids": [torch.tensor([[1]] * 4)],
        "rollout_topk_log_probs": [(train - ratios.log()).unsqueeze(-1)],
    }
    if variant == "tis":
        args = make_args(
            skip_actor_forward_only=True,
            use_tis=True,
            tis_clip=2.0,
            tis_clip_low=0.0,
            entropy_coef=0,
            observe_training_entropy=False,
        )
        monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda logits, **kw: {"log_probs": [logits]})
        function = losses.policy_loss_function
    else:
        args = _args(score_centering_is="tis")
        monkeypatch.setattr(
            score_centering_loss,
            "_candidate_log_probs",
            lambda args, batch, logits: {"selected": [logits[:, None].expand(-1, 2)]},
        )
        function = score_centering_loss.score_centering_loss_function
    mask = batch["loss_masks"][0].bool()

    def reducer(values):
        return values[mask].mean()

    results = []
    for enabled in (False, True):
        args.log_tis_metrics = enabled
        logits = train.clone().requires_grad_()
        loss, metrics = function(args, batch, logits, reducer)
        results.append((loss.detach(), torch.autograd.grad(loss, logits)[0], metrics))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert not any(key.startswith("_tis_part/") for key in results[0][2])
    _assert_stats(_final(results[1][2]), [0.25, 2, 8], [0.25, 2, 2], 1 / 3)


@pytest.mark.parametrize("loss_type,variant", [("policy_loss", "tis"), ("score_centering", "tis")])
def test_tis_logging_validation_accepts_builtin_losses(loss_type, variant):
    validate_tis_metrics_arguments(
        SimpleNamespace(
            log_tis_metrics=True,
            loss_type=loss_type,
            use_tis=True,
            score_centering_is=variant,
            tis_clip=2,
            tis_clip_low=0,
            score_centering_tis_clip=2,
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_tis": False},
        {"custom_tis_function_path": "custom:correction"},
        {"loss_type": "score_centering", "score_centering_is": "mis"},
        {"tis_clip": 0},
        {"tis_clip": math.inf},
        {"tis_clip_low": -1},
    ],
)
def test_tis_logging_validation_rejects_inapplicable_or_invalid_settings(overrides):
    values = dict(log_tis_metrics=True, loss_type="policy_loss", use_tis=True, tis_clip=2, tis_clip_low=0)
    with pytest.raises(ValueError, match="TIS|tis"):
        validate_tis_metrics_arguments(SimpleNamespace(**(values | overrides)))
