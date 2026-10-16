import pytest

from miles.rollout.staleness_metrics import batch_category_staleness_metrics, sample_category
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def _sample(group, category, versions, mask=None):
    return Sample(
        group_index=group,
        metadata={"category": category},
        tokens=[1, 2, 3, 4],
        response_length=3,
        loss_mask=mask,
        weight_versions=[WeightVersionsPerCall([WeightVersionSpan(str(v), a, b) for v, a, b in versions])],
    )


def test_group_weighting_category_fractions_and_measured_token_masks():
    samples = [
        _sample(1, "math", [(2, 1, 3), (4, 3, 4)], [1, 0, 0]),
        _sample(1, "math", [(3, 1, 4)], [1, 1, 1]),
        _sample(2, "code", [(4, 1, 4)], [0, 0, 0]),
    ]
    metrics = batch_category_staleness_metrics(samples, current_version=5)
    assert metrics["rollout/category/math/sample_fraction"] == pytest.approx(2 / 3)
    assert metrics["rollout/category/math/group_fraction"] == 0.5
    assert metrics["rollout/staleness/total/oldest/mean"] == 2
    assert metrics["rollout/staleness/total/oldest/count_3"] == 1
    assert metrics["rollout/staleness/total/loss_enabled_token_weighted_lag"] == pytest.approx(2.25)
    assert metrics["rollout/staleness/total/loss_enabled_versioned_tokens"] == 4
    assert metrics["rollout/staleness/total/first-prefill/coverage"] == 0


def test_unknown_versions_are_missing_not_zero_and_category_keys_do_not_collide():
    a = _sample(0, "math/easy", [])
    b = _sample(1, "math%2Feasy", [])
    metrics = batch_category_staleness_metrics([a, b], current_version=5)
    assert metrics["rollout/staleness/total/version_coverage"] == 0
    assert "rollout/staleness/total/oldest/mean" not in metrics
    assert "rollout/category/math%2Feasy/sample_fraction" in metrics
    assert "rollout/category/math%252Feasy/sample_fraction" in metrics


def test_invalid_future_and_mixed_sibling_category_rejected():
    with pytest.raises(ValueError, match="newer"):
        batch_category_staleness_metrics([_sample(1, "math", [(6, 1, 4)])], current_version=5)
    with pytest.raises(ValueError, match="share a category"):
        batch_category_staleness_metrics([_sample(1, "math", []), _sample(1, "code", [])], current_version=5)
    with pytest.raises(ValueError, match="string"):
        sample_category(Sample(metadata={"category": 42}))


def test_historical_prompt_spans_are_excluded_from_response_and_loss_populations():
    sample = _sample(0, "math", [(0, 0, 1), (4, 1, 4)], [1, 0, 1])
    metrics = batch_category_staleness_metrics([sample], current_version=5)
    prefix = "rollout/staleness/total"
    assert metrics[f"{prefix}/oldest/mean"] == 1
    assert metrics[f"{prefix}/generated_token_weighted_lag"] == 1
    assert metrics[f"{prefix}/generated_token_version_coverage"] == 1
    assert metrics[f"{prefix}/loss_enabled_versioned_tokens"] == 2


def test_partial_and_multimodal_provenance_publish_measured_coverage():
    a = _sample(0, "math", [(4, 1, 2)])
    b = _sample(1, "math", [(4, 1, 4)])
    b.multimodal_train_inputs = {"pixel_values": [0]}
    metrics = batch_category_staleness_metrics([a, b], current_version=5)
    assert metrics["rollout/staleness/total/generated_token_version_coverage"] == pytest.approx(1 / 6)
    assert metrics["rollout/staleness/total/loss_enabled_token_version_coverage"] == pytest.approx(1 / 6)


def test_missing_span_version_contributes_unknown_coverage_without_an_invented_lag():
    sample = _sample(0, "math", [])
    sample.weight_versions = [WeightVersionsPerCall([WeightVersionSpan(None, 1, 4)])]
    metrics = batch_category_staleness_metrics([sample], current_version=5)
    assert metrics["rollout/staleness/total/generated_token_version_coverage"] == 0
    assert "rollout/staleness/total/oldest/mean" not in metrics


def test_category_balance_separates_group_sample_and_length_populations():
    samples = [
        Sample(group_index=0, metadata={"category": "math"}, response_length=2),
        Sample(group_index=0, metadata={"category": "math"}, response_length=4),
        Sample(group_index=1, metadata={"category": "code"}, response_length=24, status=Sample.Status.TRUNCATED),
    ]
    metrics = batch_category_staleness_metrics(
        samples, current_version=None, target_group_fractions={"math": 0.25, "code": 0.25, "mcqa": 0.5}
    )
    assert metrics["rollout/category/math/sample_fraction"] == pytest.approx(2 / 3)
    assert metrics["rollout/category/math/group_fraction"] == 0.5
    assert metrics["rollout/category/math/response_token_fraction"] == 0.2
    assert metrics["rollout/category/math/response_length/mean"] == 3
    assert metrics["rollout/category/math/response_length/p90"] == pytest.approx(3.8)
    assert metrics["rollout/category/math/response_length/std"] == 1
    assert metrics["rollout/category/code/truncated_fraction"] == 1
    assert metrics["rollout/category/mcqa/sample_count"] == 0
    assert metrics["rollout/category/mcqa/response_token_fraction"] == 0
    assert metrics["rollout/category/mcqa/response_length/count"] == 0
    assert "rollout/category/mcqa/response_length/mean" not in metrics
    assert metrics["rollout/category/mcqa/group_fraction_delta"] == -0.5
    assert metrics["rollout/category/balance/total_variation"] == 0.5
    assert metrics["rollout/staleness/current_version_known"] == 0


def test_no_category_target_does_not_invent_balance_and_invalid_targets_fail():
    sample = Sample(group_index=0, metadata={"category": "math"}, response_length=0)
    metrics = batch_category_staleness_metrics([sample], current_version=None)
    assert not any("target_group_fraction" in key or "/balance/" in key for key in metrics)
    assert "rollout/category/math/response_token_fraction" not in metrics
    for targets in ({"math": 0.5}, {"math": float("nan")}, {"math": -1, "code": 2}):
        with pytest.raises(ValueError, match="target fractions"):
            batch_category_staleness_metrics([sample], current_version=None, target_group_fractions=targets)
