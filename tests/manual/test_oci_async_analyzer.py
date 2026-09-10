"""Synthetic checks for the raw-dump versus async-log qualification checker."""

from copy import deepcopy

import pytest

from tests.manual.analyze_oci_async_smoke import validate_samples


def evidence():
    samples = []
    for first, length, segments in [(10, 4, [[0, 2, 10], [2, 4, 11]]), (11, 2, [[0, 2, 11]])]:
        samples.append(
            {
                "group_index": 1,
                "first_prefill_weight_versions": [first],
                "last_forward_weight_versions": [11],
                "response_length": length,
                "response_weight_version_segments": [segments],
                "metadata": {
                    "group_ready_weight_version": 11,
                    "queue_put_weight_version": 11,
                    "drain_weight_version": 11,
                    "train_weight_version": 12,
                    "sample_staleness_reference_weight_version": first,
                },
            }
        )
    metrics = {"fully_async/train_weight_version": 12}
    for phase, value in [("total", 2), ("pre_queue", 1), ("in_queue", 1)]:
        metrics.update({f"staleness/{phase}/mean": value, f"staleness/{phase}/max": value})
    metrics.update(
        {
            "staleness/sample_lag/total/sequence_mean": 1.5,
            "staleness/token_lag/exact/mean": 8 / 6,
            "staleness/token_lag/exact/num_tokens": 6,
        }
    )
    trainer = {f"sample_staleness/s_{lag}/consumed_sequence_mass": mass for lag, mass in [(0, 0), (1, 0.5), (2, 0.5)]}
    return samples, metrics, trainer


def test_exact_token_lag_is_not_scalar_sample_lag():
    samples, metrics, trainer = evidence()
    result = validate_samples(samples, metrics, trainer, 2)
    assert result["sample_lag_counts"] == {2: 1, 1: 1}
    assert result["response_tokens"] == 6


def test_inflight_resume_concatenates_call_local_token_offsets():
    samples, metrics, trainer = evidence()
    samples[0]["response_weight_version_segments"] = [[[0, 2, 10]], [[0, 2, 11]]]
    samples[0]["first_prefill_weight_versions"] = [10, 11]
    samples[0]["last_forward_weight_versions"] = [10, 11]
    result = validate_samples(samples, metrics, trainer, 2)
    assert result["generation_calls"] == 3
    assert result["response_tokens"] == 6


@pytest.mark.parametrize("problem", ["wrong_train_version", "missing_tokens", "wrong_bin", "wrong_total", "bound"])
def test_invalid_evidence_fails(problem):
    samples, metrics, trainer = deepcopy(evidence())
    bound = 2
    if problem == "wrong_train_version":
        metrics["fully_async/train_weight_version"] = 11
    elif problem == "missing_tokens":
        samples[0]["response_weight_version_segments"][0][-1][1] = 3
    elif problem == "wrong_bin":
        trainer["sample_staleness/s_1/consumed_sequence_mass"] = 1
    elif problem == "wrong_total":
        metrics["staleness/total/mean"] = 1
    elif problem == "bound":
        bound = 1
    with pytest.raises(AssertionError):
        validate_samples(samples, metrics, trainer, bound)
