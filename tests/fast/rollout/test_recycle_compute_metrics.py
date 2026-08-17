from tests.ci.ci_register import register_cpu_ci

from miles.rollout.fully_async_telemetry import FullyAsyncPipelineTelemetry
from miles.rollout.recycle_compute_metrics import (
    ADMITTED_TOKENS_KEY,
    DISCARD_REASONS,
    DRAIN_TIME_KEY,
    DRAIN_VERSION_KEY,
    GENERATED_TOKENS_KEY,
    GROUP_GENERATION_COMPLETE_TIME_KEY,
    GROUP_GENERATION_COMPLETE_VERSION_KEY,
    GROUP_READY_TIME_KEY,
    GROUP_READY_VERSION_KEY,
    LIFECYCLE_EXACT_KEY,
    QUEUE_PUT_TIME_KEY,
    REWARD_SECONDS_KEY,
    SELECTION_METRICS_KEY,
    SAMPLE_GENERATION_COMPLETE_TIME_KEY,
    SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
    SAMPLE_REFERENCE_VERSION_KEY,
    STALE_AT_GENERATION_COMPLETION,
    STALE_DURING_QUEUE_BACKPRESSURE,
    STALE_DURING_REWARD_FINALIZE,
    STALE_IN_OUTPUT_QUEUE,
    TRAJECTORY_START_TIME_KEY,
    TRAJECTORY_START_VERSION_KEY,
    add_selection_population,
    append_final_consumed_records,
    batch_consumption_metrics,
    build_batch_consumption_snapshot,
    classify_stale_recycle_stage,
    discard_waste_metrics,
    finalize_useful_rollout_metrics,
    prequeue_phase_metrics,
    sample_lag_metrics,
    selection_population_metrics,
    straggler_collateral_indices,
    waste_vector,
)
from miles.utils.types import Sample

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def _sample(index: int, response_length: int, loss_mask: list[int]) -> Sample:
    return Sample(
        group_index=1,
        index=index,
        response_length=response_length,
        loss_mask=loss_mask,
        reward=float(index),
        metadata={},
    )


def test_useful_rollout_accounting_is_an_exact_partition() -> None:
    samples = [_sample(1, 6, [1, 1, 1, 1, 1, 0]), _sample(2, 4, [1, 1, 1, 0])]
    metrics = {
        GENERATED_TOKENS_KEY: 20,
        ADMITTED_TOKENS_KEY: 12,
        "rollout/fully_async/aborted_tokens": 1,
        "rollout/fully_async/stale_tokens": 1,
        "rollout/fully_async/age_cutoff_tokens": 2,
        "rollout/fully_async/queue_evicted_tokens": 3,
        "rollout/fully_async/dynamic_filter_tokens": 1,
    }

    finalize_useful_rollout_metrics(samples, metrics, has_custom_converter=False)

    assert metrics["rollout/fully_async/useful_rollout/loss_input_tokens"] == 8
    assert metrics["rollout/fully_async/useful_rollout/efficiency"] == 0.4
    assert metrics["rollout/fully_async/useful_rollout/postprocess_trimmed_tokens"] == 2
    assert metrics["rollout/fully_async/useful_rollout/loss_masked_tokens"] == 2
    assert metrics["rollout/fully_async/useful_rollout/accounting_error_tokens"] == 0


def test_waste_reason_metrics_have_fixed_cardinality() -> None:
    metrics = discard_waste_metrics({})

    assert metrics["rollout/fully_async/waste/all_discarded/decode_tokens"] == 0
    for reason in DISCARD_REASONS:
        assert metrics[f"rollout/fully_async/waste/{reason}/decode_tokens"] == 0


def test_waste_vector_keeps_heterogeneous_units_separate() -> None:
    sample = _sample(1, 7, [1] * 7)
    sample.prefix_cache_info.total_prompt_tokens = 11
    sample.prefix_cache_info.cached_tokens = 3
    sample.non_generation_time = 1.25
    sample.metadata[REWARD_SECONDS_KEY] = 0.5

    assert waste_vector([sample]) == {
        "decode_tokens": 7.0,
        "prefill_uncached_tokens": 8.0,
        "tool_env_seconds": 1.25,
        "reward_seconds": 0.5,
    }


def test_stale_reason_is_the_first_boundary_that_crosses_the_bound() -> None:
    common = {
        "reference_version": 10,
        "drain_version": 15,
        "bound": 1,
    }
    assert (
        classify_stale_recycle_stage(
            **common,
            generation_completion_version=12,
            group_ready_version=13,
            queue_put_version=14,
        )
        == STALE_AT_GENERATION_COMPLETION
    )
    assert (
        classify_stale_recycle_stage(
            **common,
            generation_completion_version=11,
            group_ready_version=12,
            queue_put_version=14,
        )
        == STALE_DURING_REWARD_FINALIZE
    )
    assert (
        classify_stale_recycle_stage(
            **common,
            generation_completion_version=11,
            group_ready_version=11,
            queue_put_version=12,
        )
        == STALE_DURING_QUEUE_BACKPRESSURE
    )
    assert (
        classify_stale_recycle_stage(
            **common,
            generation_completion_version=11,
            group_ready_version=11,
            queue_put_version=11,
        )
        == STALE_IN_OUTPUT_QUEUE
    )


def test_straggler_collateral_detects_crossing_during_group_wait() -> None:
    early = _sample(1, 2, [1, 1])
    straggler = _sample(2, 2, [1, 1])
    for sample in (early, straggler):
        sample.metadata["submission_weight_version"] = 10
        sample.metadata[GROUP_GENERATION_COMPLETE_VERSION_KEY] = 13
        sample.metadata[LIFECYCLE_EXACT_KEY] = True
    early.metadata[SAMPLE_GENERATION_COMPLETE_VERSION_KEY] = 11
    straggler.metadata[SAMPLE_GENERATION_COMPLETE_VERSION_KEY] = 13

    collateral = straggler_collateral_indices(
        [early, straggler],
        reference_mode="submission",
        drain_version=13,
        bound=1,
    )

    assert collateral == [early.index]


def test_nested_pass_rate_annotation_becomes_an_explicit_difficulty_proxy() -> None:
    sample = _sample(1, 7, [1] * 7)
    sample.metadata["difficulty"] = {"Qwen3-4B-Instruct-2507": {"pass_rate": 0.75, "n_samples": 8}}
    populations: dict[str, dict[str, list[float]]] = {}

    add_selection_population(populations, population_name="generated", samples=[sample])
    metrics = selection_population_metrics(populations)

    assert metrics["selection_bias/generated/difficulty/mean"] == 0.25
    assert metrics["selection_bias/generated/prompt_pass_rate/mean"] == 0.75


def test_queue_telemetry_owns_response_length_until_final_consumption() -> None:
    sample = _sample(1, 7, [1] * 7)
    populations: dict[str, dict[str, list[float]]] = {}

    add_selection_population(populations, population_name="generated", samples=[sample])
    add_selection_population(populations, population_name="consumed", samples=[sample])
    metrics = selection_population_metrics(populations)

    assert metrics["selection_bias/generated/samples"] == 1
    assert "selection_bias/generated/response_length/mean" not in metrics
    assert metrics["selection_bias/consumed/response_length/mean"] == 7


def test_custom_selection_metrics_are_aggregated_for_each_population() -> None:
    samples = [_sample(1, 7, [1] * 7), _sample(2, 7, [1] * 7)]
    samples[0].metadata[SELECTION_METRICS_KEY] = {
        "strict_math/multiple_answer_markers": 0.0,
        "strict_math/reward_disagreement": 0.0,
    }
    samples[1].metadata[SELECTION_METRICS_KEY] = {
        "strict_math/multiple_answer_markers": 1.0,
        "strict_math/reward_disagreement": 1.0,
    }
    populations: dict[str, dict[str, list[float]]] = {}

    add_selection_population(populations, population_name="generated", samples=samples)
    metrics = selection_population_metrics(populations)

    assert metrics["selection_bias/generated/strict_math/multiple_answer_markers/mean"] == 0.5
    assert metrics["selection_bias/generated/strict_math/reward_disagreement/mean"] == 0.5


def test_consumed_debug_rows_are_idempotent_for_prepared_batch_replay() -> None:
    sample = _sample(1, 7, [1] * 7)
    debug_metadata: dict = {}

    for _ in range(2):
        append_final_consumed_records(
            debug_metadata,
            [sample],
            reference_mode="completion",
            bound=2,
            training_step=3,
        )

    records = debug_metadata["recycle_compute"]["records"]
    assert len(records) == 1
    assert records[0]["disposition"] == "consumed"


def test_prequeue_phase_partition_is_additive() -> None:
    sample = _sample(1, 2, [1, 1])
    sample.metadata.update(
        {
            TRAJECTORY_START_VERSION_KEY: 2,
            SAMPLE_GENERATION_COMPLETE_VERSION_KEY: 4,
            GROUP_GENERATION_COMPLETE_VERSION_KEY: 5,
            GROUP_READY_VERSION_KEY: 7,
            TRAJECTORY_START_TIME_KEY: 10.0,
            SAMPLE_GENERATION_COMPLETE_TIME_KEY: 13.0,
            GROUP_GENERATION_COMPLETE_TIME_KEY: 15.0,
            GROUP_READY_TIME_KEY: 19.0,
            SAMPLE_REFERENCE_VERSION_KEY: 2,
            LIFECYCLE_EXACT_KEY: True,
        }
    )

    metrics = prequeue_phase_metrics([sample])

    assert metrics["staleness/pre_queue_phase/version/active/sequence_mean"] == 2
    assert metrics["staleness/pre_queue_phase/version/group_wait/sequence_mean"] == 1
    assert metrics["staleness/pre_queue_phase/version/postprocess/sequence_mean"] == 2
    assert metrics["staleness/pre_queue_phase/version/total/sequence_mean"] == 5
    assert metrics["staleness/pre_queue_phase/version/identity_max_abs_error"] == 0
    assert metrics["staleness/pre_queue_phase/wall_seconds/identity_max_abs_error"] == 0
    assert metrics["staleness/pre_queue_phase/exact_sample_frac"] == 1


def test_pipeline_snapshot_yields_same_window_for_all_throughputs() -> None:
    sample = _sample(1, 4, [1, 1, 1, 0])
    sample.metadata.update(
        {
            SAMPLE_REFERENCE_VERSION_KEY: 3,
            DRAIN_VERSION_KEY: 5,
            QUEUE_PUT_TIME_KEY: 10.0,
            DRAIN_TIME_KEY: 12.0,
        }
    )
    snapshot = build_batch_consumption_snapshot(
        [sample],
        selection_version=5,
        bound=4,
        optimizer_updates=2,
        cohort_generated_tokens=10,
    )
    metrics = batch_consumption_metrics(
        snapshot,
        train_start_version=5,
        pipeline_snapshot={
            "window_seconds": 2.0,
            "generated_tokens": 10.0,
            "completed_training_batches": 1.0,
            "accepted_tokens": 3.0,
            "accepted_tokens_available": 1.0,
            "optimizer_updates": 2.0,
            "queue_depth_time_mean": 3.0,
            "queue_depth_current": 4.0,
            "trainer_starvation_seconds": 0.25,
            "rollout_backpressure_seconds": 0.5,
            "active_group_capacity_fraction": 0.75,
        },
    )

    assert metrics["throughput/generated_tokens_per_second"] == 5
    assert metrics["throughput/accepted_tokens_per_second"] == 1.5
    assert metrics["throughput/useful_tokens_per_second"] == 1.5
    assert metrics["throughput/window_useful_efficiency"] == 0.3
    assert metrics["throughput/cohort_useful_efficiency"] == 0.3
    assert metrics["throughput/cohort_projected_useful_tokens_per_second"] == 1.5
    assert metrics["throughput/optimizer_updates_per_second"] == 1
    assert metrics["queue/consumption/wall_wait_seconds/p90"] == 2


def test_custom_converter_does_not_guess_accepted_token_throughput() -> None:
    sample = _sample(1, 4, [1, 1, 1, 0])
    snapshot = build_batch_consumption_snapshot(
        [sample],
        selection_version=5,
        bound=4,
        cohort_generated_tokens=4,
        has_custom_converter=True,
    )

    metrics = batch_consumption_metrics(
        snapshot,
        train_start_version=5,
        pipeline_snapshot={
            "window_seconds": 2.0,
            "generated_tokens": 4.0,
            "accepted_tokens_available": 0.0,
            "optimizer_updates": 1.0,
        },
    )

    assert metrics["throughput/accepted_loss_tokens_available"] == 0.0
    assert "throughput/accepted_loss_tokens" not in metrics
    assert "throughput/accepted_tokens_per_second" not in metrics
    assert metrics["throughput/generated_tokens_per_second"] == 2.0


def test_pipeline_telemetry_uses_time_weighted_queue_depth() -> None:
    now = [0.0]
    telemetry = FullyAsyncPipelineTelemetry(clock=lambda: now[0])
    telemetry.set_queue_depth(2)
    telemetry.set_active_groups(2, 4)
    now[0] = 2.0
    telemetry.set_queue_depth(4)
    telemetry.set_active_groups(4, 4)
    telemetry.add_generated_group(20)
    telemetry.add_trained_batch(accepted_tokens=6, optimizer_updates=2)
    telemetry.add_trainer_starvation(0.5)
    telemetry.add_rollout_backpressure(0.25)
    now[0] = 4.0

    snapshot = telemetry.snapshot(active_groups=3, max_active_groups=4)

    assert snapshot["queue_depth_time_mean"] == 3
    assert snapshot["generated_tokens"] == 20
    assert snapshot["accepted_tokens"] == 6
    assert snapshot["accepted_tokens_available"] == 1
    assert snapshot["optimizer_updates"] == 2
    assert snapshot["trainer_starvation_seconds"] == 0.5
    assert snapshot["rollout_backpressure_seconds"] == 0.25
    assert snapshot["rollout_idle_capacity_seconds"] == 1.0
    assert snapshot["active_group_capacity_fraction"] == 0.75
    assert snapshot["active_group_capacity_time_mean"] == 0.75


def test_exact_response_segments_produce_token_weighted_lag() -> None:
    sample = _sample(1, 5, [1] * 5)
    sample.response_weight_version_segments = [
        [[0, 2, 8], [2, 5, 9]],
    ]

    metrics = sample_lag_metrics([sample], train_version=10)

    assert metrics["staleness/token_lag/exact/covered_response_token_frac"] == 1.0
    assert metrics["staleness/token_lag/exact/num_tokens"] == 5.0
    assert metrics["staleness/token_lag/exact/mean"] == 1.4
    assert metrics["staleness/token_lag/exact/p90"] == 2.0


def test_exact_response_segments_reject_overlapping_turns() -> None:
    sample = _sample(1, 5, [1] * 5)
    sample.response_weight_version_segments = [
        [[0, 3, 8], [2, 5, 9]],
    ]

    metrics = sample_lag_metrics([sample], train_version=10)

    assert metrics["staleness/token_lag/exact/covered_response_token_frac"] == 0.0
    assert metrics["staleness/token_lag/exact/invalid_segments"] == 1.0
    assert metrics["staleness/token_lag/exact/invalid_turns"] == 1.0
