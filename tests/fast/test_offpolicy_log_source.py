from experiments.src.offpolicy_acceleration import log_source


def test_parse_log_merges_additive_rollout_streams(tmp_path):
    log = tmp_path / "job.log"
    log.write_text(
        "\n".join(
            (
                "[2026-08-13 18:31:43.300 rollout_manager] metrics.py:128 - "
                "perf 2: {'rollout/fully_async/avg_staleness': 1.0}",
                "[2026-08-13 18:31:43.310 actor_cell0_rank0] metrics.py:128 - " "perf 2: {'perf/step_time': 2.3}",
                "[2026-08-13 18:31:43.311 actor_cell1_rank0] metrics.py:128 - " "perf 2: {'perf/step_time': 2.3}",
                "[2026-08-13 18:31:43.324 rollout_manager] metrics.py:50 - "
                "rollout batch consumption 2: {'throughput/accepted_loss_tokens': 8192}",
                "[2026-08-13 18:31:43.691 rollout_manager] metrics.py:64 - "
                "rollout pipeline throughput 2: {'throughput/generated_tokens_per_second': 3542.9}",
                "[2026-08-13 18:31:44.000 actor_cell0_rank0] log_utils.py:460 - " "step 2: {'train/loss': 0.25}",
            )
        )
    )

    records = log_source.merge_step_records(log_source.parse_log(log))

    assert len(records) == 2
    rollout_record = next(record for record in records if record["step_key"] == "rollout/step")
    assert rollout_record["step"] == 2
    assert rollout_record["metrics"] == {
        "rollout/fully_async/avg_staleness": 1.0,
        "rollout/step": 2,
        "perf/step_time": 2.3,
        "throughput/accepted_loss_tokens": 8192,
        "throughput/generated_tokens_per_second": 3542.9,
    }
