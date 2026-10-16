from experiments.src.nemo_gym.bridge.client import EVENTS


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    # Completion events include excluded samples, unlike selected training batches.
    # Agent subprocess counters return once per episode through bridge.generate;
    # stateless reward requests add to the same parent counter directly.
    metrics = {f"rollout/nemo_gym/events/{key}": value for key, value in EVENTS.items()}
    EVENTS.clear()
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(metrics)
    elif metrics:
        from miles.utils.metric_utils import compute_rollout_step
        from miles.utils.tracking_utils import tracking

        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        tracking.log(args, metrics, step_key="rollout/step")
    return False
