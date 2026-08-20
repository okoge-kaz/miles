"""Shared configuration contract for fully-async rollout queues."""

QUEUE_RECYCLE_POLICY = "queue-recycle"
QUEUE_MAX_POLICY = "queue-max"
QUEUE_DROP_POLICY = "queue-drop"
FULLY_ASYNC_QUEUE_POLICIES = (QUEUE_RECYCLE_POLICY, QUEUE_MAX_POLICY, QUEUE_DROP_POLICY)


def should_prefetch_rollout_batches(args) -> bool:
    """Whether the driver may reserve the next batch during the current train step."""
    queue_type = getattr(args, "fully_async_queue_type", QUEUE_RECYCLE_POLICY)
    return not (getattr(args, "fully_async", False) and queue_type in (QUEUE_MAX_POLICY, QUEUE_DROP_POLICY))


def validate_fully_async_queue_args(args) -> None:
    """Reject configurations that change a named queue algorithm's semantics."""
    queue_type = args.fully_async_queue_type
    if queue_type != QUEUE_RECYCLE_POLICY and not args.fully_async:
        raise ValueError("--fully-async-queue-type requires --fully-async")
    if not args.fully_async:
        return
    if args.fully_async_queue_factor < 1:
        raise ValueError("--fully-async-queue-factor must be at least 1")
    if queue_type != QUEUE_DROP_POLICY and args.fully_async_queue_factor != 1:
        raise ValueError("--fully-async-queue-factor is only used by queue-drop")
    if args.max_weight_staleness is not None and args.max_weight_staleness < 0:
        raise ValueError("--max-weight-staleness must be non-negative")
    if queue_type == QUEUE_RECYCLE_POLICY:
        if args.max_weight_staleness == 0:
            raise ValueError(
                "queue-recycle requires --max-weight-staleness >= 1 because its strict gap < bound rule "
                "admits no group when the bound is 0"
            )
    elif queue_type == QUEUE_MAX_POLICY:
        if args.max_weight_staleness is None:
            raise ValueError("queue-max requires --max-weight-staleness")
        if args.staleness_reference != "prefill":
            raise ValueError("queue-max requires --staleness-reference prefill")
    elif queue_type == QUEUE_DROP_POLICY:
        if args.max_weight_staleness is not None:
            raise ValueError("queue-drop cannot be combined with --max-weight-staleness")
