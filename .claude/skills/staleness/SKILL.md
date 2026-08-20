---
name: staleness
description: Audit or change Miles fully-async staleness control, queue-recycle admission, policy-version provenance, and staleness logging. Use when modifying max-weight-staleness, staleness-reference, updates_before_train, queue lifecycle metrics, sample/token lag metrics, or documentation and tests for those concepts.
---

# Miles staleness

Preserve the distinction between the version available at dequeue and the
version that actually trains a batch. Apply the formulas below when
`--staleness-reference prefill` is selected.

## Use these symbols

- `F_i`: sample `i`'s minimum scheduler-authoritative first-prefill version.
- `F_g = min_i(F_i)`: prompt group's prefill reference used for admission.
- `G_i`: sample `i`'s last model-forward version.
- `G_g = max_i(G_i)`: group's last model-forward boundary.
- `Q_g`: version when the completed group becomes queue-ready.
- `P_g`: version when the ready group is actually put into the queue, after any
  output-capacity backpressure.
- `D_g`: version observed when the consumer dequeues/prefetches the group.
- `u_b`: `updates_before_train` for batch `b`.
- `T_b = D_g + u_b`: version that actually trains batch `b`.
- `V_t`: model-forward version that generated token `t`.
- `M`: configured `--max-weight-staleness`.

Treat versions as policy-update counters, not wall-clock timestamps.
Do not assume `G_g = Q_g`. They are equal only when no weight update lands
after the group's last model forward and before reward/finalization makes the
group queue-ready. Do not use queue-ready and queue-put interchangeably either.
Versions are nondecreasing across the distinct lifecycle boundaries:
`G_g <= Q_g <= P_g <= D_g <= T_b`; adjacent values can be numerically equal.

## Keep the event order explicit

For a steady-state prefetched batch with one update per rollout, use this
timeline:

```text
time --------------------------------------------------------------------------------->
batch n+1: prefill/decode(F...G) -> reward/finalize -> ready(Q) -> put(P)
                                                              -> [dequeue/prefetch at D] -> train at T=D+1
trainer:                                                      -> [train batch n] -> update D->D+1
```

Equivalently:

```text
prefill/decode -> [dequeue/prefetch batch n+1 | train batch n]
               -> weight update -> train batch n+1
```

For the startup batch, no previous batch exists: `u_b = 0` and `T_b = D_g`.
For the normal steady-state path with update interval one, `u_b = 1` and
`T_b = D_g + 1`. Derive `T_b` from `updates_before_train`; do not infer it from
the metric name or add an unconditional `+1`.

## Enforce the queue-recycle bound at dequeue

Use the strict queue-recycle admission rule:

```text
admit   iff D_g - F_g < M
recycle iff D_g - F_g >= M
```

The desired steady-state train-time guarantee is inclusive:

```text
T_b - F_g <= M
```

Because the admission decision uses `D_g` before the scheduled update and the
normal path has `T_b = D_g + 1`, the train-time condition is equivalent to
`D_g - F_g < M`. This is why the implementation uses `< M`; changing it to
`<= M` would normally train the equality case at `M + 1`.

Require `M >= 1` for queue-recycle because `D_g - F_g` is nonnegative and the
strict rule with `M = 0` admits no group. This requirement is mathematical; it
is not caused by the startup batch. The startup path still applies the strict
rule with `u_b = 0`, so it conservatively recycles the equality case even though
that one batch would satisfy the train-time inclusive bound.

Do not copy this strict rule to other queue types. In particular, queue-max
checks the dequeue-time gap, rejects only `D_g - F_g > M`, and admits equality.

## Log realized staleness at training

Use `T_b`, not `D_g`, for metrics that claim to describe trained or admitted
data:

```text
staleness/pre_queue = Q_g - F_g
staleness/in_queue  = T_b - Q_g
staleness/total     = T_b - F_g
staleness/rollout   = T_b - F_g  # offered group, including rejected groups
```

Record the actual training version as
`fully_async/train_weight_version = T_b`. Classify a queue-recycle rejection as
bound-exceeded when its dequeue gap is `>= M`, matching the control rule.

Do not describe `T_b - G_g` as in-queue staleness:

```text
last_forward_to_train = T_b - G_g
in_queue              = T_b - Q_g
last_forward_to_train = (Q_g - G_g) + in_queue
in_queue              = (P_g - Q_g) + (T_b - P_g)
```

The extra term contains environment, reward, and postprocessing work after the
last model forward and before queue readiness. The current `in_queue` family
starts at queue-ready, so it includes both output-capacity backpressure before
the actual put and residence after the put.

## Keep sample and exact-token lag distinct

For sample `i`, use:

```text
generation            = G_i - F_i
group_sync             = G_g - G_i
last_forward_to_train  = T_b - G_g
sample_lag/total       = T_b - F_i
sample_staleness       = T_b - F_i
```

`sample_lag/total` and `sample_staleness` intentionally share the same scalar
under the prefill reference. They serve different log products: rollout-side
distribution summaries versus trainer-side loss/objective bins.

For exact token lag, use `T_b - V_t`. This can differ across tokens in a single
response and is not interchangeable with the one-scalar-per-sample value.

## Audit changes end to end

1. Inspect `train_async.py` and `miles/rollout/fully_async_rollout.py` for the
   event order and `updates_before_train` propagation.
2. Inspect `miles/rollout/recycle_compute_metrics.py` and
   `miles/ray/rollout/train_data_conversion.py` for logging endpoints.
3. Inspect `miles/rollout/queue_policy.py` and `miles/utils/arguments.py` for
   validation and CLI wording.
4. Update `experiments/notes/telemetry.md`, other notes that state the bound,
   launch scripts, and focused tests together.
5. Keep queue-recycle-specific comparisons scoped to queue-recycle. Preserve
   queue-max and queue-drop semantics unless the request explicitly changes
   them.
6. Do not add legacy aliases or replay-state migrations unless explicitly
   requested.
7. Run the staleness-focused pytest suite in the training image and run static
   formatting checks before reporting completion.
