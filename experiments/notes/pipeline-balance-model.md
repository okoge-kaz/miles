# Producer/consumer model for fully-async staleness

This note defines the deterministic steady-state model implemented by
`experiments/tools/pipeline_balance_model.py`. It predicts the central queue
regime and deliberately does not claim to predict stochastic latency outliers.

## Rates and the existence of a steady state

Let:

- `B` be prompt groups consumed by one training update;
- `tau_T` be trainer compute seconds per update, excluding rollout starvation;
- `lambda_R` be completed rollout groups per second before queue backpressure;
- `K` be completed-group queue capacity;
- `C` be the time-mean number of active rollout groups.

The trainer service capacity and producer/consumer ratio are

```text
mu_T = B / tau_T                         groups / second
rho  = lambda_R / mu_T
     = lambda_R * tau_T / B
```

Use `perf/train_time`, not a wall step time that already contains
`perf/train_wait_time`, for `tau_T`. Likewise, a group rate observed while
`queue/rollout_backpressure_seconds` is non-zero is censored by the consumer and
is not the unconstrained `lambda_R`.

For an unbounded FIFO queue:

```text
rho < 1: queue drains; a stationary rollout-limited regime exists
rho = 1: every initial queue depth is an equilibrium; noise determines drift
rho > 1: no stationary state; queue depth grows by B * (rho - 1) per update
```

The age of the FIFO item consumed at training step `n` grows more slowly than
the queue depth sampled at that step, because that item arrived at about
`n / rho`. Its first-order slope is

```text
d training_staleness / dn = (rho - 1) / rho = 1 - 1 / rho.
```

This is the linear-growth branch used for `t1r7`. It stops at the smaller of the
configured weight-staleness bound and the finite completed-buffer envelope.

## Pre-queue and in-queue components

Let `u = min(1/tau_T, lambda_R/B)` be the actual optimizer update rate. If mean
group latency `L_g` is measured, the deterministic pre-queue approximation is

```text
PQS = u * L_g.
```

Otherwise Little's law gives `L_g = C / lambda_R`, using completed groups as
the unit, and therefore

```text
PQS = C / (B * max(1, rho)).
```

This form already includes group tail latency when `lambda_R` counts complete
prompt groups. Do not multiply by a second sample-to-group tailness factor.

For an under-produced, empty-queue deterministic pipeline:

```text
queue-recycle: IQS ~= 1    (the driver prefetched the next batch before update)
queue-max:     IQS ~= 0    (selection happens after the preceding update)
```

Finite batches and bursty group completion add a residual smaller than roughly
one update in the measured runs. The script reports the deterministic value as
a lower-envelope point and a one-update review interval rather than pretending
that rates determine the arrival-phase distribution.

For `queue-drop`, the bounded steady-state closed form is different. With
`q = K/B`:

```text
IQS = rho                              if rho < 1
IQS = (2*q + rho - 1) / (2*rho)        if rho > 1.
```

The exact `rho=1` boundary is intentionally reported as indeterminate. This
closed form must not be applied to `queue-recycle` or `queue-max`.

For overloaded FIFO policies, the useful cap envelope is

```text
h = 1 for queue-recycle, 0 for queue-max
L_natural(0) ~= PQS + h
L_queue      ~= PQS + h + K/B
L_cap        = min(max_weight_staleness, L_queue)
L(n)         ~= min(PQS + h + n*(1 - 1/rho), L_cap).
```

When the weight bound is the active cap, rejection changes rollout throughput,
so `L_cap` is an envelope rather than an exact distributional mean.

## Node scaling and ratio selection

The analysis fits two intentionally small models:

```text
tau_T(T)       = a_T/T + b_T
1/lambda_R(R)  = a_R/R + b_R.
```

The first is the fixed-work inverse-DP model plus a non-scaling floor. Perfect
linear strong scaling has `b_T=0`. The second is an inverse-capacity model plus
a concurrency/latency floor. It implies

```text
lambda_R(infinity) = 1/b_R
group latency floor = C*b_R.
```

For a fixed total node count, candidate `T:R` splits are ranked by

```text
updates/second = min(1/tau_T(T), lambda_R(R)/B).
```

The continuous balance point solves

```text
rho(T, R) = lambda_R(R) * tau_T(T) / B = 1,
T + R = total nodes.
```

With integer nodes, evaluate both neighboring splits and maximize the `min`
expression instead of rounding the continuous solution blindly. The winning
integer split may lie on either side of `rho=1`.

The throughput optimum need not have `rho>=1`: adding a second trainer node can
make the system rollout-bound while still increasing total updates per hour.
Staleness and GPU-efficiency constraints should therefore be reported next to,
not substituted for, this bottleneck calculation.

## Current 8-node check

The checked-in `sr-20260819-212906` history was summarized over the latest 50
usable updates after removing ten updates at each resume boundary. It uses
`B=C=192`, 16 samples per group, `queue-recycle`, and `K=1000`. Older history
does not contain the direct generated-group counter, so group rate below is a
token-rate proxy and the `1:7` point is marked as backpressure-censored.
This historical `s8` cohort is quarantined for controlled algorithm comparison
because its 1000-group queue binds below the intended weight-staleness bound;
it remains useful here precisely as an exploratory check of the queue-cap
branch. Corrected `s8+` experiments use a 6000-group queue.

| T:R | train compute s | effective groups/s | capacity censored | observed total staleness | model |
|---|---:|---:|:---:|---:|---:|
| 1:7 | 228.35 | 0.815 | yes | 7.18 | 7.11 cap, using the uncensored rollout fit |
| 2:6 | 118.91 | 0.912 | no | 2.14 | 2.00, review interval [2, 3] |
| 3:5 | 83.92 | 0.832 | no | 2.15 | 2.00, review interval [2, 3] |
| 4:4 | 60.21 | 0.804 | no | 2.20 | 2.00, review interval [2, 3] |

Thus `t2r6` is predicted correctly as a stationary, rollout-limited point whose
natural staleness is below the configured bound. The point error is about 0.14
updates; the observation lies inside the batching interval. Conversely, using
the backpressured `t1r7` rate directly gives the misleading result `rho ~= 1`.
Fitting rollout capacity on uncensored `R=4,5,6` points predicts `rho=1.11` at
`R=7`, a queue-cap envelope of 7.11, and the observed 7.18.

The `s8` trainer fit is

```text
tau_T(T) = 221.24/T + 7.62 seconds, R^2 = 0.9991.
```

One to four trainer nodes therefore give 3.79x speedup, or 94.8% strong-scaling
efficiency at four nodes. Within this range training is very close to inverse-DP
scaling, though the positive 7.62-second floor rules out indefinite linear
extrapolation.

The uncensored rollout fit is

```text
1/lambda_R(R) = 1.695/R + 0.832 seconds/group, R^2 = 0.874.
```

It predicts an asymptote of 1.20 groups/s and, at `C=192`, an effective
159.8-second group-latency floor. Predicted rates at `R=4,5,6,7` are
`0.796, 0.854, 0.897, 0.931` groups/s, so the incremental gain decreases from
7.2% to 5.1% to 3.8%.

Combining the two fits predicts `15.73, 16.82, 16.01, 14.93` updates/hour for
`1:7, 2:6, 3:5, 4:4`; `2:6` is the throughput choice for this measured 8-node
cohort. This is a single-model, single-task operational validation, not a
cross-model scaling law.

## Commands

Predict one point and optionally emit its trajectory:

```bash
python experiments/tools/pipeline_balance_model.py point \
  --train-compute-seconds 118.9 \
  --rollout-groups-per-second 0.912 \
  --batch-groups 192 \
  --concurrency-groups 192 \
  --queue-policy queue-recycle \
  --queue-capacity-groups 1000 \
  --max-weight-staleness 8 \
  --trajectory-csv /tmp/t2r6-staleness.csv
```

Fit the checked-in history and rank the 8-node splits:

```bash
python experiments/tools/pipeline_balance_model.py history \
  --history-csv experiments/outputs/reasoning_eval/staleness-ratio-sweep/sr-20260819-212906/analysis/partial-424-20260825-1009/staleness/training-history.csv \
  --output-dir /tmp/pipeline-balance \
  --batch-groups 192 \
  --samples-per-group 16 \
  --concurrency-groups 192 \
  --queue-policy queue-recycle \
  --queue-capacity-groups 1000 \
  --fit-staleness 8 \
  --total-nodes 8 \
  --allowed-trainer-nodes 1,2,3,4
```

The `history` command also writes five dependency-free SVG figures under
`OUTPUT_DIR/figures/`:

- `training-node-scaling.svg`: observed trainer time and the inverse-DP fit;
- `rollout-node-scaling.svg`: uncensored rollout rate, fit, censored points,
  and the fitted saturation asymptote;
- `node-ratio-throughput.svg`: measured-rate and fitted updates/hour;
- `staleness-prediction-vs-observed.svg`: the late-window validation;
- `predicted-staleness-trajectories.svg`: stationary lines or linear growth to
  the active queue/weight cap.

New runs log `throughput/generated_groups_per_second` directly. Historical
runs fall back to generated token rate divided by samples per group and mean
response length; the output records which source was used and flags
backpressure-censored rate points.
