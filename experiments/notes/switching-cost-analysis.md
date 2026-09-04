# Colocated switching-cost and throughput snapshot (2026-08-27)

This snapshot compares the matched 64-H100 runs and records the model used for
the Qwen3-8B and Qwen3-30B-A3B follow-up. Current runs are incomplete; numbers
below are descriptive, not confidence intervals.

## Throughput

For a step-aligned comparison, the table sums `perf/step_time` over steps
10--139 and divides accepted response tokens by the same wall time. This keeps
the policy-age window common, includes cold steps at chained-allocation
boundaries, and excludes scheduler/requeue and process-initialization time that
`perf/step_time` cannot observe. Every arm uses eight nodes and 64 H100s.

| arm | updates/hour | accepted response tokens/s | realized total staleness median |
|---|---:|---:|---:|
| current colocated + partial O=256 | 14.60 | 76,690 | 0.33 |
| current async T1:R7, C=4096 | 12.93 | 68,759 | 3.59 |
| current async T2:R6, C=4096 | **15.51** | **80,921** | 2.41 |
| current async T3:R5, C=4096 | 14.48 | 78,327 | 2.41 |
| current async T4:R4, C=4096 | 13.91 | 73,143 | 2.40 |
| prior colocated, no partial rollout | 11.19 | 58,344 | n/a |
| prior async T1:R7, default C=3072 | 13.30 | 69,045 | 3.83 |
| prior async T2:R6, default C=3072 | 16.13 | 84,401 | 2.14 |
| prior async T3:R5, default C=3072 | 15.05 | 81,619 | 2.15 |
| prior async T4:R4, default C=3072 | 14.56 | 75,331 | 2.17 |

The partial-rollout arm is about 30% faster than the prior colocated arm on both
updates/hour and accepted-token service rate. It is about 6% slower than the
current T2:R6 async winner. Raising async admitted concurrency from 3072 to
4096 did not improve this common-window result: the four matched async arms are
roughly 3--5% slower than their prior same-ratio controls. One seed and changing
node allocations are insufficient to interpret that small regression causally.

The prior full runs show why downstream time should not be inferred from raw
throughput alone. Using the first complete three-task AIME macro checkpoint to
cross each threshold and cumulative `perf/step_time` as wall time:

| arm | 50% macro | 55% macro | 58% macro | 60% macro |
|---|---:|---:|---:|---:|
| colocated | 5.32 h | 10.79 h | not yet observed | not yet observed |
| async T1:R7, S=4 | 4.92 h | 9.68 h | 13.63 h | not yet observed |
| async T2:R6, S=4 | **3.44 h** | **6.00 h** | 9.98 h | not yet observed |
| async T3:R5, S=4 | 3.72 h | 9.19 h | 12.58 h | not yet observed |
| async T4:R4, S=4 | 3.85 h | 7.36 h | **9.51 h** | 12.38 h |

The old evaluation inventory is incomplete beyond step 190 for these async
arms and step 160 for colocated, so a final-quality ranking is not available.
The current partial/concurrency cohort has no completed downstream suite yet.

## Measured Qwen3-4B switch

Latest-write-wins parsing gives 191 unique switches (steps 0--190). Excluding
the first switch of each allocation leaves 187 steady-state observations:

| phase | median | p10--p90 |
|---|---:|---:|
| rollout offload block | 0.468 s | -- |
| maximum trainer wake component | 3.919 s | -- |
| rollout to train active sum | 4.434 s | -- |
| train to rollout driver critical path | 13.959 s | -- |
| total active switch | **18.282 s** | 14.946--22.669 s |

About 76% of the median is on train-to-rollout. The four chained allocations
have total-switch medians of 22.11, 19.35, 17.48, and 16.13 seconds, showing a
node/allocation effect large enough to obscure modest model-size scaling.

The Qwen3-4B SGLang log reports 677,242 KV tokens and 23.25 GiB each for K and
V, or 46.5 GiB per GPU. Its 0.47-second offload proves that these 46.5 GiB are
not copied to host on every switch. Partial rollout stores token prefixes and
re-prefills them after the cache flush; KV pool size therefore affects memory
release/allocation, graph capture, and capacity, not a simple `KV bytes / PCIe
bandwidth` payload.

## Model

Let `X_tms` be the actual per-node bytes backed up by torch-memory-saver,
`W_train` the updated actor shard copied to pinned host memory, `W_roll` the
rollout weight shard, `K_pool` the allocated KV pool, and `B_h2d`, `B_d2h` the
concurrent node-level host/device rates. A phase-level model is:

```text
T(snapshot)      = alpha_snapshot + W_train / B_d2h
T(trainer wake)  = alpha_wake     + X_tms / B_h2d + T(process-group reload)
T(trainer sleep) = alpha_sleep    + X_tms / B_d2h + T(process-group destroy)
T(weight onload) = alpha_onload   + f(W_roll, allocator)
T(weight publish)= alpha_publish  + f(W_train, CUDA IPC, conversion, collectives)
T(KV onload)     = alpha_kv       + f(K_pool, CUDA graphs); not K_pool / B_h2d
```

Colocated broadcast weight publication uses CUDA IPC, so only handles cross
processes; its bulk cost belongs to device-local conversion/copies and GPU
collectives rather than the CPU/GPU bandwidth term. Trainer pause/resume and
the pinned actor snapshot are the phases expected to track CPU/GPU bandwidth.

The official BF16 safetensors totals and the follow-up parallelism give these
payload anchors:

| quantity per GPU | 4B, TP2 | 8B, TP4 | 30B-A3B follow-up |
|---|---:|---:|---:|
| total BF16 model | 8.22 GiB | 15.26 GiB | about 56.87 GiB |
| rollout weight shard | 4.11 GiB | 3.81 GiB | 7.11 GiB at TP8 |
| Adam FP32 main+m+v, ideal 64-way lower bound | 0.77 GiB | 1.43 GiB | 5.33 GiB |
| KV bytes/token/rank | 73,728 | 36,864 | runtime log required |

For 30B-A3B, active 3B controls compute but not switching storage: all expert
weights must remain available. With training TP4/EP8/ETP1, expert optimizer
state is sharded over an expert-data-parallel group of size eight on 64 GPUs:
`64 / (ETP1 * EP8 * PP1) = 8`. The dense data-parallel size is separately
`64 / (TP4 * CP1 * PP1) = 16`. The topology-based FP32 main-parameter and Adam
moment estimate is therefore close to the 5.33-GiB ideal 64-way value, not
20.5 GiB per rank. Runtime state inventory must still verify padding and buffer
overheads before this is used as a fitted predictor.

The 8B weight shard is slightly smaller than the current 4B shard because TP4
replaces TP2. Its optimizer shard is about 1.86 times larger, but that term is
small in the 4B theoretical payload. The prediction is therefore a switching
time near 4B rather than 1.86 times 4B. The 30B run should increase the
trainer-pause/resume phases materially through expert optimizer state, while
weight snapshot/onload scale only about 1.7--1.8 times at the selected
parallelism. Neither model's KV phase should scale linearly with parameter
count; fixed `sglang_mem_fraction_static` allocates from remaining HBM, and
GQA/TP changes bytes per cached token.

### Pre-measurement prediction

For capacity planning, the phase model above gives the following deliberately
wide ranges. They are engineering predictions, not confidence intervals:

| model and selected topology | predicted active switch | ratio to 4B | basis |
|---|---:|---:|---|
| 4B, train/rollout TP2 | 18.3 s measured; p10--p90 15.0--22.7 s | 1.00 | 187 steady-state observations |
| 8B, train/rollout TP4 | **18--20 s central; 15--24 s planning range** | about 1.0--1.1 | trainer weight plus ideal optimizer payload is only about 7% larger per rank; rollout weight shard is about 7% smaller |
| 30B-A3B, train TP4/EP8 and rollout TP8 | **28--38 s central; 25--45 s planning range** | about 1.5--2.1 | trainer-resident weight plus optimizer payload is roughly 2.5--2.8x per rank, while rollout weight paths are about 1.7--1.8x and fixed/KV phases do not follow total parameters |

The 30B range can be reproduced from the 4B trace by assigning 10--40% of the
14-second train-to-rollout path to trainer offload, scaling that part and the
3.9-second wake component by the roughly 2.5--2.8x trainer-state ratio, scaling
the rollout weight paths by roughly 1.7x, and leaving fixed/KV work unscaled.
The large range is the consequence of not yet having the four new subphase
timers on a production trace; it should narrow substantially after the first
30B run.

`switch_total_active_time` does not include the adjacent pinned-host actor
snapshot. Its byte-only prediction is approximately 0.93x the 4B snapshot for
8B and 1.7--1.8x for 30B. The new job records it separately so an
optimizer-step-to-generation-ready definition can add it exactly once.

## Present explanatory power and missing measurements

A one-model fixed-shape run cannot identify byte/bandwidth coefficients. On the
4B trace, univariate fits of train-to-rollout time have R-squared 0.079 for the
primary-rank trainer-sleep timer, 0.010 for weight-update time, and 0.019 for
node-wide used CPU memory. Subtracting those two primary-rank timers and fitting
a constant residual has negative R-squared. The current aggregate model thus
explains the mechanism but not the observed variance; critical-rank and
unobserved phase variation dominate.

The follow-up addresses this by recording the four train-to-rollout awaits
separately, the actor snapshot, every rank's before/after HBM log, actual SGLang
KV allocation, and a concurrent eight-GPU H2D/D2H benchmark on every node.
Still required are:

- readable matching HF and Megatron SFT checkpoints for both models;
- confirmation that 30B TP4/EP8 training and TP8 rollout is the intended
  topology, because changing it changes per-rank payloads;
- a short contemporary 4B run with the new phase contract on the same node
  pool, needed to separate code/time drift from model-size effects;
- at least one repeat per model/node allocation, or node-fixed placement, to
  estimate allocation variance;
- a small `SGLANG_MEM_FRACTION` sweep if the goal is to isolate the KV/graph
  allocation coefficient rather than only predict the production setting.

## Parallelism audit

The follow-up does not increase TP merely in proportion to total parameters.
The matched 32k-token 4B training recipe at TP2 has already observed only
0.086--0.113 GiB of instantaneous headroom around its largest 9.43-GiB
allocation. For dense 8B, TP4 makes the BF16 weight shard slightly smaller than
4B TP2 (3.81 versus 4.11 GiB) and reduces sequence-parallel activation width per
rank; TP2 would double the weight shard and materially increase activations.
TP4 is therefore the memory-equivalent starting point, not known spare TP.

For 30B-A3B, EP8/ETP1 shards the routed experts independently of dense TP. TP4
is retained for the 32k dense attention, vocabulary/log-prob, and activation
paths, not to shard the expert optimizer state. A TP2 probe is worthwhile after
checkpoint conversion because this switching-only recipe does not enable the
production training-entropy observation that caused the documented 4B peak.
It must pass a full-token HBM smoke and weight-update equality check before it
replaces TP4.

The rollout group sizes have an additional colocated-update constraint. The 8B
TP4 source needs a four-rank group to reconstruct all dense shards; the 30B EP8
source needs an eight-rank group to cover all experts. Reducing
`ROLLOUT_NUM_GPUS_PER_ENGINE` solely because one inference shard fits in HBM can
therefore make the weight handoff incomplete. It should only be changed with a
live `--check-weight-update-equal` validation, not from static HBM arithmetic.

Official architecture sources: [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json),
[Qwen3-8B safetensors index](https://huggingface.co/Qwen/Qwen3-8B/blob/main/model.safetensors.index.json),
[Qwen3-30B-A3B-Base config](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json), and
[Qwen3-30B-A3B-Base safetensors index](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/model.safetensors.index.json).
The [DGX H100 guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/dgxh100-user-guide.pdf)
documents the dual-CPU PCIe Gen5 host topology and 900-GB/s GPU-to-GPU NVLink
fabric; achievable simultaneous host/device bandwidth is measured by the job,
not assumed from that peak specification.
