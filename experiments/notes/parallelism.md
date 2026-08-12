# Parallelism: why every recipe is CP=1 with the whole context in one budget

Context parallelism was never chosen for its own sake. The only constraint is

    max_tokens_per_gpu * context_parallel_size >= rollout_max_context_len (32768)

and CP was one way to satisfy it while keeping the per-GPU token budget small.
It is the wrong way. CP splits the sequence across GPUs and pays an all-to-all
in every layer, for models that fit on one GPU without it.

Measured on Qwen3-4B-Instruct-2507, async, 1 train node + 2 rollout nodes
(jobs 15150858 / 15150860 / 15150862 against 15150863):

| | `actor_train` | `log_probs` | logged TFLOP/s | worst-GPU free |
|---|---|---|---|---|
| tp2 cp2, mtpg 16384 | 42.7-44.5 s | 12.8-13.3 s | 164 | 11.3 GB |
| tp2 cp1, mtpg 32768 | **23.4-24.1 s** | **5.8 s** | **327** | **18.3 GB** |

1.9x on training and 2.3x on log-probs, *and* more free memory -- CP's own
buffers cost more than the smaller activation budget saved. Corrected for the
4/3 undercount in `train_metric_utils.py:41` (it charges 3x forward where full
recompute costs 4x), the real figures are 218 -> 436 TFLOP/s, i.e. **22% -> 44%
MFU** against the H100 dense peak of 989.

`--recompute-granularity selective` is not the next step: at this token budget it
OOMs (job 15150865, 79.1 of 79.1 GiB). The memory freed by dropping CP has
already been spent on the larger budget, which bought more than selective would.

## What is and is not measured

Only **Qwen3-4B-Instruct-2507 under async** is measured. Every recipe has been
set to CP=1 / mtpg 32768 for consistency, and `validate.py` still enforces the
`mtpg * cp >= 32768` invariant, but the memory headroom is a per-model question:

| model | TP | measured? | note |
|---|---|---|---|
| Qwen3-1.7B | 1 | no | tightest case -- TP=1 puts weights *and* optimizer on one GPU |
| Qwen3-4B | 2 | no | same shape as 4B-Instruct, should transfer |
| Qwen3-4B-Instruct-2507 | 2 | **yes** | 18.3 GB free at peak |
| Qwen3-8B | 4 | no | per-rank weights similar to 4B at TP=2 |
| Qwen3-30B-A3B | 4 | no | already CP=1; MoE, EP=1 |

Each needs one short run before its first production job. An OOM here is loud
and immediate, not silent, so the check is cheap.

## Colocated is a separate memory question

`--colocate` sets `offload_train` and `offload_rollout` to True
(`arguments.py:2971-2975`), so SGLang releases HBM before the training phase and
the trainer sees roughly the async profile. That is the reason to expect CP=1 to
transfer.

It is still not the same: colocated cycles offload/onload every step, and
fragmentation across that cycle is exactly what the old sync settings (cp4,
mtpg 9216 -- far more conservative than async's cp2/16384) look like an
accommodation for. **In flight: job 15168705**, colocated 2 nodes at cp1 /
mtpg 32768, which either OOMs in the first step or does not.

## Node balance is an async-only question

There is no train:rollout split to tune under `--colocate` -- the same GPUs do
both, in sequence. So the colocated arm of the study does not depend on the
async balance experiments and can proceed in parallel with them. The only shared
dependency was CP, and that is now settled.

## The expandable allocator is async-only (2026-08-07)

`--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'` fixed
the trainer OOM in `_VocabParallelEntropy.forward` (a 9.24 GiB fp32 logits copy
for one 32k sample, with 8.72 GiB free and 15.16 GiB fragmented). It belongs to
the async recipe only.

Under `--colocate` the trainer dies at startup:

```
RuntimeError: TorchMemorySaver is disabled for the current process because
expandable_segments is not supported yet.
```

`--colocate` sets `offload_train=True` (`arguments.py:2985`), and
`ray/train/actor_factory.py:44-49` then LD_PRELOADs
`torch_memory_saver_hook_mode_preload` into the trainer actor with
`TMS_INIT_ENABLE=1`. torch_memory_saver replaces the allocator's segment
handling to pause/resume HBM around the rollout phase, which is the same
mechanism `expandable_segments` claims, so the two are mutually exclusive. The
async trainer never sets `offload_train`, gets no LD_PRELOAD, and is unaffected.

Cost: job 15288321 (`conv-s0-tis-lr1e-6-32768-p1`) FAILED at 4:36, and
`afterany` released p2 into the same failure before it could be cancelled.
`validate.py` now rejects `expandable_segments` in any non-async recipe.

Resolved (job 15290984, colocated 4 nodes, n=16, gbs 3072, 32k): it does not
OOM, but the margin is thin and does not improve with scale.

| | alloc | device free | headroom after |
|---|---|---|---|
| production dp16 | 9.43 GiB | 9.54 GiB | **0.113 GiB** |
| production dp16 | 9.43 GiB | 10.30 GiB | 0.873 GiB |
| 2-node smoke dp8 | 9.28 GiB | 10.16 GiB | 0.875 GiB |
| 2-node smoke dp8 | 9.28 GiB | 9.37 GiB | **0.086 GiB** |

torch_memory_saver declines the allocation because granting it would breach its
1 GiB margin; torch's caching allocator then releases cached blocks and the
allocation succeeds. `CUDA out of memory` count is 0 in both runs.

Two corrections to what was expected here. Going dp8 -> dp16 was expected to free
~3 GB per GPU through the distributed optimizer; it does not show up as device
free at this instant, because torch holds it as cache -- the observed headroom is
the same in both. And the peak does **not** grow as responses lengthen:
`10122952704 / (151936 * 4 / 2) = 33313` tokens, which is
`--rollout-max-context-len 32768` plus padding, so the allocation is already at
its ceiling. Thin, but bounded.

If it ever does OOM, the levers that do not conflict with torch_memory_saver are
a lower `MAX_TOKENS_PER_GPU` and `--log-probs-chunk-size`, not the allocator --
and either one has to be applied to every arm, since both cost throughput and
the study compares arms on wall-clock.

### Colocated training entropy is opt-in (2026-08-12)

Job 15627089 failed in `_VocabParallelEntropy.forward` while allocating
9,965,666,304 bytes (9.28125 GiB): 32,768 response tokens times the local
TP2 vocabulary of 76,032 logits times fp32. `ENTROPY_COEF=0`, so
`--observe-training-entropy` was performing a detached diagnostic calculation;
it was not part of the loss or backward pass.

An exact dumped rollout batch from that run was replayed for one deterministic
optimizer step on 8 H100s with the same 32k token cap. The table reports the
largest `nvidia-smi memory.used` sample across the eight trainers. This replay
does not instantiate SGLang, so use the absolute values only to compare the
trainer-side alternatives, not as a production colocated capacity estimate.

| training entropy | log-prob chunk | max used | change from baseline |
|---|---:|---:|---:|
| yes | disabled | 67,666 MiB | -- |
| yes | 8,192 | 62,052 MiB | -5,614 MiB |
| no | disabled | 58,098 MiB | -9,568 MiB |
| no | 8,192 | 60,188 MiB | -7,478 MiB |

Disabling diagnostic entropy gave the largest observed reduction. Combining it
with chunking did not lower the total replay peak further, although chunking
still reduces the largest individual full-vocabulary allocation and can help
fragmentation. The sync colocated default is therefore entropy observation off
and chunking off. To run an entropy diagnostic safely, set
`OBSERVE_TRAINING_ENTROPY=1 LOG_PROBS_CHUNK_SIZE=8192`.

All four replays produced identical displayed loss, policy-gradient loss, TIS,
KL, ESS, and gradient norm. Their saved gradient-norm artifacts were also
byte-identical. A standalone tensor test found chunked versus unchunked maximum
log-prob error `9.54e-7`, relative gradient L2 error `2.39e-7`, and identical
entropy. With entropy disabled, the current metric reducer emits the placeholder
`train/entropy_loss=0`; that value means "not observed", not zero policy entropy.

### The colocated arm OOMs, and its SGLang fraction was the wrong way round (2026-08-08)

`conv-s0-tis-lr5e-6-p1` and `p2` both died with

    torch.OutOfMemoryError: Tried to allocate 9.28 GiB.
    GPU 7 has 79.11 GiB total, 8.55 GiB free.
    17.75 GiB is reserved by PyTorch but unallocated.

at roughly the same training step, so the chain would have burned all 18 jobs at
the same place.

`SGLANG_MEM_FRACTION` was **0.80 in the colocated recipe and 0.70 in the async
one**, which is backwards. Under `--fully-async` the rollout engines own their
GPUs outright; under `--colocate` the same device also holds the training
weights, optimizer state and activations. The setting that can afford to be
generous was the tight one. Colocated is now **0.65**, below async's 0.70.

That frees roughly 12 GiB against a shortfall of 0.73 GiB. The margin is
deliberately large because the previously measured headroom was **0.113 GiB**
(prod, above) -- there is nothing to absorb growth -- and because response length
climbs throughout a run, so the allocation that fits at rollout 10 need not fit
at rollout 200.

The cost is a smaller KV cache, so fewer concurrent sequences and a slower
rollout. Colocated is already the slowest arm (604 s/step measured) and carries
an 18-job chain, so it absorbs this; the alternative levers do not exist.
`MAX_TOKENS_PER_GPU` is an experimental variable and changing it for one arm
would confound the comparison, and `expandable_segments` -- which the error
message itself recommends -- is unavailable here: `--colocate` sets
`offload_train=True`, which LD_PRELOADs torch_memory_saver, which refuses it.

**This is a robustness fix, not a cure.** The OOM was downstream of a training
collapse at lr 5e-6 (see notes/algorithm-ablation.md): response length tripled,
so the activations grew until they did not fit. At a learning rate where the run
does not diverge, 0.65 should hold; at one where it does, no fraction will.
