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
