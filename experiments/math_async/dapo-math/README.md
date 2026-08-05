# math_async / dapo-math

Fully-async GRPO on DAPO-Math-17K, evaluated on AIME-2024 and AIME-2025.

Every `<task>/<dataset>/` directory carries a README in this shape — published
reference points first, then the range each hyperparameter is worth searching
over. See [the contract](../../README.md#dataset-directories).

## Prior work

| system | base model | reported result | reference |
|---|---|---|---|
| DAPO | Qwen2.5-32B | **AIME 2024 = 50** | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| GRPO (DeepSeekMath) | DeepSeekMath 7B | MATH 51.7 (60.9 with self-consistency @64) | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| Dr. GRPO | 7B base | **AIME 2024 = 43.3** | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| AReaL | — | 2.77× training speedup vs. synchronous at equal GPU count | [arXiv:2505.24298](https://arxiv.org/abs/2505.24298) |
| Qwen3 (the base models here) | 0.6B–235B, dense + MoE | thinking / non-thinking in one model | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |

Three of these set the algorithm this recipe runs. **DAPO** contributes decoupled
clipping (`eps_clip` ≠ `eps_clip_high`) and dynamic sampling, both on by default
here. **Dr. GRPO** is the argument against GRPO's length and standard-deviation
normalization, which inflates the length of *wrong* answers. **AReaL** is the
async design this task implements: a staleness-aware PPO variant with an explicit
bound on how old a sample may be, which is what `--max-weight-staleness` is.

> Provenance: the titles, the headline scores and the algorithmic claims above
> were read from the arXiv abstracts. The per-hyperparameter values in the next
> table come from the papers' bodies and are **not** re-verified — treat them as
> a starting point to check, not as citations.

## Reference hyperparameters vs. this recipe

| knob | DAPO | this recipe | note |
|---|---|---|---|
| group size (`N_SAMPLES_PER_PROMPT`) | 16 | 8 | DeepSeekMath used 64 |
| prompts per rollout (`ROLLOUT_BATCH_SIZE`) | 512 | 32 | the biggest gap |
| learning rate (`LR`) | 1e-6, constant | 1e-6, constant | matches |
| clip low / high (`--eps-clip` / `-high`) | 0.2 / 0.28 | 0.2 / 0.28 | matches |
| KL coefficient | 0 | 0 | matches |
| max response (`MAX_RESPONSE_LEN`) | 20480 (16384 + 4096 overlong buffer) | 24576 | — |
| dynamic sampling | yes | yes | always on |

## Local baseline

Measured on this cluster, Qwen3-4B (thinking) on DAPO-Math, colocated:

```
raw_reward       0.84, flat over 46 steps
zero_std         21.6 of 32 groups all-correct, 2.4 all-wrong   (~76% wasted)
grad_norm        0.04
AIME24 avg@16    0.706 (step 0) -> 0.692 (step 19)
rollout time     253 s -> 761 s with dynamic sampling on (3x)
```

Qwen3-4B sits at its post-training RLVR fixed point on this dataset. It is a
collapse detector, not a model with headroom — use `qwen3-4b-instruct-2507` when
the question is whether something *learns*.

## What to search, and over what range

A hyperparameter here is any value that can move **throughput** or **downstream
performance** — batch shape, off-policy step count, learning rate, context
length, the GPU split, the engine geometry. Every row says which of the two it
moves, because that decides which lane it is swept in: `quality` and `both` rows
change what is learned and invalidate a comparison, `throughput` rows do not.
See [`miles-run-ladder`](../../../.claude/skills/miles-run-ladder/SKILL.md).

### Batch shape — the largest gap to published work

| knob | default | search | affects | why |
|---|---|---|---|---|
| `ROLLOUT_BATCH_SIZE` | 32 | **64, 128, 256, 512** | both | DAPO used 512. At 76% zero-variance groups, 32 prompts leaves ~8 contributing per step, which is where the 0.04 grad_norm comes from. This is the first thing to move. Larger also amortises the weight sync over more generation. |
| `N_SAMPLES_PER_PROMPT` | 8 | **8, 16, 32** | both | At n=8 a 1/8 or 7/8 group carries only `sqrt(0.125·0.875)=0.33` of the maximum advantage. 16 is DAPO's value: half the variance of the per-group estimate at 2× the generation cost. |
| `GLOBAL_BATCH_SIZE` | 256 | derived | quality | Must equal `rollout_batch × n / num_steps`; the invariant is asserted at startup. Also has to be divisible by `dp`, which grows with the allocation. |
| `NUM_STEPS_PER_ROLLOUT` (off-policy step) | 1 | **1, 2, 4** | both | >1 splits one rollout into several optimizer steps, so the later ones are off-policy. Throughput improves because generation is amortised over more updates; `--use-tis` is on in this task, so the ratio is corrected. Watch `dump/mean_abs_lp_diff` rise with it. |
| `OVER_SAMPLING_BATCH_SIZE` | 2× rollout batch | **1.5×, 2×, 3×** | throughput | Higher means fewer resampling rounds but more aborted generations; partial rollout recovers those, so the cost is bounded. |

### Async / off-policy — what this task exists to study

| knob | default | search | affects | why |
|---|---|---|---|---|
| `MAX_WEIGHT_STALENESS` | 2 | **1, 2, 4, unset** | both | AReaL's central quantity. Looser lets generation overlap more weight syncs (throughput) at the cost of training on older samples (quality). `unset` is miles' own default and means *no bound* — the upper end of the sweep, not a safe baseline. Read `dump/mixed_version_frac` and per-sample `weight_version_min`. |
| actor / rollout split | 1 node + 1 node | **1+1, 1+3, 2+2, 3+1** | throughput | Whichever side is starving decides. `async/queue_depth` growing means training is the bottleneck; sitting at 0 means rollout is. |
| `ASYNC_MAX_CONCURRENT_SAMPLES` | unset (= one training batch) | **1×, 2×, 4× `rollout_batch × n`** | both | Decouples generation concurrency from the training batch. Raising it fills the engines but raises average staleness. |
| `--use-tis` | on | on / off | quality | Off only as a controlled ablation, to measure what the correction is worth at a given staleness. |

### Optimization

| knob | default | search | affects | why |
|---|---|---|---|---|
| `LR` | 1e-6 | **5e-7, 1e-6, 2e-6** | quality | Matches DAPO already. Move only after the batch shape is settled — the effective step size depends on both. |
| `--eps-clip-high` | 0.28 | **0.2 (symmetric), 0.28, 0.32** | quality | DAPO's clip-higher exists to stop entropy collapse. Watch `dump/mean_entropy`. |
| `--kl-loss-coef` | 0.00 | **0, 1e-3** | quality | Both DAPO and this recipe run without a KL penalty. Reintroduce only if the policy drifts off the reference in a way the reward does not catch. |
| `--grpo-std-normalization` | on | on / off | quality | Dr. GRPO's correction is to turn this off. The cheapest published change here, targeting length inflation on wrong answers — check `dump/response_length_mean` split by reward. |

### Generation and context

| knob | default | search | affects | why |
|---|---|---|---|---|
| `MAX_RESPONSE_LEN` | 24576 | **8192, 16384, 24576, 32768** | both | Not free in either direction. Measured on Qwen3-4B-Instruct-2507, an 8192 budget truncated 18 of 32 prompts, and a truncated sample scores 0 under every rule-based verifier — so a short budget *biases* against long-solution problems rather than adding noise. Longer costs generation time roughly linearly and activation memory through `max_tokens_per_gpu × cp`. Watch `dump/truncated_frac`. |
| `ROLLOUT_MAX_CONTEXT_LEN` | 32768 | **16384, 32768, 65536** | both | The hard cap on prompt + response, and the number `MAX_TOKENS_PER_GPU × cp` has to clear (`data.py:473`). Raising it without raising that product fails at startup; raising both costs activation memory and KV cache. Below `MAX_RESPONSE_LEN` + the longest prompt it silently truncates. |
| `--rollout-temperature` | 1 | **0.8, 1.0, 1.2** | quality | Lower reduces unanimous-wrong groups; higher increases spread and therefore usable advantage. Interacts directly with `zero_std_group_frac`. |

### Throughput only

These change how fast a step is, never what it learns, so they are swept in the
`batch_short` lane and then frozen.

| knob | default | search | why |
|---|---|---|---|
| `MAX_TOKENS_PER_GPU` | per model | up until OOM | Fewer microbatches per step. Must keep `MAX_TOKENS_PER_GPU × cp ≥ ROLLOUT_MAX_CONTEXT_LEN`. Usually the cheapest win. |
| `TENSOR_PARALLEL_SIZE` / `CONTEXT_PARALLEL_SIZE` | per model | powers of 2 whose product divides the training GPUs | Sets `dp` as the remainder, which then has to divide `GLOBAL_BATCH_SIZE`. CP is what makes a long response fit at all. |
| `EXPERT_PARALLEL_SIZE` | per model | MoE only | `qwen3-30b-a3b` only. |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | per model | 1, 2, 4, 8 | Smaller engines mean more of them and more concurrency; larger ones are needed once the weights stop fitting. |
| `SGLANG_MEM_FRACTION` | 0.7 | **0.6–0.85** | KV cache size. The dashboard advisory panel reads `sglang_token_usage` and `sglang_cache_hit_rate` back. |
| `SGLANG_MAX_RUNNING_REQUESTS`, `SGLANG_CUDA_GRAPH_MAX_BS` | unset | from the observed peak | Capturing CUDA graphs above the concurrency actually reached costs startup time and memory for nothing. |

### The wider variable space

The full catalogue for the off-policy study — including what miles rejects at
startup, the algorithm/clip/IS surface, what is held fixed and why, and what has
to be recorded for a sample-efficiency claim later — is in
[`notes/off-policy-variables.md`](../../notes/off-policy-variables.md).

### Not a hyperparameter

`RM_TYPE` is a correctness setting, not something to sweep. `deepscaler` returns
0 unless the response contains a `</think>` delimiter, so a non-thinking
checkpoint needs `math`. The `qwen3-4b-instruct-2507` recipes already default to
it; everything else here is a thinking model.
