# math_sync / dapo-math

Colocated GRPO on DAPO-Math-17K, evaluated on AIME-2024 and AIME-2025.

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
normalization, which inflates the length of *wrong* answers. **AReaL** is not
implemented here — this task is the colocated baseline it is measured against;
see `math_async/dapo-math/` for the asynchronous variant.

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

Ordered by expected effect on this baseline. Change one class at a time; the
`batch` lane is for results, not for sweeps (see
[`miles-run-ladder`](../../../.claude/skills/miles-run-ladder/SKILL.md)).

### Batch shape — the largest gap to published work

| knob | default | search | why |
|---|---|---|---|
| `ROLLOUT_BATCH_SIZE` | 32 | **64, 128, 256, 512** | DAPO used 512. At 76% zero-variance groups, 32 prompts leaves ~8 contributing per step, which is where the 0.04 grad_norm comes from. This is the first thing to move. |
| `N_SAMPLES_PER_PROMPT` | 8 | **8, 16, 32** | At n=8 a 1/8 or 7/8 group carries only `sqrt(0.125·0.875)=0.33` of the maximum advantage. 16 is DAPO's value and halves the variance of the per-group estimate at 2× cost. |
| `GLOBAL_BATCH_SIZE` | 256 | keep = `rollout_batch × n / num_steps` | Not free: the four-knob invariant is asserted at startup. |
| `NUM_STEPS_PER_ROLLOUT` | 1 | **1, 2, 4** | >1 makes the later minibatch steps off-policy, and this task has **no** `--use-tis`, so only PPO clipping absorbs it. Watch `dump/mean_abs_lp_diff` as it rises. |
| `OVER_SAMPLING_BATCH_SIZE` | 2× rollout batch | **1.5×, 2×, 3×** | Higher means fewer resampling rounds but more aborted generations; partial rollout recovers those, so the cost is bounded. |

### Colocation

| knob | default | search | why |
|---|---|---|---|
| `ACTOR_NUM_NODES` × `ACTOR_GPUS_PER_NODE` | 2 × 8 | scale with the allocation | Colocated means the whole allocation trains and generates in turn; data parallelism is the remainder after `tp × cp`, so adding nodes shrinks the per-rank batch rather than making a step more informative. Raise `ROLLOUT_BATCH_SIZE` with it. |
| `SGLANG_MEM_FRACTION` | 0.7 | **0.6–0.85** | Under colocation this trades KV cache against training memory directly. The dashboard's advisory panel reads `sglang_token_usage` and `sglang_cache_hit_rate` back. |

The off-policy knobs (`MAX_WEIGHT_STALENESS`, the actor/rollout split,
`ASYNC_MAX_CONCURRENT_SAMPLES`, `--use-tis`) do not exist in this task. They are
what `math_async/dapo-math/` searches over; the comparison between the two
directories at an otherwise identical configuration is the point of having both.

### Optimization

| knob | default | search | why |
|---|---|---|---|
| `LR` | 1e-6 | **5e-7, 1e-6, 2e-6** | Matches DAPO already; move only after the batch shape is settled, since the two interact. |
| `--eps-clip-high` | 0.28 | **0.2 (symmetric), 0.28, 0.32** | DAPO's clip-higher exists to stop entropy collapse. Watch `dump/mean_entropy`. |
| `--kl-loss-coef` | 0.00 | **0, 1e-3** | Both DAPO and this recipe run without a KL penalty. Reintroduce it only if the policy drifts off the reference in a way the reward does not catch. |
| `--grpo-std-normalization` | on | on / off | Dr. GRPO's correction is to turn this off. It is the cheapest published change here and specifically targets length inflation on wrong answers — check `dump/response_length_mean` split by reward. |

### Generation

| knob | default | search | why |
|---|---|---|---|
| `MAX_RESPONSE_LEN` | 24576 | **8192, 16384, 24576, 32768** | Not a free parameter: measured on Qwen3-4B-Instruct-2507, an 8192 budget truncated 18 of 32 prompts, and a truncated sample scores 0 under every rule-based verifier, so a short budget biases against long-solution problems rather than adding noise. Watch `dump/truncated_frac`. Above 32768 also raise `ROLLOUT_MAX_CONTEXT_LEN` and re-check `max_tokens_per_gpu × cp`. |
| `--rollout-temperature` | 1 | **0.8, 1.0, 1.2** | Lower reduces the fraction of unanimous-wrong groups; higher increases spread and therefore usable advantage. Interacts directly with `zero_std_group_frac`. |

### Not a hyperparameter

`RM_TYPE` is a correctness setting, not something to sweep. `deepscaler` returns
0 unless the response contains a `</think>` delimiter, so a non-thinking
checkpoint needs `math`. The `qwen3-4b-instruct-2507` recipes already default to
it; everything else here is a thinking model.
