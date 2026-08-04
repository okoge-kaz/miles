# difficulty_filter

Select training prompts by the pass rate a *specific policy* achieves on them.

## Why

The first math runs on this cluster produced a flat learning curve, and the
rollout metrics say why:

```
raw_reward       0.84, flat over 46 steps
zero_std         21.6 of 32 groups all-correct, 2.4 all-wrong   (~76% wasted)
grad_norm        0.04
AIME24 avg@16    0.706 (step 0) -> 0.692 (step 19)
```

Three quarters of every rollout batch carried no gradient at all. In GRPO the
advantage is proportional to the group standard deviation, which for binary
rewards is exactly `sqrt(p * (1 - p))` for pass rate `p` — so a unanimous group
contributes literally nothing while costing a full group of generation.

Difficulty here is a property of the *(prompt, policy, sampling-params)* triple,
never of the prompt alone. DAPO-Math-17K is not "easy"; it is easy *for
Qwen3-4B (thinking)*, which is sitting at its post-training RLVR fixed point.
Everything in this directory is therefore keyed to a measured policy.

## Two ways to filter, and when each wins

| | online (`--dynamic-sampling-filter-path`) | offline (this directory) |
|---|---|---|
| when the cost is paid | every rollout, forever | once per (dataset, policy) |
| measured cost | rollout time 253 s -> 761 s (**3x**) | one inference job |
| adapts as the policy improves | yes | no — re-measure when stale |

For a fixed prompt set visited repeatedly, offline wins outright. The online
filter is the right tool when the policy has moved far enough that a cached
measurement no longer describes it.

## Files

| file | role |
|---|---|
| `pass_rate.py` | pure core: pass rate, the window test, the group-std identity, summary stats |
| `filters.py` | `check_pass_rate_window`, mountable via `--dynamic-sampling-filter-path` |
| `measure_pass_rate.py` | the expensive half: generate `k` samples per prompt, score, record |
| `apply_filter.py` | the cheap half: turn a measurement + a window into a prompt file |
| `run_measure.sbatch` | Slurm entry point (SGLang on 8 GPUs, then the driver) |

## Relationship to the built-in filters

`filters.check_pass_rate_window` follows the conventions in
`miles/rollout/filter_hub/dynamic_sampling_filters.py` exactly — the
`(args, samples, **kwargs)` signature, the `DynamicFilterOutput(keep, reason)`
return, and the same `_flatten_samples` handling for multi-turn generate
functions that return a list per group element.

The built-in `check_reward_nonzero_std` is the **limiting case** of this filter,
with the window `(0, 1)` open: it keeps every group that is not unanimous. The
window form additionally drops the near-degenerate groups a std test cannot
distinguish from informative ones — at `n_samples_per_prompt=8`, a 1/8 or 7/8
group passes a zero-std test but carries only `sqrt(0.125 * 0.875) = 0.33` of
the maximum advantage magnitude.

Drop reasons are bucketed (`pass_rate_all_wrong`, `pass_rate_too_low`, …) rather
than emitted as raw floats, because `base_types.MetricGatherer` creates one
counter series per distinct reason string.

## The verifier must match the policy's output format

`--rm-type deepscaler` grades the boxed answer exactly like `--rm-type math`,
but gates on a thinking delimiter first (`rm_hub/deepscaler.py:36-44`):

```python
if "</think>" in response:      model_solution = response.split("</think>")[-1]
elif "###Response" in response: model_solution = response.split("###Response")[1]
else:                           return 0        # <- no delimiter, no credit
```

A **non-thinking** policy never emits `</think>`, so every response scores 0 no
matter how correct. Measured here: Qwen3-4B-Instruct-2507 answered
`Answer: \boxed{37}` to a prompt labelled `37` and scored 0 under `deepscaler`.
Nothing errors — the sweep completes and reports a clean, entirely false
`mean_pass_rate 0.0`.

| policy | verifier |
|---|---|
| Qwen3-4B (hybrid thinking) | `deepscaler` or `math` |
| **Qwen3-4B-Instruct-2507** (non-thinking) | **`math`** |

This is not only a measurement concern. **Every `math_sync` / `math_async` /
`tool_multiturn` recipe ships `--rm-type deepscaler` except the
`qwen3-4b-instruct-2507` recipes, which default to `math` for exactly this
reason.** Pointing any of the others at a non-thinking checkpoint without
changing `--rm-type` produces reward ≡ 0, zero advantage, and a run that looks
like a model failing to learn rather than a misconfiguration.

`measure_pass_rate.py` therefore runs `verifier_preflight` before generating
anything: it scores a correct boxed answer with and without a `</think>`
delimiter plus one clearly wrong answer, and aborts with the fix when only the
thinking form is graded. It costs three verifier calls and no GPU time.

```
verifier preflight (--rm-type math): {'plain_boxed': 1.0, 'thinking_boxed': 1.0, 'clearly_wrong': 0.0}
```

## Calibration rules

Two more ways to get a filter that lies to you:

1. **Grading with a different verifier than the trainer uses.** Avoided by
   routing every response through `miles.rollout.rm_hub.batched_async_rm` with
   the recipe's own `--rm-type`, rather than reimplementing the check. That
   keeps the `boxed_` prefix handling and the `</think>` split inside the
   deepscaler reward identical on both sides.
2. **Measuring with a shorter generation budget than training.** A truncated
   sample scores 0 under every rule-based verifier, so a short
   `--max-new-tokens` does not add noise — it adds a *directional* bias that
   marks long-solution problems as too hard and drops exactly the prompts a math
   curriculum most wants to keep. Averaging cannot recover from a bias.

   `--max-new-tokens` therefore defaults to **24576**, matching
   `--rollout-max-response-len` in `experiments/math_sync`. Measured on
   Qwen3-4B-Instruct-2507 over DAPO-Math, an 8192 budget truncated **18 of 32**
   prompts (median mean response 6762 tokens); raising it to 24576 dropped
   truncation to 3.5%. `truncated_frac` is recorded per prompt and reported in
   the summary — check it before trusting a window.

`--n-samples` and `--temperature` default to 8 and 1.0 to match
`experiments/math_sync`. The measured pass rate is then the same statistic the
trainer sees per group, so a window maps onto training batches with no
rescaling.

## Usage

Measure (inference only — needs the HF weights, not the torch_dist checkpoint,
so it can run before or alongside training):

```bash
sbatch -A coreai_horizon_dilations \
  experiments/src/difficulty_filter/run_measure.sbatch
```

Resumable: results are appended and flushed per prompt and a rerun skips indices
already present, so hitting the 4 h wall just means resubmitting.

Then select a window (seconds, no GPU) — this is why the measurement records the
pass rate instead of a keep/drop decision:

```bash
python -m experiments.src.difficulty_filter.apply_filter \
  --prompt-data  $DATASET_DIR/dapo-math-17k/dapo-math-17k.jsonl \
  --pass-rates   $DATASET_DIR/difficulty/dapo-math-17k.Qwen3-4B-Instruct-2507.passrate.jsonl \
  --output       $DATASET_DIR/dapo-math-17k/dapo-math-17k-p20-80.jsonl \
  --pass-rate-min 0.2 --pass-rate-max 0.8 \
  --policy Qwen3-4B-Instruct-2507
```

The output keeps every input line verbatim and appends a `difficulty` block, so
it stays a drop-in `--prompt-data` while remaining traceable to its measurement.

Use it by pointing a recipe at the filtered file:

```bash
experiments/submit_training.sh math_sync/dapo-math/qwen3-4b-instruct-2507 filtered-math \
  --export=ALL,PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k-p20-80.jsonl
```
