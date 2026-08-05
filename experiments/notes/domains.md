# Domain characteristics, for the off-policy staleness sweep

Reference table for choosing which domains to sweep and for reading the results.
Everything here is **measured on Qwen3-4B-Instruct-2507** (n=8, temperature 1.0,
max_new_tokens 24576) unless a cell says otherwise — a pass rate is a property of
the (prompt, policy, sampling-params) triple, not of the dataset.

Input lengths are token counts after `apply_chat_template` (tools included where
the dataset ships them), over the first 1500 rows.

## The table

| domain | training set | eval set | multi-turn | reward density | input p50 / p90 / max | output p50 / p90 / mean | pass rate | zero-std |
|---|---|---|---|---|---|---|---|---|
| math | dapo-math-17k (17,398) | AIME24, AIME25 | no (1 turn) | sparse: 0/1 after long CoT | 131 / 198 / 517 | 3,493 / 8,782 / 4,347 | 0.770 | 66.3% |
| math | skywork-or1-math (105,045) | AIME24, AIME25 | no | sparse | 135 / 204 / 977 | — | — | — |
| math | nemotron-rl-math-v2 (7,732) | AIME24, AIME25 | no | sparse (judge-intended) | 110 / 197 / 1,230 | — | — | — |
| math + tool | dapo-math-17k (ReTool) | AIME24, AIME25 | **yes,真** | sparse + tool-use bonus | 131 / 198 / 517 | (longer: interleaved tool output) | — | — |
| knowledge | knowledge-mcqa (685,573) | MMLU-Pro | no | **dense**: 4-way, chance 0.25 | 269 / 394 / 1,314 | 406 / 1,548 / 805 | 0.462 | **34.0%** |
| reasoning | reasoning-gym (14,259) | ARC-AGI validation | no | sparse, bimodal per env | 117 / 405 / 9,338 | 2,059 / 12,865 / 4,419 | 0.528 | 63.5% |
| instruction following | instruction-following (46,391) | IFBench | no | medium: AND over constraints | 92 / 343 / 1,410 | — | — | — |
| structured output | structured-outputs (9,437) | (train split) | no | medium: schema validates or not | 1,307 / 2,245 / 3,601 | 417 / 1,019 / 535 | 0.663 | 86.5% |
| tool (single) | fncall-pivot (9,620) | BFCL | context only | dense: name+args match | **3,802 / 12,127 / 27,303** | 158 / 1,052 / 407 | 0.564 | **96.0%** |
| tool (conversational) | conv-tooluse (96,968) | BFCL, τ-bench | context only | dense | 3,365 / 4,368 / 7,005 | 96 / 260 / 121 | 0.314 | 88.0% |
| SWE | SWE-Pivot-v1 | — | context only | dense | — | — | — | — |
| code | competitive_coding (~41k) | LiveCodeBench-v6 | no | sparse: all tests pass | — | — | — | — |

## What "multi-turn" means here, and why it matters

Only **ReTool** is multi-turn in the sense the staleness question cares about:
the policy acts, a Python interpreter runs, and the *next* observation depends on
what the policy did. Turn count is a property of the trajectory.

`fncall-pivot` and `conv-tooluse` look multi-turn — their prompts average 17.4
and 6.9 messages — but each row is a **single-step decision**: reproduce the
expert's next action given a fixed conversation prefix. Nothing the policy emits
changes the observation. NVIDIA built them that way on purpose (the Gym
environment is literally named
`single_step_tool_use_with_argument_comparison`), and it is why they need no
sandbox. For measuring how off-policy staleness interacts with *trajectory
length*, they are long-context single-step tasks, not multi-turn ones.

So the controlled comparison for the multi-turn axis is:

```
dapo-math-17k, single turn          →  math_sync recipe
dapo-math-17k, ReTool multi-turn    →  tool_multiturn recipe, --generate-max-turns N
```

Same prompts, same answer, same verifier family — only the turn structure
differs. Sweeping `--generate-max-turns` gives turn count as a clean independent
variable.

## Reward density, ordered

Density is what determines how much signal survives a stale gradient, so it is
the second axis worth sweeping:

```
dense   knowledge-mcqa   34.0% zero-std   short outputs, chance floor 0.25
        instruction-following            AND over several checkable constraints
        reasoning-gym    63.5%           per-environment, strongly bimodal
        math (DAPO)      66.3%           0/1 after a long chain
        structured-out.  86.5%
        conv-tooluse     88.0%
sparse  fncall-pivot     96.0% zero-std   exact name+arguments match
```

`zero-std` is the fraction of 8-sample groups that are unanimous and therefore
contribute no GRPO gradient at all. **knowledge-mcqa is the only set that gives a
usable gradient out of the box**; everything else needs the pass-rate filter, and
`fncall-pivot` at 96% needs it badly.

Note the two are not the same thing: `conv-tooluse` has a *dense* reward (a short
exact match, no long chain to get through) but a *degenerate* distribution (0.314
mean with 88% unanimous). Density describes the reward function, zero-std
describes this policy's competence on it.

## Sequence-length regimes

Three distinct regimes, which matter because KV pressure and rollout latency
drive how much staleness a fully-async setup actually produces:

- **short in, long out** — math, reasoning-gym. ~130 in, 3.5k–13k out. Rollout
  time dominated by decode; this is where the 24,576 budget is needed.
- **short in, short out** — knowledge-mcqa, instruction-following. Cheapest to
  sweep, most steps per GPU-hour.
- **long in, short out** — tool-use, structured-outputs. `fncall-pivot` reaches
  **27,303 input tokens** (p90 12,127) because the whole conversation plus every
  tool signature is in the prompt, while the answer is one call. Prefill-bound,
  and the only sets that risk the 32,768 context limit in
  `--rollout-max-context-len`.

`fncall-pivot`'s p99 of 22.5k against a 32,768 context leaves ~10k for
generation. Raise `--rollout-max-context-len` before training on it.

## Verifier per dataset

| dataset | verifier | notes |
|---|---|---|
| dapo-math-17k, skywork-or1-math | `--rm-type math` | `deepscaler` returns 0 for a non-thinking policy |
| nemotron-rl-math-v2 | `--rm-type math` | NeMo-RL uses `math_with_judge`; ~10% of labels are prose a rule cannot grade |
| knowledge-mcqa | `--rm-type gpqa` | grades `Answer: C`; `plain_boxed` scores 0 |
| instruction-following | `--rm-type ifbench` | clones allenai/IFBench at runtime; needs nltk, langdetect, immutabledict, absl-py, emoji |
| reasoning-gym | `rewards.reasoning_gym_reward` | literal answer, normalised |
| structured-outputs | `rewards.structured_output_reward` | JSON Schema; some rows `$ref` into Draft-7 `definitions` |
| fncall-pivot, conv-tooluse, SWE-Pivot | `rewards.tool_call_match_reward` | same verifier NeMo-RL uses for all three |
| ReTool | `examples.retool_v2.tool_sandbox.reward_func` | returns a dict — needs `--reward-key score` |

## External difficulty labels, for validating our own measurement

Three sets ship a pass rate measured by someone else, which is the only
independent check we have on the measurement pipeline:

| dataset | reference | detail |
|---|---|---|
| skywork-or1-math | DeepSeek-R1-Distill 1.5B / 7B / 32B | `metadata.model_difficulty` |
| knowledge-mcqa | Qwen3-30B-A3B | `metadata.reward_profiles`, 5 generations |
| conv-tooluse | Qwen3-235B | `metadata.pass_rate`, **32 samples** — finer than our 8 |
