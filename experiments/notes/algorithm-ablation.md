# RL algorithm ablation

Runs inside the environment frozen by [node-ratio-procedure.md](node-ratio-procedure.md). The
algorithm set is not settled yet; this records the frame it will be run in, what
miles can express today, and what each arm costs, so that deciding the set is a
choice about science rather than about plumbing.

## The environment is frozen first, and does not move between arms

| fixed by | quantity |
|---|---|
| `node-ratio-procedure.md` | train:rollout split, staleness levels worth running |
| [rollout-scaling.md](rollout-scaling.md) | `SGLANG_MEM_FRACTION` — async 0.70, colocated 0.80 |
| the study design | rbs 256, n 8, gbs 2048, `MAX_RESPONSE_LEN` 32768, LR 1e-6, 400 rollouts |

Every arm runs at the same total GPU count, because the headline metric is
wall-clock to a quality target and a different allocation would change it for
reasons that have nothing to do with the algorithm.

## Each algorithm needs its own on-policy reference

This is the cost that makes the grid expensive, and it is not optional.
`tau_m(delta)` is defined against `Q_on*`, the converged quality of the
*on-policy* run — and that quality is a property of the algorithm, not of the
task. DAPO's plateau is not GSPO's plateau. So one reference per algorithm, run
colocated, at staleness 0 and one step per rollout.

    arms per algorithm = 1 reference (colocated) + |staleness levels| (async)

Sharing one reference across algorithms would silently compare each arm to
another algorithm's ceiling.

## What miles can express today

`RL_ALGORITHM` is derived from these, so each combination is already a distinct
checkpoint path and wandb group ([off-policy-variables.md](off-policy-variables.md)):

| knob | values | status |
|---|---|---|
| `ADVANTAGE_ESTIMATOR` | `grpo`, `gspo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `ppo` | implemented |
| `EPS_CLIP` / `EPS_CLIP_HIGH` | DAPO clip-higher is 0.2 / 0.28 | implemented |
| `EPS_CLIP_C` | dual-clip PPO, off by default | implemented |
| `IS_CORRECTION` | `none`, `tis`, `icepop`, `mis` | implemented |
| `TIS_CLIP` / `TIS_CLIP_LOW` | truncation bounds | implemented |
| `MIS_PROFILE` | token/sequence level, truncate/clip/mask, rejection sampling | implemented, YAML-driven |
| `RATIO_DENOMINATOR` | `actor`, `rollout-logprobs`, `old-actor` | implemented |
| `USE_OPSM` / `OPSM_DELTA` | off-policy sequence masking | implemented |

**CISPO, VCPO and M2PO are not in miles.** They have to be written before they
can be run, and writing a loss is not a configuration change — it needs its own
correctness check against the paper before any arm using it means anything.

## The trap this frame is designed to avoid

TIS is not a nuisance parameter: it *is* the off-policy correction, so whether an
arm survives at a given staleness is largely its doing. `--tis-clip` sat on
argparse's default of 2.0 until 2026-08-06, which put an unrecorded knob directly
on the axis being measured. It is now explicit in the recipes and in the
directory name.

Whether it interacts with the response-length axis is **an open empirical
question, not an assumption**. An earlier revision of the notes asserted a strong
interaction from the ESS of importance weights decaying in sequence length; that
argument does not apply here, because `--use-tis` is token-level
(`loss_hub/corrections.py:7`) and takes no product over the sequence. The
instrument already exists — `train/tis_clipfrac`, `train/tis`, `train/tis_abs`
are logged per step. **Read clipfrac across the length arms before crossing
`TIS_CLIP` with anything.** Multiplying the grid for a mechanism that has not
been shown to exist is the expensive mistake here.

## Before any arm runs

1. The environment is frozen and written down, with the job ids it came from.
2. The staleness levels are the measured ones. A bound above the observed natural
   maximum never binds, so its arm is bit-for-bit the uncapped run and is dropped
   rather than run — this was already true of 8, 16 and 32 in the original plan.
3. Any algorithm not already in miles is implemented and checked first.
4. `sweep.py` refuses a knob no recipe consumes. An unconsumed knob produces a
   grid of runs that differ only in their directory name.

## Reporting

Per arm, against its own algorithm's reference: `tau_m(delta)`, `S_m(delta)`, the
speedup profile over `q_p`, and the realized `P(L)` histogram — the last one
because the configured bound and the realized lag are different quantities, and
conflating them is the specific gap this study exists to close.

## The on-policy reference is invariant to ICEPOP, but not to MIS (2026-08-07)

Tier 2 compares TIS, ICEPOP and sequence-level MIS. It carries no s=0 **icepop**
arm, because at token level TIS and ICEPOP are the same loss on the colocated
on-policy run. It does carry a s=0 **mis** arm -- see the sequence-level section
below, which is why.

Measured on job 15290984 (colocated, lr 1e-6, 32k), first four training steps:

```
train/tis           0.9999980     mean importance weight
train/tis_abs       0.00941       mean |ratio - 1|
train/tis_clipfrac  4.85e-06      fraction of tokens the clip touched
```

TIS clips five tokens in a million. ICEPOP masks tokens outside a band and MIS
masks whole sequences; against a ratio distribution this tight, all three pass
every token through. The three on-policy curves would differ only by the
framework's own numerical noise, so tier 1's s=0 run is the reference for tier 2
as well -- the tiers share lr 1e-6 and 32k, so the comparison is exact.

This removes the two most expensive arms in the study: a colocated arm runs
60-71 h against the async arms' 25-35 h.

### Sequence-level does not follow from token-level

The per-token log-ratio is not zero-mean noise. It is a systematic -5.1e-04,
identical on every arm including the on-policy one, and it multiplies by the
sequence length:

| arm | length | per-token log-ratio | sequence log-ratio | sequence ratio |
|---|---|---|---|---|
| s=0 colocated | 6309 | -5.117e-04 | -3.23 | 0.040 |
| s=1 async | 6748 | -5.071e-04 | -3.42 | 0.033 |
| s=2 async | 7114 | -5.022e-04 | -3.57 | 0.028 |
| s=4 async | 7066 | -4.906e-04 | -3.47 | 0.031 |

(from `rollout/log_probs - rollout/rollout_log_probs` times
`rollout/response_lengths`.)

Against `seq-mask.yaml`'s `[0.5, 2.0]` those raw ratios are two orders of
magnitude low, so **unnormalized sequence-level masking would mask every
sequence and zero the loss**. It gets worse as responses lengthen: -2.48 at 4737
tokens, -3.57 at 7114, and around -16 at 32k. `tis_batch_normalize: true`
divides out exactly this batch-common component and is therefore not optional --
it is what makes the profile runnable at all. **Verify on a short run that the
normalized sequence ratios land inside the bounds before spending tier 2 on it.**

Because the effect is the framework's numerical mismatch rather than staleness,
it is present on the on-policy arm at full strength. A sequence-level correction
therefore does something on the s=0 arm that a token-level one does not, and the
s=0 mis arm is not redundant with the s=0 tis arm.

### A caveat on the staleness axis, still unresolved

The token-level statistic is flat across the staleness axis at this learning
rate:

| arm | tis mean | tis_abs max | clipfrac max |
|---|---|---|---|
| s=0 colocated | 0.9999980 | 0.00941 | 4.85e-06 |
| s=1 async | 0.9999999 | 0.01021 | 4.86e-06 |
| s=2 async | 1.0000033 | 0.01060 | 4.59e-06 |
| s=4 async | 0.9999988 | 0.01032 | 4.74e-06 |

At lr 1e-6 the policy moves so little per step that a lag of 4 still leaves
`pi_train / pi_rollout` within 1% of unity, and the clip fraction does not
separate the arms at all.

**This is measured over rollouts 0-11 of 300 and is not yet evidence about the
run.** Off-policy divergence is expected to appear later, not now: early on the
advantages are weak so each step moves the policy little, the entropy is still
high so the distribution is flat, and the responses are short. All three trend
the other way as training proceeds -- `response_len/mean` has already gone
4737 -> 7114. The right time to ask whether the corrections separate is when
`tis_clipfrac` starts to move, which has not happened yet.

Watch `train/tis_clipfrac` and `train/tis_abs` over tier 1. If they are still at
5e-06 and 1% at rollout 100-150, then tier 2 at lr 1e-6 would compare three
corrections that never fire, and it is worth more at lr 5e-6. Do not reorder the
tiers before that evidence exists.

### seq-mask's bounds are wrong, and the failure is documented (2026-08-07)

`experiments/configs/mis/seq-mask.yaml` carries `tis_lower_bound: 0.5` /
`tis_upper_bound: 2.0`. Those are token-level numbers and they will mask
everything.

Two independent confirmations:

- **Measured here.** The sequence-level ratio on the running arms is 0.03-0.08
  ([telemetry.md](telemetry.md)), two orders below 0.5, and it deepens with length --
  -2.48 at 4737 tokens, -3.57 at 7114.
- **Observed in VCPO** (arXiv:2602.17616, Figure 4): "Most baselines lead to
  training collapse (or crash, e.g. **Geometric MIS masks all sequences and has
  no loss**)" -- at 2048 tokens, where the effect is roughly a third of ours.

VCPO's own sweep (Appendix E.2) settles on **sequence-level TIS with c = 8.0**
as the threshold that survives longest among masking/clipping methods. That is
the order of magnitude to start from, not 2.0.

Fix the bounds against arXiv:2512.02556 section 3.1 before tier 2, and check
whether `tis_batch_normalize: true` is what the DeepSeek formulation actually
does -- if it divides out the batch-common component, the bound applies to a
normalized ratio and the numbers above do not transfer directly.

### What is actually wired, and what has never been executed (2026-08-07)

Static audit of the three corrections tier 2 needs.

| Method | Reachable | Path | Exercised |
|---|---|---|---|
| TIS | yes | `--use-tis`, default `vanilla_tis_function` | tier 1, all four arms |
| ICEPOP | yes | `--use-tis --custom-tis-function-path miles.backends.training_utils.loss_hub.corrections.icepop_function` | unit test only |
| MIS (sequence mask) | yes | `--custom-tis-function-path examples.…mis.compute_mis_weights_with_cp --custom-config-path .../seq-mask.yaml` | never |
| OPSM (DeepSeek-V3.2 §3.1) | yes, native | `--use-opsm --opsm-delta` | never |

`load_function` splits on the last `.` and calls `importlib.import_module`, so a
dotted path is the required form -- the `mis.py:func` spelling in the `--help`
text is stale. `examples.infra_features` has no `__init__.py` and resolves as a
namespace package; `find_spec` confirms all four targets resolve.

Two things that will bite when tier 2 runs:

- `icepop_function` returns `loss_masks` unchanged, so a popped token still sits
  in the `sum_of_sample_mean` denominator. The `[decouple IS and rejection]`
  path exists precisely to rebuild that denominator from a modified mask and
  ICEPOP does not use it, so popping shrinks the gradient rather than
  renormalising it. Check against the ICEPOP formulation before reading a tier-2
  result as the method's.
- `icepop_function` emits its metrics under `tis`, `tis_clipfrac`, `tis_abs` --
  the same keys as vanilla TIS. Runs are separable only by run name, not by
  series.

### Measured: what the sequence-level bound would actually mask (2026-08-07)

The per-token log-ratio is not a constant offset. From the four running arms'
`dump/dashboard/metrics.jsonl`:

| quantity | value | source |
|---|---|---|
| `E[log w]` per token | -8.5e-06 | `train/tis` - 1 |
| `E[(log w)^2]` per token | 9.6e-04 | 2 x `train/train_rollout_kl` |
| RMS per token | **0.0310** | sqrt of the above |
| response length median / p90 / p99 | 6332 / 10317 / 32768 | `rollout/response_len/*` |

The mean is three orders below the RMS, so the sequence sum is a random walk,
not a drift: `sd(sum) = 0.0310 * sqrt(L)`. `tis_level: sequence` is
`masked_sum` (`mis.py:196`), so the sequence weight is `exp` of that sum.

| L | sd(sum) | kept at c=2.0 | kept at c=8.0 | c needed to keep 90% |
|---|---|---|---|---|
| 2048 | 1.41 | 37.8% | 86.1% | 10 |
| 6332 (median) | 2.47 | 22.1% | 60.0% | 58 |
| 11069 (p90) | 3.27 | 16.8% | 47.5% | 216 |
| 32768 (p99) | 5.62 | 9.8% | 28.8% | 10300 |

**No fixed bound works.** The threshold that keeps 90% of sequences varies 177x
across our own length distribution. c = 2.0 masks four sequences in five. This is
also why VCPO settles on c = 8.0 and why it does not transfer: at their 2048
tokens c = 8.0 keeps 86%, at our median it keeps 60%.

Tokens within a sequence are positively correlated, so `sqrt(L)` is the
optimistic end -- the true `sd(sum)` lies between `sqrt(L)*sigma` and `L*sigma`.
The table is a **lower bound on the damage**.

`tis_batch_normalize` does not rescue this: normalisation happens at `mis.py:274`,
after `mask()` has already run per sequence at `mis.py:236`. The bound is applied
to the raw weight.

Going the other way, `tis_level: geometric` (`masked_mean`) needs 1776 sigma at
the median length to reach c = 2.0 -- it can never fire.

**Conclusion: drop `seq-mask` rather than retune it.** A fixed bound on a
sequence-level weight is a length filter at 32k, not a mismatch filter. It is not
in the tier-2 arm list and should not be added. `experiments/configs/mis/` is
kept only as a worked example.

### Verified on hardware: ICEPOP fires, OPSM does not (2026-08-07)

Two one-node colocated smoke runs on `interactive`, 3 rollouts at 2048 tokens
(jobs 15309933, 15309937). Both reach a training step without error.

| arm | flags as parsed | first-step metric |
|---|---|---|
| ICEPOP | `custom_tis_function_path=...icepop_function`, `tis_clip_low=0.5`, `tis_clip=5.0` | `tis_clipfrac` = 7.83e-06 |
| OPSM | `use_opsm=True`, `opsm_delta=1e-4` | `opsm_clipfrac` = **0.0** |

ICEPOP works. The pop fires on eight tokens per million, which is what the
bounds imply on-policy; whether it separates under staleness is the tier-2
question.

**OPSM's 0.0 is exact, and it is structural.** `compute_opsm_mask`
(`math_utils.py:279`) forms `seq_kl` from `full_old_log_probs - full_log_probs`,
and `old_log_probs` is the PPO reference. Under `RATIO_DENOMINATOR=actor` with
`NUM_STEPS_PER_ROLLOUT=1` the reference is a fresh forward pass of the same
weights, so `seq_kl` is identically zero and the mask never fires for any
`opsm_delta`.

DeepSeek-V3.2 masks against the **behaviour** policy. In miles that is
`--use-rollout-logprobs` (`RATIO_DENOMINATOR=rollout-logprobs`), which puts the
rollout log-probs in `old_log_probs`. **OPSM is a silent no-op without it**, and
`convergence_sweep.sh` now sets it whenever `USE_OPSM=1`.

With the reference fixed, `seq_kl` is the per-token mean log-ratio, whose spread
across sequences is `0.0310 / sqrt(L)` = 3.9e-04 at the median length. The
`opsm_delta` default of **1e-4 sits at 0.26 sigma**, so roughly 40% of sequences
exceed it and, gated on `advantage < 0`, about 20% are masked. That is a
working operating point, and it is what the TBD in the tier-2 table is now set
to.

### M2PO: the released code is not the algorithm in the paper (2026-08-07)

M2PO is Zheng, Zhao and Chen, *Prosperity before Collapse: How Far Can
Off-Policy RL Reach with Stale Data on LLMs?*, arXiv:2510.01161 (ICLR 2026).
It is the closest prior work to this study -- it claims parity with on-policy at
a staleness of 256 updates -- so it is a tier-3 arm.

The paper (eq. 4/5, Algorithm 1) describes **masking**: drop the tokens with the
largest `(log r)^2` until the batch mean falls under `tau_M2 = 0.04`, then take
an unclipped importance-weighted objective over what is left.

`verl/trainer/ppo/core_algos.py` in the authors' repo does something else:

1. it considers only **harmful** tokens -- `(A > 0, r > 1)` and `(A < 0, r < 1)`,
   the two quadrants PPO's `max()` acts on;
2. it solves for one scalar `tau` such that capping `|log r|` at `tau` brings
   their mean `(log r)^2` to the budget;
3. it turns `tau` into a **PPO clip range**, `eps_low = 1 - exp(-tau)` and
   `eps_high = exp(tau) - 1`, and clips. Nothing is masked and no token leaves
   the denominator;
4. it floors those at `miniclip_low = 0.3` / `miniclip_high = 0.5`.

So the released M2PO is an *adaptive clip range chosen by a second-moment
budget*. We implement the released version, because it is what the reported
numbers came from. Two consequences worth carrying into the writeup:

- The floors dominate whenever the batch is near-policy. `[0.7, 1.5]` is already
  wider than the DAPO clip the rest of this study runs (`[0.8, 1.28]`), so on a
  quiet batch M2PO *loosens* rather than tightens. `test_m2po_clip_bounds.py`
  pins this.
- VCPO's failed reproduction (arXiv:2602.17616, Figure 12) is described as
  masking with "max = 0.04" and reports the trusted-token fraction collapsing at
  lag 12. That is the paper's formulation, not the released one. VCPO also
  attributes the failure to a **mixed-staleness queue** versus M2PO's fixed-lag
  behaviour policy -- and miles' fully-async rollout is a mixed-staleness queue.
  We have the per-sample `weight_versions` to settle it.

**M2PO needs `--use-rollout-logprobs`, for the same reason OPSM does.** Its
`delta` is `log pi_behav - log pi_theta`, which under `RATIO_DENOMINATOR=actor`
with one step per rollout is identically zero. `arguments.py` now asserts this
rather than letting it silently no-op, and `run_identity.sh` lets `m2po` pair
with `rollout-logprobs` where the other corrections may not.

Scope note: the threshold is solved per microbatch, matching the reference,
which sits inside verl's per-microbatch `compute_policy_loss`. It is not the
global batch either there or here.

### The loss snapshot test is red before we touch anything

`tests/fast/backends/training_utils/loss/test_loss_snapshot.py` compares against
`.pt` fixtures shallow-cloned from an external artifacts repo, not from this
tree. Six cases (`grpo_b3`, `grpo_tis_b3`, `gspo_b1`, `reinforce_pp_baseline_b2`,
`grpo_kl_loss_b2`, `grpo_bshd_b3`) fail with a metric-key mismatch on committed
`af90e72e`, verified in a clean worktree (job 15311872). The fixtures predate
`rollout_ess_ratio`.

Any new key in `policy_loss_function`'s metric dict widens that mismatch. Do not
read it as a regression from the change under test -- run the same test against
HEAD in a worktree first. Metrics that only appear behind a flag no snapshot
config sets (M2PO's, OPSM's) do not affect it.

## lr 5e-6 does not collapse a learning rate, it collapses a trust region (2026-08-08)

Every tier-1 and tier-2 arm was submitted at lr 5e-6. Sixteen arms, one clean
split, and the split is **not** along the staleness axis:

| ratio denominator | arms | last step reached | `response_len/mean` | `truncated_ratio` |
|---|---|---|---|---|
| `actor` | s0/s1/s2/s4 **tis**, s1/s2/s4 **icepop** | 10-16 | 5.0k -> 18-24k | 0.37-0.67 |
| `rollout-logprobs` | s1/s2/s4 **m2po**, **none**, **none+opsm** | 60-157 | 5-8k -> 6-10k | 0.01-0.10 |

7 of 7 diverged; 9 of 9 survived. Staleness does not order anything inside
either group: the colocated s=0 TIS arm drifts the same way as s=4, only slower,
and the s=4 `none` arm is the healthiest run in the study (145 steps, +34%
length, 2% truncation).

The divergence is a length runaway, not a loss spike. `conv-s4-tis-lr5e-6-p1`
(job 15319216), one job, twelve steps:

| step | entropy | grad_norm | tis_clipfrac | tis_abs | resp_len | trunc |
|---|---|---|---|---|---|---|
| 0 | 0.289 | 0.048 | 3.7e-06 | 0.010 | 4975 | 0.000 |
| 4 | 0.234 | 0.055 | 4.0e-04 | 0.020 | 7542 | 0.063 |
| 6 | 0.205 | 0.203 | 3.0e-04 | 0.017 | 9013 | 0.118 |
| 8 | 0.197 | 0.615 | 1.2e-03 | 0.042 | 11908 | 0.214 |
| 10 | 0.309 | 1.346 | 5.5e-03 | 0.127 | 16386 | 0.381 |
| 11 | 0.508 | 2.705 | 6.4e-03 | 0.210 | 18583 | 0.458 |

`grad_norm` departs from its 0.04 baseline at step 5-6, ahead of the entropy
turn at step 9. The same arm at lr 1e-6 (job 15288366) holds `grad_norm`
0.039-0.051 and `tis_clipfrac` 3-7e-06 flat across 34 steps. So `tis_clipfrac`
rising three orders is a **symptom** of the policy leaving the rollout policy,
not the cause, and it is not evidence that TIS is at fault.

### Why the denominator, and not the learning rate, is the variable

At `--num-steps-per-rollout 1` the actor-denominator arms have **no trust region
at all**. The PPO ratio is `pi_train / pi_old` with `pi_old` recomputed by the
trainer from the same weights, so it is identically 1 and the clip is a no-op —
measured, not inferred:

| arm | `pg_clipfrac` | `ppo_kl` | `ess_ratio` |
|---|---|---|---|
| s4 tis (actor) | **0.0000** | **0.00000** | 1.0000 |
| s1 icepop (actor) | **0.0000** | **0.00000** | 1.0000 |
| s1 none (rollout-logprobs) | 0.0008 | 0.00056 | 0.9989 |
| s1 m2po (rollout-logprobs) | 0.0002 | 0.00055 | 0.9987 |

exact zeros, every step. `grpo-clip0.2-0.28` in those run names is decorative:
at k=1 the only thing between the update and vanilla REINFORCE is TIS's
one-sided c=2.0, and that fires on 4e-06 of tokens at the start. With
`rollout-logprobs` the ratio is the real off-policy `pi_train / pi_rollout`, so
the clip has something to bite on — 0.02-0.08% of tokens, which is small but is
exactly the tail that drives length growth.

**This is a hypothesis with one cell missing.** `actor` and the correction
family are confounded: every arm that collapsed also used TIS or ICEPOP, and
every survivor also used M2PO/none/OPSM. One control run separates them —
`none` + `actor` at 5e-6, or `tis` + `rollout-logprobs` — and it is worth the
four node-hours before any conclusion about the corrections is written down.
Note that k=1 makes the identity `ratio == 1` unavoidable for any
actor-denominator arm; it is a property of the schedule, not a miles defect.

### What this means for the learning-rate axis

- **There is no single ceiling.** `rollout-logprobs` arms ran 100+ steps at 5e-6
  with no sign of instability, so for them 5e-6 is a measured-good setting and
  tier 6 (1e-5) is a real question. Actor-denominator arms diverge by step 8-11.
- **1e-6 stays the common LR for the cross-arm comparison**, because a shared
  setting has to be one every arm survives. Reporting arms at different learning
  rates would confound the algorithm axis with the LR axis, which is the one
  thing tier 2 exists to avoid.
- **The gap 1e-6 -> 5e-6 is a factor of 5 with nothing in it.** If the
  actor-denominator ceiling matters, probe 2e-6 and 3e-6 on one arm (s=2 tis, 14
  steps is enough — divergence is visible by step 6) rather than inferring it.
- **`--clip-grad 1.0` never binds at 1e-6** (`grad_norm` ~0.04, 25x below it),
  and at 5e-6 it only binds after the runaway is already underway. It is not a
  safety net at these settings; do not treat it as one.
- Submitting tier 6 for `tis`/`icepop` is guaranteed waste. `convergence_sweep.sh
  --tier 6 --is m2po,none` is the only form of that submission worth GPU-hours.

Evidence: jobs 15319173/91/15319206/16 (tis s0/s1/s2/s4), 15319226/44/55
(icepop), 15319267/88/99 (m2po), 15319358/75/89 (none), 15319312/34/48
(none+opsm), all lr 5e-6; 15288366 (s4 tis lr 1e-6) as the control. Logs under
`hiso/kzk/miles/experiments/outputs/training/math/dapo-math-p10-90/qwen3-4b-instruct-2507/`.
