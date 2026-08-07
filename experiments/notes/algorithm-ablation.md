# RL algorithm ablation

Runs inside the environment frozen by `notes/node-ratio-procedure.md`. The
algorithm set is not settled yet; this records the frame it will be run in, what
miles can express today, and what each arm costs, so that deciding the set is a
choice about science rather than about plumbing.

## The environment is frozen first, and does not move between arms

| fixed by | quantity |
|---|---|
| `node-ratio-procedure.md` | train:rollout split, staleness levels worth running |
| `notes/rollout-scaling.md` | `SGLANG_MEM_FRACTION` — async 0.70, colocated 0.80 |
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
checkpoint path and wandb group (`notes/off-policy-variables.md`):

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

## The on-policy reference is invariant to the IS correction (2026-08-07)

Tier 2 compares TIS, ICEPOP and sequence-level MIS. It carries no s=0 arm,
because on the colocated on-policy run all three are the same loss.

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

**A larger caveat comes with it.** The same statistic is flat across the
staleness axis at this learning rate:

| arm | tis mean | tis_abs max | clipfrac max |
|---|---|---|---|
| s=0 colocated | 0.9999980 | 0.00941 | 4.85e-06 |
| s=1 async | 0.9999999 | 0.01021 | 4.86e-06 |
| s=2 async | 1.0000033 | 0.01060 | 4.59e-06 |
| s=4 async | 0.9999988 | 0.01032 | 4.74e-06 |

At lr 1e-6 the policy moves so little per step that a lag of 4 still leaves
`pi_train / pi_rollout` within 1% of unity, and the clip fraction does not
separate the arms at all. An IS correction that never fires cannot distinguish
itself from another one that never fires, so **tier 2 may return a null result at
this learning rate by construction**. If tier 1 confirms the pattern, tier 2 is
worth more at lr 5e-6 (tier 4's upper point) than at 1e-6, and the tier order
should be revisited rather than run as scheduled.
