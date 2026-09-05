# v0.1 ABCI integration and validation, 2026-09-05

## Integrated revisions

Created `v0.1-abci` from `v0.0-abci` at `61def269`. Fetched
`origin/merge/v0.1.0-b300-aws-pdx` at `36cc6a12`. The merge base was
`487a9dfd`, with seven ABCI-only and 232 incoming commits. Merge `4602ba6f`
contains both histories. Python model profiles, v0.1 training/rollout APIs,
configurable completed-group buffers, partial rollout, truncation loss options,
and analysis tools are included. PBS/Singularity, numeric full-node GPU views,
HPC-X's unbound node launch, system NCCL/IB, and ABCI asset paths are retained.

The runtime is the existing `miles-20260902-134430.sif`: Torch 2.13.0+cu130,
Ray 2.58.0, SGLang f994b9aed, Transformers 5.12.1, MCore 0.19.0+c160f0717.
Validation uses the staged public Qwen3-4B HF/MCore checkpoints and the
17,398-row DAPO-Math file. The historical step-4000 SFT model is not staged
here; its configurable recipe remains available.

## CPU verification

The first focused run passed 339 tests and skipped one. Its single failure
identified incoming Slurm job headers, which were ported to PBS. The wider
experiment/launcher run passed 644 tests, with six failures: two missing the
optional `tau2` package in this image, three stale model/index/fixture references,
and the container-build test missing its host Bubblewrap bind. Fixing the stale
references and supplying the real host bind passed the 32-test follow-up. The
code/STEM/multi reward suite passed 11 tests.

Initial real code-reward preflight failed: Singularity marks its root overlay
unbindable, so Bubblewrap could not bind `/usr`. The verifier now enters a
private mount namespace and changes propagation only there before building
the read-only sandbox. The fixed preflight scored Math, Python, MCQA, and
Reasoning Gym correct/wrong pairs as `[1,1,1,1,0,0,0,0]`. The Python probe also
checked that `/root/miles` was absent and only the loopback interface existed.
No process-only reward fallback is used.

## Data preparation

CPU job `2226121.pbs1` finished with exit 0 in 7m30s. It converted complete
training components, rejected evaluation-only rows during merging, and built
32,673 balanced rows: 10,891 Math, 10,891 Code, and 10,891 STEM
(10,578 MCQA plus 313 Reasoning Gym). Reasoning Gym 0.1.25 supplied all 104
task scorers. Job `2226144.pbs1` regenerated the blend with the correct
`dapo-math-17k` source label and finished with exit 0 in 2m43s.

Data and audits are under `datasets/rl/nemotron-performance-transfer/`; the
reproduction entrypoint is `experiments/prepare_abci_multi_env.sbatch`.

## Two-node bring-up and restart

`2226122.pbs1` trained rollout 0, saved model/replay state, and completed with
exit 0. It used 8 actor + 8 rollout GPUs, 8 groups x 8 responses, global batch
64, 8K response / 32K context, and concurrency 64. Inflight replay,
queue-recycle, prefill reference, maximum staleness 8, and exact token-version
segments were enabled. Replay publication preceded the model save.

`2226139.pbs1` resumed this checkpoint and trained rollouts 1--3. The Ray job
reported success. Prepared rollout 1's 363 staleness values matched exactly
before/after restart. It restored 16 pending groups, one prepared batch,
eight inflight groups, and 180,503 generated prefix tokens; it regenerated no
active group. `warm_prepared_batch_hit=1`. All four optimizer losses/gradients
were finite with positive gradient norms. Staleness means were 0, 1, 1.875,
2.125; maxima were 0, 1, 2, 3. Exact token coverage was 1 and invalid counts 0.

The current-vs-old log comparison uses retained ABCI job `2211415.pbs1` in
the `miles-v0.0` checkout. Incoming defaults expanded histogram bins from 16
to 32: 299 staleness keys become 363; four `count_ge_17` fields are replaced
by bins 17--32 and `count_ge_33`. Definitions of group/sample/token lag remain
unchanged. Incoming math recipes also select capacity 6000 at maximum
staleness >= 8 instead of the previous capacity 1000. Distribution comparisons
must account for these configuration changes.

Local evidence is under `experiments/outputs/abci-validation/`, including
`math-2node-resume-report.json`, PBS logs, live logs, checkpoints, replay
manifests, HF exports, and rollout/dashboard dumps. Reproduce the audit with
`python -m experiments.tools.analyze_abci_resume --fresh-log ... --resume-log ...
--baseline-log ... --output ...`.

## Remaining acceptance checks

The eight-node production batch (192 x 16 = 3072, 16K responses), its replay
restart, and the actual multi-environment training have not yet been accepted.
Their entrypoints are `experiments/validate_abci_math.sbatch` and
`experiments/validate_abci_multi_env.sbatch`, both with a two-hour PBS walltime.
