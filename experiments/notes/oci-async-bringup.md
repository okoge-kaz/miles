# OCI async provenance and training qualification

This is a disposable correctness check on GB200/aarch64, not a production
throughput or learning-quality result. The sync runs in
[Step4000 bring-up](oci-step4000-bringup.md) are separate evidence.

## Runtime

Base image: `experiments/outputs/miles-oci-aarch64-7035384.sqsh`, with SGLang
`a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`. The opt-in
`OCI_SGLANG_PROVENANCE_PATCH=1` applies
`experiments/container/patches/sglang-a8e5c63-prefill.patch` inside each writable
container before any Ray process starts. The helper rejects a different base
commit and prints the patch SHA-256. It does not replace the base SQSH, install
the old cw-dfw fork, or change the default sync runtime.

Qualified patch SHA-256:
`2d1bc8ed455ac758128b9febfe241a88cad2afed91eb36fd311608a2172064a1`.
All four nodes in each training job reported this same digest.

The overlay retains upstream session finalization, LoRA handling, and native
`weight_versions`. It adds a begin-session version to the existing pending
version path, scheduler-authoritative first/last/min/max forward versions, and
token spans stamped from the batch that actually produced each token. Timing
metrics can remain disabled. The validation scope is dense, single-turn,
non-speculative generation through the Python tokenizer frontend; speculative
decoding and beam search are explicitly rejected when stamping is enabled.
Other models, disaggregated serving, Rust frontend, and multimodal retraction
are not qualified by this patch or smoke.

## Resource discovery and commands

On 2026-09-10 the discovery commands in [cluster.md](cluster.md) again showed
`batch / interactive`, account `nemotron_sw_post`, and 4 nodes per user/job.
The initial single-node check qualifies only the provenance protocol; the
training wrapper explicitly requires four nodes.

```bash
sinfo -o '%P %a %l %D %G'
scontrol show partition
sacctmgr show qos format=Name,Priority,Flags,MaxWall,MaxTRES,MaxTRESPU -P
sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS

sbatch -A nemotron_sw_post -p batch --qos=interactive -N4 --time=02:00:00 \
  --export=ALL,SMOKE_TAG=oci-async-prefill-20260910-s2 \
  tests/manual/run_oci_async_smoke.sbatch
# After the first successful link, repeat the SAME command to verify resume.
```

The wrapper fixes six total updates and exits after three per invocation.
It uses 2 x 4 training GPUs (TP2/CP1/DP4), plus 8 rollout GPUs (eight TP1
engines). The batch is 4 prompts x 4 samples = GBS16, one optimizer update per
rollout, max response 8192/context 32768, LR 1e-6, GRPO/TIS, and the staged
Step4000 checkpoint/DAPO data. Queue-recycle uses prefill reference, bound 2,
16 completed groups, 32 concurrent samples, and in-place generation pause.

MCore and the inflight replay buffer are saved every update, retaining every
snapshot; HF is exported every three updates. Evaluation every three updates
uses two overlapping DAPO training rows, not a held-out benchmark. W&B runs
offline; dashboard JSONL and raw rollout/train dumps are retained. This does
not establish successful upload or step completeness in the W&B cloud service.

## Qualification results, 2026-09-10

- Allocation `7061190`, node `nvl72155-T05`: final focused run **336 passed**.
  This includes real-SGLang overlay unit checks, Miles queue/replay/staleness,
  train-data conversion, async drivers, and launcher checks. The earlier 64 Ray
  setup errors were logical-eight-GPU mock tests constrained by four visible
  physical GPUs; only those CPU-only tests unset `CUDA_VISIBLE_DEVICES`.
- GPU provenance smoke passed with the real Step4000 checkpoint. Baseline
  version 10 had `[0,8,10]`; an in-place update during 512-token generation
  produced first-prefill 10 / last-forward 11 and exact spans
  `[0,14,10], [14,512,11]`; a fresh request had `[0,8,11]`.
  Native event-derived spans placed the mixed boundary at 13 instead of 14.
  Preserve this distinction: the extra forward-stamped channel records the
  producing batch under overlap, not tokenizer response time.
- Evidence logs: `experiments/outputs/oci-async-final-unit-7061190.log` and
  `experiments/outputs/oci-prefill-gpu-smoke-v2-7061190.log`. The first GPU smoke
  exposed a component-split reference error; it is not a successful run.

Both training jobs completed with exit `0:0`, using four nodes / 16 GB200 GPUs:

| Job | Updates | Elapsed | Result |
|---|---|---|---|
| `7061490` | 0–2 | 7m50s | Initial async training, MCore/replay 0–2, HF 2, eval 2 |
| `7061593` | 3–5 | 7m28s | Actual MCore/optimizer/RNG + inflight replay resume, MCore/replay 3–5, HF 5, eval 5 |

The resume restored pending 24 / inflight 8 groups (119,572 inflight tokens)
and one prepared batch. Its `warm_prepared_batch_hit` was 1. The 16 samples in
immutable `replay_buffer_2.pt`'s prepared batch 3 match the resumed rollout 3
in tokens, response, rewards, rollout log-probabilities, and all forward-version
provenance fields. The first job recycled 17 over-bound groups while preparing
the next batch. This exercises async admission and inflight replay, not merely
an async entry point with zero lag.

All six trained steps (96 samples) passed independent raw-dump comparisons:

| Train/rollout step | Scheduled train version T | Group total lag mean | Group total lag max | Exact token lag mean |
|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 |
| 1 | 2 | 1 | 1 | 1 |
| 2 | 3 | 2 | 2 | 2 |
| 3 | 4 | 1.5 | 2 | 1.698760 |
| 4 | 5 | 1.5 | 2 | 1.236543 |
| 5 | 6 | 2 | 2 | 2 |

Sample, response-token, and loss-token provenance coverage were 100%; invalid
segments/samples and useful-token accounting error were zero. Raw group lag
decomposition, sample lag, exact token lag and trainer sample-staleness bin
mass matched the logged values. Loss and gradient norm were finite on all six
steps; step 0 had zero gradient from constant rewards within groups, while
steps 1–5 had nonzero gradients. Every optimizer step reported applied = 1.

Replay checksums, dataset fingerprint and applied version passed for all six
snapshots; each MCore snapshot has optimizer/RNG metadata and the latest-iteration
tracker is 5. Both HF exports have complete shard/header/index coverage
(16 shards, 398 tensors, 8,044,936,192 tensor bytes each). A finite 128 x 128
Q-projection probe changed from the Step4000 base to HF 2 and again to HF 5.
This is **not** a full-model finite-value scan of these two exports.

W&B offline histories `at4wta1g` (steps 0–2) and `yqin72nj` (steps 3–5) retain
all six steps of staleness, queue wait, throughput, trainer loss/staleness bins,
and policy-lag diagnostics. The analyzer merges each producer's records by its
explicit train/rollout axis; prefetched rollout 3 in the first run is not an
extra trained update. Inflight resume can create multiple generation calls for
one response: steps 3 and 4 had 19 and 24 calls respectively for 16 samples.
Exact-token validation concatenates call-local spans instead of assuming one
call per sample.

**Remaining logging issue:** the final eval point in each job is present in
dashboard JSONL and driver logs, but absent from the retained W&B offline
history (`eval/step` is empty). Eval 2 scored 0 and eval 5 scored 0.5 on the two
overlapping DAPO rows, both with truncation 0.5. The W&B eval persistence issue
is not fixed or root-caused here; staleness history completeness does not imply
eval completeness. No online upload, held-out accuracy, production batch size,
long-run stability, or production throughput is qualified.

Evidence is retained under the ignored `experiments/outputs/` directory:

- `miles-oci-async-smoke-7061490.log` and `miles-oci-async-smoke-7061593.log`.
- `oci-async-artifacts-7061600.log`: first run raw metrics and W&B history.
- `oci-async-final-artifacts-v2-7061600.log`: 7 analyzer tests, all six raw
  metric steps, resumed W&B history, and checkpoint/replay/HF checks.
- `oci-async-final-static-7061600.log`: final **129 passed**, Black/isort,
  shell syntax, clean-base overlay applicability and whitespace checks passed.
  The final `git diff --check` waited in Lustre I/O but eventually completed;
  Slurm step `7061600.6` finished with `0:0` before allocation release.

Post-hoc analysis used CPU partition / `cpu-interactive` allocation `7061600`;
the actual model/provenance/training tests used GPU `batch / interactive`.
An earlier post-hoc step `7061490.1` ended when its parent training job exited,
and an earlier analyzer assumed one generation call per response. Neither is
counted as a passing check; the CPU checks above replaced them.

## Check the actual logged values

Inside the training image, run:

```bash
python tests/manual/analyze_oci_async_smoke.py \
  --checkpoint /ckpt/training/<resolved-run-path> --steps 0 1 2 3 4 5
python tests/manual/check_oci_async_checkpoints.py \
  --checkpoint /ckpt/training/<resolved-run-path> --hf-base /ckpt/hf/iter_0004000
python tests/manual/check_oci_async_wandb.py --root /root/miles/wandb \
  --run-id at4wta1g --steps 0 1 2
python tests/manual/check_oci_async_wandb.py --root /root/miles/wandb \
  --run-id yqin72nj --steps 3 4 5
```

The resolved path relative to `/ckpt/training/` for this evidence is:

```text
math/dapo-math-p10-90-qwen3-4b-base-lr2e-5-step4000/Qwen3-4B-Base-LR2e-5-Step4000/grpo-clip0.2-0.28-tis2.0/async/off-policy/max-weight-staleness-2-from-prefill/oci-async-prefill-20260910-s2-zero-trunc-rb-inflight-concurrency-32-tbq16
```

The checker requires every requested train/rollout step, full sample and exact
token provenance coverage, no invalid token segments, zero useful-token
accounting error, and finite loss/gradient norm. It independently recomputes
group/sample/token lag from raw rollout dumps, compares trainer staleness bins,
checks `G <= Q <= P <= D <= T`, strict recycle admission `D-F < 2`, and inclusive
training lag `T-F <= 2`, where F is the group's earliest first-prefill version.
Nonzero async lag must actually be observed. The checkpoint checker verifies
replay resume state and retained artifacts; process exit 0 alone is not
qualification. The W&B checker qualifies requested training/rollout steps,
reports observed eval axes, and does not assert eval completeness.
