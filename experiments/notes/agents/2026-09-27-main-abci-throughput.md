# Main integration and filtered DAPO Math throughput on ABCI

The integration starts from ABCI `dadc0476`, merges upstream main
`23d41d711f3b80544fda655898ed4f051ed644fe`, and ports the applicable async,
staleness, replay, loss diagnostics, and routing changes from experiment branch
`f6b1a4d9`. PBS/Singularity, native InfiniBand, the checkpoint-zero fix, and
Megatron-aligned replay archive retention remain in place. OCI/Lightning launch
scripts and their machine-specific paths are not imported.

The experiment queue lives in `miles/rollout/experimental_fully_async_rollout.py`.
An explicit `--fully-async-queue-type` or `--use-replay-buffer` selects it;
upstream main's default fully-async implementation remains available. The port
uses main's worker handles, typed `RolloutDataPack`, and disposal lifecycle.

## Image build (operator action)

The old SGLang v0.5.16 image cannot run this checkout. The new base uses SGLang
v0.5.20 and PyTorch 2.13. The fork commit below is published and includes
scheduler forward provenance, response token/version segments, and provenance
retention on synthetic abort replies, using native weight-update sessions.

Run from this repository root:

```bash
PBS_PROJECT=gai51740 \
SETUP_CONTAINER_WALLTIME=02:00:00 \
DOCKER_IMAGE=radixark/miles@sha256:046650ba356dccd3393ef5da4d50199941d16b5fdbaf6104394b3a6d800a5f19 \
SGLANG_REPO=okoge-kaz/sglang \
SGLANG_BRANCH=miles-staleness-weight-boundaries-v0.5.20 \
SGLANG_COMMIT=164a43b253cceccaaab1d386550a0227419740e5 \
CONTAINER_IMAGE=/groups/gai51740/kazuki_fujii/containers/miles-main-20260927.sif \
IMPORT_OUTPUT_IMAGE=/groups/gai51740/kazuki_fujii/containers/miles-main-20260927-sg0520.sif \
bash experiments/setup.sh container --submit
```

Omit `--submit` to preview. This requests a two-hour managed CPU build job with
`qsub -P gai51740`, its node-local
scratch space, non-root access checks, and the fork's runtime provenance tests.
It publishes a separate image/link without replacing `miles.sif`. The immutable
amd64 OCI digest was resolved from the official `radixark/miles:latest` image
built on 2026-09-26. Build job `2449958.pbs1` completed with exit status 0 in
13m37s. Its local workspace was `/local/2449958.pbs1/miles-singularity-2449958.Ty40jW`.
The published SIF is 17,600,172,032 bytes, with SHA256
`bba0977739d7825d2081794aa3eeb6387e32ef5adf0fe7f5a300beafc5acbc7a`;
the fork's 10 tests passed both during build and under the ordinary-user mounts.

## Assets and comparison

The data is
`/groups/gai51740/kazuki_fujii/datasets/rl/dapo-math-p10-90-qwen3-4b-sft-iter4000/dapo-math-p10-90-qwen3-4b-sft-iter4000.jsonl`.
It contains 10,868 of 17,398 problems, with pass-rate bounds 0.1 and 0.9.
Its checked SHA256 is
`910925a5c09e55099e2ee18dc1db767eb2325e46d841359e52b41596af888d06`.
All rows have `prompt` and `label`. The matching model, recorded by the filtering
metadata, is the SFT `iter_0004000` under
`checkpoints/hf/training/reasoing-sft/Qwen3-4B-Base/LR2.0e-5-MINLR2.0e-6-SEQ32768-GBS128-MBS1-TP2-PP1-CP1-EP1-PACK1/`.
The `reasoing-sft` spelling is the actual asset path.
All eight model/tokenizer/config files, including both safetensors shards,
match the SHA256 values in the filtering metadata.

| Setting | 2:6 | 3:5 |
| --- | --- | --- |
| Exclusive nodes / total GPUs | 8 / 64 | 8 / 64 |
| Actor nodes / GPUs | 2 / 16 | 3 / 24 |
| Rollout nodes / GPUs | 6 / 48 | 5 / 40 |
| Actor TP / CP | 2 / 1 | 2 / 1 |
| Prompt groups × samples | 192 × 16 | 192 × 16 |
| Global batch / updates per rollout | 3072 / 1 | 3072 / 1 |
| Response / context limits | 16384 / 32768 | 16384 / 32768 |
| Concurrent rollout samples | 3072 | 3072 |
| Queue / staleness reference / bound | recycle / prefill / 8 | recycle / prefill / 8 |
| Measured optimizer updates | 20 | 20 |

Both retain the existing GRPO/TIS recipe, learning rate 1e-6, zero reward on
truncation, fused one-step actor logprobs, inflight replay, and exact provenance.
W&B uses entity `okoge`, project `async-rl-staleness-dapo-math-miles`.
The throughput jobs are disposable: no in-run eval or HF/model checkpoint is
scheduled in the measured interval. `NUM_ROLLOUT=300` and debug exit at 20 avoid
changing the scheduler horizon or triggering the last-rollout checkpoint.
The two-node smoke/resume jobs do save both distributed and HF checkpoints.

## Execution after the image is ready

Use a clean frozen checkout for `MILES_REPO` and one explicit image for every
job. `BENCHMARK_ID` must be new for a fresh comparison; smoke and resume share
the same ID and checkpoint directory. The converted checkpoint has its own
name, `Qwen3-4B-SFT-iter4000-main-20260927_torch_dist`.
The submit wrapper exports only the selected assets and run identity. W&B
credentials are resolved on the compute host from the owner's `~/.netrc`.

```bash
export PBS_PROJECT=gai51740
export CONTAINER_IMAGE=/groups/gai51740/kazuki_fujii/containers/miles-main-20260927.sif
export BENCHMARK_ID=main-20260927-a
bash experiments/submit_abci_math_benchmark.sh convert 2:6
bash experiments/submit_abci_math_benchmark.sh smoke 2:6
bash experiments/submit_abci_math_benchmark.sh resume 2:6
bash experiments/submit_abci_math_benchmark.sh fit 2:6
bash experiments/submit_abci_math_benchmark.sh fit 3:5
bash experiments/submit_abci_math_benchmark.sh throughput 2:6
bash experiments/submit_abci_math_benchmark.sh throughput 3:5
```

These commands preview by default. Append `--submit` to execute one stage after
the preceding stage has been checked. Confirm node joins, 192-CPU affinity,
placement, finite losses, reward/response samples, saved artifacts and actual
resume, then full-shape memory fit. Inspect the SGLang runtime tests and the
Megatron checkpoint-selection tests in the new image before training. The smoke
stage runs the latter on its allocated GPU host: the image's Transformer Engine
requires `libcuda.so.1` at import time, so these tests cannot run on the login
node even though their assertions only concern checkpoint path selection.
Job IDs, source revision and image are recorded in
`experiments/outputs/abci-throughput/jobs.tsv`; live logs are under `logs/`.

## Throughput analysis

```bash
python experiments/tools/analyze_abci_throughput.py t2r6.log t3r5.log > throughput.json
```

The analyzer requires exactly rollout IDs 0–19 with one completed optimizer
update each. It reports total wall time, seconds/update, generated tokens/s,
accepted loss tokens/s, trainer starvation and rollout backpressure. Rates use
summed counters divided by summed window durations. Both all 20 windows and
windows 1–19 are reported; the first includes pipeline startup. These are short
throughput observations, not learning-quality measurements. Throughput numbers
remain pending the GPU bring-up and measurement jobs.

## Integration validation before GPU allocation

Final CPU checks passed on Python 3.12.13, PyTorch 2.13.0+cpu,
Transformers 5.12.1, and Megatron-LM
`f148a32b4385b758b66a77c9c3ad1641f1295d4b`:

- 366 tests cover weight-update ordering, staleness and policy lag, worker RPC
  serialization, queue telemetry, default fully-async cancellation, sample
  codecs, SGLang abort provenance, PBS/Singularity integration, and analysis.
- 124 tests cover the async driver, experiment queue, replay save/restore and
  routing, checkpoint-zero selection, and checkpoint publication failures.
- 10 tests in the published SGLang fork exercise its actual scheduler/request/
  tokenizer classes, including token/version segments and synthetic aborts.
- Shell syntax, eight-node PBS previews for both splits, Python compilation,
  and Ruff undefined-name/syntax checks passed.

The applied-version commit now runs after all engines finalize their update
and before any engine resumes. A two-engine regression checks that ordering.
Main's worker-handle API is used for this notification and for replay state;
legacy Ray `.remote()` calls are not used for these handles.

Logs are retained under `experiments/outputs/abci-throughput/validation/`.
These CPU checks do not execute CUDA kernels, multi-node collectives,
checkpoint conversion, or either throughput run. GPU conversion subsequently
passed in job `2450315.pbs1` (1m47s, exit 0), producing the named checkpoint's
`release/.metadata` and `latest_checkpointed_iteration.txt = release`.
Multi-node smoke/resume and throughput measurements remain pending.
