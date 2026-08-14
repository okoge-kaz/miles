# Offline evaluation

Scoring exported HF checkpoints after the fact, rather than inside the training
loop. In-run eval is off (`EVAL_INTERVAL=0`) because it perturbs the independent
variable — see `notes/telemetry.md`.

## The procedure

Three steps, in order. Skipping step 1 fails at model load; skipping the mount in
step 2 fails at weight load, several minutes in.

### 1. Unpad the vocabulary

```
uv run --no-project python experiments/src/offline_eval/unpad_vocab.py \
    <src-hf-dir> $CKPT_ROOT/training/offline_eval_unpadded/<tag>
```

`--vocab-size 151936` is padded to `padded_vocab_size` 152064 by
`_vocab_size_with_padding` (`megatron_utils/arguments.py:35`), and
`megatron.bridge`'s `save_hf_pretrained` writes the padded tensor while leaving
`config.json` at the true 151936. sglang refuses it:

```
AssertionError: self.org_vocab_size=151936 ... loaded_weight.shape[0]=152064
```

Setting `vocab_size` to 152064 instead would load, and would be wrong. The model
ties its output projection to the embedding (`tie_word_embeddings: true`), so the
padding rows become 128 extra logits. They are not zero — measured at 1.65e-09
against 8e-2 for real rows — which puts them at a logit of about 0. Against a
peaked distribution that is ~4e-5 of the mass per token, and over a 6k-token
response it is a coin-flip whether at least one sampled id is outside the
tokenizer's range.

~16 s per checkpoint. Only the shard holding the embedding is rewritten; the
others are hard-linked where the filesystem allows and **symlinked otherwise**,
which is what makes step 2 mandatory.

### 2. Submit, with the source tree mounted

```
HISO=/lustre/fs1/portfolios/coreai/projects/coreai_horizon_dilations/users/hiso
sbatch -A coreai_horizon_dilations \
  --partition=batch,batch_short \
  --job-name="oeval-${TAG}" \
  --export=ALL,CKPT=/ckpt/training/offline_eval_unpadded/${TAG},TAG=${TAG},EXTRA_MOUNTS=${HISO}:${HISO} \
  experiments/src/offline_eval/run_eval.sbatch
```

`EXTRA_MOUNTS` must expose the **symlink target's** path, at the same path inside
the container. The targets resolve to `/lustre/fs1/...`, not the `/lustre/fsw/...`
spelling the checkpoints are usually referred to by; `Path.resolve()` picks the
former and the symlink is written with it.

`--partition=batch,batch_short` because `batch_short`'s QOS carries a
cluster-wide `GrpTRES=node=20` that other users fill (`notes/cluster.md`).
Already-pending jobs can be moved with `scontrol update jobid=<id>
partition=batch,batch_short` without a kill.

### 3. Read the result

`report.py <out-dir>` pools the years. Resumable: `measure_pass_rate.py` appends
per prompt and skips indices already present, so re-submitting after a
wall-clock kill continues rather than restarting.

## Search-R1

Search-R1 has a separate runner because the AIME client performs one generation
request per answer.  A search trajectory must instead alternate LLM generation
and retrieval, append each `<information>...</information>` observation, and
resume generation.  The offline driver imports the same
`generate_with_search.generate` and `reward_func` functions as training; it does
not maintain a second inference implementation.

Evaluate one checkpoint:

```
sbatch -A coreai_horizon_dilations \
  --export=ALL,CKPT=/ckpt/training/search_r1/.../hf/19,UNPAD_VOCAB=1 \
  experiments/src/offline_eval/run_search_r1_eval.sbatch
```

The job starts an 8-way data-parallel SGLang server and the CPU e5/wiki-18
retriever, then evaluates NQ, HotpotQA, TriviaQA, PopQA, 2WikiMultiHopQA,
MuSiQue, and Bamboogle.  Defaults match training: temperature/top-p 1.0,
512 generated tokens per LLM turn, at most three LLM turns, top-3 passages, and
outcome exact match with no format shaping.  Six sets are capped at 500 prompts;
Bamboogle contains 125, for 3,125 trajectories per checkpoint at avg@1.

```
python3 experiments/src/offline_eval/report_search_r1.py \
  $DATASET_DIR/offline_eval/search_r1/<tag>
```

In addition to EM, the report gives retriever calls per trajectory (`search`),
LLM generation calls per trajectory (`turns`), search-use and final-answer
rates, truncation, generated tokens, and injected observation tokens.  The
observation count is derived from `loss_mask == 0` and includes both retrieval
blocks and invalid-action feedback; it is therefore also a check that
environment text did not become policy loss.  Aborted trajectories are treated
as infrastructure failures, leave their prompt incomplete, and make the job
fail rather than lowering model accuracy silently.  Re-submission resumes from
the completed prompt records.

For a checkpoint series, first inspect, then probe one checkpoint end-to-end,
then fan out:

```
SOURCE_ROOT=<host Search-R1 run root> SWEEP_NAME=<name> \
  experiments/src/offline_eval/submit_search_r1_sweep.sh check
SOURCE_ROOT=<host Search-R1 run root> SWEEP_NAME=<name> \
  experiments/src/offline_eval/submit_search_r1_sweep.sh probe
SOURCE_ROOT=<host Search-R1 run root> SWEEP_NAME=<name> \
  experiments/src/offline_eval/submit_search_r1_sweep.sh all
```

`<search>` and `<information>` are deliberately ordinary text, not tokenizer
special tokens.  SGLang matches `</search>`/`</answer>` as stop strings, while
the host inserts and normally tokenizes the information block before the next
turn.  The offline driver records the tag token IDs and fails if any tag does
not encode/decode exactly; adding special tokens would change the vocabulary and
would require training new embedding rows.

## What is measured, and why it differs from in-run eval

In-run eval was AIME-2025 only at n=8 and existed to show that a run is learning.

- Generation budget 32768, not the 24576 the training recipe uses. 13.75% of
  AIME-2024 truncates at 24576 and a truncated sample scores 0 under every
  rule-based verifier, so the in-training number is depressed by the budget
  rather than by the model.
- n=16 over three years rather than n=8 over one. 30 problems put the standard
  error at ~9 points; 90 put it at ~4.5. Paired against a common baseline the
  between-problem variance cancels and the standard error falls to ~1.3.
- Sampling is whatever the comparison needs. Qwen's card reports AIME25 = 47.4
  at temperature 0.7 / top_p 0.8; the training recipe generates at 1.0 / 1.0,
  which is the default here.

All years carry the same instruction wrapper, asserted at staging time by
`prepare_aime.py --template-reference`. AIME-2024 was the exception until
2026-08-05 (bare problem text, so the boxed answer the verifier grades was never
asked for); `aime-2024.nowrapper.jsonl` is that older file, and scores measured
against it do not compare to scores measured now.

**AIME 2023 is deliberately excluded: it is in the training data.** Normalized
verbatim matching against the prompt files gives

| set | AIME-2023 overlap |
|---|---|
| dapo-math-17k | 23/30 |
| dapo-math-p10-90 (the set trained on) | 11/30 |
| deepscaler | 26/30 |

so an AIME-2023 score is a memorisation score. 2024/2025 are clean against all
three. 2026 is clean against the filtered p10-80 set but not against unfiltered
dapo-math-17k, so re-check the overlap whenever the difficulty window widens.
Matching is exact-after-normalisation and therefore a **lower** bound:
paraphrases and number-swapped variants are missed.

## Failure taxonomy (measured 2026-08-06/07, 35 jobs)

15 of 35 jobs failed. Three distinct causes, and only one of them is random.

| Cause | Count | Symptom | Fix |
|---|---|---|---|
| Vocab padding | 1 | `org_vocab_size=151936 ... loaded_weight.shape[0]=152064` at load | step 1 |
| Dangling symlink | 13 | `Rank 0 scheduler died during initialization (exit code: -3)`, `FileNotFoundError: .../model-00002-of-00003.safetensors` | step 2's `EXTRA_MOUNTS` |
| NaN at sampling | 1 | `probability tensor contains inf, nan or element < 0`; job exits **COMPLETED** with 30 generation failures | resubmit elsewhere |

The 13 were a single wave (15301356–15301426), one cause, submitted together
before any one of them had been verified. Every one of them retried clean.
**Probe one checkpoint end-to-end before fanning out** — a whole submission
sharing one config also shares one config bug.

The NaN case (job 15305744, `s4-r24`, `pool0-00707`) is worth reading carefully
because it *lies*: the sampler dies on the first batch, every request returns
`ServerDisconnectedError`, `report.py` prints an empty table, and the Slurm state
is `COMPLETED` with exit code 0:0. Nothing in `sacct` distinguishes it from a
real run. The weights were checked and are clean — every bf16 tensor in the
checkpoint has a finite exponent — and no other job on any other node reproduced
it, so it is node-local. Resubmit with `--exclude=<node>`.

The consequence for monitoring: **`sacct` state is not a completion signal
here.** Check line counts.

```
for d in $DATASET_DIR/offline_eval/*/; do
    printf '%-10s %s\n' "$(basename $d)" "$(wc -l < $d/aime26.jsonl)"
done
```

30 per benchmark is complete. A count between 1 and 29 is a job still running or
one that hit the wall clock — resubmit, it resumes. A count of 0 with a
`COMPLETED` state is the NaN case.

## Two shape choices in the sbatch

**`--dp-size 8 --tp-size 1`.** A 4B model fits on one H100, so eight independent
replicas beat one tensor-parallel engine for a pure-inference sweep — no
cross-GPU collectives on the critical path.

**Benchmarks run backgrounded, not sequentially.** A benchmark is 30 prompts, so
its last few long generations leave the eight replicas nearly idle; run
sequentially that tail is paid once per year. Overlapping them keeps the engines
fed and costs nothing, since 90 prompts still fits under `--concurrency`.
