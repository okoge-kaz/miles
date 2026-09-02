# Repository-level SWE RL

This directory contains a production-shaped Miles recipe for repository-level
SWE tasks backed by native Harbor/E2B execution. It does not depend on NeMo Gym.
The recipe exists, but it is not yet a claim that SWE RL is training-ready.

## Current status

- R2E-Gym V1, SWE-ReBench-V2, SWE-Gym, Nemotron Super `swe2`, and Nemotron
  Ultra `swe` have schema-normalized candidate paths.
- Offline contracts, production-container CPU tests, GPU metadata/reward
  plumbing, and one-task dry materialization for ReBench and SWE-Gym have
  passed.
- Historical former-cluster evidence: the pinned Harbor overlay preflight job
  `306362` passed 74 Miles/Harbor contract tests and 175 patched-Harbor tests.
  Legacy image jobs `306374`, `306375`, and `306376` respectively passed the
  CPU rollout contract, scheduler-to-Ray secret/client identity inheritance,
  and the GPU agent-metadata-to-reward path. These job IDs do not validate the
  current PBS/Singularity deployment.
- Real source-image probes passed for SWE-Gym (`306206`), R2E-Gym (`306215`),
  and SWE-ReBench-V2 (`306242`). These prove the selected image/runtime paths,
  not live hosted-E2B admission of every source row.
- No row has completed live E2B semantic admission in the checked artifacts,
  `/data/miles-swe/admitted` has no promoted training dataset, and no 4-node RL
  or downstream SWE result exists. Admission requires the E2B credential only
  in the separate Harbor controller job.
- Replay is deliberately disabled. Do not enable it until fresh and resumed
  live E2B jobs both pass; inflight replay additionally requires exact sandbox
  snapshot restoration.

Normalized source counts are documented in
[`experiments/notes/dataset-inventory.md`](../../notes/dataset-inventory.md).
They are candidate counts, not admitted training-row counts.

## Training contract

The asynchronous recipe is under
`async/swe-rebench-v2-swe-gym/qwen3-4b/`. It uses the Step4000 Qwen3-4B SFT
checkpoint and defaults to the full pinned SWE-ReBench V2 source; `SWE_DATASET`
can select SWE-Gym. `train.sh` accepts only a finalized
`/data/miles-swe/admitted/<selector>-train.jsonl` plus its digest-bound admission
summary. A normalized or dry-materialized file cannot bypass that gate.

The recipe uses four GPU nodes, 16,384 maximum response tokens, 16 samples per
prompt, maximum prefill-referenced staleness four, response-weight version
segments, W&B offline mode, and no inline evaluation. Provider credentials are
not forwarded to training or rollout workers; only the scoped Harbor `/run`
bearer token is present there. No `.env` file is loaded.

Submit production training with the adjacent `submit_when_ready.sh`. It puts a
CPU-only authenticated readiness gate ahead of the GPU job using a PBS
`afterok` dependency. The GPU allocation itself permits only a final 60-second
health check; full task prebuild belongs to the separate one-day Harbor server
allocation.

## Security and score boundaries

Harbor allocates one-use E2B execution sandboxes and verifies a patch in a
separate fresh sandbox with a late-uploaded hidden verifier. Its trusted
per-trial provider controller holds the E2B credential but runs with an exact
environment allowlist and strips the HTTP run/admin master secrets. The native
provider forwards only task/persistent environment entries into the remote
model sandbox; task trees and evidence are digest-bound. These controls still
trust the pinned Harbor controller, the Harbor task host, and
its Unix UID: owner-only permissions cannot isolate another hostile process
running under the same UID. The bearer-authenticated HTTP path is for private
cluster fabric, not direct Internet exposure; add a TLS/mTLS proxy across an
untrusted network.

`eval/swebench-verified/run.sbatch` is intentionally a hardened-local evaluator.
It uses pinned official parser/grading code but adds local path/security policy
and requires pre-bound admitted task rows, so its result is not leaderboard-
comparable. Produce an exact official score separately with the unmodified
pinned official SWE-bench evaluation harness.

Before the first live run, complete the E2B template/publisher-pin and full-scale
cost/concurrency gate, then admit tasks with the required empty-patch=0 and
oracle-patch=1 probes. Only after that should the 4-node fresh training and
external downstream evaluations be submitted.

The E2B SDK exposes runtime network policy, which this integration sets before
uploading hidden verifier material. It does not expose a no-egress policy for
hosted template builds, so the pinned source publisher and E2B build service
remain an explicit trusted boundary.
