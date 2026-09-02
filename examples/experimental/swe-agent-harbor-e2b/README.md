# SWE-Agent training with Harbor on E2B

This is the E2B backend for the existing `examples/swe-agent` Miles recipe.
Miles keeps the same `/run` agent-server API and reward contract; Harbor starts
one isolated E2B sandbox per trajectory, runs the task agent and verifier, then
returns the verifier reward to Miles.

The integration deliberately reuses Harbor's E2B provider rather than wrapping
the E2B SDK a second time. Harbor owns:

- deterministic templates from each task's Dockerfile or pinned image;
- sandbox create/start/stop and command timeouts;
- command execution and file/directory upload/download;
- agent edits and patch application through normal sandbox commands;
- required artifact collection, separate-verifier execution, and teardown.

Miles owns provider selection, rollout/session routing, reward propagation, and
the Docker/Daytona/E2B compatibility overlay for Harbor's Miles agent server.

## Requirements

Use the official Miles integration branch. It already contains Harbor's
`E2BEnvironment`; the overlay in this directory only connects that provider to
the Miles-specific server selector.

```bash
git clone https://github.com/harbor-framework/harbor.git /path/to/harbor
cd /path/to/harbor
git checkout harbor-miles-v0.20.0
git checkout 2ce5ba2af33a00c9fba0463f6403313996373f85
uv sync --extra e2b
```

The compatibility target validated here is commit
`2ce5ba2af33a00c9fba0463f6403313996373f85`. At the time of validation its
`src/harbor/environments/e2b.py` is byte-for-byte the same Git blob as Harbor
main (`c45f105ae67d3345cbd73af7a9cceec555893e02`). Harbor main has the native
`--env e2b` CLI, but not the Miles-specific dynamic `/run` session-server code;
the pinned branch is therefore still required for training.

Every training row's `metadata.instance_id` must exactly match one materialized
task directory under `HARBOR_TASKS_DIR`. Gold patches, hidden tests, expected
outputs, and verifier configuration belong only in this trusted task tree; do
not put them in model-visible prompts or Miles training-row metadata.
Materialized SWE tasks also store the prompt row's SHA-256 `task_digest` in
`task.toml`. The Miles client forwards that digest explicitly and the patched
server compares it in constant time before creating a trial. A missing, stale,
or mismatched binding is an ungraded `TaskDigestMismatch`, never reward zero.

## Start the E2B agent server

The launcher never reads `.env`. Credential-capable PBS Singularity steps also
mount the tracked comment-only `experiments/common/dotenv.disabled` over
`/root/miles/.env` read-only, so an unrelated host checkout file cannot be
discovered inside the container. The repository itself is never mounted into a
model-controlled E2B sandbox. Put `E2B_API_KEY` in the process environment
through a shell export, PBS secret injection, or your secret manager. It
checks only that the value is present and never prints it. It also exports
`PYTHON_DOTENV_DISABLED=1`, so Harbor's optional registry client cannot
implicitly discover a dotenv file later in the process.

```bash
export E2B_API_KEY=<injected-secret>
export HARBOR_ROOT=/path/to/harbor
export HARBOR_TASKS_DIR=/path/to/materialized/harbor_tasks
export HARBOR_E2B_PREBUILD_TASK_IDS_FILE=/path/to/admitted-instance-ids.txt
export HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS=/path/to/admission-a.jsonl:/path/to/admission-b.jsonl
export TRIALS_DIR=/path/to/trials
export MAX_CONCURRENT=64
# Inject a unique master per training/evaluation job and a server-only admin key.
export HARBOR_RUN_SECRET=<job-scoped-secret-at-least-32-characters>
export HARBOR_ADMIN_SECRET=<distinct-server-admin-secret-at-least-32-characters>

bash examples/experimental/swe-agent-harbor-e2b/submit_agent_server.sh
```

The launcher calls `apply_harbor_e2b_overlays.sh`, which pins the Harbor commit,
checks the SHA-256 of every overlay, accepts only an exact clean or prior
overlay stage, and verifies the resulting source tree after each application.
This supports an idempotent restart and migration from an older checkout without
silently accepting source drift. The overlays are:

- `harbor-miles-e2b.patch`: Miles session-server provider selection, task-digest
  binding, cloud-safe Docker behavior, and graceful worker cancellation;
- `harbor-swe-collect-hardening.patch`: required collect hooks and fail-closed
  artifact capture;
- `harbor-e2b-no-new-privs.patch`: opt-in `setpriv --no-new-privs` dispatch for
  UID 1000, with login/profile/BASH_ENV/loader injection disabled;
- `harbor-e2b-late-verifier-tests.patch`: direct immutable verifier images,
  post-start private-test upload, artifact readback, and template-ID pins.
- `harbor-agent-server-auth-attestation.patch`: task/session-scoped `/run` and
  `/flush` bearer authentication, server-only admin routes, disabled dashboard,
  and run-before task-tree attestation.

`HARBOR_RUN_SECRET` is a job-scoped master held by trusted Miles workers and is
never sent over HTTP or into E2B. `/run` uses an HMAC bearer bound to the client
ID, a cryptographic per-trajectory request nonce, the optional opaque
session-server instance, and the exact task ID; a captured token cannot start a
sibling trajectory or another admitted task. `/cancel` uses a distinct context
with the same exact binding, including a bounded cancellation tombstone that
closes cancel-before-registration races. `/drain` is client-scoped and cancels
only that PBS job's remaining inventory. Only these derived bearers cross the
HTTP boundary.
`/flush` uses a distinct session-only HMAC context, and a token for one session
cannot flush another. Do not share one server/master between mutually untrusted
jobs. `/clients` and `/flush_all` accept only the distinct
`HARBOR_ADMIN_SECRET`, which is present in the server process but never exported
to per-trial workers. The task/session run master is likewise stripped before
the per-trial process starts. That trusted provider-controller process retains
only an explicit environment allowlist, including `E2B_API_KEY`, because the
native SDK must create and manage the remote sandbox there. Harbor's E2B
environment passes only task/persistent environment entries to sandbox startup
and commands, so the provider key is not present in the model-controlled agent
environment. This boundary assumes the exact pinned Harbor/controller code is
trusted. Dashboard routes and the standalone dashboard port are disabled
because they expose private trial metadata; the owner-only JSONL
remains available for host-side auditing. The HTTP server binds `0.0.0.0` for
PBS workers and therefore must remain behind a private-fabric ACL. TLS/mTLS
and bearer replay defense on a hostile network are outside this deployment
boundary.

On `/flush`, the session-server overlay sends the trial worker `SIGINT` and
gives Harbor 30 seconds (configured by `HARBOR_WORKER_CANCEL_GRACE_SEC`) to run
its shielded sandbox teardown before falling back to a hard kill.

Production timeout scaling is fixed at `HARBOR_TIMEOUT_MULTIPLIER=1`. The
launcher also rejects overrides of `AGENT_TIMEOUT=3600`,
`AGENT_SETUP_TIMEOUT=1800`, `HARBOR_VERIFIER_TIMEOUT_SEC=2100`, and
`HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER=1`. The resulting upper bounds are 1,800
seconds for agent-environment start, 1,800 for agent setup, 3,600 for the
agent, 120 for collection, 1,800 for the fresh verifier environment, and 2,100
for verification: 11,220 seconds in total. A
server-owned `HARBOR_TRIAL_WALL_TIMEOUT_SEC=12600` wraps the entire trial
subprocess, including sequential phases. At that deadline the server first
invokes the same bounded Harbor cleanup and then returns the ungraded
`TrialWallTimeout` status. Miles HTTP clients use 13,200 seconds, leaving 600
seconds for cleanup and transport; the client must never time out before the
server-owned wall clock.

The collect overlay adds an opt-in `required = true` verifier hook. A
required snapshot hook that exits nonzero or raises aborts collection before
any artifact is downloaded, is not retried by output recovery, and therefore
cannot reach the fresh verifier. Existing hooks remain best-effort by default.
This is necessary for E2B because killing its sandbox before download also
makes its filesystem unavailable: SWE tasks instead freeze the unprivileged
agent and atomically snapshot the patch into a root-only path in the required
hook. The authoritative patch is written atomically below a root-only external
Git directory; agent-writable Git config, attributes, hooks, filters, and old
object databases are not trusted.

Run the included preflight directly to validate the SDK and Harbor method
surface without making an E2B API call:

```bash
HARBOR_ENV_TYPE=e2b E2B_API_KEY=present \
PYTHONPATH="$PWD:/path/to/harbor" \
/path/to/harbor/.venv/bin/python \
  examples/experimental/swe-agent-harbor-e2b/preflight.py
```

## Template admission and rollout scale

Harbor runs every trajectory in a separate subprocess, so 16 samples of a
first-seen task could otherwise race the native `alias_exists`/build sequence.
Aliases are mutable within an E2B team and are not an attestation by themselves.
Production admission builds each canonical template once while it performs the
live source/agent/empty/oracle checks. The launcher consumes the resulting
owner-only semantic admission JSONL, binds each record to the exact task ID,
task digest, immutable image, task-tree hash, and canonical build identity,
then maps every runtime alias to the stored `template_id` and `build_id`. It
checks access to each unique template ID with the SDK's ID endpoint; it does
not rebuild admitted templates. Rollout passes the exact `template_id` to
`AsyncSandbox.create` instead of resolving a mutable alias, and rollout-time
builds are forbidden.

The private pin file also binds the sorted task-ID set, sorted
`(task_id, task_digest)` set, and sorted
`(task_id, task_digest, task_tree_sha256)` runtime set. Server startup verifies
all three once, hashes every sealed task tree once, and exposes only the
resulting digests and count from the authenticated health endpoint. CPU
readiness, the final GPU fail-fast check, and offline evaluation require an
exact match to the admitted dataset summary; a server prepared from another
shard or another materializer/verifier policy cannot silently accept work.

An explicit `--allow-fresh-build` mode remains for isolated smoke tests. It
deduplicates by the exact build inputs (immutable image or Dockerfile
environment identity plus effective CPU/memory), uses a cryptographically
random one-use alias with `skip_cache=True`, and pins the returned IDs. It is
not used by the production launcher.

The fresh verifier is an immutable source image with no Dockerfile and zero
private build context. Hidden tests are uploaded only after the separate
no-network sandbox is running. The upload validates local and remote exact file
sets, rejects symlinks/special files, reads every byte back, then atomically
replaces `/tests`. Only after that succeeds is the root-owned model patch
transferred and the verifier started.

By default every direct task directory under `HARBOR_TASKS_DIR` is admitted.
For a bounded training shard, provide a newline-delimited allowlist; every task
the dataset can sample must be present:

```bash
export HARBOR_E2B_PREBUILD_TASK_IDS_FILE=/path/to/training-instance-ids.txt
export HARBOR_E2B_SEMANTIC_ADMISSION_MANIFESTS=/private/r2e.jsonl:/private/rebench.jsonl:/private/swe-gym.jsonl
export HARBOR_E2B_PREBUILD_CONCURRENCY=4
```

The sanitized admission report defaults to
`$TRIALS_DIR/e2b-template-admission.json`; it contains aggregate counts, roles,
and the nonsecret task-set/task-binding/task-runtime digests, not individual task IDs,
aliases, template IDs, hidden-derived hashes, or credentials. The separate
`$TRIALS_DIR/e2b-template-pins.json` is owner-only and
is consumed directly by the provider. A later alias reassignment cannot change
the pinned template ID. The E2B SDK cannot pin `build_id` at sandbox creation,
so anyone holding the same E2B team credentials remains inside the production
trust boundary; use a dedicated team/project credential and restrict access to
the pin file. Both producer and provider enforce a 64 MiB pin-file cap. A
realistic pretty-printed payload measured 1,796,865 bytes for 2,438 tasks and
5,608,629 bytes for 7,610 tasks (two runtime pins plus one task attestation per
task); a 32,000-task stress model was 23,584,059 bytes. The former 2 MiB cap was
therefore insufficient for the full Ultra shard. Two additional aggregate
SHA-256 bindings add only fixed-size overhead.

For private PBS deployments, use `terminus-2` as the default agent: it remains
in the Harbor host process and does not require exposing the Miles session
endpoint to a sandbox. `run_agent_server.sbatch` provides a one-day
reservation allocation with 32 host CPUs and no requested GPUs, W&B offline mode, owner-only
trial storage, and a logged non-secret connection URL. This prevents full-shard
prebuild time from consuming the GPU training allocation. The training
launcher polls the bearer-authenticated `/health` endpoint and does not start
Ray/NCCL work until the server has completed prebuild and is accepting traffic.
Set `MAX_CONCURRENT` explicitly to the same bounded value as the training
recipe's `ASYNC_MAX_CONCURRENT_SAMPLES` (currently 64); the launcher rejects a
mismatch and caps either value at 256 rather than silently imposing a server
bottleneck. Thirty-two CPUs provide a conservative one-core-per-two-inflight
starting point for async E2B orchestration, but live throughput at concurrency
64 has not yet been measured and remains a production scale gate.

Do not spend a four-node GPU allocation waiting for server prebuild. After the
server job prints its non-secret URL, submit training through the dependency
helper:

```bash
export AGENT_SERVER_URL=http://private-server-host:port
export HARBOR_RUN_SECRET=<same-job-scoped-master>
bash experiments/scripts/swe/async/swe-rebench-v2-swe-gym/\
qwen3-4b/submit_when_ready.sh
```

It submits a CPU-only reservation readiness job, then submits the GPU job with
`afterok:<readiness-job>`. Both submissions use fixed-name environment exports;
every underlying job is submitted without `-V` and receives only a fixed `-v`
allowlist, so the secret value is absent from argv and logs. The GPU job performs only a
final authenticated health check capped at 60 seconds, so a stale or restarted
server fails before Ray/NCCL initialization rather than consuming an hour of GPU
time.

## Network topology

`terminus-2` is the simplest deployment: it runs on the Harbor server host, so
E2B only receives shell/file commands and does not need an inbound route to the
Miles model endpoint. A sandbox-resident agent such as `mini-swe-agent` must be
able to reach the Miles session URL. In that case set
`MILES_ROUTER_EXTERNAL_HOST` to a hostname reachable from E2B and include it in
`HARBOR_AGENT_ALLOWED_HOSTS`; do not expose an unauthenticated session endpoint
to the public internet.

The provider defaults to E2B Cloud. `E2B_API_URL` and `E2B_SANDBOX_URL` remain
SDK-owned process settings for an E2B-compatible self-hosted deployment. They
are never copied into task metadata.

Harbor accepts either `environment/Dockerfile` in a task or a pinned
`[environment].docker_image`. For the former, native E2B builds a deterministic
template with the task environment directory as the Docker build context; for
the latter it creates the template from the source image directly and does not
execute the task Dockerfile. Docker Compose task environments are not supported
by the native E2B provider.

E2B SDK 2.25 and its template-build API do not expose a build-network policy.
Runtime `no-network` begins only after sandbox creation and does not constrain
template building. `from_image` does not replay the source Dockerfile's `RUN`,
`CMD`, or `ENTRYPOINT` as user build steps, but its complete root filesystem
(including enabled init/service hooks) remains trusted code while E2B provisions
the template. Consequently immutable digest pinning is necessary but not a
build-time no-egress proof. Production permits template creation only in the
trusted semantic-admission job, supplies no training/server secrets to that
build, records the returned IDs, and forbids rollout-time builds. Strict
build-time no-egress requires self-hosted E2B or infrastructure firewall support
covering provisioning, user steps, and finalization; hosted SDK 2.25 cannot
enforce it.

That precedence matters for SWE hardening. The agent uses
`environment/Dockerfile` with exactly one immutable `FROM` so it can remove gold
history, externalize the trusted Git directory, strip SUID/SGID/file
capabilities, and install the invalid-patch sentinel. The verifier instead sets
`[verifier.environment].docker_image` to the same immutable source image and
must not contain `tests/Dockerfile`; this prevents hidden tests and expected
outputs from ever reaching a registry/template build service. Runtime root
bootstrap performs the fresh-Git and privilege hardening after the no-network
sandbox starts.

## Training and verification

Prepare rows with the canonical `examples/swe-agent/download_and_process_data.py`
converter, using an agent supported by the Harbor checkout. Then launch
`examples/swe-agent/run.py` unchanged and set `--agent-server-url` to this
server. The training-side response and reward fields remain:

- `reward`: Harbor verifier reward, consumed by `generate.reward_func`;
- `exit_status` and `eval_report`: terminal verifier outcome;
- `agent_metrics`: turns, tool calls, and timing metrics.

For repository-level SWE, the task verifier must grade in a clean state from a
pinned base commit/image. Set `[verifier].environment_mode = "separate"` so
Harbor stops the agent sandbox, starts a fresh no-network E2B verifier sandbox,
and re-materializes only the required root-captured patch. The verifier deletes
the source image's Git object database, creates a synthetic base commit, checks
the model patch against the private oracle-derived path allowlist, applies
hidden tests, and runs model-controlled code as UID 1000 with NoNewPrivs and no
effective capabilities. Missing transport/artifact/bootstrap evidence is
ungraded; explicit invalid agent state, patch rejection, parser-invalid output,
and bounded test timeouts receive reward zero.

The model path policy is a training-pilot safety boundary: it permits only
paths touched by the trusted oracle patch. An alternative valid fix that needs
a different production file is therefore a known false negative. Do not widen
this allowlist to hidden-test, configuration, or toolchain paths merely to raise
training coverage. The policy is not applied to official downstream evaluation;
official SWE benchmark tasks must be prepared and scored independently with the
benchmark harness.

Production materialization can emit an owner-only
`miles-swe-materialization-evidence-v1` JSONL. Each row binds the instance,
task/content digests, exact OCI image digest, semantic-admission record, and a
canonical owner-only task-tree hash. Dry-run or mutable-image modes cannot emit
this evidence. Training finalization must re-hash the tree and require the
evidence rather than promoting a dry-run directory by path alone.

Start with completed-rollout replay only. Partial/inflight replay is unsafe
unless the exact sandbox snapshot, repository state, and agent state can be
restored.

Normal completion and graceful `/flush` both run Harbor's E2B `stop()` and kill
the sandbox. An agent-server `SIGKILL`, host loss, or power loss cannot run
client-side cleanup; in that case the native provider's 24-hour E2B sandbox TTL
is the final dead-man switch. Provider-side orphan monitoring is still required
for a production service.

An immutable digest proves identity, not publisher trust. Miles admission also
enforces source-schema-specific Docker Hub publishers: the exact documented
R2E-Gym repositories under `namanjain12`, SWE-Gym images under the
`xingyaoww/sweb.eval.x86_64.*` family, and SWE-ReBench images under
`swerebenchv2`. Dataset identity and the Filtered-Verified canonicalization
marker are bound before OCI resolution; another registry or namespace fails
closed. Both the agent image
build and verifier runtime must execute binaries supplied by the source image as
root before hardening completes. Production admission therefore requires a
trusted image/digest policy plus live empty=0,
oracle=1, runtime/tool, NoNewPrivs/capability, history-leak, and no-network
checks. The in-training verifier is deliberately restrictive but is not a
replacement for the official downstream SWE benchmark harness; report official
downstream scores separately.

## Current validation boundary

Unit tests cover Docker/Daytona compatibility, E2B fail-closed configuration,
graceful worker cancellation, secret non-propagation, template-ID pinning,
NoNewPrivs shell-injection resistance, native create/stop, exact private-package
upload/readback, root-only artifact transfer, required collection, malicious Git
config/attributes, fresh object databases, model path policy, official parser
failure behavior, hardlink rejection, semantic template reuse, and
separate-verifier lifecycle ordering. All five patches apply from the exact
pinned clean checkout. The latest timeout/auth/attestation snapshot passes 175
patched-Harbor tests and 74 Miles Harbor contract tests locally, including
trial-wall cleanup, ungraded timeout handling, and the worker/sandbox secret
boundary. The durable preflight reruns both sets.

`tests/slurm/test_harbor_e2b_preflight.sbatch` is the durable CPU-only validation
and makes no E2B API call. Submit the live probe only through
`tests/slurm/submit_harbor_e2b_live.sh`; its fixed-name export allowlist passes
`E2B_API_KEY` and an admitted immutable `E2B_LIVE_SOURCE_IMAGE` without copying
the rest of the submission environment. The probe creates one fresh template
by random alias, starts by returned template ID, checks
NoNewPrivs/CapEff/no-network, exercises exec/upload/download, and kills the
sandbox. No live template or sandbox is created when the key or admitted image
is absent, and absence must not be reported as live validation.
