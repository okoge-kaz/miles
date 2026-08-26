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
- verifier execution and download of `/logs/artifacts/` before teardown.

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

The launcher never reads `.env`. Put `E2B_API_KEY` in the process environment
through a shell export, Slurm secret injection, or your secret manager. It
checks only that the value is present and never prints it. It also exports
`PYTHON_DOTENV_DISABLED=1`, so Harbor's optional registry client cannot
implicitly discover a dotenv file later in the process.

```bash
export E2B_API_KEY=<injected-secret>
export HARBOR_ROOT=/path/to/harbor
export HARBOR_TASKS_DIR=/path/to/materialized/harbor_tasks
export TRIALS_DIR=/path/to/trials

bash examples/experimental/swe-agent-harbor-e2b/launch_agent_server.sh
```

The launcher applies `harbor-miles-e2b.patch` idempotently to the Harbor
checkout. The patch changes only Miles-specific session-server glue; it does
not modify Harbor's E2B implementation. It selects the provider, disables the
local Docker login/prune/compose paths for both cloud backends, and makes worker
cancellation cleanup-safe. On `/flush`, the overlay sends the trial worker
`SIGINT` and gives Harbor 30 seconds (configured by
`HARBOR_WORKER_CANCEL_GRACE_SEC`) to run its shielded sandbox teardown before
falling back to a hard kill. If the checkout has drifted, the overlay fails
instead of attempting a fuzzy source rewrite.

Run the included preflight directly to validate the SDK and Harbor method
surface without making an E2B API call:

```bash
HARBOR_ENV_TYPE=e2b E2B_API_KEY=present \
PYTHONPATH="$PWD:/path/to/harbor" \
/path/to/harbor/.venv/bin/python \
  examples/experimental/swe-agent-harbor-e2b/preflight.py
```

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

That precedence matters for SWE hardening. If an agent Dockerfile must reset or
strip gold repository history, or a verifier Dockerfile must bake hidden tests,
do not set `docker_image` on either resolved environment config. Instead emit
`environment/Dockerfile` and `tests/Dockerfile`, each `FROM` the same pinned
source-image digest. The agent Dockerfile prepares the gold-free base state; the
tests Dockerfile creates the trusted separate-verifier image.

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
Harbor stops the agent sandbox, starts a fresh E2B verifier sandbox, and
re-materializes only declared artifacts such as the captured patch. Apply that
patch, run the canonical instance-specific tests, and preserve patch/test
artifacts. E2B isolation does not by itself prevent a verifier from trusting
modified in-agent tests if shared mode is used.

Start with completed-rollout replay only. Partial/inflight replay is unsafe
unless the exact sandbox snapshot, repository state, and agent state can be
restored.

Normal completion and graceful `/flush` both run Harbor's E2B `stop()` and kill
the sandbox. An agent-server `SIGKILL`, host loss, or power loss cannot run
client-side cleanup; in that case the native provider's 24-hour E2B sandbox TTL
is the final dead-man switch. Provider-side orphan monitoring is still required
for a production service.

## Current validation boundary

Unit tests cover Docker/Daytona compatibility, E2B fail-closed configuration,
graceful worker cancellation, secret non-propagation, native create/stop,
command/patch execution, upload/download, artifact collection, and a fresh E2B
agent-to-separate-verifier lifecycle with declared-artifact transfer. The overlay
is checked and imported against the official pinned source. The durable
`tests/slurm/test_harbor_e2b_preflight.sbatch` runs the same CPU-only boundary.
A live template or sandbox is intentionally not created when `E2B_API_KEY` is
absent.
