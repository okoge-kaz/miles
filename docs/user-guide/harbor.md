---
title: Harbor
description: Train agents on mixed task suites (SWE-bench, Terminal-Bench, custom) through the Harbor framework.
---

[Harbor](https://github.com/harbor-framework/harbor) is an agent-environment
framework from the Laude Institute: agent orchestration and grading are unified
in a single `Trial.run()` call, and a task is fully described by four files
(`instruction.md`, `Dockerfile`, `test.sh`, `task.toml`), so mixed task suites —
SWE-bench, Terminal-Bench, custom tasks — train through one endpoint.

Miles integrates Harbor as an
[agent-function integration](/user-guide/environments): the agent function
hands each session's OpenAI-compatible URL to Harbor, which runs the per-task
sandbox, runs the agent against that URL, and grades the result; the grade
becomes the sample's reward through a custom reward hook.

## Try it

Two execution modes; each example README is the complete guide for its mode:

- Tasks on a **local Docker daemon** → the agent-server mode:
  [`examples/swe-agent-harbor-docker`](https://github.com/radixark/miles/tree/main/examples/swe-agent-harbor-docker).
- Tasks on a **cloud [sandbox provider](/user-guide/sandbox-providers)** → the
  in-process mode, no server in between:
  [`examples/experimental/harbor`](https://github.com/radixark/miles/tree/main/examples/experimental/harbor).

## E2B sandboxes

The Miles-specific Harbor server can use Harbor's native E2B provider without
changing the training-side `/run` or reward API. The maintained integration is
under `examples/experimental/swe-agent-harbor-e2b`: it wires the pinned Harbor
Miles server to `EnvironmentConfig(type="e2b")`, skips local Docker maintenance,
and gives Harbor a graceful cleanup window when a rollout is flushed.

E2B credentials are read only from the agent-server process environment. The
launcher never reads `.env`, never adds the key to task metadata, and never
passes it on the command line. See the example README for the exact Harbor
commit, offline preflight, network topology, and task-image requirements.
Repository-level tasks bind each prompt row to its trusted materialization with
a SHA-256 task digest; the server rejects stale or mismatched task trees before
starting a sandbox.
