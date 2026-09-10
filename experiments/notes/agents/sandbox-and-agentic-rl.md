# Work log — sandboxes and agentic RL

## 2026-08-03 — what miles supports, and what can actually run here

### miles' agentic-RL surface (from the repo)

| Recipe | Layer | External dependency |
|---|---|---|
| `examples/swe-agent/` (Harbor) | agent function | Harbor agent server + **Docker host**; sandbox must reach the session server (inbound). `MILES_ROUTER_EXTERNAL_HOST` rewrites the URL for NAT (`swe_agent_function.py:76`) |
| `examples/experimental/nemo-gym/` | agent function | NeMo-Gym agent server; sandbox via the `nemo_gym.sandbox` provider API (docker / daytona / apptainer / ecs_fargate / opensandbox) |
| `examples/experimental/openenv/` | agent function | OpenEnv env server + Docker, **or** per-episode Daytona sandboxes, **or** `TB2_MODE=local` (degraded) |
| `examples/retool_v2/` | generate function | none — local subprocess Python tool |
| `examples/experimental/strands_sglang/` | generate function | strands / strands-sglang; subprocess interpreter |
| `examples/experimental/search-r1/` | generate function | local faiss retriever (GPU) **or** a serper.dev API key |
| `examples/experimental/tau-bench/` | generate function | external LLM API for the user simulator (litellm) |
| `examples/experimental/multi_agent/` | multi-agent function | none |

miles itself has no sandbox abstraction; the only provider it implements directly
is Daytona, inside the experimental OpenEnv example. Everything else is delegated
to the connector.

### What this cluster permits

Given no Docker, no Apptainer and an empty `/etc/subuid` (see
`notes/agents/cluster.md`):

| Option | Viable here |
|---|---|
| Harbor / OpenEnv docker mode | no — needs a Docker host off-cluster |
| Daytona (OpenEnv path) | yes — egress verified; outbound-only, but metered and quota-limited |
| Internal sandbox service via NeMo-Gym's provider API | **yes** — endpoint reachable (HTTP 401, key pending) |
| Apptainer provider | no — not installed |
| enroot-in-enroot, colocated on the GPU node | yes — pattern proven by another team; no CPU isolation possible without root |
| `TB2_MODE=local` / retool subprocess | yes — weak isolation, fine for wiring checks |

Onboarding to the internal service is a spreadsheet row plus a short post in the
support channel; details deliberately not recorded here.

### Colocation vs disaggregation — for the planned study

Points gathered from internal discussion, worth re-verifying by measurement:

- Colocated sandboxes compete with training; the memory watchdog can kill
  inference workers when sandbox limits are tightened.
- Without root there is **no way to enforce per-sandbox CPU limits** (no cgroup
  delegation; `taskset` reportedly ineffective; ray can only advise). This is a
  structural property of the HPC scheduler, not a tuning problem — a reportable
  finding in itself.
- Stalls have been observed at large global batch sizes with colocated sandboxes.
- The disaggregated service side reports peaks in the low thousands of concurrent
  sandboxes with request/limit tuning giving several-fold headroom.

Design implication recorded here so it is not lost: run **both arms on the same
cluster** (cw-dfw satisfies this) and change only the sandbox provider, keeping
harness, task images, policy and evaluation identical.

### Not verified

- No agentic recipe has been run on this cluster.
- The internal service has only been probed for reachability (HTTP 401); no
  sandbox has been created, so latency and concurrency behaviour are unmeasured.
