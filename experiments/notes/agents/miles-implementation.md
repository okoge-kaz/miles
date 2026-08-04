# Work log — miles implementation

## 2026-08-03 — code reading for the experiments/ scripts

Repo state: `main` at `5400905d`.

### Launch path

Two launcher styles coexist:

- `scripts/run-*.sh` — bash arrays (`MODEL_ARGS`, `CKPT_ARGS`, …) then
  `ray start --head` + `ray job submit -- python3 train.py`. `experiments/*/train.sh`
  follows this shape; only the paths differ.
- `scripts/run_*.py` — typer + dataclass (`miles/utils/typer_utils.py:25`,
  `dataclass_cli`, env prefix `MILES_SCRIPT_`). The shared implementation is
  `miles/utils/external_utils/command_utils.py:109` (`execute_train`): it kills
  stale processes, starts ray unless `MILES_SCRIPT_EXTERNAL_RAY=1`, assembles a
  runtime-env JSON (PYTHONPATH, `CUDA_DEVICE_MAX_CONNECTIONS`, NVLS detection,
  NCCL socket vars) and submits the job. `ExecuteTrainConfig.num_nodes` defaults
  to `$SLURM_JOB_NUM_NODES` (`:104`), so the `.py` launchers are already
  Slurm-aware.

`train.py` itself is 158 lines: placement groups → rollout manager (router +
SGLang engines) → training models → weight sync → loop.

### Plug points

`docs/user-guide/environments.md` defines three nested layers; verified against
the flags in `miles/utils/arguments.py`:

| Flag | Default implementation |
|---|---|
| `--rollout-function-path` | `miles/rollout/sglang_rollout.py` |
| `--custom-generate-function-path` | `generate_hub/single_turn.py` |
| `--custom-agent-function-path` | consumed by `generate_hub/agentic_tool_call.py` |
| `--rm-type` / `--custom-rm-path` | `rm_hub/` (`deepscaler`, `math`, `dapo`, `gpqa`, `f1`, `ifbench`, `remote_rm`, …) |

`generate_hub/multi_turn.py` is the framework-side multi-turn loop the ReTool v2
example rides on; the example supplies only tool specs, a tool executor and a
reward function. `generate_hub/agentic_tool_call.py` documents the agent-function
contract (`base_url, prompt, request_kwargs, metadata` → optional metadata dict).

### Batch-size invariant

`miles/utils/arguments.py:3057`:

```
global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout
```

and it asserts consistency if `--global-batch-size` is also given. Dynamic
sampling adds `over_sampling_batch_size` (`:3072`), defaulted to
`rollout_batch_size` and asserted `>=` it. In fully-async mode
`--async-max-concurrent-samples` (`:630`) decouples generation concurrency from
the training batch, with a floor of `n_samples_per_prompt` (`:3048`).

Relevance to the throughput study: **concurrent sandboxes** =
`max(async_max_concurrent_samples, over_sampling_batch_size × n_samples_per_prompt)`,
not simply `rollout_batch_size × n_samples_per_prompt`.

### Routing / session

- `miles/ray/rollout/router_manager.py:37` — `assert not has_pd_disaggregation`
  for the miles router: **PD disaggregation requires the SGLang router**.
- `miles/rollout/session/core.py:307` — the session server injects
  `X-SMG-Routing-Key: <session_id>`; with router policy `manual` or
  `consistent_hashing` (`generate_utils/generate_endpoint_utils.py:37`) this pins
  a session to one engine. `--use-session-server` auto-selects policy `manual`
  (`backends/sglang_utils/arguments.py:159`).
- `--session-server-port <start> <end>` launches one server per port; the
  session URL carries the affinity (`router_manager.py:96`).

### PD disaggregation

`--prefill-num-servers N` → `SglangConfig.from_prefill_num_servers`
(`backends/sglang_utils/sglang_config.py:170`), mutually exclusive with the YAML
`sglang_config` and with `--rollout-external` (`arguments.py:3129`). KV transfer
is handled by the SGLang Model Gateway.

### Not verified

- No run has been executed yet; every claim above is from reading the code, not
  from observing behaviour.
- FSDP backend paths (`scripts/run_*_fsdp.py`) were not read.
