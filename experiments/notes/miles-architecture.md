# miles architecture — the short version

## The loop

Every job is a loop over four objects (`docs/user-guide/concepts.md`):

```python
for it in range(num_rollout):
    prompts   = dataset.sample(rollout_batch_size)
    responses = sglang.generate(prompts, n=n_samples_per_prompt)   # 1. Sample
    rewards   = reward_fn(prompts, responses, labels)              # 2. Score
    for step in range(num_steps_per_rollout):                      # 3. Optimize
        loss = grpo_loss(actor, ref, pack(..., size=global_batch_size))
    p2p_weight_transfer(actor → sglang_engines)                    # 4. Sync
```

| Object | Where it lives |
|---|---|
| Prompt dataset | JSONL (`--prompt-data`) |
| Rollout | SGLang engines behind a router |
| Reward | `--rm-type` (built-in) or `--custom-rm-path` |
| Actor | Megatron `torch_dist` (or HF under FSDP) |
| Reference | frozen copy from `--ref-load`, for KL |

**Four-knob invariant** — miles refuses to start if it does not hold:

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

## Directory map

```
train.py / train_async.py       drivers (synchronous / fully-async)
miles/
  ray/            placement_group.py (GPU allocation, --colocate), actor groups
    rollout/      rollout_manager, rollout_server, server_group, router_manager
  rollout/        sglang_rollout.py (default rollout fn), fully_async_rollout.py
    generate_hub/ single_turn.py, multi_turn.py, agentic_tool_call.py
    rm_hub/       deepscaler, math, dapo, gpqa, f1, ifbench, remote_rm
    filter_hub/   dynamic-sampling filters
    session/      session server: TITO recording for multi-turn / agentic runs
  backends/       megatron_utils, sglang_utils (SglangConfig, PD), training_utils
  router/         miles' own lightweight router
  utils/arguments.py   every CLI flag and its validation — the de-facto spec
scripts/          per-model launchers (.sh with arg groups, .py with typer)
tools/            checkpoint conversion and quantization
examples/         recipes (retool_v2, swe-agent, openenv, nemo-gym, …)
```

## Plug points

Three nested layers plus the reward hook. Replacing an outer layer takes over
everything the inner ones would do (`docs/user-guide/environments.md`).

| Flag | Replaces | Default |
|---|---|---|
| `--rollout-function-path` | batching, filtering, the whole rollout | `miles/rollout/sglang_rollout.py` |
| `--custom-generate-function-path` | how one sample is generated | `generate_hub/single_turn.py` |
| `--custom-agent-function-path` | the agent loop; session server records tokens | — |
| `--rm-type` / `--custom-rm-path` | scoring | `deepscaler` etc. |

Our two recipes sit at different depths:

- `math/sync/<model>` — everything default, `--rm-type deepscaler`.
- `tool_multiturn/<model>` — `--custom-generate-function-path
  miles.rollout.generate_hub.multi_turn.generate` plus tool specs, a tool
  executor and `--custom-rm-path`; the framework still owns turn scheduling,
  token concatenation and loss masking.

## Execution-mode axes

| Axis | Flag |
|---|---|
| Colocated vs disaggregated placement | `--colocate` |
| Sync vs fully-async | `--fully-async` + `train_async.py`, `--async-max-concurrent-samples` |
| Prefill/decode disaggregation | `--prefill-num-servers`, or `server_groups` in the SGLang YAML |
| Session / TITO recording | `--use-session-server`, `--tito-model` |
| Dynamic sampling | `--dynamic-sampling-filter-path`, `--over-sampling-batch-size` |
| Training backend | Megatron (default) or FSDP |

## Launch path

```
run.sbatch  → one srun task per Slurm node (pyxis) → train.sh
                                                      ├─ source scripts/models/<type>.sh
                                                      ├─ source experiments/common/ray_cluster.sh
                                                      │    ├─ node 0: start Ray head
                                                      │    └─ other nodes: join and wait
                                                      └─ node 0: ray job submit -- python3 train[_async].py
                                                               ├─ create_placement_groups()
                                                               ├─ create_rollout_manager() → router + SGLang engines
                                                               ├─ create_training_models() → actor + reference
                                                               └─ rollout → reward → train → weight sync
```

This is the maintained `experiments/scripts/**/run.sbatch` path. It starts and
joins the multi-node Ray cluster explicitly through
`experiments/common/ray_cluster.sh`; these recipes do not use
`MILES_SCRIPT_EXTERNAL_RAY`. The separate `.py` launchers under top-level
`scripts/` use `miles/utils/external_utils/command_utils.py` (`execute_train`)
and retain their own optional external-Ray mode. Evidence from one launcher
family must not be used to describe the other.
