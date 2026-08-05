# Datasets — where they live and how to look at them

For the full staged inventory by genre — every path, row count, verifier and
whether it has been verified — see [dataset-inventory.md](dataset-inventory.md).
This document covers the math sets in use plus the mechanics of reading a JSONL.

## Location

Host: `/lustre/fsw/portfolios/coreai/users/kfujii/datasets` → mounted at `/data`
inside the container.

| Dataset | Path | Used by |
|---|---|---|
| DAPO-Math-17K | `/data/dapo-math-17k/dapo-math-17k.jsonl` | source for the difficulty-filtered set below |
| DAPO-Math p10-80 | `/data/dapo-math-p10-80/dapo-math-p10-80.jsonl` | training prompts (both recipes). 3962 of 17398, pass rate 0.1–0.8 measured with Qwen3-4B-Instruct-2507 |
| AIME-2025 | `/data/aime-2025/aime-2025.jsonl` | in-training eval, the only one |
| AIME-2023 / 24 / 25 / 26 | `/data/aime-20{23,24,25,26}/…` | offline eval on saved checkpoints (`src/offline_eval`) |

Both are pulled by `experiments/setup/download_assets.sbatch` from
`zhuzilin/dapo-math-17k` and `zhuzilin/aime-2024` — the same repos the upstream
miles scripts use, so the field names match what `run-qwen3-4B.sh` expects.

## Inspecting the files

`jq` is available on the login node, so no container is needed for a quick look.

```bash
D=/lustre/fsw/portfolios/coreai/users/kfujii/datasets/dapo-math-17k/dapo-math-17k.jsonl

wc -l "$D"                       # number of prompts
ls -lh "$D"

jq -r 'keys | @csv' "$D" | sort | uniq -c        # which keys exist, and how consistently
jq -s '.[0]' "$D" | head -50                     # first record, pretty-printed
jq -r '.prompt' "$D" | head -3                   # the prompt field only
jq -r '.label' "$D" | head -10                   # the ground-truth answers
```

Useful sanity checks before a run:

```bash
# every row has a prompt and a label?
jq -r 'select(.prompt == null or .label == null) | "BAD"' "$D" | wc -l

# rough prompt-length distribution in characters (a proxy for tokens)
jq -r '.prompt | tostring | length' "$D" | sort -n | awk '
  {a[NR]=$1} END {print "n="NR, "p50="a[int(NR*0.5)], "p95="a[int(NR*0.95)], "max="a[NR]}'
```

For a real token count (this is what `--rollout-max-prompt-len` is compared
against), use the tokenizer from the HF checkpoint inside the container:

```bash
srun -A coreai_horizon_dilations -p cpu -n1 --time=20 \
  --container-image=/lustre/fsw/portfolios/coreai/users/kfujii/container/miles-latest.sqsh \
  --container-mounts=/lustre/fsw/portfolios/coreai/users/kfujii/datasets:/data,/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/hf:/ckpt/hf \
  python3 - <<'PY'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/ckpt/hf/Qwen3-4B")
lens = []
with open("/data/dapo-math-17k/dapo-math-17k.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 2000:      # sample, not the whole file
            break
        lens.append(len(tok(json.loads(line)["prompt"])["input_ids"]))
lens.sort()
print("n", len(lens), "p50", lens[len(lens)//2], "p95", lens[int(len(lens)*0.95)], "max", lens[-1])
PY
```

## How miles reads a JSONL dataset

`miles/utils/data.py:186` (`Dataset`) reads the file line by line and maps four
CLI flags onto record keys:

| Flag | Default | Meaning |
|---|---|---|
| `--input-key` | `input` | the prompt field (our scripts pass `prompt`) |
| `--label-key` | `None` | ground truth handed to the reward function (`label`) |
| `--metadata-key` | `metadata` | free-form dict carried through to the reward hook / agent function |
| `--tool-key` | `None` | per-sample tool definitions; merged into `metadata["tools"]` |

Notes that matter in practice:

- `--apply-chat-template` turns the raw string into a conversation and renders it
  with the tokenizer's chat template (`_build_messages`, `miles/utils/data.py:133`).
  Without it, the prompt string is sent verbatim.
- A record whose prompt exceeds `--rollout-max-prompt-len` is dropped at load
  time, so a silently shrinking dataset usually means that flag is too small.
- `metadata` is the channel agentic recipes use: e.g. the SWE recipes put
  `instance_id` there and the agent function reads it.
- `--rollout-shuffle` shuffles per epoch with `--rollout-seed`; the sampling
  order otherwise follows file order.

## Adding your own dataset

Write a JSONL where each line is one prompt:

```json
{"prompt": "…question…", "label": "42", "metadata": {"source": "my-set"}}
```

then point `--prompt-data` at it and set `--input-key` / `--label-key` to match.
The reward side decides what `label` means — `--rm-type deepscaler` treats it as
the reference answer for a math verifier; a custom `--custom-rm-path` can treat
it as anything.
