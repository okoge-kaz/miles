# Datasets — where they live and how to look at them

For the full staged inventory by genre — every path, row count, verifier and
whether it has been verified — see [dataset-inventory.md](dataset-inventory.md).
This document covers the math sets in use plus the mechanics of reading a JSONL.

## Location

Host: `/lustre/fsw/portfolios/coreai/users/kfujii/datasets` → mounted at `/data`
inside the container.

| Dataset | Path | Used by |
|---|---|---|
| DAPO-Math-17K | `/data/dapo-math-17k/dapo-math-17k.jsonl` | 17,398-row source for the policy-specific difficulty filters |
| DAPO-Math SFT p10-90 | `/data/dapo-math-p10-90-<model>/…jsonl` | policy-specific outputs for Qwen3 4B (10,891), 8B (9,816), and 30B-A3B (7,425) |
| AIME-2024 / 25 / 26 | prepared by `experiments/scripts/reasoning_eval/prepare-aime-data.sbatch` | reportable offline checkpoint evaluation; maintained training recipes set `EVAL_INTERVAL=0` |
| MATH-500 | `/data/math-500/math-500.jsonl` | training-aligned offline diagnostic only; 500 canonical eval-only rows |

DAPO-Math-17K is listed in `experiments/setup/manifests/datasets.txt` and staged
through `experiments/setup/download/stage_all.sh` from
`zhuzilin/dapo-math-17k` — the same repo the upstream Miles scripts use. The
policy-specific p10-90 files are produced from it by
`experiments/tools/difficulty_filter`. The former
`/data/dapo-math-p10-80/dapo-math-p10-80.jsonl` path is not staged on the current
cluster and is not a maintained-recipe input.

The reportable AIME path is `experiments/scripts/reasoning_eval/`: its setup job
prepares pinned NeMo Skills AIME24/25/26 data and its evaluator records dataset,
image, checkpoint and sampling-protocol provenance. Current-refactor job 306691
completed the full 30-prompt AIME24 set with one repeat and wrote checksummed
artifacts. The YAML `experiments/configs/eval_aime.yaml` is a separate,
training-aligned diagnostic contract. Job 306776 generated and scored two AIME24
rows through that YAML-entry path, but AIME25/26 and the full YAML-config suite
remain unverified. Unified-runner job 307365 separately completed two-prompt,
16-repeat current-SFT smokes for all three AIME years and published checksummed
artifacts; those small samples validate execution, not full-set accuracy.

CPU data-contract job 306823 separately completed ten tests and audited the
actual AIME24/25/26 artifacts: 30 canonical eval-only rows per year with matching
source and output SHA-256 provenance. Job 306822 completed four tests and audited
the actual 500-row MATH-500 artifact plus its source provenance. These jobs
validate prepared data and config contracts, not model generation or benchmark
scores; the generation evidence remains 306691/306776/307365 as scoped above.

Difficulty filtering for all three SFT policies is complete. The 8B first pass
had one failed prompt group and was resumed to the full 17,398 measurements before
the 9,816-row output was finalized. Conversion/filter completion does not by
itself prove that a corresponding 8B or 30B-A3B RL recipe can forward, backward,
checkpoint and resume.

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
srun -A coreai_horizon_dilations -p cpu --qos=cpu-interactive -n1 --time=20 \
  --container-image=/lustre/fsw/portfolios/coreai/users/kfujii/containers/miles-search-r1-b300-20260815.sqsh \
  --container-mounts=/lustre/fsw/portfolios/coreai/users/kfujii/datasets:/data,/lustre/fsw/portfolios/coreai/users/kfujii/checkpoints/huggingface/Qwen3-4B-Base/LR2.0e-5-SEQ32768-GBS128-MBS1-TP1-PP1-CP1-EP1-PACK1-standard-cp-STEPS4000:/ckpt/hf \
  python3 - <<'PY'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/ckpt/hf/iter_0004000")
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
