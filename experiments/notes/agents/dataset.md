# Work log — dataset

## 2026-08-03 — data sources and the ingestion path

### Chosen sources

`experiments/setup/download_assets.sbatch` pulls:

- `zhuzilin/dapo-math-17k` → `$DATASET_DIR/dapo-math-17k/dapo-math-17k.jsonl`
- `zhuzilin/aime-2024` → `$DATASET_DIR/aime-2024/aime-2024.jsonl`

These are the repos the upstream launchers use (`scripts/run-qwen3-4B.sh`,
`examples/retool_v2/run_retool_multi_turn.py`), so the `--input-key prompt`
`--label-key label` mapping in our scripts matches without adaptation. The quick
start additionally mentions `BytedTsinghua-SIA/DAPO-Math-17K`; the `zhuzilin`
mirrors were chosen because the example scripts reference them directly.

Destination is `/lustre/fsw/portfolios/coreai/users/kfujii/datasets`, mounted at
`/data` in the container.

### Ingestion path in miles

`miles/rollout/data_source.py:49` builds a `Dataset`
(`miles/utils/data.py:186`) with:

| Constructor arg | CLI flag | Default |
|---|---|---|
| `prompt_key` | `--input-key` | `input` |
| `label_key` | `--label-key` | `None` |
| `metadata_key` | `--metadata-key` | `metadata` |
| `tool_key` | `--tool-key` | `None` |
| `max_length` | `--rollout-max-prompt-len` | — |

Behaviour worth remembering:

- `_build_messages` (`miles/utils/data.py:133`) converts the raw field into a
  conversation when `--apply-chat-template` or multimodal keys are set; the chat
  template is then rendered with `add_generation_prompt=True`.
- `tools` may be a JSON string, list or ndarray; it is normalised and merged into
  `metadata["tools"]` (`data.py:211`).
- `get_samples` (`data_source.py:88`) expands each prompt into
  `n_samples_per_prompt` deep copies sharing a `group_index` — the GRPO group.
- Epoch rollover reshuffles when `--rollout-shuffle` is set.

### Inspection recipes

`jq` is present on the login node (`/usr/bin/jq`), so a first look needs no
container. Recipes are written up in `notes/datasets.md` (key histogram, null
checks, character-length quantiles, and a tokenizer-based token-length sample
that runs inside the container).

### Not verified

- **The actual field names in the downloaded files have not been inspected** —
  the data has not been downloaded yet. `prompt` / `label` is inferred from the
  upstream launch scripts. Run the key histogram from `notes/datasets.md` right
  after the download job and correct the flags if it disagrees.
- Token-length distribution unknown, so `--rollout-max-prompt-len` is left at its
  default; long prompts would be silently dropped at load time.
