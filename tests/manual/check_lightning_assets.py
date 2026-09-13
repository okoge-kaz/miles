"""Check the staged Lightning policy, unfiltered math data, and Megatron release."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--megatron-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.hf_checkpoint / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 0
    index = json.loads((args.hf_checkpoint / "model.safetensors.index.json").read_text())
    assert not any(key.startswith("mtp.") for key in index["weight_map"])
    assert all((args.hf_checkpoint / name).is_file() for name in set(index["weight_map"].values()))
    assert (args.megatron_checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == "release"
    assert (args.megatron_checkpoint / "release" / ".metadata").is_file()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
    rows = [json.loads(line) for line in args.prompt_data.read_text().splitlines() if line.strip()]
    lengths = [
        len(
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True
            )
        )
        for row in rows
    ]
    assert len(rows) == 17398
    assert min(lengths) > 0
    assert max(lengths) <= 2048, "Increase the recipe's prompt/context budget"
    assert all(row["label"] not in (None, "") for row in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "max_prompt_tokens": max(lengths),
                "mean_prompt_tokens": sum(lengths) / len(rows),
                "policy_tensors": len(index["weight_map"]),
                "release": str(args.megatron_checkpoint),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
