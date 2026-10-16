import argparse
import ast
import json
from pathlib import Path

from experiments.src.datasets.common.parallel import add_parallel_arguments
from experiments.src.datasets.common.prepare import convert
from experiments.src.datasets.instruction_following.download import IF_TRAIN_DATASET, IF_TRAIN_REVISION
from experiments.src.nemo_gym.bridge.payloads import instruction_metadata


def adapt_if_training(row: dict) -> dict:
    constraints = ast.literal_eval(row["ground_truth"])
    if not isinstance(constraints, list) or len(constraints) != 1:
        raise ValueError(f"Unexpected IF training constraint shape: {row['key']}")
    constraint = constraints[0]
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"IF training input must end with a user message: {row['key']}")
    ids, kwargs = constraint["instruction_id"], constraint["kwargs"]
    if not ids or len(ids) != len(kwargs):
        raise ValueError(f"Mismatched IF training constraint arguments: {row['key']}")
    metadata = {
        "source": "if-multi-constraints-train",
        "source_row_id": row["key"],
        "category": "ifevalg",
        "dataset_repo": IF_TRAIN_DATASET,
        "dataset_revision": IF_TRAIN_REVISION,
        "verifier": "ifeval_g",
        "instruction_id_list": ids,
        "kwargs": kwargs,
        "prompt_text": messages[-1]["content"],
        "eval_only": False,
    }
    instruction_metadata(metadata)
    return {"prompt": messages, "label": json.dumps(constraint, ensure_ascii=False), "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IF multi-constraint training rows")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_parallel_arguments(parser)
    args = parser.parse_args()
    print(
        json.dumps(
            convert(args.input, args.output, adapt_if_training, num_proc=args.num_proc, chunk_size=args.chunk_size),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
