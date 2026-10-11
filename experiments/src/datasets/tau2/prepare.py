import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from experiments.src.datasets.common.io import read_rows, write_jsonl
from experiments.src.datasets.common.openai_responses import to_chat_messages
from experiments.src.datasets.common.parallel import add_parallel_arguments, map_rows
from experiments.src.datasets.common.prepare import sha256


def adapt_tau2(row: dict, *, split: str) -> dict:
    if split not in {"train", "test"}:
        raise ValueError("Tau2 needs an explicit official train/test split")
    config = row["config"]
    if config["domain"] not in {"airline", "retail", "telecom"}:
        raise ValueError("This CPU-only recipe supports the three original Tau2 domains")
    if "NL_ASSERTION" in (row["task"].get("evaluation_criteria") or {}).get("reward_basis", []):
        raise ValueError("Use NeMo Gym prepared no-judge rows; NL_ASSERTION needs a separate judge")
    native = deepcopy(row)
    native["config"]["task_split_name"] = split
    prompt = to_chat_messages(native["responses_create_params"]["input"])
    if not prompt:
        raise ValueError("Native NeMo Gym row has no policy prompt")
    return {
        "prompt": prompt,
        "label": "NeMo Gym Tau2 terminal reward",
        "metadata": {
            "source": f"nemo-gym-tau2-{config['domain']}",
            "category": "tool",
            "verifier": "nemo_gym_tau2",
            "eval_only": split == "test",
            "stateful_environment": True,
            "interaction_mode": "stateful_multi_turn_user_simulator_environment",
            "tau_task_id": str(row["task"]["id"]),
            "tau_domain": config["domain"],
            "tau_split": split,
            "nemo_gym": {"service": "tau2", "request": native},
        },
    }


def official_splits() -> dict:
    # Run preparation in the pinned NeMo Gym CPU runtime, where Tau owns its data.
    import os
    import nemo_gym

    root = Path(nemo_gym.__file__).resolve().parent.parent
    os.environ["TAU2_DATA_DIR"] = str(root / "responses_api_agents/tau2/tau2_data")
    from tau2.runner.helpers import load_tasks

    result = {}
    for domain in ("airline", "retail", "telecom"):
        splits = {
            split: {str(task.id) for task in load_tasks(task_set_name=domain, task_split_name=split)}
            for split in ("train", "test")
        }
        if splits["train"] & splits["test"]:
            raise ValueError(f"Official Tau2 split overlap: {domain}")
        result[domain] = splits
    return result


def _assign_splits(rows, splits):
    seen = set()
    for row in rows:
        domain, task_id = row["config"]["domain"], str(row["task"]["id"])
        identity = domain, task_id
        if identity in seen:
            raise ValueError(f"Duplicate task: {identity}")
        seen.add(identity)
        matching = [split for split, ids in splits[domain].items() if task_id in ids]
        if len(matching) != 1:
            raise ValueError(f"Task has no unambiguous official split: {identity}")
        yield row, matching[0]


def _adapt_split_task(item):
    row, split = item
    return split, adapt_tau2(row, split=split)


def main():
    parser = argparse.ArgumentParser(
        description="Convert native NeMo Gym Tau2 rows using official disjoint train/test task IDs"
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    add_parallel_arguments(parser)
    args = parser.parse_args()
    tasks = _assign_splits(read_rows(args.input), official_splits())
    outputs = {"train": [], "test": []}
    with map_rows(tasks, _adapt_split_task, num_proc=args.num_proc, chunk_size=args.chunk_size) as rows:
        for split, row in rows:
            outputs[split].append(row)
    if not all(outputs.values()):
        raise ValueError("Input must include both official training and held-out tasks")
    counts = {
        "train": write_jsonl(args.train_output, outputs["train"]),
        "test": write_jsonl(args.validation_output, outputs["test"]),
    }
    digests = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.input}
    import nemo_gym

    root = Path(nemo_gym.__file__).resolve().parent.parent
    sources = json.loads((root / "miles-runtime.json").read_text())["sources"]
    for split, destination in (("train", args.train_output), ("test", args.validation_output)):
        manifest = {
            "rows": counts[split],
            "split": split,
            "source_sha256": digests,
            "sources": sources,
            "sha256": sha256(destination),
            "adapter": "adapt_tau2",
            "conversion_schema": 2,
        }
        destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": counts, "source_sha256": digests, "sources": sources}, indent=2))


if __name__ == "__main__":
    main()
