"""Check the real Math/Code/STEM reward dispatcher, including sandbox execution."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.src.reward_sets.math_code_stem import reward
from miles.utils.types import Sample


def _reasoning_sample(data: Path) -> Sample:
    with data.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["metadata"]["source_dataset"] == "simple_equations":
                return Sample(
                    response=f"<answer>{row['label']}</answer>",
                    label=row["label"],
                    metadata=row["metadata"],
                )
    raise ValueError(f"No simple_equations training sample in {data}")


def _mcqa_sample(data: Path) -> Sample:
    with data.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["metadata"]["template_id"] == "mcqa_benchmark_002":
                return Sample(response=f"Answer: {row['label']}", label=row["label"], metadata=row["metadata"])
    raise ValueError(f"No mcqa_benchmark_002 training sample in {data}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = [
        Sample(response=r"Answer: \boxed{4}", label="4", metadata={"verifier": "math"}),
        Sample(
            response=(
                "```python\nimport os, socket\n"
                "assert not os.path.exists('/root/miles')\n"
                "assert [name for _, name in socket.if_nameindex()] == ['lo']\n"
                "a, b = map(int, input().split())\nprint(a + b)\n```"
            ),
            metadata={
                "verifier": "python_code",
                "unit_tests": {"inputs": ["1 2\n", "4 5\n"], "outputs": ["3\n", "9\n"]},
            },
        ),
        _mcqa_sample(args.data_root / "nemotron-rl-mcqa/miles-train.jsonl"),
        _reasoning_sample(args.data_root / "nemotron-rl-reasoninggym/miles-train.jsonl"),
    ]
    samples[-1].metadata = {**samples[-1].metadata, "verifier": "reasoning_gym"}
    wrong = [Sample(response="definitely-wrong", label=sample.label, metadata=sample.metadata) for sample in samples]
    wrong[1].response = "```python\nprint(0)\n```"
    scores = asyncio.run(reward(SimpleNamespace(), samples + wrong))
    expected = [1.0] * len(samples) + [0.0] * len(wrong)
    report = {"verifiers": [sample.metadata["verifier"] for sample in samples], "scores": scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert scores == expected, report
    print(json.dumps(report))


if __name__ == "__main__":
    main()
