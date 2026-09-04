"""Convert a raw math dataset into the JSONL shape miles' rollout expects.

Output rows look like the DAPO-Math file that the recipes already read:

    {"prompt": [{"role": "user", "content": "<instruction + problem>"}],
     "label": "<ground-truth answer>",
     "metadata": {"source": "<dataset>"}}

`--input-key prompt --label-key label --apply-chat-template` then works
unchanged, and `--rm-type deepscaler` compares the boxed answer to `label`.

Usage:
    python build_math_jsonl.py --dataset deepscaler \
        --input  /data/deepscaler-preview/deepscaler.json \
        --output /data/deepscaler-preview/deepscaler-miles.jsonl
"""

import argparse
import json
from pathlib import Path

# Same instruction the DAPO-Math file carries, so the prompt distribution the
# policy sees does not change between datasets.
INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: \\boxed{$Answer} where $Answer is "
    "the answer to the problem.\n\n"
)
REMINDER = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def read_rows(path: Path):
    if path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        yield from (data if isinstance(data, list) else [data])
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        yield from table.to_pylist()
    else:
        raise ValueError(f"unsupported input: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, nargs="+", help="one or more files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", required=True, help="tag written into metadata.source")
    ap.add_argument("--problem-key", default="problem")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--limit", type=int, default=None)
    # Some sources already carry their own answer-format instruction. Nemotron-RL-Math-v2
    # rows begin "Your task is to find the solution to this math problem. Make sure your
    # answer is inside \\boxed{}.", so prepending INSTRUCTION as well gives the policy two
    # different format directives in one prompt.
    ap.add_argument(
        "--no-instruction",
        action="store_true",
        help="use the problem text verbatim; the source already states the answer format",
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    with out.open("w") as f:
        for path in args.input:
            for row in read_rows(Path(path)):
                problem = row.get(args.problem_key)
                answer = row.get(args.answer_key)
                # A row without a checkable ground truth would train against a
                # reward that is always zero, so drop it loudly rather than
                # silently feeding it in.
                if not problem or answer in (None, ""):
                    skipped += 1
                    continue
                if isinstance(answer, list):
                    answer = answer[0] if answer else None
                    if answer is None:
                        skipped += 1
                        continue
                content = str(problem) if args.no_instruction else INSTRUCTION + str(problem) + REMINDER
                f.write(
                    json.dumps(
                        {
                            "prompt": [{"role": "user", "content": content}],
                            "label": str(answer),
                            "metadata": {"source": args.dataset},
                        }
                    )
                    + "\n"
                )
                kept += 1
                if args.limit and kept >= args.limit:
                    break
            if args.limit and kept >= args.limit:
                break

    print(f"{args.dataset}: wrote {kept} rows to {out} (skipped {skipped} without problem/answer)")


if __name__ == "__main__":
    main()
