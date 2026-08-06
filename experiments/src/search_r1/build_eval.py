"""Convert the FlashRAG QA sets into the shape Search-R1's rollout and reward expect.

    python -m experiments.src.search_r1.build_eval \
        --flashrag-dir /data/flashrag-datasets --out-dir /data/search-r1/eval

The raw FlashRAG files carry `question` and `golden_answers` and nothing else.
Pointing an eval config straight at them fails twice over, and only one of the
two failures is loud:

  * `generate_with_search.generate` uses `sample.prompt` verbatim as the text it
    sends to sglang -- no template is applied anywhere in the rollout. A bare
    question therefore never tells the model that `<search>` exists, so it
    answers from memory and the retriever is never exercised. Nothing errors.
  * `reward_func` reads `sample.label["ground_truth"]` and `compute_score_em`
    then reads `["target"]` from that. A bare list raises
    `TypeError: list indices must be integers or slices, not str`.

So the eval rows are built to match the *training* parquet exactly: `prompt` is
the one-message chat list carrying Search-R1's instruction block, and
`reward_model` is `{"ground_truth": {"target": [...]}}`. Same `--input-key`,
same `--label-key`, same template as training -- which is the point, since an
eval that renders its prompts differently from training is measuring a different
task.
"""

import argparse
import json
from pathlib import Path

# Verbatim from PeterJinGo/nq_hotpotqa_train's `prompt` column. Copied rather
# than paraphrased: this string is what the policy is trained against, and the
# published Search-R1 numbers are for this wording.
PROMPT_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call a search engine by <search> query "
    "</search> and it will return the top searched results between <information> "
    "and </information>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> "
    "Beijing </answer>. Question: {question}\n"
)

# The seven sets Search-R1 reports, and which file each ships its eval split in.
# nq and hotpotqa are in-domain (the training mix); the rest are transfer.
SPLITS = {
    "nq": "nq/test.jsonl",
    "hotpotqa": "hotpotqa/dev.jsonl",
    "triviaqa": "triviaqa/test.jsonl",
    "popqa": "popqa/test.jsonl",
    "2wikimultihopqa": "2wikimultihopqa/dev.jsonl",
    "musique": "musique/dev.jsonl",
    "bamboogle": "bamboogle/test.jsonl",
}


def convert_row(row):
    question = (row.get("question") or "").strip()
    answers = row.get("golden_answers") or []
    if isinstance(answers, str):
        answers = [answers]
    answers = [str(a) for a in answers if str(a).strip()]
    if not question or not answers:
        return None
    return {
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        # Nested exactly as compute_score_em reads it: label["ground_truth"]["target"].
        "reward_model": {"ground_truth": {"target": answers}, "style": "rule"},
        "metadata": {"source": row.get("id"), "question": question},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flashrag-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=500,
        help="cap per set (0 = no cap). Each prompt is a multi-turn episode with "
        "retrieval between turns, so this is far dearer than a math eval.",
    )
    args = ap.parse_args()

    root, out_dir = Path(args.flashrag_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rel in SPLITS.items():
        src = root / rel
        if not src.exists():
            print(f"  {name:18s} MISSING {src}")
            continue
        kept = skipped = 0
        dst = out_dir / f"{name}-miles.jsonl"
        with src.open() as fin, dst.open("w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                converted = convert_row(json.loads(line))
                if converted is None:
                    skipped += 1
                    continue
                fout.write(json.dumps(converted) + "\n")
                kept += 1
                if args.limit and kept >= args.limit:
                    break
        print(f"  {name:18s} {kept:5d} rows -> {dst}" + (f" (skipped {skipped})" if skipped else ""))


if __name__ == "__main__":
    main()
