"""Convert the FlashRAG QA sets into the shape Search-R1's rollout and reward expect.

    python -m experiments.src.datasets.search_r1.build_eval \
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
task. All seven outputs are staged before publication; a deterministic checksum
manifest is replaced last and is the readiness record consumed by setup and
evaluation jobs.
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

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

SOURCE_REPOSITORY = "RUC-NLPIR/FlashRAG_datasets"
SOURCE_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
MANIFEST_NAME = "search-r1-eval.manifest.json"


def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_split(source: Path, output: Path, limit: int) -> tuple[int, int]:
    kept = 0
    skipped = 0
    with source.open(encoding="utf-8") as input_stream, output.open(
        "w", encoding="utf-8"
    ) as output_stream:
        for line in input_stream:
            line = line.strip()
            if not line:
                continue
            converted = convert_row(json.loads(line))
            if converted is None:
                skipped += 1
                continue
            output_stream.write(json.dumps(converted) + "\n")
            kept += 1
            if limit and kept >= limit:
                break
    return kept, skipped


def build_datasets(flashrag_dir: Path, out_dir: Path, limit: int) -> dict[str, Any]:
    """Build every reported split and publish a checksum manifest last."""
    sources = {name: flashrag_dir / relative for name, relative in SPLITS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing FlashRAG source files: {', '.join(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    with tempfile.TemporaryDirectory(prefix=".build-eval-", dir=out_dir) as temporary:
        staging_dir = Path(temporary)
        datasets: dict[str, dict[str, Any]] = {}
        for name, source in sources.items():
            output_name = f"{name}-miles.jsonl"
            staged_output = staging_dir / output_name
            kept, skipped = _convert_split(source, staged_output, limit)
            datasets[name] = {
                "source": SPLITS[name],
                "output": output_name,
                "rows": kept,
                "skipped": skipped,
                "sha256": _sha256(staged_output),
            }

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "limit": limit,
            "datasets": datasets,
        }
        staged_manifest = staging_dir / MANIFEST_NAME
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # The manifest is the commit record. Remove it before replacing any
        # JSONL so an interrupted publication cannot look complete.
        manifest_path.unlink(missing_ok=True)
        for name in SPLITS:
            output_name = datasets[name]["output"]
            os.replace(staging_dir / output_name, out_dir / output_name)
        os.replace(staged_manifest, manifest_path)
    return manifest


def validate_datasets(out_dir: Path) -> dict[str, Any]:
    """Validate all published files against the deterministic manifest."""
    manifest_path = out_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Search-R1 eval manifest schema")
    if manifest.get("source_repository") != SOURCE_REPOSITORY:
        raise ValueError("unexpected Search-R1 source repository")
    if manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("unexpected Search-R1 source revision")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(SPLITS):
        raise ValueError("Search-R1 eval manifest does not cover every benchmark")
    for name in SPLITS:
        record = datasets[name]
        output_name = f"{name}-miles.jsonl"
        if record.get("output") != output_name:
            raise ValueError(f"unexpected output name for {name}")
        output = out_dir / output_name
        if _sha256(output) != record.get("sha256"):
            raise ValueError(f"checksum mismatch for {output}")
        with output.open(encoding="utf-8") as stream:
            rows = sum(bool(line.strip()) for line in stream)
        if rows != record.get("rows"):
            raise ValueError(f"row-count mismatch for {output}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--flashrag-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="cap per set (0 = no cap). Each prompt is a multi-turn episode with "
        "retrieval between turns, so this is far dearer than a math eval.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.validate_only:
        manifest = validate_datasets(args.out_dir)
        print(
            f"validated {len(manifest['datasets'])} Search-R1 eval sets "
            f"under {args.out_dir}"
        )
        return
    if args.flashrag_dir is None:
        raise ValueError("--flashrag-dir is required unless --validate-only is used")

    manifest = build_datasets(args.flashrag_dir, args.out_dir, args.limit)
    for name, record in manifest["datasets"].items():
        output = args.out_dir / record["output"]
        suffix = f" (skipped {record['skipped']})" if record["skipped"] else ""
        print(f"  {name:18s} {record['rows']:5d} rows -> {output}{suffix}")
    validate_datasets(args.out_dir)


if __name__ == "__main__":
    main()
