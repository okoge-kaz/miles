"""Stage one AIME year as a miles prompt file, cross-checked against a second source.

The offline evaluation reports across AIME years (2023–2026), so the years have to
be *identically formatted*. A year that carries a different instruction wrapper is
not a harder year; it is a differently-prompted year, and the difference lands in
the reported score with no metric that says so.

Two sources are read for every year, because a single wrong label is invisible:
the model answers correctly, the verifier scores 0, and the year looks hard.

    primary     AI-MO/aimo-validation-aime   parquet, AoPS-derived, 2022-2024
    cross-check gneubig/aime-1983-2024       csv, AoPS-derived, 1983-2024

Answers are compared after normalization; disagreements abort. Problems present in
only one source are written but reported by index, so a single-sourced label is a
stated fact rather than a silent one.

Usage:

    uv run --with pandas --with pyarrow python -m experiments.src.offline_eval.prepare_aime \\
      --year 2023 --out-dir $DATASET_DIR/aime-2023

Years outside the sources' coverage (2025, 2026) were staged from their own
per-year HF releases and are not reproducible through this script; it refuses
rather than emitting a partial file.
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

# Byte-identical to aime-2025 / aime-2026. Any drift here silently makes the years
# incomparable, so it lives in one place and is asserted against an existing year.
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to "
    "the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

PRIMARY_URL = (
    "https://huggingface.co/datasets/AI-MO/aimo-validation-aime/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
CROSSCHECK_URL = (
    "https://huggingface.co/datasets/gneubig/aime-1983-2024/resolve/main/"
    "AIME_Dataset_1983_2024.csv"
)

PROBLEMS_PER_PART = 15
PARTS = ("I", "II")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def normalize_answer(answer) -> str:
    """AIME answers are integers 0-999; sources differ on zero padding and type."""
    text = str(answer).strip()
    return str(int(text)) if text.lstrip("-").isdigit() else text


def load_primary(year: int) -> dict[tuple[str, int], dict]:
    import pandas as pd

    frame = pd.read_parquet(io.BytesIO(fetch(PRIMARY_URL)))
    # The AoPS url is the only place the parquet records year, part and number.
    frame["year"] = frame.url.str.extract(r"/(\d{4})_AIME")[0].astype(int)
    frame["part"] = frame.url.str.extract(r"AIME_(I{1,2})_")[0]
    frame["number"] = frame.url.str.extract(r"Problem_(\d+)")[0].astype(int)
    rows = frame[frame.year == year]
    return {
        (row.part, int(row.number)): {"problem": row.problem, "answer": normalize_answer(row.answer)}
        for row in rows.itertuples()
    }


def load_crosscheck(year: int) -> dict[tuple[str, int], str]:
    import pandas as pd

    frame = pd.read_csv(io.BytesIO(fetch(CROSSCHECK_URL)))
    rows = frame[frame.Year == year]
    return {
        (row.Part, int(getattr(row, "_3"))): normalize_answer(row.Answer)
        for row in rows.itertuples()
        if row.Part in PARTS
    }


def build_records(year: int, primary: dict[tuple[str, int], dict]) -> list[dict]:
    records = []
    for part in PARTS:
        for number in range(1, PROBLEMS_PER_PART + 1):
            entry = primary[(part, number)]
            records.append(
                {
                    "prompt": [
                        {"role": "user", "content": PROMPT_PREFIX + entry["problem"].strip() + PROMPT_SUFFIX}
                    ],
                    "label": entry["answer"],
                    "metadata": {"source": f"aime-{year}", "part": part, "number": number},
                }
            )
    return records


def assert_template_matches(reference: Path) -> None:
    """Guard against drift from a year that is already staged."""
    with reference.open() as handle:
        content = json.loads(handle.readline())["prompt"][0]["content"]
    if not (content.startswith(PROMPT_PREFIX) and content.endswith(PROMPT_SUFFIX)):
        raise SystemExit(f"prompt template does not match {reference}; refusing to write a mismatched year")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--template-reference",
        type=Path,
        default=None,
        help="an already-staged year to assert the prompt wrapper against",
    )
    args = parser.parse_args()

    if args.template_reference:
        assert_template_matches(args.template_reference)

    primary = load_primary(args.year)
    expected = len(PARTS) * PROBLEMS_PER_PART
    if len(primary) != expected:
        raise SystemExit(f"primary source has {len(primary)} problems for {args.year}, expected {expected}")

    crosscheck = load_crosscheck(args.year)
    disagreements = [
        (key, primary[key]["answer"], crosscheck[key]) for key in primary if key in crosscheck and primary[key]["answer"] != crosscheck[key]
    ]
    if disagreements:
        for key, mine, theirs in disagreements:
            print(f"DISAGREE {args.year} {key[0]}-{key[1]}: primary={mine} crosscheck={theirs}")
        raise SystemExit("answers disagree between sources; not writing a file that may be mislabelled")

    single_sourced = sorted(key for key in primary if key not in crosscheck)
    print(f"cross-checked {len(primary) - len(single_sourced)}/{len(primary)} answers")
    for part, number in single_sourced:
        print(f"SINGLE-SOURCED {args.year} {part}-{number}: answer={primary[(part, number)]['answer']}")

    records = build_records(args.year, primary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = args.out_dir / f"aime-{args.year}.jsonl"
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    raw = args.out_dir / f"aime-{args.year}.raw.jsonl"
    with raw.open("w") as handle:
        for (part, number), entry in sorted(primary.items()):
            handle.write(
                json.dumps(
                    {"problem": entry["problem"], "answer": entry["answer"], "part": part, "number": number},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {len(records)} problems to {out}")
    print(f"wrote raw source to {raw}")


if __name__ == "__main__":
    main()
