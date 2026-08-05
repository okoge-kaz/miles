"""Build the BFCL eval file: join each category's questions to its answers.

BFCL ships questions and ground truth as separate files keyed by id, and splits
the benchmark across ~26 category files, so it needs its own builder rather than
an entry in convert.py's adapter table.

    python -m experiments.src.nemo_blends.build_bfcl \
        --bfcl-dir /data/bfcl --output /data/bfcl/bfcl-ast-miles.jsonl

Only the **AST** categories are included. The `exec_*` ones call live APIs, and
`multi_turn_*` needs a stateful environment -- neither is gradable here, and
including them would silently push the score down by the fraction of rows the
harness cannot judge rather than by anything the policy did.
"""

import argparse
import json
from pathlib import Path

# Graded by comparing the emitted call against a set of allowed values.
AST_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    # Empty ground truth: the correct behaviour is to call nothing. Kept because
    # spurious tool use is a real failure mode and these are the only rows that
    # measure it.
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
]

# Excluded, with the reason, so a future reader does not "fix" the omission:
#   exec_*          executes real API calls
#   multi_turn_*    needs a stateful environment, not a single-step comparison
#   rest, sql       REST/SQL execution
#   chatable        no verifiable ground truth


def load_json_lines(path: Path):
    """BFCL files are JSON-lines despite the .json extension."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_chat(question):
    """`question` is a list of message *lists* (one per turn block); the AST
    categories carry exactly one, so it is flattened."""
    messages = []
    for block in question or []:
        if isinstance(block, list):
            for m in block:
                if isinstance(m, dict) and m.get("content") is not None:
                    messages.append({"role": m.get("role") or "user", "content": str(m["content"])})
        elif isinstance(block, dict) and block.get("content") is not None:
            messages.append({"role": block.get("role") or "user", "content": str(block["content"])})
    return messages


def to_tools(functions):
    """BFCL's function specs -> the ChatCompletions shape Qwen's template renders."""
    out = []
    for fn in functions or []:
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        spec = {"name": fn["name"], "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}
        if fn.get("description"):
            spec["description"] = fn["description"]
        out.append({"type": "function", "function": spec})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfcl-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--categories", nargs="+", default=AST_CATEGORIES)
    args = ap.parse_args()

    root = Path(args.bfcl_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0
    per_category = {}
    with out.open("w") as fout:
        for category in args.categories:
            qpath = root / f"BFCL_v3_{category}.json"
            apath = root / "possible_answer" / f"BFCL_v3_{category}.json"
            if not qpath.exists():
                print(f"  {category:24s} MISSING questions")
                continue

            # The irrelevance categories legitimately have no answer file: the
            # expected behaviour is an empty call set.
            answers = {}
            if apath.exists():
                answers = {r["id"]: r.get("ground_truth", []) for r in load_json_lines(apath)}
            elif "irrelevance" not in category and "relevance" not in category:
                print(f"  {category:24s} MISSING answers, skipped")
                continue

            n = 0
            for row in load_json_lines(qpath):
                rid = row.get("id")
                messages = to_chat(row.get("question"))
                if not messages:
                    skipped += 1
                    continue
                truth = answers.get(rid, [])
                fout.write(
                    json.dumps(
                        {
                            "prompt": messages,
                            "label": rid,
                            # `tools` sits at the top level so --tool-key picks it
                            # up; ground_truth rides in metadata for the verifier.
                            "tools": to_tools(row.get("function")),
                            "metadata": {
                                "source": "bfcl",
                                "category": category,
                                "id": rid,
                                "ground_truth": truth,
                            },
                        }
                    )
                    + "\n"
                )
                kept += 1
                n += 1
            per_category[category] = n

    for category, n in per_category.items():
        print(f"  {category:24s} {n}")
    print(f"\nbfcl: wrote {kept} rows to {out} (skipped {skipped}); "
          f"use --custom-rm-path experiments.src.nemo_blends.grid_and_ast.bfcl_ast_reward --tool-key tools")


if __name__ == "__main__":
    main()
