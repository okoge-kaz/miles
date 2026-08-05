#!/usr/bin/env python3
"""Preflight for SWE-RL: train on SWE-Gym, evaluate on SWE-bench Verified.

Answers the three questions that decide whether the run is even possible,
before any GPU is allocated:

  1. Contamination  - does any train repo appear in SWE-bench Verified?
                      (overlapping repos are dropped, not just reported)
  2. Grading        - does the official ``swebench`` package carry eval specs
                      for each instance? Instances it does not know error at
                      ``make_test_spec`` and score 0, which silently poisons
                      GRPO with all-zero rewards.
  3. Sandbox images - is there a prebuilt .sif for each instance, and what
                      ``container_formatter`` template reproduces its name?

Emits filtered train/eval JSONL in Miles prompt-data format plus a coverage
report. Nothing here needs a GPU, a network, or the NeMo-Gym server.

Usage:
    python preflight.py \
        --sif-dir /lustre/.../sif \
        --train-input /lustre/.../swe-gym-local-2401-localrepos.jsonl \
        --eval-input princeton-nlp/SWE-bench_Verified \
        --out-dir ./data
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Docker tags disallow consecutive underscores, so image builders re-encode the
# "__" in an instance id. Two spellings are in use in the shared image pool:
# "_1776_" (SWE-bench official, swerebench, and part of the xingyaoww set) and
# "_s_" (the 2401-instance SWE-Gym set). Both must be tried.
DUNDER_TAGS = ("_1776_", "_s_")


def _norm(text: str) -> str:
    """Lowercase and strip every non-alphanumeric character.

    Makes filename matching independent of the ``/`` -> ``_``, ``:`` -> ``_``,
    ``__`` -> ``_1776_`` mangling that differs between image publishers.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_instances(source: str, split: str, limit: int | None = None) -> list[dict]:
    """Load raw instances from a local JSONL or a HuggingFace dataset id."""
    if Path(source).exists():
        rows = []
        with open(source) as f:
            for line in f:
                if limit is not None and len(rows) >= limit:
                    break
                row = json.loads(line)
                # Accept both raw instances and already-converted Miles rows.
                rows.append(row.get("metadata", row))
        return rows

    from datasets import load_dataset

    dataset = load_dataset(source, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def build_sif_index(sif_dir: Path) -> dict[str, Path]:
    """Map a normalized filename to its path, for substring lookup."""
    return {_norm(p.stem): p for p in sif_dir.glob("*.sif")}


def id_spellings(instance_id: str) -> list[str]:
    """Every filename spelling an instance id is known to appear under."""
    return [instance_id] + [instance_id.replace("__", tag) for tag in DUNDER_TAGS]


def find_sif(instance_id: str, index: dict[str, Path]) -> Path | None:
    """Locate the .sif for an instance id under any known id spelling."""
    for candidate in id_spellings(instance_id):
        key = _norm(candidate)
        for name, path in index.items():
            if key in name:
                return path
    return None


def infer_formatter(instance_id: str, sif_path: Path) -> str | None:
    """Reconstruct a ``container_formatter`` template from one matched file.

    Returns e.g. ``/path/xingyaoww_sweb.eval.x86_64.{instance_id_tag}.sif``,
    or None when the id is not recoverable from the filename verbatim.
    """
    name = sif_path.name
    for candidate in id_spellings(instance_id):
        placeholder = "{instance_id}" if candidate == instance_id else "{instance_id_tag}"
        for spelling in (candidate, candidate.lower()):
            if spelling in name:
                return str(sif_path.parent / name.replace(spelling, placeholder))
    return None


def check_eval_specs(instances: list[dict]) -> tuple[set[str], dict[str, str]]:
    """Return (instance ids the swebench package can grade, per-id errors)."""
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError:
        print("  ! swebench package not importable - skipping grading check.")
        print("    Install it in this venv to make this check meaningful:")
        print("        pip install swebench")
        return {i["instance_id"] for i in instances}, {}

    gradable, errors = set(), {}
    for instance in instances:
        try:
            make_test_spec(instance)
            gradable.add(instance["instance_id"])
        except Exception as e:  # noqa: BLE001 - a scan reports, it does not crash
            errors[instance["instance_id"]] = f"{type(e).__name__}: {e}"
    return gradable, errors


def to_miles_rows(instances: list[dict], subset: str, split: str) -> list[dict]:
    """Convert to Miles prompt data, matching download_and_process_data.py."""
    rows = []
    for instance in instances:
        metadata = dict(instance)
        metadata["subset"] = subset
        metadata["split"] = split
        rows.append({"prompt": instance.get("problem_statement", ""), "metadata": metadata})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows):5d} rows -> {path}")


def report_stage(label: str, kept: int, total: int) -> None:
    pct = 100.0 * kept / total if total else 0.0
    print(f"  {label:<38} {kept:5d} / {total:5d}  ({pct:5.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sif-dir", required=True, type=Path)
    parser.add_argument("--train-input", required=True, help="SWE-Gym JSONL path or HF dataset id")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-input", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--eval-subset-size", type=int, default=100, help="held-out dev slice for per-checkpoint eval")
    parser.add_argument("--limit", type=int, default=None, help="cap inputs for a fast dry run")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nIndexing .sif files under {args.sif_dir} ...")
    sif_index = build_sif_index(args.sif_dir)
    print(f"  {len(sif_index)} .sif files found")
    if not sif_index:
        print("  ! no .sif files - check the path before going further.")

    print("\n=== EVAL: SWE-bench Verified ===")
    eval_instances = load_instances(args.eval_input, args.eval_split, args.limit)
    print(f"  loaded {len(eval_instances)} instances")
    eval_repos = {i["repo"] for i in eval_instances}
    print(f"  {len(eval_repos)} distinct repos")

    eval_gradable, eval_errors = check_eval_specs(eval_instances)
    report_stage("gradable by swebench package", len(eval_gradable), len(eval_instances))

    eval_with_sif, eval_missing_sif, formatter = [], [], None
    for instance in eval_instances:
        sif = find_sif(instance["instance_id"], sif_index)
        if sif is None:
            eval_missing_sif.append(instance["instance_id"])
            continue
        formatter = formatter or infer_formatter(instance["instance_id"], sif)
        eval_with_sif.append(instance)
    report_stage("has a prebuilt .sif", len(eval_with_sif), len(eval_instances))

    if eval_missing_sif:
        print(f"  ! {len(eval_missing_sif)} Verified instances have no .sif, e.g.:")
        for iid in eval_missing_sif[:5]:
            print(f"      {iid}")
        print("    Build them before evaluating, e.g.:")
        print("      apptainer build <sif-dir>/swebench_sweb.eval.x86_64.<id_tag>.sif \\")
        print("          docker://swebench/sweb.eval.x86_64.<id_tag>:latest")
        print("    Budget ~1-2 GiB per instance.")

    eval_ready = [i for i in eval_with_sif if i["instance_id"] in eval_gradable]
    report_stage("EVAL READY (gradable + .sif)", len(eval_ready), len(eval_instances))

    print("\n=== TRAIN: SWE-Gym ===")
    train_instances = load_instances(args.train_input, args.train_split, args.limit)
    print(f"  loaded {len(train_instances)} instances")

    train_repos = Counter(i["repo"] for i in train_instances)
    overlap = eval_repos & set(train_repos)
    if overlap:
        print(f"  ! CONTAMINATION: {len(overlap)} repo(s) shared with Verified - dropping them:")
        for repo in sorted(overlap):
            print(f"      {repo}  ({train_repos[repo]} instances)")
    else:
        print("  no repo overlap with Verified")
    train_clean = [i for i in train_instances if i["repo"] not in overlap]
    report_stage("after contamination filter", len(train_clean), len(train_instances))

    train_gradable, train_errors = check_eval_specs(train_clean)
    report_stage("gradable by swebench package", len(train_gradable), len(train_clean))
    if train_errors:
        by_repo = Counter(i["repo"] for i in train_clean if i["instance_id"] in train_errors)
        print("  ! ungradable repos (these would score a constant 0 in training):")
        for repo, count in by_repo.most_common(10):
            print(f"      {repo:<40} {count}")

    train_ready = []
    train_missing_sif = 0
    for instance in train_clean:
        if instance["instance_id"] not in train_gradable:
            continue
        sif = find_sif(instance["instance_id"], sif_index)
        if sif is None:
            train_missing_sif += 1
            continue
        formatter = formatter or infer_formatter(instance["instance_id"], sif)
        train_ready.append(instance)
    if train_missing_sif:
        print(f"  ! {train_missing_sif} gradable instances have no .sif")
    report_stage("TRAIN READY (clean + gradable + .sif)", len(train_ready), len(train_instances))

    print("\n=== OUTPUT ===")
    write_jsonl(args.out_dir / "train_swegym.jsonl", to_miles_rows(train_ready, "gym", "train"))
    write_jsonl(args.out_dir / "eval_verified_full.jsonl", to_miles_rows(eval_ready, "verified", "test"))

    # Deterministic dev slice: every Nth instance of the id-sorted list, so it
    # spans repos and stays identical across runs without an RNG seed.
    ordered = sorted(eval_ready, key=lambda i: i["instance_id"])
    stride = max(1, len(ordered) // max(1, args.eval_subset_size))
    dev = ordered[::stride][: args.eval_subset_size]
    write_jsonl(args.out_dir / "eval_verified_dev.jsonl", to_miles_rows(dev, "verified", "test"))

    report = {
        "sif_dir": str(args.sif_dir),
        "sif_files_indexed": len(sif_index),
        "container_formatter": formatter,
        "eval": {
            "loaded": len(eval_instances),
            "gradable": len(eval_gradable),
            "with_sif": len(eval_with_sif),
            "ready": len(eval_ready),
            "missing_sif_examples": eval_missing_sif[:20],
        },
        "train": {
            "loaded": len(train_instances),
            "contaminated_repos": sorted(overlap),
            "after_contamination_filter": len(train_clean),
            "gradable": len(train_gradable),
            "missing_sif": train_missing_sif,
            "ready": len(train_ready),
            "ready_repos": dict(Counter(i["repo"] for i in train_ready).most_common()),
        },
    }
    report_path = args.out_dir / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  wrote report              -> {report_path}")

    print("\n=== NEXT ===")
    if formatter:
        print(f"  Inferred from one matched file: {formatter}")
        print("  Train and eval images carry different publisher prefixes, so a single")
        print("  template cannot address both. Run link_sifs.py and use its output as")
        print("  container_formatter instead.")
    else:
        print("  ! Could not infer a naming pattern - inspect a .sif filename by hand.")

    blocked = not eval_ready or not train_ready
    if blocked:
        print("  ! BLOCKED: fix the coverage gaps above before running the golden scan.")
    else:
        print(f"  Ready: {len(train_ready)} train / {len(eval_ready)} eval instances.")
        print("  Next: link_sifs.py, then GOLDEN=1 sbatch sbatch/gym_server.sbatch")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
