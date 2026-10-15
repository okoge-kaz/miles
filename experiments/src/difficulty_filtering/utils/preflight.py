import argparse
import json
from pathlib import Path

from experiments.src.difficulty_filtering.core.artifacts import sha256_file, write_json
from experiments.src.difficulty_filtering.core.records import iter_prompts


def validate_source(path, backend, receipts=()):
    path = Path(path)
    before = path.stat()
    for receipt in receipts:
        if receipt["path"] == str(path.resolve()) and receipt["backend"] == backend:
            if receipt["sha256"] != sha256_file(path):
                raise ValueError("Input content differs from the full validation receipt")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Input changed while verifying its validation receipt")
            return receipt
    expected_verifier = {"code_gen": "python_code", "mcqa": "mcqa_regex", "instruction_following": "ifeval_g"}
    count = 0
    for index, row in enumerate(iter_prompts(path)):
        count += 1
        metadata = row.get("metadata") or {}
        if backend in expected_verifier and metadata.get("verifier") != expected_verifier[backend]:
            raise ValueError(f"Wrong verifier for {backend} at {path}:{index + 1}")
        if backend == "code_gen":
            tests = metadata.get("unit_tests")
            if (
                not isinstance(tests, dict)
                or not tests.get("inputs")
                or len(tests["inputs"]) != len(tests.get("outputs", []))
            ):
                raise ValueError(f"Missing or inconsistent code unit tests at {path}:{index + 1}")
        if backend == "instruction_following":
            ids, kwargs = metadata.get("instruction_id_list"), metadata.get("kwargs")
            if not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs):
                raise ValueError(f"Missing IF constraints at {path}:{index + 1}")
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed during preflight")
    return {"path": str(path.resolve()), "rows": count, "backend": backend, "bytes": after.st_size, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(
        description="Validate every source row and verifier metadata before PBS submission"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["deepscaler", "code_gen", "mcqa", "instruction_following"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Full validation receipts; reuse only after checking the content SHA256")
    args = parser.parse_args()
    receipts = json.loads(args.cache.read_text()) if args.cache else []
    write_json(args.output, validate_source(args.input, args.backend, receipts))


if __name__ == "__main__":
    main()
