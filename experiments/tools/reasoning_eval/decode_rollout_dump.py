#!/usr/bin/env python3
"""Convert per-iteration Miles rollout dumps into readable JSONL and Markdown."""

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="dump/rollout_data/<rollout_id>.pt files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", help="Prefix output filenames, useful when combining multiple runs")
    parser.add_argument("--tokenizer", type=Path, help="Optional HF tokenizer directory for token-ID verification")
    parser.add_argument("--include-prompt-token-ids", action="store_true")
    parser.add_argument("--include-logprobs", action="store_true")
    parser.add_argument("--examples-per-category", type=int, default=4)
    parser.add_argument("--markdown-response-chars", type=int, default=12_000)
    args = parser.parse_args()
    if args.examples_per_category < 0:
        parser.error("--examples-per-category must be nonnegative")
    if args.markdown_response_chars < 1:
        parser.error("--markdown-response-chars must be positive")
    return args


def _load_tokenizer(path: Path | None):
    if path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)


def _response_token_ids(sample: dict[str, Any]) -> list[int]:
    response_length = int(sample["response_length"])
    tokens = sample["tokens"]
    if response_length < 0 or response_length > len(tokens):
        raise ValueError(f"Invalid response length {response_length} for {len(tokens)} tokens")
    return tokens[-response_length:] if response_length else []


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return repr(value)


def _decode_sample(
    sample: dict[str, Any],
    *,
    rollout_id: int,
    position: int,
    tokenizer,
    include_prompt_token_ids: bool,
    include_logprobs: bool,
) -> dict[str, Any]:
    response_token_ids = _response_token_ids(sample)
    response_length = len(response_token_ids)
    prompt_token_ids = sample["tokens"][:-response_length] if response_length else sample["tokens"]
    saved_response = sample.get("response", "")
    decoded_response = (
        tokenizer.decode(response_token_ids, skip_special_tokens=False) if tokenizer is not None else None
    )
    row = {
        "rollout_id": rollout_id,
        "training_step": rollout_id + 1,
        "position": position,
        "sample_index": sample.get("index"),
        "group_index": sample.get("group_index"),
        "prompt": _json_safe(sample.get("prompt")),
        "label": _json_safe(sample.get("label")),
        "response": saved_response,
        "response_token_ids": response_token_ids,
        "response_length": response_length,
        "prompt_length": len(prompt_token_ids),
        "total_length": len(sample["tokens"]),
        "status": sample.get("status"),
        "reward": _json_safe(sample.get("reward")),
        "response_sha256": hashlib.sha256(saved_response.encode("utf-8", errors="surrogatepass")).hexdigest(),
        "weight_versions": _json_safe(sample.get("weight_versions", [])),
        "first_prefill_weight_versions": _json_safe(sample.get("first_prefill_weight_versions", [])),
        "min_forward_weight_versions": _json_safe(sample.get("min_forward_weight_versions", [])),
        "max_forward_weight_versions": _json_safe(sample.get("max_forward_weight_versions", [])),
        "last_forward_weight_versions": _json_safe(sample.get("last_forward_weight_versions", [])),
        "response_weight_versions": _json_safe(sample.get("response_weight_versions", [])),
        "response_weight_version_segments": _json_safe(sample.get("response_weight_version_segments", [])),
    }
    if tokenizer is not None:
        row["decoded_from_token_ids"] = decoded_response
        row["token_decode_matches_saved_response"] = decoded_response == saved_response
    if include_prompt_token_ids:
        row["prompt_token_ids"] = prompt_token_ids
    if include_logprobs:
        row["rollout_log_probs"] = _json_safe(sample.get("rollout_log_probs"))
    return row


def _percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    ordered = sorted(values)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _numeric_reward(row: dict[str, Any]) -> float | None:
    reward = row["reward"]
    return float(reward) if isinstance(reward, (int, float)) else None


def _outcome_category(row: dict[str, Any]) -> str:
    reward = _numeric_reward(row)
    if reward is None:
        return "reward missing or non-scalar"
    if reward > 0:
        return "reward positive"
    if reward == 0 and row["status"] == "truncated":
        return "reward zero (truncated)"
    if reward == 0:
        return "reward zero (not truncated)"
    return "reward negative"


def _summary(
    *,
    source: Path,
    rollout_id: int,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    lengths = [row["response_length"] for row in rows]
    rewards = [reward for row in rows if (reward := _numeric_reward(row)) is not None]
    response_counts = collections.Counter(row["response_sha256"] for row in rows)
    decode_checks = [row.get("token_decode_matches_saved_response") for row in rows]
    return {
        "source": str(source),
        "rollout_id": rollout_id,
        "training_step": rollout_id + 1,
        "metadata": _json_safe(metadata),
        "sample_count": len(rows),
        "response_length": {
            "mean": statistics.fmean(lengths) if lengths else None,
            "min": min(lengths, default=None),
            "p10": _percentile(lengths, 0.10),
            "median": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "max": max(lengths, default=None),
        },
        "status_counts": dict(collections.Counter(row["status"] for row in rows)),
        "outcome_counts": dict(collections.Counter(_outcome_category(row) for row in rows)),
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "reward_positive_count": sum(reward > 0 for reward in rewards),
        "unique_response_count": len(response_counts),
        "largest_exact_response_multiplicity": max(response_counts.values(), default=0),
        "token_decode_checked": any(check is not None for check in decode_checks),
        "token_decode_match_count": sum(check is True for check in decode_checks),
        "token_decode_mismatch_count": sum(check is False for check in decode_checks),
    }


def _evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def _example_categories(rows: list[dict[str, Any]], count: int) -> dict[str, list[dict[str, Any]]]:
    by_length = sorted(rows, key=lambda row: (row["response_length"], row["position"]))
    completed = [row for row in by_length if row["status"] == "completed"]
    truncated = [row for row in by_length if row["status"] == "truncated"]
    positive = [row for row in by_length if _outcome_category(row) == "reward positive"]
    zero_truncated = [row for row in by_length if _outcome_category(row) == "reward zero (truncated)"]
    zero_not_truncated = [row for row in by_length if _outcome_category(row) == "reward zero (not truncated)"]
    response_counts = collections.Counter(row["response_sha256"] for row in rows)
    frequent = sorted(
        rows,
        key=lambda row: (-response_counts[row["response_sha256"]], row["response_length"], row["position"]),
    )
    frequent_unique = []
    seen_hashes = set()
    for row in frequent:
        if row["response_sha256"] in seen_hashes:
            continue
        seen_hashes.add(row["response_sha256"])
        frequent_unique.append({**row, "exact_multiplicity": response_counts[row["response_sha256"]]})
    return {
        "shortest completed": completed[:count],
        "length quantiles": _evenly_spaced(by_length, count),
        "longest completed": list(reversed(completed[-count:])) if count else [],
        "truncated": list(reversed(truncated[-count:])) if count else [],
        "reward positive": _evenly_spaced(positive, count),
        "reward zero (truncated)": _evenly_spaced(zero_truncated, count),
        "reward zero (not truncated)": _evenly_spaced(zero_not_truncated, count),
        "frequent exact responses": frequent_unique[:count],
    }


def _preview_tokens(token_ids: list[int], limit: int = 96) -> str:
    if len(token_ids) <= limit * 2:
        return json.dumps(token_ids)
    omitted = len(token_ids) - 2 * limit
    return f"{json.dumps(token_ids[:limit])[:-1]}, ... {omitted} omitted ..., {json.dumps(token_ids[-limit:])[1:]}"


def _preview_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n... {len(text) - 2 * half} characters omitted ...\n\n{text[-half:]}"


def _write_markdown(
    path: Path,
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    examples_per_category: int,
    response_char_limit: int,
) -> None:
    categories = _example_categories(rows, examples_per_category)
    with path.open("w", encoding="utf-8") as output:
        output.write(f"# Rollout {summary['rollout_id']}\n\n")
        output.write("```json\n")
        output.write(json.dumps(summary, ensure_ascii=False, indent=2))
        output.write("\n```\n")
        for category, examples in categories.items():
            output.write(f"\n## {category}\n")
            for row in examples:
                output.write(
                    f"\n### position={row['position']} sample={row['sample_index']} "
                    f"group={row['group_index']} length={row['response_length']} "
                    f"status={row['status']} reward={row['reward']}"
                )
                if "exact_multiplicity" in row:
                    output.write(f" exact_multiplicity={row['exact_multiplicity']}")
                output.write("\n\nPrompt:\n\n```text\n")
                output.write(str(row["prompt"]))
                output.write("\n```\n\nResponse token IDs:\n\n```text\n")
                output.write(_preview_tokens(row["response_token_ids"]))
                output.write("\n```\n\nResponse:\n\n```text\n")
                output.write(_preview_text(row["response"], response_char_limit))
                output.write("\n```\n")


def _output_stem(*, tag: str | None, rollout_id: int) -> str:
    return f"{tag}-rollout-{rollout_id}" if tag else f"rollout-{rollout_id}"


def decode_file(path: Path, *, args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rollout_id = int(payload["rollout_id"])
    rows = [
        _decode_sample(
            sample,
            rollout_id=rollout_id,
            position=position,
            tokenizer=tokenizer,
            include_prompt_token_ids=args.include_prompt_token_ids,
            include_logprobs=args.include_logprobs,
        )
        for position, sample in enumerate(payload["samples"])
    ]
    summary = _summary(
        source=path,
        rollout_id=rollout_id,
        metadata=payload.get("metadata") or {},
        rows=rows,
    )
    stem = _output_stem(tag=args.tag, rollout_id=rollout_id)
    jsonl_path = args.output_dir / f"{stem}.samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    summary_path = args.output_dir / f"{stem}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(
        args.output_dir / f"{stem}.examples.md",
        summary=summary,
        rows=rows,
        examples_per_category=args.examples_per_category,
        response_char_limit=args.markdown_response_chars,
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(args.tokenizer)
    summaries = [decode_file(path, args=args, tokenizer=tokenizer) for path in args.inputs]
    index_name = f"{args.tag}-index.json" if args.tag else "index.json"
    (args.output_dir / index_name).write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
