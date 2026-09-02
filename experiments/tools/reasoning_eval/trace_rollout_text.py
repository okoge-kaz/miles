#!/usr/bin/env python3
"""Trace qualitative response modes across per-iteration rollout dumps."""

import argparse
import collections
import concurrent.futures
import csv
import functools
import json
import re
from pathlib import Path
from typing import Any

import torch

EXACT_TIME_LOW = re.compile(r"\bgiven time low\b", re.IGNORECASE)
EXPLICIT_GUESS = re.compile(
    r"\b(?:i|we)(?:'ll)? (?:(?:will|would|might|may|shall) )?(?:just )?guess\b|"
    r"\bi(?:'m| am) going to (?:just )?guess\b|"
    r"\bwe(?:'re| are) going to (?:just )?guess\b|"
    r"\b(?:let'?s|let us) (?:just )?guess\b|"
    r"\b(?:my|our|best) guess (?:is|would be)\b",
    re.IGNORECASE,
)
SELF_GUESS_MENTION = re.compile(
    r"\b(?:i|we)(?:'ll)? (?:(?:can|could|will|would|might|may|shall) )?(?:just )?guess\b|"
    r"\bi(?:'m| am) going to (?:just )?guess\b|"
    r"\bwe(?:'re| are) going to (?:just )?guess\b|"
    r"\b(?:let'?s|let us) (?:just )?guess\b|"
    r"\b(?:my|our|best) guess (?:is|would be)\b",
    re.IGNORECASE,
)
GUESS_LIKE = re.compile(
    r"\b(?:i|we)(?:'ll)? (?:(?:will|would|might|may|shall) )?(?:just )?guess\b|"
    r"\bi(?:'m| am) going to (?:just )?guess\b|"
    r"\bwe(?:'re| are) going to (?:just )?guess\b|"
    r"\b(?:let'?s|let us) (?:just )?guess\b|"
    r"\b(?:my|our|best) guess (?:is|would be)\b|"
    r"\b(?:probably|maybe) (?:the )?answer\b|"
    r"\banswer (?:is )?(?:probably|maybe)\b|\bplaceholder\b|"
    r"\bnot (?:sure|confident)\b|\bi(?:'ll| will| am going to) (?:go with|choose)\b|"
    r"\bi think (?:the )?answer (?:is|might be)\b",
    re.IGNORECASE,
)
TIME_PRESSURE = re.compile(
    r"\bgiven time\b|\btime (?:is )?(?:low|limited|short)\b|\bout of time\b|"
    r"\btime out\b|\btime constraints?\b|\blimited time\b",
    re.IGNORECASE,
)
ABANDONED_SOLUTION = re.compile(
    r"\bplaceholder\b|\b(?:cannot|can not|can't|unable to) solve\b|"
    r"\b(?:cannot|can not|can't|unable to) (?:complete|derive|determine)\b|"
    r"\bi(?:'m| am) stuck\b|\bgiven (?:the )?(?:uncertainty|difficulty|complexity)\b|"
    r"\bwithout (?:enough|full|further) (?:time|calculation|derivation)\b",
    re.IGNORECASE,
)
GIVE_UP = re.compile(
    r"\bcannot solve\b|\b(?:cannot|can not|can't) complete\b|\bi regret\b|"
    r"\bi(?:'m| am) stuck\b|"
    r"\btoo (?:complex|difficult|messy)\b|\bguess\b|\bplaceholder\b|\bhard\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=3,
        help="Full-text audit examples per category and iteration; 0 writes every match",
    )
    args = parser.parse_args()
    if args.start < 0 or args.end < args.start:
        parser.error("require 0 <= --start <= --end")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.examples_per_category < 0:
        parser.error("--examples-per-category must be nonnegative")
    return args


def _numeric_reward(sample: dict[str, Any]) -> float | None:
    reward = sample.get("reward")
    return float(reward) if isinstance(reward, (int, float)) else None


def _response_tokens(sample: dict[str, Any]) -> list[int]:
    length = int(sample["response_length"])
    if length < 0 or length > len(sample["tokens"]):
        raise ValueError(f"Invalid response length {length} for {len(sample['tokens'])} tokens")
    return sample["tokens"][-length:] if length else []


def _first_prefill_version(sample: dict[str, Any]) -> int | None:
    versions = sample.get("first_prefill_weight_versions") or []
    return int(versions[0]) if versions else None


def _is_token_loop(tokens: list[int]) -> bool:
    if len(tokens) < 1_000:
        return False
    most_common_count = collections.Counter(tokens).most_common(1)[0][1]
    return most_common_count / len(tokens) >= 0.4


def _example(sample: dict[str, Any], *, rollout_id: int, category: str) -> dict[str, Any]:
    response = sample.get("response", "")
    return {
        "rollout_id": rollout_id,
        "training_step": rollout_id + 1,
        "category": category,
        "sample_index": sample.get("index"),
        "group_index": sample.get("group_index"),
        "response_length": sample.get("response_length"),
        "status": sample.get("status"),
        "reward": sample.get("reward"),
        "first_prefill_weight_version": _first_prefill_version(sample),
        "prompt": sample.get("prompt"),
        "response": response,
    }


def _scan_file(
    path: Path,
    *,
    examples_per_category: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rollout_id = int(payload["rollout_id"])
    samples = payload["samples"]
    if not samples:
        raise ValueError(f"Rollout dump contains no samples: {path}")
    group_rewards: dict[int, set[float]] = collections.defaultdict(set)
    for sample in samples:
        reward = _numeric_reward(sample)
        if reward is not None:
            group_rewards[int(sample["group_index"])].add(reward)

    counts = collections.Counter()
    category_versions: dict[str, list[int]] = collections.defaultdict(list)
    examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    total_response_tokens = 0
    for sample in samples:
        response = sample.get("response", "")
        tokens = _response_tokens(sample)
        total_response_tokens += len(tokens)
        reward = _numeric_reward(sample)
        mixed_group = len(group_rewards[int(sample["group_index"])]) > 1
        exact_time_low = bool(EXACT_TIME_LOW.search(response))
        explicit_guess = bool(EXPLICIT_GUESS.search(response))
        self_guess_mention = bool(SELF_GUESS_MENTION.search(response))
        guess_like = bool(GUESS_LIKE.search(response))
        time_pressure = bool(TIME_PRESSURE.search(response))
        abandoned_solution = bool(ABANDONED_SOLUTION.search(response))
        short = len(tokens) <= 128
        short_or_medium = len(tokens) <= 512
        short_guess_candidate = short_or_medium and (
            self_guess_mention or time_pressure or abandoned_solution
        )
        guess_audit_candidate = (
            self_guess_mention or guess_like or time_pressure or abandoned_solution
        )
        categories = {
            "exact_time_low": exact_time_low,
            "exact_time_low_short": exact_time_low and short,
            "explicit_guess": explicit_guess,
            "explicit_guess_le512": explicit_guess and short_or_medium,
            "self_guess_mention": self_guess_mention,
            "self_guess_mention_le512": self_guess_mention and short_or_medium,
            "guess_like": guess_like,
            "guess_like_le512": guess_like and short_or_medium,
            "short_guess_candidate": short_guess_candidate,
            "guess_audit_candidate": guess_audit_candidate,
            "short_nonmatch_review": short_or_medium and not guess_audit_candidate,
            "time_pressure": time_pressure,
            "time_pressure_short": time_pressure and short,
            "give_up": bool(GIVE_UP.search(response)),
            "short": short,
            "truncated": sample.get("status") == "truncated",
            "token_loop": _is_token_loop(tokens),
        }
        for category, matched in categories.items():
            if not matched:
                continue
            counts[category] += 1
            if mixed_group:
                counts[f"{category}_mixed_group"] += 1
            if reward is not None and reward > 0:
                counts[f"{category}_reward_positive"] += 1
                if mixed_group:
                    counts[f"{category}_positive_mixed_group"] += 1
            version = _first_prefill_version(sample)
            if version is not None:
                category_versions[category].append(version)
            if examples_per_category == 0 or len(examples[category]) < examples_per_category:
                examples[category].append(_example(sample, rollout_id=rollout_id, category=category))

    row: dict[str, Any] = {
        "rollout_id": rollout_id,
        "training_step": rollout_id + 1,
        "sample_count": len(samples),
        "group_count": len(group_rewards),
        "all_zero_group_count": sum(rewards == {0.0} for rewards in group_rewards.values()),
        "mixed_reward_group_count": sum(len(rewards) > 1 for rewards in group_rewards.values()),
        "all_one_group_count": sum(rewards == {1.0} for rewards in group_rewards.values()),
        "response_length_mean": total_response_tokens / len(samples),
    }
    for category in (
        "exact_time_low",
        "exact_time_low_short",
        "explicit_guess",
        "explicit_guess_le512",
        "self_guess_mention",
        "self_guess_mention_le512",
        "guess_like",
        "guess_like_le512",
        "short_guess_candidate",
        "guess_audit_candidate",
        "short_nonmatch_review",
        "time_pressure",
        "time_pressure_short",
        "give_up",
        "short",
        "truncated",
        "token_loop",
    ):
        versions = category_versions[category]
        row[category] = counts[category]
        row[f"{category}_mixed_group"] = counts[f"{category}_mixed_group"]
        row[f"{category}_reward_positive"] = counts[f"{category}_reward_positive"]
        row[f"{category}_positive_mixed_group"] = counts[f"{category}_positive_mixed_group"]
        row[f"{category}_prefill_version_min"] = min(versions, default=None)
        row[f"{category}_prefill_version_max"] = max(versions, default=None)
    flattened_examples = [example for category_examples in examples.values() for example in category_examples]
    return row, flattened_examples


def main() -> None:
    args = parse_args()
    paths = [args.rollout_dir / f"{rollout_id}.pt" for rollout_id in range(args.start, args.end + 1)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rollout dumps: {missing[:5]}")
    scan_file = functools.partial(
        _scan_file,
        examples_per_category=args.examples_per_category,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(scan_file, paths))
    rows = sorted((row for row, _examples in results), key=lambda row: row["rollout_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    examples_path = args.output.with_suffix(".examples.jsonl")
    with examples_path.open("w", encoding="utf-8") as output:
        for _row, examples in results:
            for example in examples:
                output.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
