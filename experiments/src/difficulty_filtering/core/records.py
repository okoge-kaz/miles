import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def validate_prompt(row, index):
    prompt, label = row["prompt"], row["label"]
    if isinstance(prompt, str):
        valid_prompt = bool(prompt.strip())
    else:
        valid_prompt = (
            isinstance(prompt, list)
            and bool(prompt)
            and all(
                isinstance(message, dict)
                and isinstance(message.get("content"), str)
                and message.get("role") in {"system", "user", "assistant"}
                for message in prompt
            )
        )
    if not valid_prompt or isinstance(label, bool) or not isinstance(label, (str, int, float)) or str(label) == "":
        raise ValueError(f"Invalid prompt/label at index {index}")


def iter_prompts(path):
    count = 0
    # JSONL uses physical LF/CRLF boundaries. str.splitlines() also splits valid
    # JSON string contents such as U+0085 and U+2028 in Code and IF prompts.
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{index + 1}: {error}") from error
            validate_prompt(row, index)
            count += 1
            yield row
    if not count:
        raise ValueError("Input contains no prompts")


def read_prompts(path):
    return list(iter_prompts(path))


def row_hash(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def shard_indices(total, num_shards, shard_index):
    if not 0 <= shard_index < num_shards <= total:
        raise ValueError("Require 0 <= shard_index < num_shards <= prompt count")
    # Contiguous, balanced ranges retain the original source indices.
    size, remainder = divmod(total, num_shards)
    start = shard_index * size + min(shard_index, remainder)
    return range(start, start + size + (shard_index < remainder))


def keep_record(record):
    """Unscored siblings exclude the group; IF uses the native fractional reward."""
    if any(reward is None for reward in record["rewards"]):
        return False
    if record.get("selection_rule") == "reward_variance":
        return max(record["rewards"]) > min(record["rewards"])
    return 0 < record["n_correct"] < record["n_samples"]


def reward_statistics(rewards, threshold):
    scored = [reward for reward in rewards if reward is not None]
    complete = len(scored) == len(rewards)
    n_correct = sum(reward >= threshold for reward in scored) if complete else None
    return {
        "n_correct": n_correct,
        "pass_rate": n_correct / len(rewards) if complete else None,
        "mean_reward": sum(scored) / len(scored) if scored else None,
        "unscored_samples": len(rewards) - len(scored),
    }


def build_record(index, row, rewards, completion_tokens, truncated, config, diagnostics=None):
    record = {
        "index": index,
        "prompt_sha256": row_hash(row),
        "n_samples": config.n_samples,
        **reward_statistics(rewards, config.correct_threshold),
        "selection_rule": config.selection_rule,
        "verifier_diagnostics": diagnostics or [{} for _ in rewards],
        "rewards": rewards,
        "completion_tokens": completion_tokens,
        "truncated": truncated,
        "response_len_mean": sum(completion_tokens) / config.n_samples,
        "truncated_frac": sum(truncated) / config.n_samples,
    }
    validate_record(record, row, config)
    return record


def validate_record(record, row, config):
    if record["prompt_sha256"] != row_hash(row) or record["n_samples"] != config.n_samples:
        raise ValueError("Measurement source or sample count mismatch")
    for name in ("rewards", "completion_tokens", "truncated"):
        if len(record[name]) != config.n_samples:
            raise ValueError(f"Incomplete response group: {name}")
    rewards = record["rewards"]
    for value in rewards:
        if value is None and config.reward_backend != "deepscaler":
            continue
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Rewards must be finite values in [0, 1], or an explicit unscored result")
        if config.reward_backend != "instruction_following" and value not in (0, 1):
            raise ValueError("Binary verifier returned a fractional reward")
    if record["selection_rule"] != config.selection_rule or len(record["verifier_diagnostics"]) != config.n_samples:
        raise ValueError("Selection rule or verifier diagnostics mismatch")
    if any(type(value) is not int or value < 0 for value in record["completion_tokens"]):
        raise ValueError("Invalid completion token count")
    if any(type(value) is not bool for value in record["truncated"]):
        raise ValueError("Invalid truncation status")
    if config.zero_reward_on_truncated and any(
        truncated and reward != 0 for truncated, reward in zip(record["truncated"], rewards, strict=True)
    ):
        raise ValueError("Truncated responses must have reward zero")
    expected = {
        **reward_statistics(rewards, config.correct_threshold),
        "response_len_mean": sum(record["completion_tokens"]) / config.n_samples,
        "truncated_frac": sum(record["truncated"]) / config.n_samples,
    }
    if any(record[name] != value for name, value in expected.items()):
        raise ValueError("Inconsistent measurement statistics")


def summarize(records, n_samples):
    records = list(records)
    histogram = [0] * (n_samples + 1)
    scored_rewards, failure_kinds = [], Counter()
    complete = [record for record in records if all(reward is not None for reward in record["rewards"])]
    for record in records:
        scored_rewards.extend(reward for reward in record["rewards"] if reward is not None)
        for detail in record.get("verifier_diagnostics", []):
            if detail.get("failure_kind"):
                failure_kinds[detail["failure_kind"]] += 1
    for record in complete:
        histogram[record["n_correct"]] += 1
    all_zero = sum(all(value == 0 for value in record["rewards"]) for record in complete)
    all_one = sum(all(value == 1 for value in record["rewards"]) for record in complete)
    constant = sum(len(set(record["rewards"])) == 1 for record in complete)
    kept = sum(keep_record(record) for record in records)
    return {
        "total": len(records),
        "scored_groups": len(complete),
        "unscored_groups": len(records) - len(complete),
        "unscored_samples": len(records) * n_samples - len(scored_rewards),
        "kept": kept,
        "kept_fraction": kept / max(1, len(records)),
        "all_zero": all_zero,
        "all_one": all_one,
        "constant_reward": constant,
        "constant_partial": constant - all_zero - all_one,
        "all_zero_fraction": all_zero / max(1, len(complete)),
        "all_one_fraction": all_one / max(1, len(complete)),
        "all_wrong": histogram[0],
        "all_correct": histogram[-1],
        "n_correct_histogram": histogram,
        "mean_pass_rate": sum(record["pass_rate"] for record in complete) / len(complete) if complete else None,
        "mean_reward": sum(scored_rewards) / len(scored_rewards) if scored_rewards else None,
        "mean_truncated_frac": sum(record["truncated_frac"] for record in records) / max(1, len(records)),
        "response_tokens": sum(sum(record["completion_tokens"]) for record in records),
        "failure_kinds": dict(failure_kinds),
    }
