"""Compare stock math and strict-math rewards on a saved inference dump."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.rm_hub.math_utils import grade_answer_verl

from experiments.src.staleness_ratio.strict_math_reward import (
    eos_tokens_from_config,
    score_strict_math_sample,
    strict_terminal_answer_error,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True)
    parser.add_argument(
        "--strict-records",
        help="Optional saved strict rewards; verify that recomputed rewards match them.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokenizer-config",
        help="Tokenizer config file or directory used to resolve terminal EOS strings.",
    )
    parser.add_argument(
        "--audit-dir",
        help="Write full accepted/rejected JSONL audit artifacts here.",
    )
    return parser.parse_args()


def _finish_type(finish_reason):
    if isinstance(finish_reason, dict):
        return finish_reason.get("type")
    return finish_reason


def _rate(count, total):
    return count / total if total else 0.0


_BOXED_START = re.compile(r"\\boxed\s*\{")
_ANSWER_LIKE_MARKER = re.compile(
    r"^[ \t]*(?:#+[ \t]*)?(?:final[ \t]+)?answer[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_ANSWER_LIKE_BOX = re.compile(
    r"\b(?:answer|result|minimum|maximum|largest|smallest|final)\b",
    re.IGNORECASE,
)


def _boxed_values(response):
    values = []
    for match in _BOXED_START.finditer(response):
        depth = 1
        index = match.end()
        while index < len(response):
            if response[index] == "\\":
                index += 2
                continue
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
                if depth == 0:
                    values.append(
                        {
                            "start": match.start(),
                            "end": index + 1,
                            "value": response[match.end() : index],
                        }
                    )
                    break
            index += 1
    return values


def _normalized_box(value):
    return re.sub(r"[\s,$]", "", value).lower()


def _audit_fields(response):
    boxes = _boxed_values(response)
    terminal_value = boxes[-1]["value"] if boxes else None
    terminal_normalized = _normalized_box(terminal_value) if terminal_value else None
    earlier_boxes = boxes[:-1]
    duplicate_terminal_boxes = sum(
        _normalized_box(box["value"]) == terminal_normalized for box in earlier_boxes
    )
    answer_like_boxes = 0
    for box in earlier_boxes:
        line_start = response.rfind("\n", 0, box["start"]) + 1
        prefix = response[line_start : box["start"]]
        answer_like_boxes += bool(_ANSWER_LIKE_BOX.search(prefix))

    words = re.findall(r"[a-z0-9_]+|\\[a-z]+", response.lower())
    ngram_size = 12
    ngrams = [
        tuple(words[index : index + ngram_size])
        for index in range(max(0, len(words) - ngram_size + 1))
    ]
    ngram_repeat_fraction = (
        (len(ngrams) - len(set(ngrams))) / len(ngrams) if ngrams else 0.0
    )

    return {
        "box_count": response.count(r"\boxed"),
        "balanced_box_count": len(boxes),
        "box_values": [box["value"] for box in boxes],
        "duplicate_terminal_box_count": duplicate_terminal_boxes,
        "answer_like_box_before_terminal_count": answer_like_boxes,
        "answer_like_marker_count": len(_ANSWER_LIKE_MARKER.findall(response)),
        "think_open_count": response.count("<think>"),
        "think_close_count": response.count("</think>"),
        "twelve_word_ngram_repeat_fraction": ngram_repeat_fraction,
    }


def _write_jsonl(path, records):
    with path.open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    eos_tokens = (
        eos_tokens_from_config(args.tokenizer_config)
        if args.tokenizer_config
        else ()
    )
    response_rows = [json.loads(line) for line in Path(args.responses).open()]
    strict_rows = (
        [json.loads(line) for line in Path(args.strict_records).open()]
        if args.strict_records
        else []
    )
    strict_by_index = {row["index"]: row for row in strict_rows}
    if len(strict_by_index) != len(strict_rows):
        raise ValueError("strict records contain duplicate prompt indices")

    totals = Counter()
    rejected_correct = Counter()
    prompt_metrics = []
    audit_records = []
    for row in response_rows:
        strict_record = strict_by_index.get(row["index"])
        strict_rewards = []
        math_rewards = []
        for sample_index, (response, finish_reason, completion_tokens) in enumerate(
            zip(
                row["responses"],
                row["finish_reasons"],
                row["completion_tokens"],
                strict=True,
            )
        ):
            finish_type = _finish_type(finish_reason)
            status = "truncated" if finish_type == "length" else "completed"
            sample = SimpleNamespace(response=response, label=row["label"], status=status)
            strict_reward = int(score_strict_math_sample(sample, eos_tokens))
            math_reward = int(bool(grade_answer_verl(response, row["label"])))
            strict_rewards.append(strict_reward)
            math_rewards.append(math_reward)

            totals["samples"] += 1
            totals["strict_correct"] += strict_reward
            totals["math_correct"] += math_reward
            totals["completed"] += finish_type == "stop"
            totals["truncated"] += finish_type == "length"
            totals["single_box"] += response.count(r"\boxed") == 1
            format_error = strict_terminal_answer_error(response, eos_tokens)
            totals["strict_format"] += format_error is None and finish_type == "stop"
            if math_reward and not strict_reward:
                reason = "truncated" if finish_type == "length" else format_error or "unknown"
                rejected_correct[reason] += 1

            audit_records.append(
                {
                    "sample_id": f"prompt-{row['index']:04d}-sample-{sample_index:02d}",
                    "prompt_index": row["index"],
                    "sample_index": sample_index,
                    "label": row["label"],
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                    "math_correct": bool(math_reward),
                    "strict_format_valid": format_error is None and finish_type == "stop",
                    "strict_format_error": (
                        "truncated" if finish_type == "length" else format_error
                    ),
                    "reward": bool(strict_reward),
                    **_audit_fields(response),
                    "prompt": row["prompt_text"],
                    "response": response,
                }
            )

        if strict_record is not None and strict_rewards != strict_record["rewards"]:
            raise ValueError(f"strict reward mismatch for prompt index {row['index']}")
        prompt_metrics.append(
            {
                "strict_any": any(strict_rewards),
                "strict_informative": min(strict_rewards) != max(strict_rewards),
                "math_any": any(math_rewards),
                "math_informative": min(math_rewards) != max(math_rewards),
            }
        )

    prompts = len(prompt_metrics)
    samples = totals["samples"]
    summary = {
        "prompts": prompts,
        "samples": samples,
        "samples_per_prompt": samples / prompts if prompts else 0.0,
        "strict_accuracy": _rate(totals["strict_correct"], samples),
        "math_accuracy": _rate(totals["math_correct"], samples),
        "accuracy_delta_strict_minus_math": _rate(
            totals["strict_correct"] - totals["math_correct"], samples
        ),
        "strict_correct": totals["strict_correct"],
        "math_correct": totals["math_correct"],
        "strict_format_rate": _rate(totals["strict_format"], samples),
        "single_box_rate": _rate(totals["single_box"], samples),
        "truncated_rate": _rate(totals["truncated"], samples),
        "strict_pass_at_n": _rate(sum(row["strict_any"] for row in prompt_metrics), prompts),
        "math_pass_at_n": _rate(sum(row["math_any"] for row in prompt_metrics), prompts),
        "strict_informative_group_rate": _rate(
            sum(row["strict_informative"] for row in prompt_metrics), prompts
        ),
        "math_informative_group_rate": _rate(
            sum(row["math_informative"] for row in prompt_metrics), prompts
        ),
        "math_correct_rejected_by_strict": sum(rejected_correct.values()),
        "rejected_correct_reasons": dict(sorted(rejected_correct.items())),
    }
    if args.audit_dir:
        audit_dir = Path(args.audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        accepted = [record for record in audit_records if record["reward"]]
        rejected = [record for record in audit_records if not record["reward"]]
        format_rejected_correct = [
            record
            for record in rejected
            if record["math_correct"] and not record["strict_format_valid"]
        ]
        repetition_candidates = [
            record
            for record in accepted
            if record["duplicate_terminal_box_count"]
            or record["answer_like_box_before_terminal_count"]
            or record["answer_like_marker_count"] > 1
            or record["twelve_word_ngram_repeat_fraction"] >= 0.1
        ]
        summary.update(
            {
                "accepted_with_duplicate_terminal_box": sum(
                    record["duplicate_terminal_box_count"] > 0 for record in accepted
                ),
                "accepted_with_two_or_more_duplicate_terminal_boxes": sum(
                    record["duplicate_terminal_box_count"] >= 2 for record in accepted
                ),
                "accepted_repetition_candidate_count": len(repetition_candidates),
                "samples_with_think_tags": sum(
                    bool(record["think_open_count"] or record["think_close_count"])
                    for record in audit_records
                ),
            }
        )
        _write_jsonl(audit_dir / "all-samples.jsonl", audit_records)
        _write_jsonl(audit_dir / "accepted.jsonl", accepted)
        _write_jsonl(audit_dir / "rejected.jsonl", rejected)
        _write_jsonl(
            audit_dir / "format-rejected-math-correct.jsonl", format_rejected_correct
        )
        _write_jsonl(
            audit_dir / "accepted-repetition-candidates.jsonl", repetition_candidates
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
