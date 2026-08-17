"""Audit terminal-answer formatting in saved Miles rollout ``.pt`` files."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

from experiments.src.staleness_ratio.strict_math_reward import (
    eos_tokens_from_config,
    strict_terminal_answer_error,
)


_ANSWER_LINE = re.compile(r"^[ \t]*Answer[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)
_BOXED_PREFIX = r"\boxed{"
_ANSWER_PREFIX = re.compile(r"[ \t]*\$?[ \t]*")
_ANSWER_SUFFIX = re.compile(
    r"[ \t]*(?:\$[ \t]*)?(?:\.[ \t]*)?"
    r"(?:(?:✅|✔️|☑️|🎉|🚀)[ \t]*)?"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout",
        action="append",
        required=True,
        metavar="STEP:PATH",
        help="Training step and saved rollout path; repeat for multiple steps.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tokenizer-config",
        help="Tokenizer config file or directory used to resolve terminal EOS strings.",
    )
    return parser.parse_args()


def _balanced_boxed_span(response, start):
    if not response.startswith(_BOXED_PREFIX, start):
        return None
    depth = 0
    for index in range(start + len(r"\boxed"), len(response)):
        character = response[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
            if depth < 0:
                return None
    return None


def _answer_line_info(response, marker, eos_tokens):
    line_end = response.find("\n", marker.end())
    if line_end < 0:
        line_end = len(response)
    content_end = line_end
    while True:
        content = response[marker.start() : content_end].rstrip()
        matching_tokens = [
            token for token in eos_tokens if token and content.endswith(token)
        ]
        if not matching_tokens:
            break
        token = max(matching_tokens, key=len)
        content_end = response.rfind(token, marker.start(), content_end)
    box_start = response.find(_BOXED_PREFIX, marker.end(), content_end)
    if box_start < 0:
        return {
            "line": response[marker.start() : line_end],
            "valid": False,
            "box_value": None,
        }
    span = _balanced_boxed_span(response, box_start)
    if span is None or span[1] > content_end:
        return {
            "line": response[marker.start() : line_end],
            "valid": False,
            "box_value": None,
        }
    prefix_valid = _ANSWER_PREFIX.fullmatch(response[marker.end() : box_start]) is not None
    suffix_valid = _ANSWER_SUFFIX.fullmatch(response[span[1] : content_end]) is not None
    return {
        "line": response[marker.start() : line_end],
        "valid": prefix_valid and suffix_valid,
        "box_value": response[box_start + len(_BOXED_PREFIX) : span[1] - 1],
    }


def _ngram_repeat_fraction(text, size=12):
    words = re.findall(r"[a-z0-9_]+|\\[a-z]+", text.lower())
    ngrams = [
        tuple(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    ]
    if not ngrams:
        return 0.0
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def _max_consecutive_identical_line_run(text):
    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    longest = 0
    current = 0
    previous = None
    for line in lines:
        if line == previous:
            current += 1
        else:
            previous = line
            current = 1
        longest = max(longest, current)
    return longest


def _status_value(status):
    value = getattr(status, "value", status)
    return str(value).lower()


def _record(step, ordinal, sample, eos_tokens=()):
    response = str(sample.get("response") or "")
    status = _status_value(sample.get("status"))
    markers = list(_ANSWER_LINE.finditer(response))
    answer_lines = [
        _answer_line_info(response, marker, eos_tokens) for marker in markers
    ]
    current_error = strict_terminal_answer_error(response, eos_tokens)
    current_format_accept = status == "completed" and current_error is None

    separator = None
    separator_repeat_fraction = None
    separator_line_run = None
    two_answer_candidate = False
    two_answer_candidate_error = "answer_marker_count_not_two"
    if len(markers) == 2:
        first_line_end = response.find("\n", markers[0].end())
        if first_line_end < 0:
            first_line_end = len(response)
        else:
            first_line_end += 1
        separator = response[first_line_end : markers[1].start()]
        separator_repeat_fraction = _ngram_repeat_fraction(separator)
        separator_line_run = _max_consecutive_identical_line_run(separator)
        final_segment_valid = (
            strict_terminal_answer_error(
                response[markers[1].start() :], eos_tokens
            )
            is None
        )
        checks = (
            (status == "completed", "truncated"),
            (answer_lines[0]["valid"], "first_answer_line_invalid"),
            (final_segment_valid, "final_answer_invalid"),
            (bool(separator.strip()), "empty_separator"),
            (separator_repeat_fraction < 0.2, "separator_ngram_repetition"),
            (separator_line_run < 3, "separator_line_repetition"),
        )
        two_answer_candidate_error = next(
            (reason for valid, reason in checks if not valid),
            None,
        )
        two_answer_candidate = two_answer_candidate_error is None

    proposed_format_accept = current_format_accept or two_answer_candidate
    original_math_correct = bool(sample.get("reward"))
    metadata = sample.get("metadata") or {}
    return {
        "sample_id": f"rollout-{step}-ordinal-{ordinal:04d}",
        "rollout_step": step,
        "ordinal": ordinal,
        "sample_index": sample.get("index"),
        "group_index": sample.get("group_index"),
        "label": sample.get("label"),
        "status": status,
        "response_length": sample.get("response_length"),
        "original_math_correct": original_math_correct,
        "answer_marker_count": len(markers),
        "answer_lines": answer_lines,
        "current_one_answer_format_accept": current_format_accept,
        "current_one_answer_error": (
            "truncated" if status != "completed" else current_error
        ),
        "current_one_answer_reward": original_math_correct and current_format_accept,
        "two_answer_candidate_accept": two_answer_candidate,
        "two_answer_candidate_error": two_answer_candidate_error,
        "proposed_up_to_two_format_accept": proposed_format_accept,
        "proposed_up_to_two_reward": original_math_correct and proposed_format_accept,
        "separator": separator,
        "separator_char_count": len(separator) if separator is not None else None,
        "separator_line_count": len(separator.splitlines()) if separator is not None else None,
        "separator_twelve_word_ngram_repeat_fraction": separator_repeat_fraction,
        "separator_max_consecutive_identical_line_run": separator_line_run,
        "submission_weight_version": metadata.get("submission_weight_version"),
        "sample_generation_complete_weight_version": metadata.get(
            "sample_generation_complete_weight_version"
        ),
        "weight_versions": sample.get("weight_versions"),
        "prompt": sample.get("prompt"),
        "response": response,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    per_step = {}
    for specification in args.rollout:
        step_text, path_text = specification.split(":", maxsplit=1)
        step = int(step_text)
        payload = torch.load(
            path_text,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        records = [
            _record(step, ordinal, sample, eos_tokens)
            for ordinal, sample in enumerate(payload["samples"])
        ]
        all_records.extend(records)
        per_step[step] = records
        del payload

    _write_jsonl(output_dir / "all-classified.jsonl", all_records)
    two_answer_review = [
        record for record in all_records if record["answer_marker_count"] == 2
    ]
    _write_jsonl(output_dir / "exactly-two-answer-review.jsonl", two_answer_review)
    _write_jsonl(
        output_dir / "exactly-two-answer-accepted.jsonl",
        [record for record in two_answer_review if record["two_answer_candidate_accept"]],
    )
    _write_jsonl(
        output_dir / "exactly-two-answer-rejected.jsonl",
        [record for record in two_answer_review if not record["two_answer_candidate_accept"]],
    )
    remaining_format_rejected_correct = [
        record
        for record in all_records
        if record["original_math_correct"]
        and not record["current_one_answer_format_accept"]
    ]
    _write_jsonl(
        output_dir / "format-rejected-math-correct.jsonl",
        remaining_format_rejected_correct,
    )
    long_responses = [
        record for record in all_records if (record["response_length"] or 0) >= 16_384
    ]
    _write_jsonl(output_dir / "long-responses.jsonl", long_responses)

    summary = {}
    for step, records in per_step.items():
        counts = Counter()
        for record in records:
            counts["samples"] += 1
            counts["completed"] += record["status"] == "completed"
            counts["truncated"] += record["status"] != "completed"
            counts["original_math_correct"] += record["original_math_correct"]
            counts["current_format_accept"] += record[
                "current_one_answer_format_accept"
            ]
            counts["current_reward"] += record["current_one_answer_reward"]
            counts["exactly_two_answers"] += record["answer_marker_count"] == 2
            counts["two_answer_candidate_accept"] += record[
                "two_answer_candidate_accept"
            ]
            counts["proposed_format_accept"] += record[
                "proposed_up_to_two_format_accept"
            ]
            counts["proposed_reward"] += record["proposed_up_to_two_reward"]
            counts["long_response"] += (record["response_length"] or 0) >= 16_384
        summary[str(step)] = dict(counts)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
