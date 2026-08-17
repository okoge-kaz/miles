"""Compare prompt-only thinking-format interventions on one frozen policy."""

import argparse
import asyncio
import copy
import json
import os
import random
import re
import statistics
from collections import defaultdict

import aiohttp
from transformers import AutoTokenizer

from experiments.src.difficulty_filter.measure_pass_rate import generate_group, load_prompts
from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.metric_utils import has_repetition


FORMAT_INSTRUCTION = r"""Follow this output format exactly:
1. Put all reasoning inside a <think>...</think> block.
2. After </think>, output exactly one final line of the form Answer: \boxed{...}.
3. Stop immediately after that line. Do not emit any other \boxed command."""

VARIANTS = ("native", "system", "prefix", "system_prefix")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-key", default="prompt")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=3600)
    return parser.parse_args()


def _with_system_instruction(prompt):
    if not isinstance(prompt, list):
        raise ValueError("format pilot requires conversation-format prompts")
    messages = copy.deepcopy(prompt)
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        messages[0]["content"] = f"{content}\n\n{FORMAT_INSTRUCTION}".strip()
    else:
        messages.insert(0, {"role": "system", "content": FORMAT_INSTRUCTION})
    return messages


def build_variant_prompt(tokenizer, prompt, variant):
    if not isinstance(prompt, list):
        raise ValueError("format pilot requires conversation-format prompts")
    messages = _with_system_instruction(prompt) if "system" in variant else prompt
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "prefix" in variant:
        text += "<think>\n"
    return text


def _boxed_span(text):
    start = text.find("\\boxed")
    if start < 0:
        return None
    left = text.find("{", start + len("\\boxed"))
    if left < 0:
        return None
    depth = 0
    for index in range(left, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
    return None


def _finish_type(meta):
    finish_reason = meta.get("finish_reason")
    if isinstance(finish_reason, dict):
        return finish_reason.get("type")
    return finish_reason


def analyze_response(response, label, meta):
    close_count = response.count("</think>")
    post_think = response.rsplit("</think>", 1)[-1] if close_count else ""
    post_box_count = post_think.count("\\boxed")
    span = _boxed_span(post_think) if post_box_count == 1 else None
    answer_prefix = False
    terminal_box = False
    if span is not None:
        before = post_think[: span[0]]
        after = post_think[span[1] :]
        answer_prefix = re.fullmatch(r"\s*Answer\s*:\s*", before, flags=re.IGNORECASE) is not None
        terminal_box = after.strip() == ""

    strict_format = close_count == 1 and post_box_count == 1 and answer_prefix and terminal_box
    finish_type = _finish_type(meta)
    return {
        "close_count": close_count,
        "has_close": close_count > 0,
        "single_close": close_count == 1,
        "box_count": response.count("\\boxed"),
        "post_box_count": post_box_count,
        "single_post_box": post_box_count == 1,
        "answer_prefix": answer_prefix,
        "terminal_box": terminal_box,
        "strict_format": strict_format,
        "math_reward": int(grade_answer_verl(response, label)),
        "deepscaler_reward": int(get_deepscaler_rule_based_reward(response, label)),
        "strict_reward": int(strict_format and get_deepscaler_rule_based_reward(response, label)),
        "completion_tokens": int(meta.get("completion_tokens", 0)),
        "finish_type": finish_type,
        "truncated": finish_type == "length",
        "repetition": has_repetition(response),
        "comma_loop": re.search(r",{8,}", response) is not None,
    }


async def _measure_one(session, args, tokenizer, row, variant, semaphore):
    prompt_text = build_variant_prompt(tokenizer, row["prompt"], variant)
    sampling_params = {
        "n": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "sampling_seed": args.seed + row["index"],
    }
    async with semaphore:
        outputs = await generate_group(
            session,
            args.server_url,
            prompt_text,
            sampling_params,
            args.request_timeout,
        )
    samples = []
    for output in outputs:
        response = output["text"]
        meta = output.get("meta_info", {})
        samples.append({"response": response, **analyze_response(response, row["label"], meta)})
    return {
        "index": row["index"],
        "variant": variant,
        "label": row["label"],
        "prompt_text": prompt_text,
        "samples": samples,
    }


def _rate(samples, key):
    return sum(bool(sample[key]) for sample in samples) / max(1, len(samples))


def _group_variance_rate(records, reward_key):
    informative = 0
    for record in records:
        rewards = [sample[reward_key] for sample in record["samples"]]
        informative += int(min(rewards) != max(rewards))
    return informative / max(1, len(records))


def summarize(records):
    by_variant = defaultdict(list)
    for record in records:
        by_variant[record["variant"]].append(record)

    summary = {}
    boolean_metrics = (
        "has_close",
        "single_close",
        "single_post_box",
        "answer_prefix",
        "terminal_box",
        "strict_format",
        "math_reward",
        "deepscaler_reward",
        "strict_reward",
        "truncated",
        "repetition",
        "comma_loop",
    )
    for variant in VARIANTS:
        variant_records = by_variant[variant]
        samples = [sample for record in variant_records for sample in record["samples"]]
        lengths = [sample["completion_tokens"] for sample in samples]
        stats = {
            "num_prompts": len(variant_records),
            "num_samples": len(samples),
            **{f"{key}_rate": _rate(samples, key) for key in boolean_metrics},
            "mean_completion_tokens": statistics.fmean(lengths) if lengths else 0.0,
            "median_completion_tokens": statistics.median(lengths) if lengths else 0.0,
            "max_completion_tokens": max(lengths, default=0),
            "deepscaler_informative_group_rate": _group_variance_rate(
                variant_records, "deepscaler_reward"
            ),
            "strict_informative_group_rate": _group_variance_rate(variant_records, "strict_reward"),
        }
        summary[variant] = stats
    return summary


def _print_summary(summary):
    columns = (
        "has_close_rate",
        "strict_format_rate",
        "math_reward_rate",
        "deepscaler_reward_rate",
        "strict_reward_rate",
        "truncated_rate",
        "repetition_rate",
        "mean_completion_tokens",
        "strict_informative_group_rate",
    )
    print("\n=== prompt-format pilot ===", flush=True)
    print("variant".ljust(16) + "".join(column[:12].rjust(14) for column in columns), flush=True)
    for variant in VARIANTS:
        values = "".join(f"{summary[variant][column]:14.4f}" for column in columns)
        print(variant.ljust(16) + values, flush=True)


async def main_async(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    rows = load_prompts(args.prompt_data, args.input_key, args.label_key)
    if args.num_prompts > len(rows):
        raise ValueError(f"requested {args.num_prompts} prompts from a dataset of {len(rows)}")
    selected = sorted(random.Random(args.seed).sample(rows, args.num_prompts), key=lambda row: row["index"])
    print(
        f"selected dataset indices: {[row['index'] for row in selected]}\n"
        f"variants={VARIANTS} n_samples={args.n_samples} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            _measure_one(session, args, tokenizer, row, variant, semaphore)
            for row in selected
            for variant in VARIANTS
        ]
        records = []
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            records.append(await task)
            if completed % max(1, len(tasks) // 10) == 0:
                print(f"completed {completed}/{len(tasks)} prompt-variants", flush=True)

    records.sort(key=lambda record: (record["index"], VARIANTS.index(record["variant"])))
    summary = summarize(records)
    os.makedirs(args.output_dir, exist_ok=True)
    records_path = os.path.join(args.output_dir, "responses.jsonl")
    with open(records_path, "w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as output:
        json.dump({"args": vars(args), "summary": summary}, output, indent=2)
    _print_summary(summary)
    print(f"responses -> {records_path}", flush=True)


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
