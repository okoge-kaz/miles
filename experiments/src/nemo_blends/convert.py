"""Convert Nemotron-3-Super RL blend components into the prompt JSONL miles reads.

One adapter per component. Every adapter returns the same shape, which is what
`--input-key prompt --label-key label --apply-chat-template` already expects:

    {"prompt": [{"role": ..., "content": ...}, ...],
     "label": "<ground truth the verifier compares against>",
     "metadata": {...}}

`metadata` is passed through to the reward function (`rm_hub.async_rm` forwards
it), so anything a verifier needs beyond the label travels with the row.

Components are registered rather than hard-coded into one function because the
seven usable blend members differ only in field names, not in shape -- adding
the next one is a dict entry, not a new script. Run with --list to see them.

    python -m experiments.src.nemo_blends.convert --dataset knowledge-mcqa \
        --input /data/nemotron-rl-mcqa/data/*.parquet \
        --output /data/nemotron-rl-mcqa/knowledge-mcqa-miles.jsonl
"""

import argparse
import glob
import json
import os
import string
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from experiments.src.nemo_blends.responses_api import (  # noqa: E402
    expected_action_signature,
    to_chat_messages,
    to_chat_tools,
)

# The instruction DAPO-Math carries. Datasets whose prompts do not already state
# an answer format need it, or the policy has no reason to emit \boxed{} and a
# boxed-answer verifier scores every response 0.
BOXED_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: \\boxed{$Answer} where $Answer is "
    "the answer to the problem.\n\n"
)
BOXED_REMINDER = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def read_rows(paths):
    for path in paths:
        p = Path(path)
        if p.suffix == ".jsonl":
            with p.open() as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        elif p.suffix == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(p)
            for i in range(pf.num_row_groups):
                yield from pf.read_row_group(i).to_pylist()
        else:
            raise ValueError(f"unsupported input: {p}")


def _chat_from_responses_create_params(row):
    """NeMo Gym rows carry the ready-made chat under responses_create_params.input."""
    params = row.get("responses_create_params") or {}
    messages = params.get("input")
    if not messages:
        return None
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        # A leading empty system turn adds nothing and costs prompt tokens.
        if content is None or (role == "system" and not str(content).strip()):
            continue
        out.append({"role": role or "user", "content": str(content)})
    return out or None


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


def adapt_knowledge_mcqa(row):
    """nvidia/Nemotron-RL-knowledge-mcqa -> --rm-type gpqa.

    The prompt already ends with "The last line of your response should be in
    the following format: 'Answer: A/B/...'", which is the shape
    `gpqa._extract_letter_from_response` matches, so nothing is added to it.
    """
    prompt = _chat_from_responses_create_params(row)
    label = row.get("expected_answer")
    if not prompt or not label:
        return None

    # `options` is a list of {letter: text} with unused letters set to null.
    # Passing the live letters keeps the extractor from accepting a letter the
    # question never offered.
    valid = []
    for entry in row.get("options") or []:
        if isinstance(entry, dict):
            valid.extend(k.upper() for k, v in entry.items() if v is not None)
    valid = sorted(set(valid)) or list(string.ascii_uppercase[:8])

    meta = {"source": "knowledge-mcqa", "valid_letters": valid}
    # Shipped difficulty: [{"model_hf_path": ..., "num_generations": 5, "pass_rate": ...}]
    if row.get("reward_profiles"):
        meta["reward_profiles"] = row["reward_profiles"]
    return {"prompt": prompt, "label": str(label).strip().upper(), "metadata": meta}


def adapt_skywork_or1_math(row):
    """Skywork/Skywork-OR1-RL-Data (math split) -> --rm-type math.

    Two things the raw row does not give us:
      * the question carries no answer-format instruction, so the boxed wrapper
        is added -- without it the policy has no reason to emit \\boxed{} and a
        boxed-answer verifier scores everything 0;
      * `reward_model.ground_truth` is a JSON *array encoded as a string*
        ('["15625"]'), not a bare answer.
    """
    messages = row.get("prompt")
    if not messages:
        return None
    question = " ".join(str(m.get("content", "")) for m in messages if m.get("role") != "system").strip()
    if not question:
        return None

    truth = ((row.get("reward_model") or {}).get("ground_truth")) or ""
    if isinstance(truth, str) and truth.strip().startswith("["):
        try:
            decoded = json.loads(truth)
            truth = decoded[0] if decoded else ""
        except json.JSONDecodeError:
            pass
    if isinstance(truth, list):
        truth = truth[0] if truth else ""
    truth = str(truth).strip()
    if not truth:
        return None

    meta = {"source": "skywork-or1-math", "data_source": row.get("data_source")}
    extra = row.get("extra_info") or {}
    # Per-model difficulty labels for three DeepSeek-R1-Distill sizes: an
    # independent reference for whatever pass rate we measure ourselves.
    if extra.get("model_difficulty"):
        meta["model_difficulty"] = extra["model_difficulty"]

    return {
        "prompt": [{"role": "user", "content": BOXED_INSTRUCTION + question + BOXED_REMINDER}],
        "label": truth,
        "metadata": meta,
    }


def adapt_reasoning_gym(row):
    """nvidia/Nemotron-RL-ReasoningGym-v1 -> rewards.reasoning_gym_reward.

    104 procedurally generated environments whose ground truth is the literal
    answer string, so nothing but the answer needs to travel with the row.
    """
    prompt = _chat_from_responses_create_params(row) or (
        [{"role": "user", "content": str(row["question"])}] if row.get("question") else None
    )
    answer = row.get("answer")
    if not prompt or answer in (None, ""):
        return None
    src = row.get("metadata") or {}
    return {
        "prompt": prompt,
        "label": str(answer),
        "metadata": {
            "source": "reasoning-gym",
            # 104 environments in one file: keeping the generator name lets a
            # pass-rate report be broken down per environment instead of averaged
            # into a single meaningless number.
            "source_dataset": src.get("source_dataset"),
            "difficulty": src.get("difficulty"),
        },
    }


def adapt_structured_outputs(row):
    """nvidia/...instruction_following-structured_outputs -> rewards.structured_output_reward."""
    prompt = _chat_from_responses_create_params(row)
    schema = row.get("schema_str")
    if not prompt or not schema:
        return None
    return {
        "prompt": prompt,
        # The schema is both the label and what the verifier validates against.
        "label": schema,
        "metadata": {
            "source": "structured-outputs",
            "schema_str": schema,
            "schema_fields_count": row.get("schema_fields_count"),
        },
    }


def adapt_instruction_following(row):
    """nvidia/Nemotron-RL-instruction_following -> --rm-type ifbench.

    The row is already in IFEval/IFBench shape; miles' ifbench reward reads
    `instruction_id_list`, `prompt_text`, `kwargs` and `record_id` off the
    metadata (rm_hub/ifbench.py:131-151), so this is a rename and nothing more.
    """
    prompt = _chat_from_responses_create_params(row) or (
        [{"role": "user", "content": str(row["prompt"])}] if row.get("prompt") else None
    )
    instruction_ids = row.get("instruction_id_list")
    if not prompt or not instruction_ids:
        return None
    return {
        "prompt": prompt,
        # ifbench decides from metadata; the label is unused but kept readable.
        "label": ";".join(str(x) for x in instruction_ids),
        "metadata": {
            "source": "instruction-following",
            "instruction_id_list": list(instruction_ids),
            "prompt_text": str(row.get("prompt") or ""),
            "kwargs": row.get("kwargs") or [],
            "record_id": row.get("id") or 0,
        },
    }


def _adapt_expert_action(row, source):
    """Shared by the two tool-use components: identical shape, different origin."""
    params = row.get("responses_create_params") or {}
    messages = to_chat_messages(params.get("input"))
    expected = row.get("expected_action")
    if not messages or not expected:
        return None
    signature = expected_action_signature(expected)
    if signature is None:
        return None

    meta = {
        "source": source,
        "expected_action": expected,
        # Per-row tool signatures: these datasets span 838 domains, so the tool
        # set differs line by line and cannot come from a single global spec.
        "tools": to_chat_tools(params.get("tools")),
        "expected_kind": signature["kind"],
    }
    # conv-tooluse ships a Qwen3-235B pass rate over 32 samples -- an independent
    # difficulty reference to check our own measurement against.
    for key in ("pass_rate", "pass_rate_total", "pass_rate_passed"):
        if row.get(key) is not None:
            meta[key] = row[key]
    return {
        "prompt": messages,
        "label": signature.get("name") or signature["kind"],
        "metadata": meta,
    }


def adapt_conv_tooluse(row):
    """nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1 -> rewards.tool_call_match_reward."""
    return _adapt_expert_action(row, "conv-tooluse")


def adapt_fncall_pivot(row):
    """nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 -> rewards.tool_call_match_reward."""
    return _adapt_expert_action(row, "fncall-pivot")


def adapt_arc_agi(row):
    """nvidia/Nemotron-RL-ARC-AGI-v1 (transductive) -> grid_and_ast.arc_agi_reward.

    Deterministic and judge-free: the policy prints the output grid in
    `\\boxed{}` and it is compared cell-for-cell. The row also carries a
    continuous `difficulty` plus a bucket, so this is the one set where a
    difficulty window can be cut without measuring anything first.
    """
    prompt = _chat_from_responses_create_params(row)
    expected = row.get("expected_output")
    if not prompt or not expected:
        return None
    return {
        "prompt": prompt,
        "label": json.dumps(expected),
        "metadata": {
            "source": "arc-agi",
            "expected_output": expected,
            "task_id": row.get("task_id"),
            "difficulty": row.get("difficulty"),
            "difficulty_bucket": row.get("difficulty_bucket"),
        },
    }


def adapt_swe_pivot(row):
    """nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1 -> rewards.tool_call_match_reward.

    The SWE stage that needs no container. NeMo-RL's own stage2_swe1.yaml runs
    exactly this environment
    (`swe_pivot_single_step_tool_use_with_argument_comparison`) and only stage 2.2
    reaches for per-instance .sif images -- so the row is the same shape as
    fncall-pivot, and the same verifier grades it.
    """
    return _adapt_expert_action(row, "swe-pivot")


def adapt_competitive_coding(row):
    """nvidia/Nemotron-RL-coding-competitive_coding -> code_exec.code_exec_reward."""
    prompt = _chat_from_responses_create_params(row)
    tests = (row.get("verifier_metadata") or {}).get("unit_tests")
    if not prompt or not isinstance(tests, dict) or not tests.get("inputs"):
        return None
    return {
        "prompt": prompt,
        # The tests are the ground truth; the label is only for readability.
        "label": f"{len(tests['inputs'])} tests",
        "metadata": {
            "source": "competitive-coding",
            "unit_tests": tests,
            "problem_id": (row.get("verifier_metadata") or {}).get("problem_id"),
            "dataset": row.get("dataset"),
        },
    }


def adapt_mmlu_pro(row):
    """TIGER-Lab/MMLU-Pro -> --rm-type gpqa. Eval only.

    Up to ten options, against MMLU's four, so the valid letters have to be
    derived per row rather than assumed.
    """
    question = row.get("question")
    options = list(row.get("options") or [])
    answer = row.get("answer")
    if not question or not options or not answer:
        return None
    letters = [string.ascii_uppercase[i] for i in range(len(options))]
    rendered = "\n".join(f"{ltr}. {opt}" for ltr, opt in zip(letters, options))
    content = (
        "Answer the following multiple choice question. The last line of your "
        f"response should be in the following format: 'Answer: {'/'.join(letters[:4])}...' "
        f"(e.g. 'Answer: {letters[0]}').\n\n{question}\n\n{rendered}"
    )
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": str(answer).strip().upper(),
        "metadata": {
            "source": "mmlu-pro",
            "valid_letters": letters,
            "choices": options,
            "category": row.get("category"),
        },
    }


ADAPTERS = {
    "knowledge-mcqa": (adapt_knowledge_mcqa, "gpqa"),
    "skywork-or1-math": (adapt_skywork_or1_math, "math"),
    "reasoning-gym": (adapt_reasoning_gym, "custom:experiments.src.nemo_blends.rewards.reasoning_gym_reward"),
    "structured-outputs": (
        adapt_structured_outputs,
        "custom:experiments.src.nemo_blends.rewards.structured_output_reward",
    ),
    "instruction-following": (adapt_instruction_following, "ifbench"),
    "conv-tooluse": (adapt_conv_tooluse, "custom:experiments.src.nemo_blends.rewards.tool_call_match_reward"),
    "fncall-pivot": (adapt_fncall_pivot, "custom:experiments.src.nemo_blends.rewards.tool_call_match_reward"),
    "swe-pivot": (adapt_swe_pivot, "custom:experiments.src.nemo_blends.rewards.tool_call_match_reward"),
    "arc-agi": (adapt_arc_agi, "custom:experiments.src.nemo_blends.grid_and_ast.arc_agi_reward"),
    "competitive-coding": (adapt_competitive_coding, "custom:experiments.src.nemo_blends.code_exec.code_exec_reward"),
    "mmlu-pro": (adapt_mmlu_pro, "gpqa"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help=f"one of: {', '.join(sorted(ADAPTERS))}")
    ap.add_argument("--input", nargs="+", help="files or globs")
    ap.add_argument("--output")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="show registered adapters and their --rm-type")
    args = ap.parse_args()

    if args.list:
        for name, (_, rm) in sorted(ADAPTERS.items()):
            print(f"  {name:22s} --rm-type {rm}")
        return
    for required in ("dataset", "input", "output"):
        if not getattr(args, required):
            ap.error(f"--{required} is required")
    if args.dataset not in ADAPTERS:
        ap.error(f"unknown dataset {args.dataset}; choose from {', '.join(sorted(ADAPTERS))}")

    adapt, rm_type = ADAPTERS[args.dataset]
    paths = sorted(p for pattern in args.input for p in glob.glob(pattern)) or args.input

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with out.open("w") as f:
        for row in read_rows(paths):
            converted = adapt(row)
            if converted is None:
                skipped += 1
                continue
            f.write(json.dumps(converted) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break

    print(f"{args.dataset}: wrote {kept} rows to {out} (skipped {skipped}), use --rm-type {rm_type}")


if __name__ == "__main__":
    main()
