"""Pure row adapters for NeMo Gym, GPQA, and LiveCodeBench datasets."""

from __future__ import annotations

import hashlib
import json
import random
import string
from typing import Any

from experiments.src.protocols.openai_responses import (
    expected_action_signature,
    to_chat_messages,
    to_chat_tools,
)

BOXED_INSTRUCTION = (
    "Solve the following math problem step by step. The last line of your response "
    "must be of the form Answer: \\boxed{$Answer}, where $Answer is the answer.\n\n"
)
BOXED_REMINDER = '\n\nPut the final answer on its own line in the form `Answer: \\boxed{...}`.'


def _chat_from_params(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    params = row.get("responses_create_params") or {}
    messages = to_chat_messages(params.get("input"))
    messages = [
        message
        for message in messages
        if message.get("content") or message.get("tool_calls") or message.get("role") == "tool"
    ]
    return messages or None


def _base_metadata(row: dict[str, Any], *, source: str, verifier: str) -> dict[str, Any]:
    metadata = {"source": source, "verifier": verifier}
    for key in (
        "dataset",
        "pass_rate",
        "pass_rate_total",
        "pass_rate_passed",
        "trajectory_id",
        "uuid",
        "id",
    ):
        if row.get(key) is not None:
            metadata[key] = row[key]
    if row.get("agent_ref"):
        metadata["agent_ref"] = row["agent_ref"]
    return metadata


def _with_tools(converted: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    tools = to_chat_tools((row.get("responses_create_params") or {}).get("tools"))
    if tools:
        converted["tools"] = tools
    return converted


def _answer_from_ground_truth(raw: Any) -> str:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                return text
        else:
            return text
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return "" if raw is None else str(raw).strip()


def adapt_knowledge_mcqa(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    label = str(row.get("expected_answer") or "").strip().upper()
    if not prompt or not label:
        return None
    valid_letters = sorted(
        {
            key.upper()
            for option in row.get("options") or []
            if isinstance(option, dict)
            for key, value in option.items()
            if value is not None
        }
    )
    template = row.get("template_metadata") or {}
    metadata = _base_metadata(row, source="knowledge-mcqa", verifier="mcqa_regex")
    metadata.update(
        {
            "valid_letters": valid_letters or list(string.ascii_uppercase[:8]),
            "output_regex": template.get("output_regex"),
            "template_id": template.get("template_id"),
            "reward_profiles": row.get("reward_profiles") or [],
        }
    )
    return {"prompt": prompt, "label": label, "metadata": metadata}


def adapt_skywork_or1_math(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("prompt") or []
    question = " ".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") != "system"
    ).strip()
    label = _answer_from_ground_truth((row.get("reward_model") or {}).get("ground_truth"))
    if not question or not label:
        return None
    metadata = _base_metadata(row, source="skywork-or1-math", verifier="math")
    extra = row.get("extra_info") or {}
    if extra.get("model_difficulty"):
        metadata["model_difficulty"] = extra["model_difficulty"]
    return {
        "prompt": [{"role": "user", "content": BOXED_INSTRUCTION + question + BOXED_REMINDER}],
        "label": label,
        "metadata": metadata,
    }


def adapt_skywork_or1_code(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = row.get("prompt") or []
    raw_tests = (row.get("reward_model") or {}).get("ground_truth")
    if isinstance(raw_tests, str):
        try:
            unit_tests = json.loads(raw_tests)
        except json.JSONDecodeError:
            return None
    else:
        unit_tests = raw_tests
    io_tests = (
        isinstance(unit_tests, dict)
        and bool(unit_tests.get("inputs"))
        and len(unit_tests.get("inputs") or []) == len(unit_tests.get("outputs") or [])
    )
    harness_tests = (
        isinstance(unit_tests, dict)
        and isinstance(unit_tests.get("entry_point"), str)
        and isinstance(unit_tests.get("test_code"), str)
        and bool(unit_tests["entry_point"].strip())
        and bool(unit_tests["test_code"].strip())
    )
    if not prompt or not (io_tests or harness_tests):
        return None
    metadata = _base_metadata(row, source="skywork-or1-code", verifier="python_code")
    metadata["unit_tests"] = unit_tests
    extra = row.get("extra_info") or {}
    if extra.get("model_difficulty"):
        metadata["model_difficulty"] = extra["model_difficulty"]
    return {
        "prompt": prompt,
        "label": f"{len(unit_tests['inputs'])} tests" if io_tests else "published test harness",
        "metadata": metadata,
    }


def adapt_dapo_math(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    label = _answer_from_ground_truth((row.get("reward_model") or {}).get("ground_truth"))
    if not prompt or not label:
        return None
    prompt = [dict(message) for message in prompt if isinstance(message, dict)]
    for message in reversed(prompt):
        if message.get("role") == "user":
            message["content"] = str(message.get("content") or "") + BOXED_REMINDER
            break
    metadata = _base_metadata(row, source="dapo-math-17k", verifier="math")
    return {"prompt": prompt, "label": label, "metadata": metadata}


def adapt_reasoning_gym(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    if not prompt and row.get("question"):
        prompt = [{"role": "user", "content": str(row["question"])}]
    answer = row.get("answer")
    source_metadata = row.get("metadata") or {}
    if not prompt or not source_metadata.get("source_dataset"):
        return None
    metadata = _base_metadata(row, source="reasoning-gym", verifier="reasoning_gym")
    metadata.update(
        {
            "source_dataset": source_metadata.get("source_dataset"),
            "difficulty": source_metadata.get("difficulty"),
            "verifier_metadata": source_metadata,
            "question": str(row.get("question") or ""),
            "reference_answer": answer,
        }
    )
    return {"prompt": prompt, "label": "" if answer is None else str(answer), "metadata": metadata}


def adapt_structured_outputs(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    schema = row.get("schema_str")
    if not prompt or not schema:
        return None
    metadata = _base_metadata(row, source="structured-outputs", verifier="json_schema")
    metadata.update(
        {
            "schema_str": schema,
            "schema_type": row.get("schema_type"),
            "schema_fields_count": row.get("schema_fields_count"),
        }
    )
    return {"prompt": prompt, "label": schema, "metadata": metadata}


def adapt_instruction_following(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    if not prompt and row.get("prompt"):
        prompt = [{"role": "user", "content": str(row["prompt"])}]
    instruction_ids = row.get("instruction_id_list") or []
    if not prompt or not instruction_ids:
        return None
    prompt_text = str(row.get("prompt") or prompt[-1].get("content") or "")
    metadata = _base_metadata(row, source="instruction-following", verifier="ifeval_g")
    metadata.update(
        {
            "instruction_id_list": list(instruction_ids),
            "prompt_text": prompt_text,
            "kwargs": row.get("kwargs") or [],
            "record_id": row.get("id") or 0,
        }
    )
    return {
        "prompt": prompt,
        "label": ";".join(str(item) for item in instruction_ids),
        "metadata": metadata,
    }


def adapt_ifbench(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one held-out IFBench prompt without exposing it to training."""
    prompt_text = str(row.get("prompt") or "").strip()
    instruction_ids = row.get("instruction_id_list") or []
    kwargs = row.get("kwargs") or []
    if not prompt_text or not instruction_ids or len(kwargs) != len(instruction_ids):
        return None
    metadata = _base_metadata(row, source="ifbench", verifier="ifbench")
    metadata.update(
        {
            "instruction_id_list": list(instruction_ids),
            "prompt_text": prompt_text,
            "kwargs": list(kwargs),
            "record_id": row.get("key"),
            "eval_only": True,
        }
    )
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "label": ";".join(str(item) for item in instruction_ids),
        "metadata": metadata,
    }


def _adapt_expert_action(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    expected = row.get("expected_action")
    signature = expected_action_signature(expected)
    if not prompt or signature is None:
        return None
    metadata = _base_metadata(row, source=source, verifier="expert_action")
    metadata.update({"expected_action": expected, "expected_kind": signature["kind"]})
    converted = {
        "prompt": prompt,
        "label": signature.get("name") or signature["kind"],
        "metadata": metadata,
    }
    return _with_tools(converted, row)


def adapt_conv_tooluse(row: dict[str, Any]) -> dict[str, Any] | None:
    return _adapt_expert_action(row, source="conv-tooluse")


def adapt_fncall_pivot(row: dict[str, Any]) -> dict[str, Any] | None:
    return _adapt_expert_action(row, source="fncall-pivot")


def adapt_swe_pivot(row: dict[str, Any]) -> dict[str, Any] | None:
    return _adapt_expert_action(row, source="swe-pivot")


def adapt_competitive_coding(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    unit_tests = (row.get("verifier_metadata") or {}).get("unit_tests")
    if not prompt or not isinstance(unit_tests, dict) or not unit_tests.get("inputs"):
        return None
    metadata = _base_metadata(row, source="competitive-coding", verifier="python_code")
    metadata.update(
        {
            "unit_tests": unit_tests,
            "problem_id": (row.get("verifier_metadata") or {}).get("problem_id"),
        }
    )
    return {
        "prompt": prompt,
        "label": f"{len(unit_tests['inputs'])} tests",
        "metadata": metadata,
    }


def adapt_gpqa(row: dict[str, Any]) -> dict[str, Any] | None:
    letters = list("ABCD")
    question = str(row.get("Question") or row.get("problem") or "").strip()
    if not question:
        return None

    # NeMo Skills' checked-in GPQA eval assets were produced directly from
    # Idavidrein/gpqa with a seeded shuffle. Preserve that published ordering.
    # Raw HF CSV rows, on the other hand, need their correct-first choice list
    # shuffled here so the label position does not leak the answer.
    if row.get("expected_answer") is not None:
        choices = [str(row.get(letter) or "").strip() for letter in letters]
        answer = str(row.get("expected_answer") or "").strip().upper()
        if answer not in letters or any(not choice for choice in choices):
            return None
        source_format = "nemo-skills-preprocessed"
    else:
        correct = str(row.get("Correct Answer") or "").strip()
        distractors = [str(row.get(f"Incorrect Answer {index}") or "").strip() for index in (1, 2, 3)]
        if not correct or any(not item for item in distractors):
            return None
        choices = [correct, *distractors]
        seed = int(hashlib.sha256(question.encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(choices)
        answer = letters[choices.index(correct)]
        source_format = "idavidrein-csv"

    rendered = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices, strict=True))
    content = (
        "Answer the following multiple choice question. End with exactly "
        "'Answer: A', 'Answer: B', 'Answer: C', or 'Answer: D'.\n\n"
        f"{question}\n\n{rendered}"
    )
    metadata = _base_metadata(row, source="gpqa", verifier="gpqa")
    metadata.update(
        {
            "valid_letters": letters,
            "choices": choices,
            "category": row.get("High-level domain") or row.get("Subdomain"),
            "subdomain": row.get("Subdomain"),
            "record_id": row.get("Record ID"),
            "source_format": source_format,
            "eval_only": True,
        }
    )
    if source_format == "nemo-skills-preprocessed":
        metadata.update(
            {
                "category": row.get("subset_for_metrics"),
                "subdomain": row.get("subset_for_metrics"),
                "explanation": row.get("explanation"),
                "writer_difficulty": row.get("difficulty"),
            }
        )
    for source_key, target_key in (
        ("Expert Validator Accuracy", "expert_accuracy"),
        ("Non-Expert Validator Accuracy", "non_expert_accuracy"),
        ("Writer's Difficulty Estimate", "writer_difficulty"),
    ):
        try:
            metadata[target_key] = float(row[source_key])
        except (KeyError, TypeError, ValueError):
            pass
    return {"prompt": [{"role": "user", "content": content}], "label": answer, "metadata": metadata}


def adapt_livecodebench(row: dict[str, Any]) -> dict[str, Any] | None:
    question = str(row.get("question_content") or "").strip()
    if not question:
        return None
    starter_code = str(row.get("starter_code") or "")
    content = (
        "You will be given a programming problem. Generate a correct Python program "
        "that matches the specification and passes all tests.\n\n"
        f"Question:\n{question}\n\n"
    )
    if starter_code:
        content += (
            "Use the following starter code and enclose the completed solution in a Python code block.\n"
            f"```python\n{starter_code}\n```"
        )
    else:
        content += (
            "Read input from stdin and write the answer to stdout. Return only the complete program "
            "inside a Python code block."
        )
    metadata = _base_metadata(row, source="livecodebench", verifier="livecodebench")
    metadata.update(
        {
            "question_id": row.get("question_id"),
            "platform": row.get("platform"),
            "contest_id": row.get("contest_id"),
            "contest_date": row.get("contest_date"),
            "difficulty": row.get("difficulty"),
            "starter_code": starter_code,
            "public_test_cases": row.get("public_test_cases"),
            "private_test_cases": row.get("private_test_cases"),
            "lcb_metadata": row.get("metadata"),
            "eval_only": True,
        }
    )
    return {"prompt": [{"role": "user", "content": content}], "label": row.get("question_id"), "metadata": metadata}


def _adapt_nano_math(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    label = _answer_from_ground_truth(row.get("expected_answer"))
    if not prompt:
        return None
    if not label:
        metadata = _base_metadata(row, source=str(row.get("dataset")), verifier="missing_ground_truth")
        metadata.update(
            {
                "support_status": "unverifiable_source_row",
                "reason": "the referenced Skywork source row has an empty ground truth",
            }
        )
        return {"prompt": prompt, "label": "", "metadata": metadata}
    metadata = _base_metadata(row, source=str(row.get("dataset")), verifier="math")
    return {"prompt": prompt, "label": label, "metadata": metadata}


def adapt_nano_workbench(row: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _chat_from_params(row)
    if not prompt:
        return None
    metadata = _base_metadata(row, source="nano-workbench", verifier="nemo_gym_environment")
    metadata.update(
        {
            "support_status": "requires_environment",
            "required_backend": "NeMo Gym Workbench server with five databases and 27 tools",
            "environment_name": row.get("environment_name"),
            "category": row.get("category"),
            "ground_truth": row.get("ground_truth"),
        }
    )
    converted = {
        "prompt": prompt,
        "label": json.dumps(row.get("ground_truth") or [], ensure_ascii=False),
        "metadata": metadata,
    }
    return _with_tools(converted, row)


def adapt_nano(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Adapt a restored Nano row and return its ready/environment/unverifiable route."""
    dataset = str(row.get("dataset") or "")
    if dataset == "nano_v3_sft_profiled_workbench":
        return adapt_nano_workbench(row), "environment"
    if dataset == "nano_v3_sft_profiled_comp_coding_50tests":
        return adapt_competitive_coding(row), "ready"
    if dataset == "nano_v3_sft_profiled_instruction_following":
        return adapt_instruction_following(row), "ready"
    if dataset == "nano_v3_sft_profiled_stem_mcqa":
        return adapt_knowledge_mcqa(row), "ready"
    if dataset == "nano_v3_sft_profiled_structured_outputs":
        return adapt_structured_outputs(row), "ready"
    if dataset in {"nano_v3_sft_profiled_dapo17k", "nano_v3_sft_profiled_skywork_no_omni"}:
        converted = _adapt_nano_math(row)
        route = (
            "unverifiable"
            if converted and converted["metadata"]["verifier"] == "missing_ground_truth"
            else "ready"
        )
        return converted, route
    return None, "unverifiable"


ADAPTERS = {
    "competitive-coding": adapt_competitive_coding,
    "conv-tooluse": adapt_conv_tooluse,
    "dapo-math": adapt_dapo_math,
    "fncall-pivot": adapt_fncall_pivot,
    "gpqa": adapt_gpqa,
    "ifbench": adapt_ifbench,
    "instruction-following": adapt_instruction_following,
    "knowledge-mcqa": adapt_knowledge_mcqa,
    "livecodebench": adapt_livecodebench,
    "reasoning-gym": adapt_reasoning_gym,
    "skywork-or1-code": adapt_skywork_or1_code,
    "skywork-or1-math": adapt_skywork_or1_math,
    "structured-outputs": adapt_structured_outputs,
    "swe-pivot": adapt_swe_pivot,
}
