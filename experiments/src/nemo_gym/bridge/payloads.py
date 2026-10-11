import json
import re
from copy import deepcopy

AGENT_VERIFIERS = {"nemo_gym_tau2", "nemo_gym_workplace"}
JUDGE_VERIFIERS = {"nemo_gym_math_judge", "nemo_gym_equivalence_judge", "nemo_gym_rubric_judge"}


def qa_references(label) -> list[str]:
    labels = label if isinstance(label, list) else [label]
    if not labels or any(not isinstance(value, str) or not value.strip() for value in labels):
        raise ValueError("QA requires nonempty reference answer strings")
    return list(dict.fromkeys(labels))


def final_answer(text: str) -> str:
    # The opening tag may be in the generation prompt, outside sample.response.
    text = text.rsplit("</think>", 1)[-1]
    # An unfinished reasoning block contains no final answer to grade.
    return "" if "<think>" in text else text


def qa_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?:<\|(?:im_end|endoftext)\|>\s*)+$", "", text)
    answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    # Wrap only the final answer in the native format; normalization belongs to NeMo Gym.
    return json.dumps({"answer": answers[-1].strip() if answers else text}, ensure_ascii=False)


def code_metadata(metadata: dict) -> dict:
    tests = metadata["unit_tests"]
    if not isinstance(tests, dict) or tests.get("test_code") or tests.get("entry_point"):
        raise ValueError(
            "NeMo Gym code_gen requires native inputs/outputs tests; convert published harness data explicitly"
        )
    inputs, outputs = tests.get("inputs"), tests.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
        raise ValueError("Code needs matching nonempty inputs/outputs")
    if any(not isinstance(value, str) for value in [*inputs, *outputs]):
        raise ValueError("NeMo Gym code_gen test inputs/outputs must be strings")
    return {"unit_tests": deepcopy(tests)}


def instruction_metadata(metadata: dict) -> dict:
    ids, kwargs = metadata["instruction_id_list"], deepcopy(metadata["kwargs"])
    if not ids or len(ids) != len(kwargs):
        raise ValueError("Instruction ids and kwargs must be nonempty and aligned")
    # Published Nemotron rows wrap these string arguments in singleton lists.
    for instruction, arguments in zip(ids, kwargs, strict=True):
        if instruction == "count:count_increment_word" and arguments:
            for name in ("keyword1", "keyword2"):
                if isinstance(arguments.get(name), list) and len(arguments[name]) == 1:
                    arguments[name] = arguments[name][0]
    return {
        "instruction_id_list": ids,
        "kwargs": kwargs,
        "prompt": metadata["prompt_text"],
        "grading_mode": "fraction",
    }


def verify_payload(sample) -> tuple[str, dict]:
    metadata = sample.metadata or {}
    verifier = metadata["verifier"]
    params = {
        "input": sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": sample.prompt}]
    }
    text = final_answer(str(sample.response or ""))
    if verifier == "qa_exact_match":
        text = qa_response(text)
    elif verifier in {"ifeval_g", "ifbench"}:
        # Qwen reasoning is outside the final answer being evaluated.
        text = text.replace("<|im_end|>", "").strip()
        if text.startswith("<answer>") and text.endswith("</answer>"):
            text = text[len("<answer>") : -len("</answer>")].strip()
    response = {
        "id": "miles",
        "created_at": 0.0,
        "model": "miles-policy",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "miles-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }
    body = deepcopy((metadata.get("nemo_gym") or {}).get("request") or {})
    body.update(responses_create_params=params, response=response)
    if verifier in {"ifeval_g", "ifbench"}:
        vm = instruction_metadata(metadata)
        service = "ifbench" if verifier == "ifbench" else "instruction_following"
        body.update({"id": 0, **(vm if service == "ifbench" else {"verifier_metadata": vm})})
    elif verifier == "qa_exact_match":
        service = "hotpotqa_qa"
        body["expected_answer"] = qa_references(sample.label)[0]
    elif verifier == "reasoning_gym":
        service = "reasoning_gym"
        body.update(
            question=metadata["question"],
            answer=metadata["reference_answer"],
            metadata={**metadata["verifier_metadata"], "source_dataset": metadata["source_dataset"]},
        )
    elif verifier == "calendar_constraints":
        service = "calendar"
        body["exp_cal_state"] = metadata["expected_calendar_state"]
    elif verifier == "python_code":
        service = "code_gen"
        body["verifier_metadata"] = code_metadata(metadata)
    elif verifier == "json_schema":
        service = "structured_outputs"
        if body.get("response_mode", "text") != "text":
            raise ValueError("Structured output scoring currently accepts text responses only")
        schema = metadata.get("schema_str") or sample.label
        if not isinstance(schema, (str, dict)):
            raise ValueError("Structured output requires a schema string or object")
        body.update(
            schema_str=schema if isinstance(schema, str) else json.dumps(schema),
            schema_type=metadata.get("schema_type") or "json",
            response_mode="text",
        )
    elif verifier == "nemo_gym_cpp":
        service = "competitive_coding_challenges"
        native = metadata["nemo_gym"]["request"]
        if not native.get("problem_id"):
            raise ValueError("C++ needs a native NeMo Gym problem_id and staged CCC test metadata")
    elif verifier == "mcqa_regex":
        service = "mcqa"
        expected = str(sample.label or "").strip().upper()
        letters = metadata.get("valid_letters") or list("ABCD")
        if len(expected) != 1 or expected not in letters:
            raise ValueError("MCQA requires a reference letter in valid_letters")
        body.update(
            expected_answer=expected,
            options=[{letter: letter} for letter in letters],
            grading_mode="lenient_answer_colon",
        )
        pattern = metadata.get("output_regex")
        if pattern:
            if not isinstance(pattern, str) or re.compile(pattern).groups != 1:
                raise ValueError("NeMo Gym MCQA output_regex must have exactly one capture group")
            body["template_metadata"] = {"output_regex": pattern}
    elif verifier in JUDGE_VERIFIERS:
        service = {
            "nemo_gym_math_judge": "math_with_judge",
            "nemo_gym_equivalence_judge": "equivalence_llm_judge",
            "nemo_gym_rubric_judge": "multichallenge",
        }[verifier]
        if verifier == "nemo_gym_rubric_judge":
            body.update(rubric=deepcopy(metadata["rubric"]), context=metadata["context"])
        else:
            if not isinstance(sample.label, str) or not sample.label:
                raise ValueError("LLM equivalence / math judging requires a nonempty reference answer")
            body["expected_answer"] = sample.label
            if verifier == "nemo_gym_math_judge":
                body["question"] = metadata.get("question") or next(
                    message["content"] for message in reversed(params["input"]) if message["role"] == "user"
                )
    else:
        raise ValueError(f"Unsupported NeMo Gym verifier: {verifier}")
    return service, body
