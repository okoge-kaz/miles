import asyncio

from experiments.src.nemo_gym.bridge.client import request
from experiments.src.nemo_gym.bridge.payloads import AGENT_VERIFIERS, qa_references, verify_payload


async def _score(sample) -> float | None:
    if (sample.metadata or {}).get("verifier") in AGENT_VERIFIERS:
        return (sample.metadata.get("nemo_gym_result") or {}).get("reward")
    service, body = verify_payload(sample)
    record = {}
    reward_key = "answer_em" if service == "hotpotqa_qa" else "reward"
    references = qa_references(sample.label) if service == "hotpotqa_qa" else [None]
    # Native HotpotQA accepts one reference. A confirmed exact match is sufficient;
    # an exhausted request makes the sample unscored, never a negative example.
    for reference in references:
        payload = {**body, "expected_answer": reference} if reference is not None else body
        result = await request(service, "/verify", payload, diagnostics=record, reward_key=reward_key)
        if result is None or result[reward_key] == 1:
            break
    if result is not None:
        detail = result.get("metadata") or {}
        record["scoring_reason"] = (
            "answer_em"
            if service == "hotpotqa_qa"
            else detail.get("scoring_reason")
            or result.get("error_type")
            or ("passed" if result["reward"] == 1 else "verifier_score")
        )
        if service == "competitive_coding_challenges" and result["reward"] == 0:
            tests = (result.get("details") or {}).get("test_case_results") or {}
            outputs = [output for subtask in tests.values() for output in subtask.get("outputs", [])]
            record["scoring_reason"] = (
                "compile_error"
                if any(output.get("compile_success") is False for output in outputs)
                else "test_failure"
            )
    sample.metadata = {**(sample.metadata or {}), "nemo_gym_result": record}
    return None if result is None else float(result[reward_key])


async def reward(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return await asyncio.gather(*(_score(sample) for sample in sample_or_samples))
    return await _score(sample_or_samples)
