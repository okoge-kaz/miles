"""
Simple single-turn generation.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)

    # Resume from persisted token ids. Text can be empty for special tokens, so
    # response_length is the authoritative indication that a prefix exists.
    if sample.response_length > 0:
        input_ids = sample.tokens
        persisted_response_tokens = len(sample.tokens) - len(prompt_ids)
        if persisted_response_tokens != sample.response_length:
            raise RuntimeError(
                "Cannot resume single-turn generation from an inconsistent token prefix: "
                f"response_length={sample.response_length}, persisted_response_tokens={persisted_response_tokens}"
            )
        sampling_params["max_new_tokens"] -= persisted_response_tokens

        assert sampling_params["max_new_tokens"] >= 0
        if sampling_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    payload, halt_status = compute_request_payload(
        args, input_ids=input_ids, sampling_params=sampling_params, multimodal_inputs=sample.multimodal_inputs
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    output = await post(url, payload, headers=compute_routing_headers(args, sample))
    await update_sample_from_response(args, sample, payload=payload, output=output)

    return GenerateFnOutput(samples=sample)
