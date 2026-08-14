import asyncio
import logging
import time
import uuid
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from miles.rollout.base_types import (
    GenerateFnInput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.generate_hub.single_turn import generate
from miles.rollout.generate_utils.generate_endpoint_utils import policy_uses_routing_key
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.recycle_compute_metrics import (
    GROUP_GENERATION_COMPLETE_TIME_KEY,
    GROUP_GENERATION_COMPLETE_VERSION_KEY,
    LIFECYCLE_EXACT_KEY,
    SAMPLE_GENERATION_COMPLETE_TIME_KEY,
    SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
    TRAJECTORY_START_TIME_KEY,
    TRAJECTORY_START_VERSION_KEY,
    add_apportioned_reward_seconds,
    stamp_sample_lifecycle_boundary,
)
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _flat_generation_output(output: Sample | list[Sample]) -> list[Sample]:
    return output if isinstance(output, list) else [output]


def _finish_generation_lifecycle(
    output: Sample | list[Sample],
    *,
    start_version: int | None,
    start_time: float | None,
    lifecycle_version_provider: Callable[[], int] | None,
) -> None:
    if lifecycle_version_provider is None or start_version is None or start_time is None:
        return
    samples = _flat_generation_output(output)
    for generated_sample in samples:
        stamp_sample_lifecycle_boundary(
            [generated_sample],
            version_key=TRAJECTORY_START_VERSION_KEY,
            version=start_version,
            time_key=TRAJECTORY_START_TIME_KEY,
            wall_time=start_time,
        )
    stamp_sample_lifecycle_boundary(
        samples,
        version_key=SAMPLE_GENERATION_COMPLETE_VERSION_KEY,
        version=lifecycle_version_provider(),
        time_key=SAMPLE_GENERATION_COMPLETE_TIME_KEY,
        wall_time=time.time(),
    )


class GenerateState:
    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.generate_fn_semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = compute_sampling_params(
            args,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
        )

        self.generate_function = load_generate_function(args.custom_generate_function_path) or generate

        self.reset()

    def reset(self) -> None:
        self.aborted = False


async def generate_and_rm(
    state: GenerateState,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    *,
    on_generation_complete: Callable[[Sample | list[Sample]], None] | None = None,
    lifecycle_version_provider: Callable[[], int] | None = None,
) -> Sample | list[Sample]:
    args = state.args

    def finish_generation(output: Sample | list[Sample], start_version: int | None, start_time: float | None) -> None:
        _finish_generation_lifecycle(
            output,
            start_version=start_version,
            start_time=start_time,
            lifecycle_version_provider=lifecycle_version_provider,
        )
        if on_generation_complete is not None:
            on_generation_complete(output)

    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        start_version = lifecycle_version_provider() if lifecycle_version_provider is not None else None
        start_time = time.time() if lifecycle_version_provider is not None else None
        finish_generation(sample, start_version, start_time)
        return sample

    # generate
    log_prefix = f"[sample={getattr(sample, 'index', '?')}]"
    logger.debug(f"{log_prefix} Waiting for semaphore...")
    async with state.generate_fn_semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            start_version = lifecycle_version_provider() if lifecycle_version_provider is not None else None
            start_time = time.time() if lifecycle_version_provider is not None else None
            finish_generation(sample, start_version, start_time)
            return sample

        logger.debug(f"{log_prefix} Acquired semaphore, calling generate_function")
        start_version = lifecycle_version_provider() if lifecycle_version_provider is not None else None
        start_time = time.time() if lifecycle_version_provider is not None else None
        output = await state.generate_function(
            GenerateFnInput(
                state=state,
                sample=sample,
                sampling_params=deepcopy(sampling_params),
                evaluation=evaluation,
            )
        )
        sample = output.samples
        logger.debug(f"{log_prefix} generate_function returned")

    finish_generation(sample, start_version, start_time)

    # TODO change to `if not args.group_rm: do reward model` for more clarity after the refactor below
    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # TODO: unify the two branches into one if we decide to use list as output type
    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        if samples_need_reward:
            reward_start = time.monotonic()
            await batched_async_rm(args, samples_need_reward, inplace_set_reward_field=True)
            add_apportioned_reward_seconds(samples_need_reward, time.monotonic() - reward_start)
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            reward_start = time.monotonic()
            sample.reward = await async_rm(args, sample)
            add_apportioned_reward_seconds([sample], time.monotonic() - reward_start)

    logger.debug(f"{log_prefix} generate_and_rm complete")
    return sample


async def generate_and_rm_group(
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    *,
    lifecycle_version_provider: Callable[[], int] | None = None,
    on_group_generation_complete: Callable[[list[Sample]], None] | None = None,
) -> list[Sample]:
    args = state.args

    if state.aborted:
        return group

    remaining_generation_tasks = len(group)
    group_completion_version: int | None = None
    group_completion_time: float | None = None
    completed_generation_samples: list[Sample] = []

    def record_generation_completion(output: Sample | list[Sample]) -> None:
        nonlocal remaining_generation_tasks, group_completion_version, group_completion_time
        completed_generation_samples.extend(_flat_generation_output(output))
        remaining_generation_tasks -= 1
        if remaining_generation_tasks == 0:
            if lifecycle_version_provider is not None:
                group_completion_version = lifecycle_version_provider()
                group_completion_time = time.time()
            if on_group_generation_complete is not None:
                on_group_generation_complete(list(completed_generation_samples))

    if policy_uses_routing_key(args):
        for sample in group:
            if sample.routing_key is None:
                sample.routing_key = str(uuid.uuid4())

    log_prefix = f"[group indices={[getattr(s, 'index', '?') for s in group]}]"
    logger.debug(f"{log_prefix} Starting group with {len(group)} samples")
    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = args.rollout_seed + idx
        tasks.append(
            asyncio.create_task(
                generate_and_rm(
                    state,
                    sample,
                    current_sampling_params,
                    evaluation=evaluation,
                    on_generation_complete=record_generation_completion,
                    lifecycle_version_provider=lifecycle_version_provider,
                )
            )
        )

    group = await asyncio.gather(*tasks)
    logger.debug(f"{log_prefix} [group] All {len(group)} samples completed")
    if lifecycle_version_provider is not None:
        exact = group_completion_version is not None and group_completion_time is not None
        if not exact:
            group_completion_version = lifecycle_version_provider()
            group_completion_time = time.time()
        flat_group = [sample for item in group for sample in (item if isinstance(item, list) else [item])]
        stamp_sample_lifecycle_boundary(
            flat_group,
            version_key=GROUP_GENERATION_COMPLETE_VERSION_KEY,
            version=group_completion_version,
            time_key=GROUP_GENERATION_COMPLETE_TIME_KEY,
            wall_time=group_completion_time,
        )
        for sample in flat_group:
            sample.metadata[LIFECYCLE_EXACT_KEY] = exact
    if state.aborted:
        return group

    if args.group_rm:
        reward_start = time.monotonic()
        await batched_async_rm(args, group, inplace_set_reward_field=True)
        flat_group = [sample for item in group for sample in (item if isinstance(item, list) else [item])]
        add_apportioned_reward_seconds(flat_group, time.monotonic() - reward_start)

    return group


def compute_sampling_params(
    args,
    *,
    # after unifying configuration, this can be further refactored
    temperature,
    top_p,
    top_k,
    max_new_tokens,
):
    return dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )


class InferenceRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self.eval_prompt_dataset_cache = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        from miles.rollout.inference_rollout.inference_rollout_train import generate_rollout_async

        output, aborted_samples = await generate_rollout_async(
            self.state, input.rollout_id, self.data_source.get_samples
        )
        self.data_source.add_samples(aborted_samples)
        return output

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        from miles.rollout.inference_rollout.inference_rollout_eval import eval_rollout_single_dataset

        assert not self.state.args.group_rm, "Group RM is not supported for eval rollout"

        coros = []
        for dataset_cfg in getattr(self.state.args, "eval_datasets", []) or []:
            coros.append(eval_rollout_single_dataset(self.state, dataset_cfg, self.eval_prompt_dataset_cache))
        results_list = await asyncio.gather(*coros)
        results = {k: v for r in results_list for k, v in r.items()}
        return RolloutFnEvalOutput(data=results)
