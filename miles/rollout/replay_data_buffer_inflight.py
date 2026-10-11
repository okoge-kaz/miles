import asyncio
from collections import deque

from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.http_utils import post
from miles.utils.types import Sample


def validate_inflight_group(group: list[Sample]) -> None:
    for sample in group:
        if (
            not isinstance(sample, Sample)
            or sample.generate_function_path not in (None, "miles.rollout.generate_hub.single_turn.generate")
            or sample.multimodal_inputs
        ):
            raise ValueError("Inflight ReplayDataBuffer requires standard single-turn text generation")
        if sample.response_length and (
            len(sample.tokens) < sample.response_length
            or sample.rollout_log_probs is None
            or len(sample.rollout_log_probs) != sample.response_length
            or (sample.loss_mask is not None and len(sample.loss_mask) != sample.response_length)
        ):
            raise ValueError("Inflight prefix requires aligned tokens, rollout logprobs, and loss masks")


def needs_continuation(entry: DataBufferInput, *, group_rm: bool) -> bool:
    return any(
        sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or (group_rm and sample.reward is None)
        for sample in entry.group
    )


async def _abort_requests(args) -> None:
    # Router discovery imports generation runtime dependencies; pure checkpoint
    # readers do not need them. No per-sample policy-version requests are made.
    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls

    urls = await get_worker_urls(args)
    await asyncio.gather(*(post(f"{url}/abort_request", {"abort_all": True}) for url in urls))


class InflightReplay:
    def __init__(self, owner):
        self._owner = owner
        self.pending: deque[DataBufferInput] = deque()
        self._restart_worker = False

    async def capture(self) -> None:
        owner = self._owner
        worker = owner._worker
        self._restart_worker = worker is not None and not worker.done()
        if worker is not None:
            if worker.done():
                worker.result()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            owner._worker = None

        # Cancelling the scheduler wait does not cancel its generation tasks.
        # Abort returns their measured partial outputs instead of losing tokens.
        owner.state.aborted = True
        unfinished = [task for task in owner._active if not task.done()]
        if unfinished:
            await _abort_requests(owner.args)
            results = await asyncio.gather(*unfinished, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result

        for task, group in list(owner._active.items()):
            if task.cancelled():
                raise RuntimeError("Inflight generation was cancelled without returning its token prefix")
            entry = owner._collect_group_result(task, group)
            validate_inflight_group(entry.group)
            if needs_continuation(entry, group_rm=owner.args.group_rm):
                self.pending.append(entry)
            elif task in owner._pending_ready or owner._output._preput_filter(entry):
                # A blocked put has already run the reward/dynamic filters.
                # Permit the finite active tail to exceed capacity at this cut.
                await owner._output.restore_entry(entry, wait_for_capacity=False)
            owner._active.pop(task)
            owner._pending_ready.pop(task, None)
            owner._original_prompts.pop(task, None)
        owner._scheduler = make_submission_scheduler(owner.args, default="sample")

    def resume(self) -> None:
        self._owner.state.reset()
        if self._restart_worker and self._owner._worker is None:
            self._owner._worker = asyncio.create_task(self._owner._worker_loop())
        self._restart_worker = False
