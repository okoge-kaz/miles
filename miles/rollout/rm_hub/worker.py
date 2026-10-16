import asyncio
from collections import deque
from time import monotonic

import ray

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward, get_gemma_math_reward
from miles.rollout.rm_hub.f1 import f1_score
from miles.rollout.rm_hub.gpqa import compute_gpqa_reward
from miles.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo
from miles.rollout.rm_hub.math_utils import grade_answer_verl

BUILTIN_REWARD_TYPES = frozenset({"deepscaler", "gemma_math", "dapo", "math", "f1", "gpqa", "ifbench"})


def _compute_reward(rm_type, response, label, metadata):
    if rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    if rm_type == "gemma_math":
        return get_gemma_math_reward(response, label)
    if rm_type == "dapo":
        return compute_score_dapo(response, label)
    if rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    if rm_type == "f1":
        return f1_score(response, label)[0]
    if rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    if rm_type == "ifbench":
        # Optional dependency; its import may prepare the official IFBench package.
        from miles.rollout.rm_hub.ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class RewardWorker:
    def ready(self):
        return True

    def score(self, rm_type, response, label, metadata):
        started = monotonic()
        reward = _compute_reward(rm_type, response, label, metadata)
        # GPQA may infer this field from choices. Preserve that existing side effect
        # without returning the entire metadata or mutating a cancelled sample.
        updates = (
            {"correct_answer": metadata["correct_answer"]}
            if rm_type == "gpqa" and metadata and "correct_answer" in metadata
            else {}
        )
        return reward, monotonic() - started, updates


class RewardExecutor:
    """Bound CPU reward RPCs independently of rollout production and buffer reads."""

    def __init__(self, workers):
        self._workers = list(workers)
        self._available = deque(workers)
        self._slots = asyncio.Semaphore(max(1, len(workers)))
        self._pending = set()
        self._closed = False
        self._ifbench_ready = False
        self._ifbench_lock = asyncio.Lock()
        self._counts = [0, 0.0, 0.0, 0.0]

    async def score(self, rm_type, response, label, *, metadata=None):
        if self._closed:
            raise RuntimeError("Reward executor is closed")
        started = monotonic()
        if not self._workers:
            reward = _compute_reward(rm_type, response, label, metadata)
            self._record(0.0, monotonic() - started, monotonic() - started)
            return reward
        await self._slots.acquire()
        if self._closed:
            self._slots.release()
            raise RuntimeError("Reward executor is closed")
        worker = self._available.popleft()
        task = asyncio.create_task(self._complete(worker, rm_type, response, label, metadata, monotonic() - started))
        self._pending.add(task)
        task.add_done_callback(self._finished)
        # A cancelled sample must not release a worker that is still computing.
        # Only the awaiting sample writes its reward; late results are discarded.
        reward, updates = await asyncio.shield(task)
        if metadata is not None:
            for key, value in updates.items():
                metadata.setdefault(key, value)
        return reward

    async def _score_worker(self, worker, rm_type, response, label, metadata):
        if rm_type == "ifbench" and not self._ifbench_ready:
            # Let one worker finish optional dependency setup before other workers
            # import it. This task stays alive if the awaiting sample is cancelled.
            async with self._ifbench_lock:
                if not self._ifbench_ready:
                    result = await worker.score.remote(rm_type, response, label, metadata)
                    self._ifbench_ready = True
                    return result
        return await worker.score.remote(rm_type, response, label, metadata)

    async def _complete(self, worker, rm_type, response, label, metadata, queue_seconds):
        started = monotonic()
        try:
            reward, compute_seconds, updates = await self._score_worker(worker, rm_type, response, label, metadata)
            self._record(queue_seconds, monotonic() - started, compute_seconds)
            return reward, updates
        finally:
            self._available.append(worker)
            self._slots.release()

    def _finished(self, task):
        self._pending.discard(task)
        if not task.cancelled():
            task.exception()  # Retrieve errors even when the sample has been cancelled.

    def _record(self, queue_seconds, rpc_seconds, compute_seconds):
        self._counts[0] += 1
        self._counts[1] += queue_seconds
        self._counts[2] += rpc_seconds
        self._counts[3] += compute_seconds

    def collect_metrics(self):
        count, queue, rpc, compute = self._counts
        self._counts = [0, 0.0, 0.0, 0.0]
        metrics = {"rollout/reward/completed_samples": count}
        if count:
            metrics.update(
                {
                    "perf/reward_queue_time_mean": queue / count,
                    "perf/reward_rpc_time_mean": rpc / count,
                    "perf/reward_compute_time_mean": compute / count,
                }
            )
        return metrics

    async def close(self):
        self._closed = True
        for worker in self._workers:
            ray.kill(worker, no_restart=True)
        await asyncio.gather(*self._pending, return_exceptions=True)


_executor = None


async def initialize_reward_workers(num_workers):
    global _executor
    if _executor is not None:
        raise RuntimeError("Reward workers already initialized in this process")
    if num_workers < 0:
        raise ValueError("rm_num_workers must be nonnegative")
    if num_workers == 0:
        return
    workers = [RewardWorker.remote() for _ in range(num_workers)]
    try:
        await asyncio.gather(*(worker.ready.remote() for worker in workers))
    except BaseException:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        raise
    _executor = RewardExecutor(workers)


async def score_reward(rm_type, response, label, *, metadata=None):
    if _executor is None:
        return _compute_reward(rm_type, response, label, metadata)
    return await _executor.score(rm_type, response, label, metadata=metadata)


def collect_reward_metrics():
    return _executor.collect_metrics() if _executor is not None else {}


async def close_reward_workers():
    global _executor
    executor, _executor = _executor, None
    if executor is not None:
        await executor.close()
