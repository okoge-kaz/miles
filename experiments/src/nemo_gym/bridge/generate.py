import argparse
import asyncio
import logging

from experiments.src.nemo_gym.bridge.client import EVENTS
from experiments.src.nemo_gym.bridge.payloads import AGENT_VERIFIERS
from miles.rollout.agentic.agent_function import AGENT_FUNCTION_MODES

logger = logging.getLogger(__name__)


async def generate(input):
    # SGLang/session dependencies are needed only in the training worker.
    from miles.rollout.generate_hub.agentic_tool_call import generate as generate_agent
    from miles.rollout.generate_hub.single_turn import generate as generate_single_turn
    from miles.utils.types import Sample

    metadata = input.sample.metadata or {}
    if metadata.get("verifier") not in AGENT_VERIFIERS:
        return await generate_single_turn(input)
    for attempt in range(3):
        result = await generate_agent(input)
        samples = result.samples if isinstance(result.samples, list) else [result.samples]
        # The agent return is copied to every turn. Consume it once per episode,
        # including discarded attempts, and keep it out of replayed sample metadata.
        episode_events = next(
            (s.metadata["nemo_gym_events"] for s in samples if "nemo_gym_events" in (s.metadata or {})), {}
        )
        EVENTS.update(episode_events)
        for sample in samples:
            if sample.metadata:
                sample.metadata.pop("nemo_gym_events", None)
        if samples and all(
            sample.status not in (Sample.Status.ABORTED, Sample.Status.FAILED)
            and (sample.metadata.get("nemo_gym_result") or {}).get("status") == "scored"
            for sample in samples
        ):
            return result
        logger.warning(
            "NeMo Gym episode failed: sample=%s attempt=%d/3; discarding incomplete episode",
            input.sample.index,
            attempt + 1,
        )
        if attempt < 2:
            await asyncio.sleep(2**attempt)
    for sample in samples:
        sample.reward = None
        sample.status = Sample.Status.FAILED
        record = {
            **((sample.metadata or {}).get("nemo_gym_result") or {}),
            "status": "unscored",
            "reward": None,
            "episode_attempts": 3,
        }
        record.setdefault("failure_kind", "episode_incomplete")
        sample.metadata = {**(sample.metadata or {}), "nemo_gym_result": record}
    return result


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", default="experiments.src.nemo_gym.bridge.agent.run")
    parser.add_argument("--custom-agent-function-mode", choices=AGENT_FUNCTION_MODES, default="subproc")
    parser.add_argument("--max-seq-len", type=int, default=None)


generate.add_arguments = add_arguments
