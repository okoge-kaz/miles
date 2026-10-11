import logging
from copy import deepcopy

import aiohttp

from experiments.src.nemo_gym.bridge.client import endpoints, request
from experiments.src.nemo_gym.bridge.payloads import AGENT_VERIFIERS

logger = logging.getLogger(__name__)


async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
    if metadata.get("verifier") not in AGENT_VERIFIERS:
        raise ValueError("NeMo Gym agent needs converted native NeMo Gym rows")
    native = metadata["nemo_gym"]
    service = native["service"]
    if service not in {"tau2", "workplace"}:
        raise ValueError(f"Unsupported NeMo Gym agent service: {service}")
    body = deepcopy(native["request"])
    bridge = endpoints()["policy_bridge"].rstrip("/")
    timeout = aiohttp.ClientTimeout(total=30)
    diagnostics = {"service": service, "status": "unscored", "reward": None}
    rollout_id = None
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        try:
            async with session.post(
                bridge + "/sessions",
                json={"base_url": base_url, "sampling_params": request_kwargs},
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                rollout_id = (await response.json())["rollout_id"]
            body["_ng_rollout_id"] = rollout_id
            # Retry the entire episode in generate(), with a fresh Miles session.
            result = await request(service, "/run", body, attempts=1, diagnostics=diagnostics)
            diagnostics["rollout_id"] = rollout_id
            if result is not None:
                diagnostics["termination_reason"] = (result.get("result") or {}).get("termination_reason")
        except (aiohttp.ClientError, TimeoutError):
            diagnostics.update(status="unscored", reward=None, failure_kind="policy_bridge_unavailable")
        finally:
            if rollout_id is not None:
                try:
                    async with session.delete(bridge + "/sessions/" + rollout_id, allow_redirects=False) as response:
                        response.raise_for_status()
                except (aiohttp.ClientError, TimeoutError):
                    logger.warning("NeMo Gym session cleanup failed: rollout_id=%s", rollout_id)
    return {"nemo_gym_result": diagnostics}
