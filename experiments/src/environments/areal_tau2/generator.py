"""Generate AReaL Tau2 rollouts with an official user simulator and terminal reward."""

from __future__ import annotations

from experiments.src.environments.areal_tau2.runtime import AReaLTau2Session
from experiments.src.environments.tau_bench.generator import _add_arguments, _generate_tau
from experiments.src.protocols.areal_tau2 import AREAL_TAU2_VERIFIER
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Run one isolated AReaL task/DB episode through the shared Tau policy loop."""

    metadata = input.sample.metadata if isinstance(input.sample.metadata, dict) else {}
    if metadata.get("verifier") != AREAL_TAU2_VERIFIER:
        raise ValueError(f"AReaL Tau2 generator rejects verifier {metadata.get('verifier')!r}")
    return await _generate_tau(input, session_factory=AReaLTau2Session)


generate.add_arguments = _add_arguments
generate.supports_inflight_replay = True

__all__ = ["generate"]
