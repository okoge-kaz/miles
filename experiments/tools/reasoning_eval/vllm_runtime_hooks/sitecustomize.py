"""Keep the vLLM compile cache stable when only its startup timeout changes."""

from __future__ import annotations

import vllm.envs


_DEFAULT_ENGINE_READY_TIMEOUT_SECONDS = 600
_original_compile_factors = vllm.envs.compile_factors


def _compile_factors_with_stable_startup_timeout() -> dict[str, object]:
    """Normalize the operational startup deadline in compiled-graph identity."""
    factors = _original_compile_factors()
    if "VLLM_ENGINE_READY_TIMEOUT_S" in factors:
        factors["VLLM_ENGINE_READY_TIMEOUT_S"] = _DEFAULT_ENGINE_READY_TIMEOUT_SECONDS
    return factors


vllm.envs.compile_factors = _compile_factors_with_stable_startup_timeout
