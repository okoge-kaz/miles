"""Import compatibility for the pinned pre-LiteLLM-optional Tau environment."""

from __future__ import annotations

import sys
import types
from typing import Any


def install_litellm_import_stub() -> None:
    """Make Tau's human/local-policy paths importable without LiteLLM."""

    try:
        __import__("litellm")
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("litellm")

    class UnavailableError(RuntimeError):
        pass

    def completion(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("LiteLLM is unavailable; use the local-policy Tau user backend")

    module.completion = completion
    module.ServiceUnavailableError = UnavailableError
    module.InternalServerError = UnavailableError
    sys.modules["litellm"] = module
