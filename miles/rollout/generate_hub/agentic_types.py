"""Lightweight result contract for custom agent-environment functions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentFunctionOutput:
    """Metadata returned by an agent function plus an explicit abort signal."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    aborted: bool = False

    @classmethod
    def abort(cls, metadata: Mapping[str, Any] | None = None) -> AgentFunctionOutput:
        """Build an ungraded result that the generate layer must discard."""
        return cls(metadata={} if metadata is None else metadata, aborted=True)
