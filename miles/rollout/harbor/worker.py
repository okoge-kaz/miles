"""Cancellation-safe lifecycle helpers for Harbor trial worker processes."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

_DEFAULT_CANCEL_GRACE_SECONDS = 30.0


class HarborWorkerProcess(Protocol):
    """Subprocess operations needed to stop one Harbor trial worker."""

    returncode: int | None

    def send_signal(self, sig: signal.Signals) -> None: ...

    def kill(self) -> None: ...

    def wait(self) -> Awaitable[int]: ...


def _cancel_grace_seconds(environ: Mapping[str, str] | None = None) -> float:
    source = os.environ if environ is None else environ
    raw = source.get("HARBOR_WORKER_CANCEL_GRACE_SEC", str(_DEFAULT_CANCEL_GRACE_SECONDS))
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise ValueError(f"HARBOR_WORKER_CANCEL_GRACE_SEC must be positive, got {raw!r}") from exc
    if seconds <= 0:
        raise ValueError(f"HARBOR_WORKER_CANCEL_GRACE_SEC must be positive, got {raw!r}")
    return seconds


async def stop_harbor_trial_worker(
    process: HarborWorkerProcess,
    *,
    grace_seconds: float | None = None,
    wait_for: Callable[[Awaitable[int], float], Awaitable[int]] = asyncio.wait_for,
) -> None:
    """Ask a worker to unwind Harbor cleanup, then kill it after a deadline."""
    if process.returncode is not None:
        return

    timeout = _cancel_grace_seconds() if grace_seconds is None else grace_seconds
    if timeout <= 0:
        raise ValueError(f"grace_seconds must be positive, got {timeout!r}")
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await wait_for(process.wait(), timeout)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        await process.wait()
