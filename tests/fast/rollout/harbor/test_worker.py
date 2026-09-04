from __future__ import annotations

import signal
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.rollout.harbor.worker import stop_harbor_trial_worker


@pytest.mark.asyncio
async def test_worker_gets_graceful_interrupt_before_kill() -> None:
    process = MagicMock(returncode=None)
    process.wait = AsyncMock(return_value=0)

    await stop_harbor_trial_worker(process, grace_seconds=5)

    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.wait.assert_awaited_once_with()
    process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_worker_is_killed_only_after_cleanup_grace_expires() -> None:
    process = MagicMock(returncode=None)
    process.wait = AsyncMock(return_value=0)
    waits = 0

    async def timeout_once(awaitable, timeout: float) -> int:
        nonlocal waits
        waits += 1
        assert timeout == 5
        awaitable.close()
        raise TimeoutError

    await stop_harbor_trial_worker(
        process,
        grace_seconds=5,
        wait_for=timeout_once,
    )

    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.kill.assert_called_once_with()
    process.wait.assert_awaited_once_with()
    assert waits == 1


@pytest.mark.asyncio
async def test_finished_worker_is_unchanged() -> None:
    process = MagicMock(returncode=0)
    process.wait = AsyncMock()

    await stop_harbor_trial_worker(process, grace_seconds=5)

    process.send_signal.assert_not_called()
    process.kill.assert_not_called()
    process.wait.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_exit_race_is_reaped() -> None:
    process = MagicMock(returncode=None)
    process.send_signal.side_effect = ProcessLookupError
    process.wait = AsyncMock(return_value=0)

    await stop_harbor_trial_worker(process, grace_seconds=5)

    process.kill.assert_not_called()
    process.wait.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_worker_exit_race_after_grace_timeout_is_reaped() -> None:
    process = MagicMock(returncode=None)
    process.kill.side_effect = ProcessLookupError
    process.wait = AsyncMock(return_value=0)

    async def timeout(awaitable, timeout: float) -> int:
        assert timeout == 5
        awaitable.close()
        raise TimeoutError

    await stop_harbor_trial_worker(
        process,
        grace_seconds=5,
        wait_for=timeout,
    )

    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.kill.assert_called_once_with()
    process.wait.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_invalid_grace_is_rejected_without_signalling() -> None:
    process = MagicMock(returncode=None)
    process.wait = AsyncMock()

    with pytest.raises(ValueError, match="grace_seconds"):
        await stop_harbor_trial_worker(process, grace_seconds=0)

    process.send_signal.assert_not_called()
    process.wait.assert_not_awaited()
