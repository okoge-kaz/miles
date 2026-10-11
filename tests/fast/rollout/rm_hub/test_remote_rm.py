import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web

from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.rm_hub import async_rm
from miles.utils.types import Sample


@asynccontextmanager
async def reward_server(handler):
    app = web.Application()
    app.router.add_post("/score", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield SimpleNamespace(
            custom_rm_path=None,
            rm_type="remote_rm",
            rm_url=f"http://127.0.0.1:{port}/score",
            rm_max_retries=1,
            rm_request_timeout=0.1,
            reward_key=None,
        )
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("status", [400, 429, 503])
async def test_remote_http_failure_is_retried_then_excluded(status, caplog):
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.Response(status=status)

    async with reward_server(handler) as args:
        sample = Sample(prompt="question", response="answer", label="reference", index=3)
        sample.reward = await async_rm(args, sample)
        assert len(requests) == 2
        assert requests[0] == {"prompt": "question", "response": "answer", "label": "reference"}
        assert sample.reward is None
        assert sample.metadata["remote_reward_error"]["http_status"] == status
        filtered = apply_preput_filters(args, None, [sample])
        assert not filtered.keep and filtered.reason == "group_has_missing_reward"
        assert "sample excluded" in caplog.text and "attempts=2" in caplog.text


@pytest.mark.parametrize("failure", ["http", "timeout", "connection"])
async def test_remote_retry_recovers_without_inventing_a_reward(failure):
    requests = 0

    async def handler(request):
        nonlocal requests
        requests += 1
        if requests == 1:
            if failure == "http":
                return web.Response(status=503)
            if failure == "timeout":
                await asyncio.sleep(0.05)
            if failure == "connection":
                request.transport.close()
                return web.Response()
        return web.json_response(0.75)

    async with reward_server(handler) as args:
        args.rm_request_timeout = 0.02 if failure == "timeout" else 0.1
        sample = Sample(response="answer")
        assert await async_rm(args, sample) == 0.75
        assert requests == 2
        assert "remote_reward_error" not in sample.metadata
