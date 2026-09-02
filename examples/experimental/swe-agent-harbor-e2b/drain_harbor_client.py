#!/usr/bin/env python3
"""Cancel the active Harbor trials owned by one PBS/Ray client.

All configuration, including the HMAC master, comes from the process
environment. The secret is never accepted on argv or printed.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from miles.rollout.harbor.auth import derive_harbor_drain_bearer

_MAX_RESPONSE_BYTES = 64 * 1024
_REQUEST_TIMEOUT_SECONDS = 30


def _service_url() -> str:
    value = os.environ.get("AGENT_SERVER_URL", "")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("AGENT_SERVER_URL must be a credential-free HTTP(S) origin")
    return value.rstrip("/")


def drain() -> int:
    """Drain one client inventory and return the server-reported count."""

    client_id = os.environ.get("HARBOR_CLIENT_ID", "")
    secret = os.environ.get("HARBOR_RUN_SECRET", "")
    bearer = derive_harbor_drain_bearer(secret, client_id=client_id)
    payload = json.dumps(
        {"client_id": client_id},
        separators=(",", ":"),
    ).encode()
    request = Request(
        f"{_service_url()}/drain",
        data=payload,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        encoded = response.read(_MAX_RESPONSE_BYTES + 1)
        if response.status != 200 or len(encoded) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Harbor drain returned an invalid response")
    result = json.loads(encoded)
    cancelled = result.get("cancelled") if isinstance(result, dict) else None
    if isinstance(cancelled, bool) or not isinstance(cancelled, int) or cancelled < 0:
        raise RuntimeError("Harbor drain response has no valid cancelled count")
    return cancelled


def main() -> None:
    try:
        cancelled = drain()
    except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Harbor client drain failed: {type(exc).__name__}") from exc
    print(f"Harbor client drain completed: cancelled={cancelled}")


if __name__ == "__main__":
    main()
