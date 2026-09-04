"""Validate process-environment-only Harbor secret inheritance through Ray."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
from pathlib import Path

import ray


def _secret_digest() -> str:
    secret = os.environ.get("HARBOR_RUN_SECRET")
    if secret is None or not 32 <= len(secret) <= 4096:
        raise RuntimeError("Ray worker lacks a valid Harbor run secret")
    return hashlib.sha256(secret.encode()).hexdigest()


def _client_id() -> str:
    client_id = os.environ.get("HARBOR_CLIENT_ID")
    if (
        client_id is None
        or not 1 <= len(client_id) <= 128
        or not client_id[0].isalnum()
        or any(
            not (character.isalnum() or character in "._-")
            for character in client_id
        )
    ):
        raise RuntimeError("Ray worker lacks a valid Harbor client ID")
    return client_id


@ray.remote
def _remote_contract() -> tuple[str, str]:
    return _secret_digest(), _client_id()


def _assert_secret_absent_from_process_argv() -> None:
    encoded = os.environ["HARBOR_RUN_SECRET"].encode()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            payload = cmdline.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if encoded in payload:
            raise RuntimeError("Harbor run secret appeared in a process command line")


def _assert_dashboard_is_loopback_only(port: int = 8265) -> None:
    listeners: list[str] = []
    for table, family in (
        (Path("/proc/net/tcp"), socket.AF_INET),
        (Path("/proc/net/tcp6"), socket.AF_INET6),
    ):
        if not table.is_file():
            continue
        for line in table.read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            local_address, local_port = fields[1].split(":")
            if fields[3] != "0A" or int(local_port, 16) != port:
                continue
            if family == socket.AF_INET:
                packed = bytes.fromhex(local_address)[::-1]
            else:
                packed = b"".join(
                    bytes.fromhex(local_address[offset : offset + 8])[::-1]
                    for offset in range(0, 32, 8)
                )
            listeners.append(socket.inet_ntop(family, packed))
    if not listeners:
        raise RuntimeError("Ray dashboard did not open its expected listener")
    if not all(ipaddress.ip_address(address).is_loopback for address in listeners):
        raise RuntimeError("Ray dashboard opened a non-loopback listener")


def main() -> None:
    ray.init(address="auto")
    local_digest = _secret_digest()
    local_client_id = _client_id()
    remote_digest, remote_client_id = ray.get(_remote_contract.remote())
    if (local_digest, local_client_id) != (remote_digest, remote_client_id):
        raise RuntimeError("Ray worker did not inherit the Harbor client contract")
    _assert_dashboard_is_loopback_only()
    _assert_secret_absent_from_process_argv()
    print("ray-secret-inheritance-ok")


if __name__ == "__main__":
    main()
