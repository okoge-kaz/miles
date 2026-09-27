import asyncio
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DYNAMIC_PORT_START = 20000
_MAX_PORT = 65535


@dataclass
class PortAllocator:
    port_start: int = field(
        default_factory=lambda: int(os.environ.get("MILES_WORKER_PORT_START", _DYNAMIC_PORT_START))
    )
    port_end: int = field(default_factory=lambda: int(os.environ.get("MILES_WORKER_PORT_END", _MAX_PORT)))
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if not 1024 <= self.port_start <= self.port_end <= _MAX_PORT:
            raise ValueError(f"Invalid worker port range [{self.port_start}, {self.port_end}]")

    async def alloc(self, actor, *, node_ip: str, consecutive: int = 1) -> int:
        if not 1 <= consecutive <= self.port_end - self.port_start + 1:
            raise ValueError(f"Port block size {consecutive} does not fit the worker port range")
        async with self._lock:
            # Configure a range outside the node's ip_local_port_range and Ray's
            # worker ports. OCI uses ephemeral ports from 9000, not 32768.
            start_port = self._next_port_of_ip.get(node_ip, self.port_start)
            if start_port + consecutive - 1 > self.port_end:
                start_port = self.port_start
            port: int = await actor._get_free_port_block.remote(
                start_port=start_port,
                count=consecutive,
            )
            if not self.port_start <= port <= self.port_end - consecutive + 1:
                raise RuntimeError(
                    f"Free port block [{port}, {port + consecutive - 1}] on {node_ip} is outside "
                    f"the configured worker range [{self.port_start}, {self.port_end}]"
                )
            self._next_port_of_ip[node_ip] = port + consecutive
            return port
