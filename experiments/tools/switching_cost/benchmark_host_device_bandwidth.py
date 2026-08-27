import argparse
import json
import os
import socket
import time
from collections.abc import Callable

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure concurrent pinned-host transfers on every visible GPU in one node."
    )
    parser.add_argument("--bytes-per-gpu", type=int, default=1024**3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if args.bytes_per_gpu <= 0:
        parser.error("--bytes-per-gpu must be positive")
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups non-negative")
    return args


def _synchronize_all(num_gpus: int) -> None:
    for device_index in range(num_gpus):
        torch.cuda.synchronize(device_index)


def _measure_direction(
    copy_once: Callable[[], None],
    *,
    num_gpus: int,
    bytes_per_gpu: int,
    repeats: int,
    warmups: int,
) -> dict[str, float]:
    for _ in range(warmups):
        copy_once()
        _synchronize_all(num_gpus)

    start = time.perf_counter()
    for _ in range(repeats):
        copy_once()
        _synchronize_all(num_gpus)
    elapsed = time.perf_counter() - start

    aggregate_bytes = bytes_per_gpu * num_gpus * repeats
    return {
        "elapsed_s": elapsed,
        "aggregate_gib_per_s": aggregate_bytes / elapsed / 1024**3,
        "per_gpu_gib_per_s": aggregate_bytes / num_gpus / elapsed / 1024**3,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    num_gpus = torch.cuda.device_count()
    host_buffers = [torch.empty(args.bytes_per_gpu, dtype=torch.uint8, pin_memory=True) for _ in range(num_gpus)]
    device_buffers = [
        torch.empty(args.bytes_per_gpu, dtype=torch.uint8, device=f"cuda:{device_index}")
        for device_index in range(num_gpus)
    ]
    streams = [torch.cuda.Stream(device_index) for device_index in range(num_gpus)]

    def copy_host_to_device() -> None:
        for device_index in range(num_gpus):
            with torch.cuda.device(device_index), torch.cuda.stream(streams[device_index]):
                device_buffers[device_index].copy_(host_buffers[device_index], non_blocking=True)

    def copy_device_to_host() -> None:
        for device_index in range(num_gpus):
            with torch.cuda.device(device_index), torch.cuda.stream(streams[device_index]):
                host_buffers[device_index].copy_(device_buffers[device_index], non_blocking=True)

    result = {
        "event": "host_device_bandwidth",
        "hostname": socket.gethostname(),
        "slurm_node_id": os.environ.get("SLURM_NODEID"),
        "num_gpus": num_gpus,
        "gpu_names": [torch.cuda.get_device_name(device_index) for device_index in range(num_gpus)],
        "bytes_per_gpu": args.bytes_per_gpu,
        "repeats": args.repeats,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host_to_device": _measure_direction(
            copy_host_to_device,
            num_gpus=num_gpus,
            bytes_per_gpu=args.bytes_per_gpu,
            repeats=args.repeats,
            warmups=args.warmups,
        ),
        "device_to_host": _measure_direction(
            copy_device_to_host,
            num_gpus=num_gpus,
            bytes_per_gpu=args.bytes_per_gpu,
            repeats=args.repeats,
            warmups=args.warmups,
        ),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
