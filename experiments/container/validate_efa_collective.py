#!/usr/bin/env python3
"""Measure a multi-node NCCL all-reduce launched by Slurm."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _mapped_network_libraries() -> list[str]:
    mapped_paths: set[str] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[-1]
        if "libnccl-net" in path or "libfabric.so" in path:
            mapped_paths.add(path)
    return sorted(mapped_paths)


def _transport_evidence(rank: int) -> dict[str, object]:
    mapped_libraries = _mapped_network_libraries()
    ofi_plugin_mapped = any(path.endswith("/libnccl-net-ofi.so") for path in mapped_libraries)
    aws_libfabric_mapped = any(
        path.startswith("/opt/amazon/efa/lib/") and "/libfabric.so" in path
        for path in mapped_libraries
    )
    transport = os.environ.get("MILES_NCCL_TRANSPORT")
    nccl_net = os.environ.get("NCCL_NET")
    nccl_net_plugin = os.environ.get("NCCL_NET_PLUGIN")
    nccl_tuner_plugin = os.environ.get("NCCL_TUNER_PLUGIN")
    nccl_ib_disable = os.environ.get("NCCL_IB_DISABLE")
    fi_provider = os.environ.get("FI_PROVIDER")

    if transport == "efa":
        valid = all(
            (
                nccl_net == "AWS Libfabric",
                nccl_net_plugin == "ofi",
                nccl_tuner_plugin == "ofi",
                nccl_ib_disable == "0",
                fi_provider == "efa",
                ofi_plugin_mapped,
                aws_libfabric_mapped,
            )
        )
    elif transport == "tcp":
        valid = all(
            (
                nccl_net == "Socket",
                nccl_net_plugin in (None, ""),
                nccl_tuner_plugin in (None, ""),
                nccl_ib_disable == "1",
                fi_provider in (None, ""),
                not ofi_plugin_mapped,
            )
        )
    else:
        valid = False

    return {
        "rank": rank,
        "host": os.uname().nodename,
        "transport": transport,
        "nccl_net": nccl_net,
        "nccl_net_plugin": nccl_net_plugin,
        "nccl_tuner_plugin": nccl_tuner_plugin,
        "nccl_ib_disable": nccl_ib_disable,
        "fi_provider": fi_provider,
        "ofi_plugin_mapped": ofi_plugin_mapped,
        "aws_libfabric_mapped": aws_libfabric_mapped,
        "mapped_network_libraries": mapped_libraries,
        "valid": valid,
    }


def _validate_transport(rank: int, world_size: int) -> None:
    evidence: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(evidence, _transport_evidence(rank))
    gathered_evidence = [item for item in evidence if item is not None]
    if rank == 0:
        for item in gathered_evidence:
            print(f"NCCL transport evidence {json.dumps(item, sort_keys=True)}", flush=True)

    if len(gathered_evidence) != world_size or not all(item["valid"] for item in gathered_evidence):
        raise RuntimeError(f"NCCL transport validation failed: {gathered_evidence}")


def main() -> None:
    rank = int(os.environ.get("RANK", os.environ["SLURM_PROCID"]))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["SLURM_LOCALID"]))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ["SLURM_NTASKS"]))

    visible_devices = torch.cuda.device_count()
    if visible_devices == 0:
        raise RuntimeError("no CUDA device is visible")
    device_index = 0 if visible_devices == 1 else local_rank
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    element_count = int(os.environ.get("EFA_TEST_ELEMENTS", str(64 * 1024 * 1024)))
    warmup_iterations = int(os.environ.get("EFA_TEST_WARMUP", "5"))
    measured_iterations = int(os.environ.get("EFA_TEST_ITERATIONS", "20"))
    tensor = torch.ones(element_count, dtype=torch.float32, device=device)

    for _ in range(warmup_iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    _validate_transport(rank, world_size)

    durations: list[float] = []
    for _ in range(measured_iterations):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)

    all_durations: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(all_durations, durations)
    if rank == 0:
        flattened = [duration for values in all_durations if values for duration in values]
        mean_seconds = statistics.fmean(flattened)
        size_bytes = tensor.numel() * tensor.element_size()
        algorithm_gbps = size_bytes / mean_seconds / 1e9
        bus_gbps = algorithm_gbps * 2 * (world_size - 1) / world_size
        print(
            "NCCL all-reduce "
            f"transport={os.environ.get('MILES_NCCL_TRANSPORT', 'unknown')} "
            f"ranks={world_size} bytes={size_bytes} iterations={measured_iterations} "
            f"mean_ms={mean_seconds * 1e3:.3f} "
            f"algorithm_GBps={algorithm_gbps:.3f} bus_GBps={bus_gbps:.3f}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
