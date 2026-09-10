"""Check all GPUs and cross-node NCCL; launch with torchrun on each Slurm node."""

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        value = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(value)
        expected = world * (world + 1) / 2
        assert value.item() == expected, (rank, value.item(), expected)
        print(
            f"NCCL PASS host={socket.gethostname()} rank={rank}/{world} "
            f"gpu={torch.cuda.get_device_name(local_rank)} sum={value.item()}",
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
