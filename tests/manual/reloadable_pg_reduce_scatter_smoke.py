from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.tensor_parallel import mappings

import miles.utils.reloadable_process_group as reloadable_pg_module
from miles.utils.reloadable_process_group import (
    ReloadableProcessGroup,
    destroy_process_groups,
    monkey_patch_torch_dist,
    reload_process_groups,
)


def _assert_runtime() -> None:
    assert torch.__version__ == "2.13.0+cu130", torch.__version__
    mapping_path = Path(inspect.getfile(mappings)).resolve()
    miles_path = Path(inspect.getfile(reloadable_pg_module)).resolve()
    assert mapping_path.is_relative_to(Path("/root/Megatron-LM")), mapping_path
    assert miles_path.is_relative_to(Path("/root/miles")), miles_path
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    for name in (
        "NCCL_NET",
        "NCCL_NET_PLUGIN",
        "NCCL_TUNER_PLUGIN",
        "FI_PROVIDER",
    ):
        assert name not in os.environ, (name, os.environ.get(name))

    expected_cpu_count = int(os.environ.get("RPG_EXPECTED_CPU_COUNT", "192"))
    affinity = os.sched_getaffinity(0)
    assert len(affinity) == expected_cpu_count, (len(affinity), sorted(affinity))


def _run_reduce_scatter(group: ReloadableProcessGroup, phase: str) -> None:
    rank = dist.get_rank()
    input_tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    input_tensor += rank * 100
    actual = mappings._reduce_scatter_along_first_dim(input_tensor, group)
    expected_values = (
        [100.0, 102.0, 104.0, 106.0]
        if rank == 0
        else [108.0, 110.0, 112.0, 114.0]
    )
    expected = torch.tensor(expected_values, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        f"RPG_REDUCE_SCATTER_PHASE_OK phase={phase} rank={rank} "
        f"values={actual.tolist()}",
        flush=True,
    )


def main() -> None:
    _assert_runtime()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    captured_reduce_scatter = mappings.dist_reduce_scatter_func
    monkey_patch_torch_dist()
    assert mappings.dist_reduce_scatter_func is captured_reduce_scatter

    group = dist.new_group(ranks=[0, 1], backend="nccl")
    assert isinstance(group, ReloadableProcessGroup), type(group)
    for method_name in ("reduce_scatter_single", "reduce_scatter_single_coalesced"):
        assert method_name in ReloadableProcessGroup.__dict__, method_name
        assert getattr(ReloadableProcessGroup, method_name) is not getattr(
            dist.ProcessGroup, method_name
        )

    _run_reduce_scatter(group, "initial")
    dist.barrier()
    destroy_process_groups()
    dist.barrier()
    reload_process_groups()
    dist.barrier()
    _run_reduce_scatter(group, "reloaded")
    dist.barrier()

    destroy_process_groups()
    dist.destroy_process_group()
    if int(os.environ["RANK"]) == 0:
        print(
            "RPG_MEGATRON_SEQUENCE_PARALLEL_SMOKE_OK "
            f"node_rank={os.environ.get('MILES_NODE_RANK', 'unknown')} "
            "phases=initial,reloaded",
            flush=True,
        )


if __name__ == "__main__":
    main()
