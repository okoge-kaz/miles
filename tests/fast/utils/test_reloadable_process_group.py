from unittest.mock import MagicMock

import torch.distributed as dist

from miles.utils.reloadable_process_group import ReloadableProcessGroup


def test_reduce_scatter_single_forwards_to_inner_group(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(ReloadableProcessGroup, "GROUPS", {})
    inner_group = MagicMock()
    expected_work = inner_group.reduce_scatter_single.return_value
    reloadable_group = ReloadableProcessGroup(inner_group, inner_args=(), inner_kwargs={})
    output_tensor = MagicMock()
    input_tensor = MagicMock()
    options = MagicMock()

    actual_work = reloadable_group.reduce_scatter_single(output_tensor, input_tensor, options)

    assert actual_work is expected_work
    inner_group.reduce_scatter_single.assert_called_once_with(output_tensor, input_tensor, options)


def test_all_process_group_collectives_are_overridden():
    lifecycle_methods = {"rank", "size", "name", "abort", "shutdown", "bound_device_id"}
    inherited_collectives = []

    for name in dir(dist.ProcessGroup):
        if name.startswith("__") or name in lifecycle_methods:
            continue
        if not callable(getattr(dist.ProcessGroup, name, None)):
            continue
        if getattr(ReloadableProcessGroup, name) is getattr(dist.ProcessGroup, name):
            inherited_collectives.append(name)

    assert inherited_collectives == []
