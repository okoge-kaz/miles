from unittest.mock import patch

import torch

from miles.utils.tensor_backper import TensorBackuper


def _make_backuper(*, track_transfer_nbytes: bool):
    source = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    backuper = TensorBackuper.create(
        source_getter=lambda: source.items(),
        single_tag=None,
        track_transfer_nbytes=track_transfer_nbytes,
    )
    backuper._backups["actor"]["weight"] = torch.empty_like(source["weight"])
    return backuper


def test_backup_transfer_bytes_are_only_collected_when_enabled():
    with patch("torch.cuda.synchronize"):
        disabled = _make_backuper(track_transfer_nbytes=False)
        disabled.backup("actor")
        enabled = _make_backuper(track_transfer_nbytes=True)
        enabled.backup("actor")

    assert disabled.backup_transfer_nbytes("actor") == 0
    assert enabled.backup_transfer_nbytes("actor") == 6 * torch.float32.itemsize
