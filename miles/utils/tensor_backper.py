from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, single_tag, *, track_transfer_nbytes: bool = False):
        if single_tag is None:
            return _TensorBackuperNormal(
                source_getter=source_getter,
                track_transfer_nbytes=track_transfer_nbytes,
            )
        else:
            return _TensorBackuperNoop(
                source_getter=source_getter,
                single_tag=single_tag,
                track_transfer_nbytes=track_transfer_nbytes,
            )

    def __init__(self, source_getter: _SourceGetter, *, track_transfer_nbytes: bool):
        self._source_getter = source_getter
        self._track_transfer_nbytes = track_transfer_nbytes

    @property
    @abstractmethod
    def backup_tags(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup_transfer_nbytes(self, tag: str) -> int:
        """Return bytes copied from the source tensors by one ``backup`` call."""
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter, *, track_transfer_nbytes: bool):
        super().__init__(source_getter=source_getter, track_transfer_nbytes=track_transfer_nbytes)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self._backup_transfer_nbytes: dict[str, int] = {}

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        should_measure = self._track_transfer_nbytes and tag not in self._backup_transfer_nbytes
        transfer_nbytes = 0
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
            if should_measure:
                transfer_nbytes += param.numel() * param.element_size()
        torch.cuda.synchronize()
        if should_measure:
            self._backup_transfer_nbytes[tag] = transfer_nbytes

    def backup_transfer_nbytes(self, tag: str) -> int:
        return self._backup_transfer_nbytes.get(tag, 0)

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()


class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag, *, track_transfer_nbytes: bool):
        super().__init__(source_getter=source_getter, track_transfer_nbytes=track_transfer_nbytes)
        self._single_tag = single_tag
        # Sanity check for safety
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        ans = dict(self._source_getter())
        ans = {k: v.detach() for k, v in ans.items()}
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        return ans

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        torch.cuda.synchronize()

    def backup_transfer_nbytes(self, tag: str) -> int:
        assert tag == self._single_tag
        return 0

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        torch.cuda.synchronize()


def _compute_hash_dict(tensors: dict[str, torch.Tensor]):
    return {k: _compute_hash_tensor(v) for k, v in tensors.items()}


def _compute_hash_tensor(x: torch.Tensor):
    # Not a real/good hash, but pretty fast
    x = x.contiguous().view(-1).view(torch.uint8)

    alignment = 4
    if (remainder := (x.numel() % alignment)) != 0:
        x = torch.nn.functional.pad(x, (0, alignment - remainder))

    x = x.view(torch.uint32).sum()
    return x.item()
