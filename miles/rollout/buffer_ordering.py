import heapq
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from miles.rollout.filter_hub.common_filters import GroupWeightVersionStats


@dataclass(frozen=True)
class BufferedGroup:
    sequence: int
    entry: Any
    stats: GroupWeightVersionStats


class OrderedGroupBuffer:
    def __init__(self, order: str = "fifo"):
        if order not in ("fifo", "oldest-policy-first"):
            raise ValueError(f"Unknown data buffer order: {order}")
        self.order = order
        self._fifo = deque()
        self._heap = []
        self._sequence = 0
        self._versions = Counter()
        self._oldest = []
        self._version_sum = 0
        self._versioned_groups = 0

    def push(self, entry, stats: GroupWeightVersionStats) -> None:
        value = BufferedGroup(self._sequence, entry, stats)
        self._sequence += 1
        if self.order == "fifo":
            self._fifo.append(value)
        else:
            heapq.heappush(self._heap, (*self._key(value), value))
        version = stats.oldest_version
        if version is not None:
            if not self._versions[version]:
                heapq.heappush(self._oldest, version)
            self._versions[version] += 1
            self._version_sum += version
            self._versioned_groups += 1
            if len(self._oldest) > 2 * len(self._versions) + 64:
                self._oldest = list(self._versions)
                heapq.heapify(self._oldest)

    @staticmethod
    def _key(value):
        version = value.stats.oldest_version
        return (version is None, version if version is not None else 0, value.sequence)

    def pop(self) -> BufferedGroup:
        value = self._fifo.popleft() if self.order == "fifo" else heapq.heappop(self._heap)[-1]
        version = value.stats.oldest_version
        if version is not None:
            self._versions[version] -= 1
            self._version_sum -= version
            self._versioned_groups -= 1
            if not self._versions[version]:
                del self._versions[version]
        return value

    def snapshot(self) -> tuple[BufferedGroup, ...]:
        # Snapshot cost is paid only when saving. Heap ordering is frozen into disk order.
        return tuple(self._fifo) if self.order == "fifo" else tuple(v[-1] for v in sorted(self._heap))

    def lag_summary(self, current_version: int | None) -> dict[str, float]:
        if current_version is None or not self._versions:
            return {}
        while self._oldest and self._oldest[0] not in self._versions:
            heapq.heappop(self._oldest)
        return {
            "buffer_avg_staleness": current_version - self._version_sum / self._versioned_groups,
            "buffer_max_staleness": float(current_version - self._oldest[0]),
        }

    def __len__(self) -> int:
        return len(self._fifo) if self.order == "fifo" else len(self._heap)
