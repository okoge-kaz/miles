from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp


def load_common_state(checkpoint_dir: str | Path) -> dict:
    """Read shared state from legacy common.pt or the current distributed format."""
    checkpoint_dir = Path(checkpoint_dir)
    legacy = checkpoint_dir / "common.pt"
    if legacy.is_file():
        return torch.load(legacy, map_location="cpu", weights_only=False)

    reader = dist_cp.FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()
    common_keys = [
        key
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("common_state/") and isinstance(value, dist_cp.metadata.BytesStorageMetadata)
    ]
    if len(common_keys) != 1:
        raise ValueError(f"Expected one distributed common state in {checkpoint_dir}, found {common_keys}")
    key = common_keys[0]
    state = {key: None}
    dist_cp.load(
        state,
        storage_reader=reader,
        planner=dist_cp.DefaultLoadPlanner(flatten_state_dict=False, flatten_sharded_tensors=False),
        no_dist=True,
    )
    common = state[key]
    # MCore stores ShardedObjects as a list of their data at the unique shard key.
    if isinstance(common, list) and len(common) == 1:
        common = common[0]
    if not isinstance(common, dict):
        raise ValueError(f"Distributed common state is not a dictionary: {checkpoint_dir}")
    return common
