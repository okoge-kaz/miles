from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed.checkpoint as dist_cp

from miles.utils.torch_dist_checkpoint import load_common_state


@pytest.mark.parametrize("legacy", [False, True])
def test_load_common_state_preserves_args_and_iteration_without_model_weights(tmp_path, legacy):
    common = {"args": Namespace(num_layers=36, hidden_size=2560), "iteration": 7, "checkpoint_version": 3.0}
    if legacy:
        torch.save(common, tmp_path / "common.pt")
    else:
        dist_cp.save(
            {"common_state/shard_0_1": [common], "model.weight": torch.ones(5)},
            checkpoint_id=tmp_path,
            planner=dist_cp.DefaultSavePlanner(flatten_state_dict=False),
            no_dist=True,
        )
    assert load_common_state(tmp_path) == common


def test_missing_common_state_is_rejected(tmp_path):
    dist_cp.save({"model.weight": torch.ones(5)}, checkpoint_id=tmp_path, no_dist=True)
    with pytest.raises(ValueError, match="Expected one distributed common state"):
        load_common_state(tmp_path)
