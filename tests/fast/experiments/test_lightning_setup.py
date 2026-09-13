import json

import torch
from safetensors.torch import save_file
from scripts.download.checkpoints.nemotron_3_5_lightning_30b_a3b import MODEL_NAME, prepare_policy_view


def test_policy_view_preserves_source_and_indexes_every_trainable_tensor(tmp_path):
    source = tmp_path / MODEL_NAME
    source.mkdir()
    config = {"num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"], "num_hidden_layers": 52}
    (source / "config.json").write_text(json.dumps(config))
    tensors = {
        "backbone.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        "backbone.bias": torch.zeros(2, dtype=torch.float32),
        "mtp.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    # A mixed shard ensures exclusion does not accidentally drop backbone weights.
    save_file(tensors, source / "model.safetensors")
    save_file({"mtp.extra": torch.ones(1)}, source / "model-mtp.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": dict(dict.fromkeys(tensors, "model.safetensors"), **{"mtp.extra": "model-mtp.safetensors"})}
        )
    )
    (source / "tokenizer.json").write_text("{}")
    original_files = {path.name: path.read_bytes() for path in source.iterdir()}

    prepare_policy_view(tmp_path)
    prepare_policy_view(tmp_path)

    policy = tmp_path / f"{MODEL_NAME}-policy"
    policy_config = json.loads((policy / "config.json").read_text())
    assert policy_config == dict(config, num_nextn_predict_layers=0, mtp_layers_block_type=[])
    index = json.loads((policy / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"backbone.weight", "backbone.bias"}
    assert index["metadata"]["total_size"] == 20
    assert (policy / "model.safetensors").is_symlink()
    assert (policy / "model.safetensors").resolve() == source / "model.safetensors"
    assert not (policy / "model-mtp.safetensors").exists()
    assert (policy / "tokenizer.json").read_text() == "{}"
    assert (policy / ".download_complete").is_file()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original_files
